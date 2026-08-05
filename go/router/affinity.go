package router

// P7.6(B1):亲和选路——本地命中闭环的调度半边(docs/architecture/scheduling.md
// 「缓存命中感知调度」)。设计参考(CLAUDE.md reference 强制查阅):
//   - Dynamo lib/kv-router/src/protocols.rs::WorkerSelectionResult
//     (overlap_blocks/effective_overlap_blocks):KV-aware 路由把命中量化为一等
//     输入。借鉴:热路径按「本地前缀得分 = 命中前缀中 LocalHit 连续块数」打分。
//     差异:Dynamo 的 overlap 来自 engine 私有 KV 索引(events 旁路、最终一致);
//     lake 读存储池权威位置视图的镜像(强一致源),且得分是「真·本机 HBM 命中」
//     (D-direct 零传输),不是 routing-to-history 近似。
//   - SGLang sgl-model-gateway/src/policies/cache_aware.rs::CacheAwarePolicy:
//     失衡时退最短队列。借鉴:亲和得分最高者 in-flight 超护栏则降级次优,
//     全部过载落冷路径——命中收益不能无界压垮单点。差异:SGLang 的近似树按
//     请求文本历史猜亲和;lake 的负载信号是本注册表 per-node in-flight 原子
//     计数(即时、精确),非最短队列近似。
//   - Dynamo protocols.rs::stable_routing_id 注释(HRW rendezvous 使 cache
//     assignment 在 worker churn 下最小迁移):冷路径用 rendezvous(HRW)一致性
//     哈希,同键同节点、加节点最小迁移;权重 1/(1+inflight) 做负载倾斜。

import (
	"encoding/binary"
	"hash/fnv"
	"sort"

	lakepb "github.com/chengda-wu/lake/go/pb"
)

// defaultAffinityGuard 亲和路径单节点 in-flight 护栏默认值:命中得分最高的
// 节点在途 ≥ 护栏则降级次优;全部过载落冷路径(加权 HRW 天然倾向低载节点)。
// 原型默认 8——远小于真实 batch 容量,仅防单点饥饿;P7 校准前不做自适应。
const defaultAffinityGuard = 8

// anchorDepth 冷路径路由键的有界前缀深度:深度 1 会被全局系统前缀打成热点
// (所有会话共享前几个 block,同锚点 → 同节点);深度 8 以内大概率含会话
// 分叉块,可锚定会话级亲和;更深不再增加亲和收益,反而受尾部长尾影响。
const anchorDepth = 8

// prefixAnchorHash 有界深度前缀锚点 = hashes[min(len,anchorDepth)-1]。
// 与池侧 B2-a 预放置(rust/controlplane/src/placement.rs::preplace_on_register
// 的链锚点)同源:同锚点 ⇒ 负载均衡时 Router 冷路径与池侧预放置选同一家节点。
func prefixAnchorHash(hashes [][]byte) []byte {
	if len(hashes) == 0 {
		return nil
	}
	i := len(hashes)
	if i > anchorDepth {
		i = anchorDepth
	}
	return hashes[i-1]
}

// fnv1a64 FNV-1a 64(key_len_u64_le ‖ key ‖ node)——字节布局与 Rust 权威
// rust/controlplane/src/placement.rs::hrw_score 完全一致(同测试向量
// fnv1a64("lake") = 0xCE697C88B1B1D97E),保证池侧 HRW 预放置的家节点与
// 本函数在负载全 0 时一致。标准库 fnv.New64a 常数相同,仅补长度前缀。
func fnv1a64(key []byte, node string) uint64 {
	h := fnv.New64a()
	var lb [8]byte
	binary.LittleEndian.PutUint64(lb[:], uint64(len(key)))
	_, _ = h.Write(lb[:])
	_, _ = h.Write(key)
	_, _ = h.Write([]byte(node))
	return h.Sum64()
}

// pickNodeForRequest P7.6(B1)两段式亲和选路(纯内存,µs 级,守住 D-direct
// 模式选择 <5ms 的 SLO 预算):
//
//  1. 热路径——逐 ready 节点在镜像上算本地前缀得分(命中前缀中在请求方本机
//     L0 的连续块数,PrefixLookup 首个 miss 截断 ⇒ 天然连续前缀语义);得分
//     >0 的最高者胜出,其 in-flight ≥ 护栏则降级次优,全部过载落冷路径。
//     平局按节点 id 序(与 readyIDs 排序一致,确定性)。
//  2. 冷路径——负载权重 rendezvous(HRW):score = fnv1a64(routingKey‖node)
//     × 1/(1+inflight),取 max。同键同节点(确定性);加节点最小迁移(HRW
//     性质);负载越高权重越低(倾斜但不钉死,避免冷启动全 0 时退化为纯负载
//     轮转而丢失键亲和)。
//
// 抢占重跑沿用「Submit 前选一次」(P6.4 现状):被抢占者重跑不重选节点,
// retry 重选是未来工作(需与 F4 重路由统一设计)。
func (s *Server) pickNodeForRequest(model string, hashes [][]byte, routingKey []byte) string {
	if s.nodes == nil {
		return "worker-0"
	}
	ids := s.nodes.readyIDs()
	if len(ids) == 0 {
		return "worker-0"
	}
	if len(ids) == 1 {
		return ids[0]
	}
	guard := int64(s.cfg.AffinityInFlightGuard)
	if guard <= 0 {
		guard = defaultAffinityGuard
	}

	// 热路径:镜像逐节点查 L0 命中(纯内存;镜像 miss 不查权威——选路热路径
	// 不打 RPC,权威回退只在 prefixHint 档做一次)。
	type cand struct {
		id    string
		score int
	}
	var cands []cand
	if s.mirror != nil {
		for _, id := range ids {
			blocks, _, _ := s.mirror.PrefixLookup(model, "", lakepb.PoolKind_TARGET, hashes, id)
			score := 0
			for _, b := range blocks {
				if !b.LocalHit {
					break
				}
				score++
			}
			if score > 0 {
				cands = append(cands, cand{id: id, score: score})
			}
		}
	}
	if len(cands) > 0 {
		sort.SliceStable(cands, func(i, j int) bool {
			if cands[i].score != cands[j].score {
				return cands[i].score > cands[j].score
			}
			return cands[i].id < cands[j].id
		})
		for _, c := range cands {
			if s.nodes.inFlight(c.id) < guard {
				return c.id
			}
		}
		// 全部命中节点过载 → 冷路径(加权 HRW 会自然绕开高载)。
	}

	// 冷路径:加权 HRW(键亲和 × 负载倾斜)。
	var best string
	var bestScore float64
	for _, id := range ids {
		w := 1.0 / float64(1+s.nodes.inFlight(id))
		score := float64(fnv1a64(routingKey, id)) * w
		if best == "" || score > bestScore {
			best, bestScore = id, score
		}
	}
	return best
}
