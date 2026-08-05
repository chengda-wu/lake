package router

// P7.6(B1)亲和选路单测:镜像 L0 得分偏好、护栏降级、HRW 确定性与最小迁移、
// in-flight 记账。harness 级 RR/亲和对照在 bench_p76_test.go。

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"testing"
	"time"

	lakepb "github.com/chengda-wu/lake/go/pb"
)

// newAffinityServer 裸 Server(无 CP/worker 依赖)供纯选路测试。
func newAffinityServer(nodeIDs ...string) *Server {
	return &Server{
		cfg:    Config{NodeRole: string(RoleHybrid), MaxInFlight: 64, AffinityInFlightGuard: 4},
		mirror: NewViewMirror(),
		nodes:  newNodeRegistry(nodeIDs...),
	}
}

// regL0 往镜像注册一块(model "m",L0 落在 nodes)。
func regL0(m *ViewMirror, seq uint64, hash string, nodes ...string) {
	locs := make([]*lakepb.Location, 0, len(nodes))
	for _, n := range nodes {
		locs = append(locs, &lakepb.Location{Tier: lakepb.Tier_L0, NodeId: n, SegmentId: 1})
	}
	if err := m.Apply(&lakepb.ViewUpdate{Seq: seq, Events: []*lakepb.ViewEvent{{
		Kind:      lakepb.ViewEvent_REGISTERED,
		Id:        &lakepb.KVBlockID{ModelId: "m", PoolKind: lakepb.PoolKind_TARGET, BlockHash: []byte(hash)},
		Locations: locs,
	}}}); err != nil {
		panic(err)
	}
}

func TestAffinityPrefersL0Node(t *testing.T) {
	s := newAffinityServer("worker-0", "worker-1", "worker-2")
	hashes := [][]byte{[]byte("h0"), []byte("h1"), []byte("h2")}
	// 整链在 worker-1 的 L0;worker-2 只有首块(得分 1);worker-0 零命中。
	regL0(s.mirror, 1, "h0", "worker-1", "worker-2")
	regL0(s.mirror, 2, "h1", "worker-1")
	regL0(s.mirror, 3, "h2", "worker-1")

	if got := s.pickNodeForRequest("m", hashes, []byte("key-x")); got != "worker-1" {
		t.Fatalf("亲和得分最高应为 worker-1, got %s", got)
	}
}

func TestAffinityGuardDowngradesOverloaded(t *testing.T) {
	s := newAffinityServer("worker-0", "worker-1", "worker-2")
	hashes := [][]byte{[]byte("h0"), []byte("h1")}
	regL0(s.mirror, 1, "h0", "worker-1", "worker-2")
	regL0(s.mirror, 2, "h1", "worker-1") // worker-1 得分 2,worker-2 得分 1

	// 得分最高者 in-flight 打满护栏(4)→ 降级到次优 worker-2。
	for i := 0; i < 4; i++ {
		s.nodes.incInFlight("worker-1")
	}
	if got := s.pickNodeForRequest("m", hashes, []byte("key-x")); got != "worker-2" {
		t.Fatalf("护栏降级应选次优 worker-2, got %s", got)
	}

	// 全部命中节点过载 → 冷路径加权 HRW(worker-0 在途 0,权重 1)。
	for i := 0; i < 4; i++ {
		s.nodes.incInFlight("worker-2")
	}
	if got := s.pickNodeForRequest("m", hashes, []byte("key-x")); got != "worker-0" {
		t.Fatalf("全过载应落冷路径(在途 0 的 worker-0 权重最高), got %s", got)
	}
}

func TestAffinityColdHRWDeterministic(t *testing.T) {
	s := newAffinityServer("worker-0", "worker-1", "worker-2")
	// 空镜像(无命中)→ 全部走冷路径;同键同节点(纯函数)。
	for i := 0; i < 8; i++ {
		key := []byte(fmt.Sprintf("key-%d", i))
		first := s.pickNodeForRequest("m", nil, key)
		for j := 0; j < 4; j++ {
			if got := s.pickNodeForRequest("m", nil, key); got != first {
				t.Fatalf("key %q 第 %d 次选路漂移: %s → %s", key, j, first, got)
			}
		}
	}
	// 键空间应散布到 ≥2 节点(HRW 非恒等)。
	seen := map[string]bool{}
	for i := 0; i < 64; i++ {
		seen[s.pickNodeForRequest("m", nil, []byte(fmt.Sprintf("spread-%d", i)))] = true
	}
	if len(seen) < 2 {
		t.Fatalf("64 键只落在 %d 个节点,HRW 散布异常", len(seen))
	}
}

func TestAffinityHRWMinimalMigration(t *testing.T) {
	s := newAffinityServer("worker-0", "worker-1", "worker-2")
	rnd := rand.New(rand.NewSource(7))
	const keys = 2000
	before := make(map[string]string, keys)
	for i := 0; i < keys; i++ {
		k := string([]byte{byte(i), byte(i >> 8), byte(rnd.Intn(256))})
		before[k] = s.pickNodeForRequest("m", nil, []byte(k))
	}
	s.nodes.add("worker-3")
	moved := 0
	for k, prev := range before {
		if got := s.pickNodeForRequest("m", nil, []byte(k)); got != prev {
			moved++
		}
	}
	// HRW 性质:迁移 ≈ 1/(N+1) = 25%(3→4);宽界 35% 防实现退化成取模。
	if moved > keys*35/100 {
		t.Fatalf("加节点迁移 %d/%d (%.1f%%),超 HRW 宽界 35%%(期望 ≈25%%)",
			moved, keys, float64(moved)/keys*100)
	}
	if moved == 0 {
		t.Fatal("加节点零迁移——新节点未进入 HRW 候选")
	}
}

// TestFnv1a64CrossLanguageVectors 钉死与 Rust 权威(placement.rs::hrw_score)
// 的字节布局一致性:原语层 fnv1a 经 std lib 核对("lake" 向量与 Rust
// fnv1a64_matches_go_reference_vectors 同值),组合层钉绝对值防回归。
func TestFnv1a64CrossLanguageVectors(t *testing.T) {
	if got := fnv1a64([]byte("anchor-0"), "n0"); got != 0xc8e4e7b063e4ef7f {
		t.Fatalf("fnv1a64(anchor-0,n0) = %#x, want 0xc8e4e7b063e4ef7f", got)
	}
	if got := fnv1a64([]byte("key!"), "worker-1"); got != 0xd485e7e9b3d80a5f {
		t.Fatalf("fnv1a64(key!,worker-1) = %#x, want 0xd485e7e9b3d80a5f", got)
	}
}

// TestInFlightAccounting 全链路记账:请求执行中在途=1,完成后归零。
func TestInFlightAccounting(t *testing.T) {
	h := newP76Harness(t, 2, false)
	started := make(chan string, 1)
	release := make(chan struct{})
	var once sync.Once
	h.srv.worker = fakeWorker{generate: func(_ context.Context, req *lakepb.GenerateRequest) (*lakepb.GenerateResponse, error) {
		once.Do(func() { started <- req.GetRequesterNodeId() })
		<-release
		hashes := ChainBlockHashes(req.GetPromptTokens(), BlockSize)
		h.cp.registerBlocks(req.GetModelId(), hashes, req.GetRequesterNodeId())
		return &lakepb.GenerateResponse{RequestId: req.GetRequestId(), Mode: req.GetExecMode()}, nil
	}}

	body := `{"model":"m","messages":[{"role":"user","content":"inflight 记账探针请求"}],"max_tokens":2}`
	done := make(chan int, 1)
	go func() {
		rec := postChatP76(t, h.srv, body, "")
		done <- rec.Code
	}()

	var node string
	select {
	case node = <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker 未收到请求")
	}
	waitFor(t, 2*time.Second, "在途计数升 1", func() bool {
		return h.srv.nodes.inFlight(node) == 1
	})
	close(release)
	if code := <-done; code != http.StatusOK {
		t.Fatalf("status = %d", code)
	}
	waitFor(t, 2*time.Second, "在途计数归零", func() bool {
		return h.srv.nodes.inFlight(node) == 0
	})
}
