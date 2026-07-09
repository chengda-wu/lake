# CLAUDE.md

本文件给 Claude Code（及任何接手的开发者/AI）提供 lake 仓库的工作指引。
先读 [`docs/00-plan.md`](docs/00-plan.md) 了解全貌与当前阶段，再读本文件了解**怎么做**。

## 这是什么仓库

lake 是一个**探索性**仓库，目标是设计并验证一套**彻底的存算分离**大模型推理系统。
所有有状态物（权重、KV cache、调度队列）从算力路径剥离，算力节点可随时销毁/拉起。

**当前阶段**：P0（特性设计）基本完成 → 即将进入 P1（架构设计）→ P2（模块划分 + 技术选型）。
仓库现在是**设计文档为主**，配一份早期单进程 Python 原型（`src/`，将被三语言子项目取代）。

## 文档结构（务必遵守分类）

```
docs/
├── 00-plan.md          # 路线图主线（阶段、任务、状态），改阶段进度只动这里
├── features/           # P0：做什么（特性 / SLO / 非功能）
├── architecture/       # P1：怎么搭
└── research/           # 相关工作
```

- 特性 → `features/`，架构 → `architecture/`，文献 → `research/`。不要把新文档堆在 `docs/` 根。
- 文档间用**相对路径**链接（如 `[../features/features.md]`），跨目录引用要带对路径。
- 改动设计后，检查所有相关文档的内部链接是否仍有效（grep 旧路径）。

## 已确立的设计原则（改动前必须遵守，如要推翻需与用户确认）

1. **混合执行模式**：请求不固定走 P→D。Router 按 `f(请求, 集群状态) → (模式, 节点)` 在三种模式间逐请求选路：
   - **PD 分离** / **混部**（同节点完成）/ **D-direct**（前缀 KV 已被存储池放置在执行节点 HBM，零/极小传输直跳）。
   - **关键区分**：Pool 命中（前缀 KV 在分布式池，省重算但需传输）≠ 本地命中（前缀 KV 已被存储池放置在某执行节点 HBM，可 D-direct，零/极小传输）。
   - 混部与 D-direct **不违背存算分离**——KV 仍归存储池权威、写回 Pool；分离的是"存储与计算"，不是"P 与 D 必须分机"。
   - 详见 [`docs/features/features.md`](docs/features/features.md) "执行模式"节。

2. **失败处理不设降级链**：执行失败（故障/超时）→ 触发 F4 故障恢复重新路由 → Router 依最新状态重选模选点。
   不写 mode-to-mode 的预设 fallback 阶梯。模式选择是纯函数，失败即重跑该函数。

3. **过载控制归 gateway，推理系统只管执行**：
   - 限并发、拒请求、按优先级丢弃、准入控制 → gateway / 外部控制面。
   - Worker **不得**为保 SLO 自行降 batch size 或丢请求。
   - 推理系统的过载职责是**上报信号**（队列长度、in-flight、剩余容量），供 gateway 决策。
   - 可用性 SLO 仅针对"已准入、非过载"请求；过载拒绝不计推理系统失败率。
   - 详见 [`docs/features/slo.md`](docs/features/slo.md) "过载控制"节、[`docs/features/nonfunctional.md`](docs/features/nonfunctional.md)。

4. **存储池是长期存续、模型无关的独立基础设施，并统一管理全部分层**（F11 + F3）：
   - 同一池同时承载多个 `(model_id, revision)` 的 KV/权重，可对接任意模型；模型上下线与池生命周期解耦。
   - 池不解释张量布局，按**不透明字节块**存取 → 接入新模型只需注册 `model_id` 命名空间。
   - **统一编址 L0–L4**：GPU HBM(L0) → 主机 RAM(L1) → 本地 NVMe(L2) → 远端内存池(L3) → 对象存储(L4) 全部由存储池统一管理放置/驱逐/副本/冷热/生命周期。计算节点不拥有任何内存——HBM/RAM/NVMe 是池的物理载体，不是 worker 私有状态。因此不存在计算层私有的易失缓存（如 APC），所有 KV 位置均为存储池权威元数据。
   - **冷热与生命周期**：L0/L1 做缓存副本、L2/L3/L4 间按移动、L4 永久权威；冷热按"引用数>0 冻结 + 热度分(LFU-Aging) + 前缀亲和"判定；迁移主动为主（按热度 promotion/demotion + L0 预放置）+ 被动兜底（读 miss 回填/写满驱逐）；迁移/GC/碎片整理共享后台带宽池（<10%）。
   - **放置与 batch 职责边界（方案 Z）**：存储池按热度主动预放置 KV 到 HBM 并发布位置视图；调度器读视图组 batch（本地命中优先→D-direct，缺失补拉），不反向指挥放置。单向耦合。
   - 具备：按模型配额（软/硬 + 借用）、池容量扩缩（一致性哈希最小迁移）、GC（冷块/孤儿块/级联删/旧 revision）、碎片整理（逻辑共置 + 物理压实，后台节流可暂停）。
   - 触硬配额 → 池返回**写入背压信号**上传，请求级 shedding 仍归 gateway。
   - 详见 [`docs/architecture/kv-cache-pool.md`](docs/architecture/kv-cache-pool.md)。

5. **技术选型（已定，P2 落地）**：

   | 层 | 语言 |
   |----|------|
   | 存储层（KV Pool / Weight Cache / Tiered Store） | Rust |
   | 控制面（Router / Scheduler / 元数据） | Go |
   | 计算层（Prefill / Decode / Draft） | Python + Triton |
   | 元数据 | etcd · SSOT 用 S3/MinIO · 跨语言 RPC 用 gRPC+Protobuf（大块 KV 走 RDMA 旁路） |

   目录划分见 [`docs/00-plan.md`](docs/00-plan.md) P2 节。`src/` 是早期单进程原型，**不要**在三语言子项目就位后继续往 `src/` 加功能。

## SLO 是架构硬约束

SLO 数值是 draft（待 P7 校准），但**约束关系是硬的**：TTFT/ITL/冷启动等目标倒逼架构取舍（如 D-direct 模式选择开销 < 5ms，否则吃掉本地命中省传输的收益）。
改架构设计时，对照 [`docs/features/slo.md`](docs/features/slo.md) 检查是否仍满足 SLO 预算。

## 工作约定

- **职责边界优先**：遇到"某能力归谁"的问题，先判断是否越界（推理系统只管执行 vs. gateway 管过载；F4 管故障恢复 vs. Router 管重决策）。宁可少做，不要越界。
- **文档先行**：新增能力先写进 `features/` 或 `architecture/`，再考虑代码。代码服务于验证文档里的假设。
- **保持文档一致**：改了一处设计，同步更新所有引用它的文档（features ↔ architecture ↔ plan），并在 `00-plan.md` 勾选状态。
- **链接要活**：移动文档用 `git mv` 保留历史；移动后 grep 修复所有内部链接。
- **不自行决定技术选型**：语言/框架/存储选型由用户定。新增依赖或换语言前问用户。
- **职责外的事不擅自做**：过载 shedding、鉴权计费等外部控制面职责，不在推理系统内实现。

## Git 约定

- 主分支 `main`，直接提交推送（仓库当前无 PR 流程）。
- 提交信息中文，开头用 `docs(P0):` / `feat:` / `fix:` 等前缀，结尾附 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 推送走 SSH：`git@github.com:chengda-wu/lake.git`。本地已配 `origin`。
- 用户尚未配置全局 git 身份，仓库本地配置为 `witcher` / `witcher@users.noreply.github.com`，提交时用 `git -c commit.gpgsign=false commit`。

## 原型运行

早期单进程原型（验证前缀复用逻辑，非生产）：

```bash
python3 -m src
# 期望输出：两个请求共享前缀，第二个命中 KV Pool（reused=3, prefill=1）
```

三语言子项目就位后此原型将被取代，勿在此基础上扩展。
