# Vendored upstream pin

本目录是 [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) 的 **in-tree vendor**（非 git submodule），供 lake P4 KV Pool 复用（见 [#20](https://github.com/chengda-wu/lake/issues/20)、`docs/research/3rdparty-reference.md`「代码级复用策略」B）。

| crate | 上游路径 | 许可 |
|-------|----------|------|
| `kvbm-logical` | `lib/kvbm-logical/` | Apache-2.0（见各 crate `LICENSE`） |
| `dynamo-tokens` | `lib/tokens/` | Apache-2.0（见各 crate `LICENSE`） |

## Pin

| 字段 | 值 |
|------|-----|
| Upstream | https://github.com/ai-dynamo/dynamo |
| Commit | `f5b1c1cceaee8374e3e6134f43f8aa1a0a225f9c` |
| 对应本地 submodule | `3rdparty/dynamo` @ 同上 SHA（vendor 时一致） |
| Vendor 日期 | 2026-07-21 |

## 工具链 / MSRV

- vendor crate 使用 **`edition = "2024"`**（上游 let chains）。
- **钉死工具链**：`rust/rust-toolchain.toml` → `1.96.1`（与 `3rdparty/dynamo/rust-toolchain.toml` 对齐）；CI 用 `dtolnay/rust-toolchain@1.96.1`，**不用**滚动 `@stable`。
- 本地若仍是旧 toolchain（无 2024 edition），`cargo build -p kvbm-logical` 会直接失败——`rustup toolchain install 1.96.1`（或按该文件自动安装），勿降 edition 回写上游语义。
- lake 业务 crate 仍可为 `edition = "2021"`；仅 vendor 成员要求 2024。

## CI / clippy

- **业务 crate**：`cargo clippy --workspace --all-targets --exclude kvbm-logical --exclude dynamo-tokens -- -D warnings`。
- **vendor 排除 `-D warnings`**：本树约定业务源码不改；滚动 lint / 新 stable 若 deny 进 vendor，会逼改 fork 或堆 `allow`。排除后 lint 洁净度与「我们的 CI」解耦；vendor 正确性靠下方单测门禁。
- **vendor 单测**：`cargo test -p dynamo-tokens -p kvbm-logical`（约 500，含 proptest）在 P4.1 起作为**每 PR** rust job 门禁，锁「业务源码未改 + 构建接入仍绿」。P4.2 链业务依赖后若嫌慢，可改 label / `schedule` 触发，勿静默删门禁。

## 本树相对上游的改动

### P4.1（构建接入）

`Cargo.toml`：去 workspace 继承、填实版本、改 path；`edition = "2024"`；`dynamo-tokens` → `path = "../dynamo-tokens"`。见下节偏差表。

### P4.2（最小源码改造，#20）

| 项 | 上游 | vendor | 说明 |
|----|------|--------|------|
| `InactiveIndex` + backends | `pub(crate)` | `pub` + crate root re-export | lake controlplane 薄驱动复用，不用 `BlockStore` |
| `mark_present` / `mark_absent` | `pub(crate)` | `pub` | 无 BlockStore 时由 CP 标 presence |
| `LogicalLayoutHandle` | `G1..G4` | `L0..L3` | lake 统一编址；vendor 内原无引用 |
| `EventsManager` | 可选挂 registry | **仍保留模块**；lake **不接线** | 物理删除字段留后续小 PR |
| `check_presence` 注释 | store-shadow | 注明 lake 同锁线性 | 实现未改 |

其余 `src/` 业务逻辑与上游 pin 一致（除上表）。

## Cargo.toml 相对上游的版本/feature 偏差

填实依赖时对齐 **lake workspace 风格** 与上游 crate 意图，而非逐字复制 dynamo workspace 根：

| 依赖 | 上游 | vendor | 说明 |
|------|------|--------|------|
| `tokio` | `=1.48.0` + `full` | `"1"` + `rt-multi-thread/macros/net/signal/sync/time` | 与 lake 根 `tokio` 同风格；显式含 `sync`/`time`，不靠 `tokio-stream` 间接补齐。dev-deps 仍 `features = ["full"]`。 |
| `bytes` | crate 局部 `"1.10"`（覆盖根 `"1.9"`） | `"1.10"` | 对齐上游 crate 局部约束（曾误填根值 1.9，已改）。semver 解析结果通常与 lock 一致。 |
| `tracing-subscriber` | workspace `"0.3"`（dev） | **省略** | 上游/vendor 源码均无引用，属死依赖；曾误填 `"0.1"`（像抄了 `tracing` 版本号），已删除而非改成 0.3。 |

其余填实版本（`dashmap`/`parking_lot`/`prometheus`/…）与上游 workspace 根定义对齐。re-vendor 时保留上表策略，勿盲目改回 `=1.48.0`+`full`。

## Re-vendor 约定

1. 更新 `3rdparty/dynamo` 到目标 commit（或从上游检出同等树）。
2. 同步拷贝 `lib/kvbm-logical` → `rust/vendor/kvbm-logical`、`lib/tokens` → `rust/vendor/dynamo-tokens`（保留 lake 侧 `Cargo.toml` 填实版本与 path 依赖；冲突时以「能编过 + 单测绿」为准手工合并）。
3. 刷新本文件的 Commit / Vendor 日期；`LICENSE` 若上游变更一并更新。
4. `cd rust && cargo test -p dynamo-tokens -p kvbm-logical` + `cargo fmt --check` + 业务 crate clippy（`--exclude` vendor，见上节）全绿后再提交。


## LICENSE 说明

各 crate 与 `rust/vendor/LICENSE` 使用**纯 Apache-2.0 全文**（附 NVIDIA 版权说明）。
已去掉上游根 `LICENSE` 开头针对 `lib/llm/tests/data/deepseek-v3.2` 的 NOTICE——vendor
树不含该测试数据，避免误导。

上游 `lib/kvbm-logical/AGENTS.md`（指向 Claude Code 规则的 symlink）不纳入 vendor，
与 lake「忽略 submodule 自带 `.claude`/agent 规则」约定一致。
