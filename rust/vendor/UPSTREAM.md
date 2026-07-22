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

- vendor crate 使用 **`edition = "2024"`**（上游 let chains），需要较新的 **Rust stable**（CI：`dtolnay/rust-toolchain@stable`）。
- 本地若仍是旧 toolchain（无 2024 edition），`cargo build -p kvbm-logical` 会直接失败——先 `rustup update stable`，勿降 edition 回写上游语义。
- lake 业务 crate 仍可为 `edition = "2021"`；仅 vendor 成员要求 2024。

## 本树相对上游的改动（P4.1 / GitHub PR #21）

**业务源码未改**（`src/` / tests / benches 与 `3rdparty/dynamo` 对应路径 `diff` 应为空）。  
`Cargo.toml` 是**构建接入层**：去 workspace 继承、填实版本、改 path——其中若干约束相对上游有意偏离，见下节，勿读成「Cargo.toml 也字节级一致」。

- `edition = "2024"`（上游 let chains 需要）。
- `dynamo-tokens` → `path = "../dynamo-tokens"`。

源码级改造（`InactiveIndex` 提 `pub`、拆 `EventsManager`、tier G1–G4→L0–L3、`check_presence` 线性一致等）**不在本 pin**，随 #20 后续切片（P4.2+）按需改。

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
4. `cd rust && cargo test -p dynamo-tokens -p kvbm-logical` + workspace fmt/clippy 全绿后再提交。


## LICENSE 说明

各 crate 与 `rust/vendor/LICENSE` 使用**纯 Apache-2.0 全文**（附 NVIDIA 版权说明）。
已去掉上游根 `LICENSE` 开头针对 `lib/llm/tests/data/deepseek-v3.2` 的 NOTICE——vendor
树不含该测试数据，避免误导。

上游 `lib/kvbm-logical/AGENTS.md`（指向 Claude Code 规则的 symlink）不纳入 vendor，
与 lake「忽略 submodule 自带 `.claude`/agent 规则」约定一致。
