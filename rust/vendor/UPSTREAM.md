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

## 本树相对上游的改动（P4.1 / PR #21）

**仅构建接入，业务源码未改**（与 `3rdparty/dynamo` 对应路径 `diff` 应为空）：

- `Cargo.toml`：去掉 dynamo workspace 继承，填实依赖版本；`dynamo-tokens` 改为 `path = "../dynamo-tokens"`。
- `edition = "2024"`（上游 let chains 需要）。

源码级改造（`InactiveIndex` 提 `pub`、拆 `EventsManager`、tier G1–G4→L0–L3、`check_presence` 线性一致等）**不在本 pin**，随 #20 后续切片（P4.2+）按需改。

## Re-vendor 约定

1. 更新 `3rdparty/dynamo` 到目标 commit（或从上游检出同等树）。
2. 同步拷贝 `lib/kvbm-logical` → `rust/vendor/kvbm-logical`、`lib/tokens` → `rust/vendor/dynamo-tokens`（保留 lake 侧 `Cargo.toml` 填实版本与 path 依赖；冲突时以「能编过 + 单测绿」为准手工合并）。
3. 刷新本文件的 Commit / Vendor 日期；`LICENSE` 若上游变更一并更新。
4. `cd rust && cargo test -p dynamo-tokens -p kvbm-logical` + workspace fmt/clippy 全绿后再提交。
