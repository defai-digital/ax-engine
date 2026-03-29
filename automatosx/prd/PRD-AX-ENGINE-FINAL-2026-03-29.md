# PRD：AX Engine 最終整合 PRD

**Date:** 2026-03-29  
**Status:** Finalized / Active Implementation Baseline  
**Role:** 單一執行來源文件（source of truth）  

## 1. 目的

這份文件整合下列內容為單一最終 PRD：

- 現有 `automatosx/prd/` 內所有 AX Engine PRD / PLAN
- `Qwen 3.5 Prefill Performance Bottleneck Analysis`
- 目前程式碼狀態所反映的最新事實

本文件的目標不是把舊 PRD 原封不動拼接，而是：

- 保留仍然成立的決策
- 修正已被新實作推翻的敘述
- 將已完成、已過時、仍待完成的工作重新分層
- 定義接下來唯一應該執行的優先順序

---

## 2. 吸收與取代的文件

本文件吸收下列 PRD / PLAN 的有效內容：

- `PRD-AX-ENGINE-v2.0.md`
- `PRD-AX-ENGINE-v3.md`
- `PRD-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PLAN-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PRD-AX-ENGINE-PERF.md`
- `PRD-AX-ENGINE-PERF-RECOVERY.md`
- `PLAN-AX-ENGINE-PERF-RECOVERY-2026-03-22.md`
- `PRD-KERNEL-PERF-IMPROVEMENT-2026-03-22.md`
- `PRD-MATMUL-PARITY-2026-03-26.md`
- `PRD-GPU-PROFILING-ATTRIBUTION.md`
- `PRD-HEURISTIC-POLICY-AND-QWEN35-RUNTIME-2026-03-28.md`
- `PRD-PREFILL-GAP-CLOSURE-VS-LLAMA-CPP-AND-MISTRAL-RS-2026-03-29.md`
- `PRD-QWEN35-SINGLE-CB-DECODE.md`
- `PRD-QWEN35-RECURRENT-OWNERSHIP-AND-PREFILL-SHAPE-2026-03-29.md`
- `PRD-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`
- `PLAN-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`
- `PRD-QWEN35-PREFILL-DTYPE-AUDIT-AND-RECOVERY-2026-03-29.md`
- `PRD-QWEN35-RECURRENT-RUNTIME-CONTRACT-REWRITE-2026-03-29.md`
- `PRD-MOE-SUPPORT-2026-03-26.md`
- `PRD-REFACTOR-MODEL-DEDUP.md`
- `PRD-INFERENCE-ROUTING-2026-03-27.md`
- `qwen35-focus-support.md`

從這份文件起：

- 本文件是主文件
- 舊文件改為背景、證據、或局部設計參考
- 若舊文件與本文件衝突，以本文件為準

---

## 2.1 文件狀態總表

### 已完成並吸收

- `PRD-QWEN35-RECURRENT-OWNERSHIP-AND-PREFILL-SHAPE-2026-03-29.md`
- `PRD-PREFILL-GAP-CLOSURE-VS-LLAMA-CPP-AND-MISTRAL-RS-2026-03-29.md`
- heuristic-first policy / autotune bridge / prefill gap attribution / Qwen3.5 fast-path guardrail
- recurrent hot-path scratch reuse 第一輪
- unified recurrent batch CPU workspace reuse 已落地
- unified recurrent projection CPU norm scratch reuse 與 non-SSM output scratch reuse 已落地
- unified recurrent projection metadata 已改為 fixed-size stack arrays，不再為 4-way projection bookkeeping 建立小型 `Vec`
- batch logits GPU path 第一輪與 correctness tests
- batch logits GPU scratch reuse 已落地
- batch logits rollout gate 已收斂為明確 runtime toggle
- `Qwen3 / Gemma3 / Qwen3.5 / LLaMA` 模組化重構第一輪
- `shared.rs` / `prefill_schedule.rs` 結構拆分第一輪
- `Qwen3.5` partial prefill schedule 執行入口已收斂，不再由 batch runtime 重複組裝 builder/execute 樣板
- `Qwen3.5` 頂層 prefill route plan 已存在，layer routing 不再完全散落在 batch runtime 內
- `build_qwen35_prefill_schedule(...)` 等價頂層 builder 已存在，但目前是 route-first / summary-first 版本
- `build_qwen35_prefill_schedule(...)` 已升級為 top-level full-flow phase schedule，runtime 直接消費 layer phase flow
- `build_qwen35_prefill_schedule(...)` 現在已額外輸出 flattened execution step flow，runtime 直接 iterate builder steps，而不是在 batch hot path 內重新拼 phase 次序
- `Qwen3.5` unified batch hot path 主函式已收斂為 route / phase dispatch，recurrent layer orchestration 已集中成 layer-level helper
- `Qwen3.5` prefill builder 的 recurrent phase flags 已由 runtime 實際消費，不再只是結構欄位
- `Qwen3.5` prefill backend-state-batch auto policy 已改為 backend-owned-first，不再只偏好短 prompt
- `ax-engine-bench` / `prefill_profile` / `prefill_gap` / `profile` / `soak` / `ax-engine-cli` 已輸出或顯示 normalized prefill runtime metadata，不再只依賴 `prefill_plan` 字串做 route 推斷

### 仍是有效 backlog

- `PRD-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`
- `PLAN-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`
- `PRD-QWEN35-PREFILL-DTYPE-AUDIT-AND-RECOVERY-2026-03-29.md`
- `PRD-QWEN35-RECURRENT-RUNTIME-CONTRACT-REWRITE-2026-03-29.md`
- `PRD-AX-ENGINE-PERF-RECOVERY.md`
- `PRD-MATMUL-PARITY-2026-03-26.md`
- `PRD-MOE-SUPPORT-2026-03-26.md`
- `PRD-REFACTOR-MODEL-DEDUP.md`
- `PRD-INFERENCE-ROUTING-2026-03-27.md`

### 已過時但保留歷史背景

- `PRD-QWEN35-SINGLE-CB-DECODE.md`
- `PRD-AX-ENGINE-v2.0.md`
- `PRD-AX-ENGINE-v3.md`
- `PRD-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PLAN-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PRD-AX-ENGINE-PERF.md`
- `PLAN-AX-ENGINE-PERF-RECOVERY-2026-03-22.md`
- `PRD-KERNEL-PERF-IMPROVEMENT-2026-03-22.md`

這個狀態表的用途只有一個：

- 讓之後所有討論都不需要再回頭判斷哪份文件才算主線

---

## 3. 執行摘要

AX Engine 現在的主戰場，已經不是「再補一個單點 kernel 就會追上 llama.cpp」。

真正的問題排序如下：

1. `Qwen3.5` 的 recurrent runtime contract 仍然太 CPU-shaped
2. `Qwen3.5` prefill 缺少顯式 schedule / graph-IR 路徑
3. unified prefill hot path 仍有可避免的 recurrent temp buffer 配置與同步成本
4. dense model prefill 與 `llama.cpp` 的差距，主因更像 execution graph / runtime，而不是再加 profile knob
5. benchmark / prefill-profile / prefill-gap / profile / soak / CLI 的 runtime metadata 已對齊第一輪，但更完整 artifact pipeline 仍可再收斂

這表示 AX 現階段不應把主要時間放在：

- 再加更多 env var / profile knobs
- 再做一輪沒有 runtime cleanup 支撐的 micro-fusion
- 把 Qwen3.5 問題簡化成「單純 GPU vs CPU」二元敘述

本文件的核心結論是：

- `Qwen3.5` decode 單一 command buffer 的方向已大致成立，不再是主瓶頸
- `Qwen3.5` 現在最大瓶頸是 recurrent ownership、scratch lifecycle、prefill schedule、batched logits
- dense prefill gap 的主因是 runtime / graph shape，不再是 profile-first policy
- 量測系統必須報告 route、ownership、command-buffer shape，不能只看 tok/s

---

## 4. 目前已成立的事實

### 4.1 已完成或已大致完成

- `Qwen3.5` decode 單一 command buffer 方向已大致達標，相關舊 PRD 的「main function pending」不再是目前 source of truth
- `Qwen3.5` recurrent ownership / prefill shape 第一輪收斂已完成
- heuristic-first policy、autotune bridge、prefill gap attribution CLI、Qwen3.5 fast-path guardrail 已落地
- Metal parity phase 已大致完成；剩餘問題縮小到少數明確 backlog
- GPU profiling attribution 的第一階段已有明確方向：至少要先把 GPU decode 歸到可用 bucket，而不是全部落在 `Other`
- unified recurrent prefill 的 recurrent projection scratch reuse 已落地，per-layer allocation 已從主熱路徑縮小
- unified recurrent batch layer 已改用 reusable CPU workspace，不再每層新建 recurrent `Vec<f32>` buffers
- unified recurrent projection / non-SSM output 路徑的 CPU temporary buffers 已改走 reusable scratch，不再以 `to_vec()` / `vec![..]` 為每層配置模型
- unified recurrent projection bookkeeping 已改為固定大小陣列，移除每層小型 metadata `Vec` 配置
- `write_all_batch_logits(...)` 已有 GPU batch path、correctness tests、reusable Metal scratch、以及明確 runtime gate
- `Qwen3.5` prefill schedule 已有部分 graph-IR：
  - full-attention schedule builder
  - recurrent tail schedule builder
  - partial schedule execution entrypoints 已集中到 `prefill_schedule/qwen35.rs`
  - top-level prefill route plan 已可輸出每層 full-attention / recurrent route
  - top-level builder 已可輸出 schedule summary
  - runtime 已改由 builder 提供的 route / phase flags 驅動 full-attention 與 recurrent 執行路徑
- model code management 第一輪已完成：
  - `qwen35.rs` 已拆模組
  - `qwen3.rs` 已拆模組
  - `gemma3.rs` 已拆模組
  - `llama.rs` 已拆模組
  - `shared.rs` 與 `prefill_schedule.rs` 已拆成職責子模組

### 4.2 仍未完成

- `Qwen3.5` recurrent state 尚未達到真正的 device-primary / backend-owned-by-default hot path
- unified prefill recurrent 路徑仍未對所有分支完全消除 temp / scratch churn
- 尚未有單一扁平化 full-op graph，把 full-attention op schedule + recurrent core + recurrent tail 全部收成一個跨 layer 的 prefill op graph
- dense prefill graph/runtime convergence 尚未完成
- tooling/runtime alignment 的 bench / prefill-profile / prefill-gap / profile / soak / CLI 第一輪已完成，但 artifact pipeline 尚未完全收斂
- MoE 目前 CPU decode 可用，但 GPU decode 尚未真正接線
- model dedup 仍停在 Phase 2 完成，Phase 3-6 尚未做完
- inference routing 是完整提案，但尚未是已整合功能

---

## 4.3 當前程式碼證據要點

這裡只保留最影響決策的事實，不做逐行 code review。

- `Qwen3.5` prefill 已先嘗試 unified GPU path，而不是直接走舊 serial fallback
- recurrent path 已存在：
  - fused single-slot GPU path
  - GPU QKV handoff path
  - backend-owned slot-buffer path
  - CPU / generic fallback path
- recurrent ownership 已有 `CpuMaterialized` / `BackendOwned` / `Split` 概念，不再是單純 CPU-only 模型
- unified recurrent prefill 仍在 layer loop 中建立 temp Metal buffers
- `LLaMA / Phi` 已有 prefill schedule builder 與 multi-CB execution；`Qwen3.5` 現在已有 top-level full-flow phase schedule、flattened execution step flow、局部 op schedule builder、集中化 execution entrypoints，且 runtime 已直接消費 builder steps，但尚未收斂成單一扁平化 full-op graph
- batch logits 已有 GPU batch projection、correctness tests、reusable scratch、以及明確 runtime gate
- bench/profile/soak/CLI tooling 現在已直接輸出或顯示 `prefill_mode`、`prefill_route_family/detail`、`prefill_attention_route`、`prefill_qkv_plan`、`prefill_split_rope_append`

這些事實的直接含義是：

- `Qwen3.5` 的問題已不再能被簡化成「沒有 GPU path」
- 但也還不能宣稱已經完成 device-primary runtime

---

## 5. 修正後的問題陳述

### 5.1 Qwen3.5 prefill 的正確問題描述

以下是目前應採用的版本。

#### A. recurrent layer 仍然可能走高度碎片化的 fallback-heavy 路徑

`Qwen3.5` 的 recurrent layer 在最差情況下，仍可能出現：

- CB1：norm + input projections
- GPU readback
- CPU-side matmul / recurrent fallback 或 backend-shaped recurrent path
- CB2：residual + FFN
- 額外 readback / alias / handoff

但這不應再被描述成「每個 recurrent layer 永遠固定有 6 個 CPU-GPU sync points」。

原因是目前程式碼已經存在：

- fused single-slot GPU recurrent path
- GPU QKV handoff path
- backend-owned slot-buffer path

所以更精確的說法是：

- 舊的 worst-case recurrent 路徑仍然存在
- 但它已不是唯一實作
- 現在真正的工作是把它從常見路徑降級成窄 fallback

#### B. recurrent state 不是純 CPU-owned，但也還不是正確的 device-primary contract

舊敘述「`ModelKv::Qwen35` state 是 CPU-owned」現在過度簡化。

目前實際狀態是：

- KV 層已經有 `CpuMaterialized` / `BackendOwned` ownership tracking
- Metal backend 已有 reusable slot buffers
- recurrent path 已能在某些情況下維持 backend-owned carry-over

真正問題不是「完全沒有 GPU ownership」；而是：

- ownership 還是 hybrid
- CPU materialization 邊界仍太寬
- slot/native contract 還沒有成為預設 hot path

#### C. recurrent temp buffer allocation 已有第一輪修正，但仍是高優先級 hot-path 問題

目前 unified prefill recurrent path 已開始使用 reusable scratch，但尚未在所有 recurrent 分支徹底消除 temp / scratch churn。這仍然是高 ROI 問題。

#### D. backend fallback recurrent path 仍然是 CPU-shaped

在 fallback 或 backend generic path 裡，仍可看到：

- token-by-token sequential update
- `repeat_heads(...)` 類 CPU heap allocation
- CPU transpose in `dequant_matmul_token_major(...)`

這些成本仍然真實存在，但它們應被定位為：

- fallback / generic path 的 ceiling
- 不是最終想保留的主路徑

#### E. Qwen3.5 已有 top-level full-flow phase schedule，但尚未完成扁平化 full-op graph 收斂

`LLaMA / Phi` 已有 pre-computed prefill schedule 與 multi-CB execution path。`Qwen3.5` 目前已經有：

- top-level prefill route plan
- `build_qwen35_prefill_schedule(...)` top-level full-flow phase schedule
- full-attention prefill schedule builder
- recurrent tail prefill schedule builder
- schedule summary / route summary
- builder-driven runtime phases / flags

但它目前仍不是等價於 `LLaMA / Phi` 的 single flattened full-op prefill graph。現況已經超過 route-first / summary-first builder，因為 runtime 已直接消費 full-flow phase schedule；剩下未完成的是跨 layer 的扁平化 op graph 收斂。

這使得：

- prefill 仍依賴 inline Rust branching
- route 難以 inspect
- 之後要做真正的 schedule-level pipelining 會被卡住

#### F. batch logits 已完成可控 rollout，但尚未是最終 dense-wide 共用終態

目前 all-logits emission 已有：

- GPU batch projection
- correctness tests
- reusable Metal scratch
- 明確 runtime gate

所以它已不再是未完成 blocker；剩餘工作是之後是否把這條路徑再往 shared dense output-head 收斂，而不是繼續把它當成 Qwen3.5 專屬緊急修復項。

#### G. serial fallback path 仍有大量 CPU buffer churn

這是真問題，但應明確標記為 fallback-only，而不是代表首選路徑的成本模型。

---

## 6. 產品主張

AX Engine 成功的條件不是「能跑很多模型」，而是：

- 在刻意支援的 native model set 上，AX 的路徑是可解釋、可量測、可維護的 Apple-Silicon-native path
- 這條 native path 的勝利，來自 runtime / memory / schedule 的整體設計，而不是一堆不可維護的 profile 例外
- compatibility fallback 可以存在，但不能定義產品核心

對當前版本來說，這個主張先收斂成兩件事：

1. 把 `Qwen3.5` 做成真的 native-fast prefill / decode 路徑
2. 把 dense prefill gap 的 root cause 從「印象」變成固定 artifact 與 execution plan

---

## 7. 主要目標

### 7.1 Primary Goals

- 讓 `Qwen3.5` recurrent hot path 轉為 device-primary / backend-owned contract
- 消除 unified prefill recurrent path 中可避免的 per-layer temp allocation
- 為 `Qwen3.5` 建立可 inspect、可 benchmark 的 prefill schedule builder
- 讓 dense prefill gap closure 以 graph/runtime convergence 為主，而不是 profile-first 調參
- 讓 benchmark / profile / CLI 對 runtime path 的描述一致

### 7.2 Secondary Goals

- 收斂 heuristic / profile / autotune 的責任邊界
- 完成 kernel backlog 中真正仍有 ROI 的項目
- 將 MoE GPU decode、model dedup、routing 收斂為次主線工作

---

## 8. 非目標

- 不把 `Qwen3.5` 問題再定義成單純的「缺一個 kernel」
- 不再擴張 profile-first runtime policy
- 不為了 ownership 問題啟動一個無邊界的全域 KV rewrite
- 不在 runtime surface 未穩定前，把更多 decode fusion prototype 推成主線
- 不把 speculative decoding 當成當前主戰場
- 不讓 routing feature 蓋過 native path 的產品定位

---

## 9. 核心決策

### ADR-FINAL-001：本文件是唯一主 PRD

所有舊 PRD 改為：

- 背景材料
- 證據材料
- 子設計參考

不再作為同級主文件。

### ADR-FINAL-002：Qwen3.5 的主瓶頸已從 decode CB count 轉為 recurrent ownership + prefill schedule

`single-CB decode` 是已經大致證明方向的里程碑，不再是主 backlog。

### ADR-FINAL-003：recurrent end state 必須是 slot-native、device-primary、backend-owned-first

CPU materialization 只能是：

- 顯式 fallback
- correctness fallback
- 受控的 bridge

不能再是 hot-path baseline。

### ADR-FINAL-004：prefill 改進以 schedule / route / execution shape 為核心

`Qwen3.5` 與 dense prefill 的下一輪工作，優先順序是：

- contract
- scratch
- schedule
- route output

不是先調更多 profile。

### ADR-FINAL-005：hot-path allocation elimination 是硬需求，不是微優化

`MetalBuffer::new(...)` 若仍在 recurrent layer main loop 出現，就代表 runtime surface 尚未收斂。

### ADR-FINAL-006：量測必須揭露 route、ownership、command-buffer shape

所有 perf 工作至少要可回答：

- 走的是哪個 route
- recurrent state owner 是什麼
- 是否有 CPU materialization
- command-buffer shape 如何

### ADR-FINAL-007：dense prefill gap 的主因是 runtime / graph convergence

對 `Llama3 / Qwen3 / Gemma3` 類 dense model，剩餘 gap 的優先方向是：

- execution graph shape
- schedule builder
- runtime truthfulness

不是更多 per-model profile。

### ADR-FINAL-008：MoE / dedup / routing 是 active tracks，但不是當前 perf-critical critical path

它們要做，但不能干擾 `Qwen3.5` 與 dense prefill 主線。

---

## 10. 工作流與優先順序

## Phase 0：Baseline Refresh and Truth Capture

### Goal

把所有後續工作固定在目前真實實作上，而不是舊敘述上。

### Tasks

- 更新 `Qwen3.5-9B` / `Qwen3.5-27B` prefill / decode baseline
- 記錄 recurrent route metadata：
  - `qkv-fast`
  - `qkv-handoff`
  - backend-owned carry-over
  - CPU fallback
- 記錄 command-buffer shape
- 記錄 logits path 與 ownership path
- 對 dense representative models 更新 prefill gap attribution artifact

### Outputs

- route-aware benchmark matrix
- ownership / command-buffer matrix
- current blockers note

### Exit Gate

- 後續每個 change 都能對照同一組 baseline artifact

## Phase 1：GPU-Primary Recurrent Slot Contract

### Goal

把 `Qwen3.5` recurrent state 從 hybrid bridge 收斂成真正的 backend-owned-first contract。

### Tasks

- 縮小 CPU materialization 邊界
- 明確化 slot lifecycle semantics
- 讓 branch / snapshot / rollback 與 slot-native contract 對齊
- 保留 correctness fallback，但不保留 CPU-shaped hot path baseline

### Main Files

- `crates/ax-engine-core/src/kv/qwen35_kv.rs`
- `crates/ax-engine-core/src/backend/metal.rs`
- `crates/ax-engine-core/src/model/qwen35.rs`

### Exit Gate

- recurrent slot 在常見 prefill/decode path 中可保持 backend-owned
- CPU materialization event 變成可量測的例外，而非預設

## Phase 2：Recurrent Scratch Reuse and Allocation Elimination

### Goal

消除 unified prefill recurrent path 中的 per-layer temp MetalBuffer allocation。

### Tasks

- 把 recurrent projection temp buffers 移入 reusable batch scratch
- 以 stable shape key 管理 scratch
- 保留 fused / handoff / fallback 差異，但不再每層新建 buffer

### Main Files

- `crates/ax-engine-core/src/model/qwen35.rs`
- `crates/ax-engine-core/src/backend/metal.rs`

### Exit Gate

- recurrent main prefill path 不再在 layer loop 內呼叫 `MetalBuffer::new(...)`
- 所有 recurrent 分支都改走 reusable scratch 或明確 fallback gate

## Phase 3：Qwen3.5 Prefill Schedule Builder

### Current Status

- 部分完成
- top-level full-flow phase schedule 已存在
- full-attention / recurrent-tail partial builders 已存在
- runtime 已直接消費 builder phases / flags
- 尚未完成 single flattened full-op graph

### Goal

建立 `build_qwen35_prefill_schedule(...)`，讓 Qwen3.5 prefill 有可 inspect 的 graph-IR。

### Tasks

- 將現有 full-attention / recurrent-tail schedule builder 收斂為單一頂層入口
- 定義 Qwen3.5 prefill schedule representation
- 支援 full-attention 與 recurrent phase
- 將 schedule encode 到一個或多個 command buffers
- 輸出 schedule summary / route summary

### Main Files

- `crates/ax-engine-core/src/model/prefill_schedule.rs`
- `crates/ax-engine-core/src/model/qwen35.rs`

### Exit Gate

- `Qwen3.5` prefill 不再只靠 inline Rust branching
- schedule 可被 benchmark、profile、review

## Phase 4：Batched Logits Stabilization

### Current Status

- 大致完成
- GPU batch path、correctness tests、reusable scratch、runtime gate 已落地
- 剩餘工作偏向 dense shared convergence，不再是 Qwen3.5 緊急 blocker

### Goal

修正 batched output-head path，或把 blocker 縮到非常窄的明確 gate。

### Tasks

- 穩定 `write_all_batch_logits(...)`
- 對照保守 reference path 做 correctness matrix
- 只在數值穩定時開啟 batched path

### Main Files

- `crates/ax-engine-core/src/model/qwen35.rs`
- `crates/ax-engine-core/src/model/shared.rs`

### Exit Gate

- all-logits emission 要嘛可安全 rollout，要嘛有窄且明確的 gate reason

## Phase 5：Dense Prefill Graph / Runtime Convergence

### Goal

對 `Llama3 / Qwen3 / Gemma3` 類 dense model，把 prefill gap closure 的主線收斂到 execution graph / runtime。

### Tasks

- 固化 dense prefill route attribution
- 把 schedule / graph shape 與 benchmark artifact 綁定
- 讓 heuristic policy 只負責 deterministic selection，不再背負 runtime architecture
- 確認 `llama.cpp` gap 的主因排序與 route 可解釋

### Main Files

- `crates/ax-engine-core/src/model/prefill_schedule.rs`
- `crates/ax-engine-bench/src/prefill_gap.rs`
- runtime policy / profile 相關模組

### Exit Gate

- dense prefill gap 有明確 attribution 與持續縮小計畫

## Phase 6：Runtime / Tooling Alignment

### Goal

讓 CLI、bench、profile、interactive mode 描述的是同一條 runtime path。

### Tasks

- 對齊 decode runner / decode mode selection
- profile 不得改變 production decode shape
- perf run 不得 silent fallback
- benchmark / profile output 顯示 decode mode、fallback reason、route metadata

### Exit Gate

- performance claim、benchmark output、runtime path 三者一致

## Phase 7：Secondary Tracks

### Goal

在不干擾主線的前提下，推進次主線工作。

### Included Tracks

- kernel backlog
- MoE GPU decode wiring
- model forward-pass dedup
- inference routing

---

## 11. 優先級矩陣

### P0

- `Qwen3.5` device-primary recurrent ownership
- recurrent hot-path allocation elimination
- `build_qwen35_prefill_schedule(...)`
- route / ownership / command-buffer truth capture

### P1

- dense prefill graph/runtime convergence
- artifact pipeline alignment
- kernel backlog 中仍有 ROI 的項目：
  - G21
  - G22 redesign
  - G13 redesign
  - G14 redesign
- fallback allocation cleanup

### P2

- MoE GPU decode wiring
- model dedup Phase 3-6
- inference routing

### P3

- 更遠期的 Apple10 / Metal 4 capability tier
- prefix cache / paged KV 擴張
- speculative decoding 重設計

---

## 11.1 立即執行 backlog

以下 backlog 是從現在開始可直接排工的順序，不依賴再開新 PRD。

1. 基線與 artifact 固化
   - 更新 `Qwen3.5-9B` / `Qwen3.5-27B` prefill / decode baseline
   - 更新 dense representative model prefill gap attribution artifact
2. recurrent allocation elimination
   - 補齊剩餘 recurrent 分支的 reusable scratch
3. ownership contract 收斂
   - backend-owned-first
   - 明確 CPU materialization event
4. `build_qwen35_prefill_schedule(...)`
   - 從 top-level full-flow phase schedule + flattened execution steps 再收斂成單一 flattened op graph
   - schedule representation
   - route summary
   - split points
5. dense output-head / shared convergence
   - 將已穩定的 batch logits 路徑視情況往 shared dense path 收斂
6. dense prefill graph/runtime convergence
7. secondary tooling alignment
   - artifact pipeline
8. secondary tracks

---

## 12. 仍然有效的 kernel backlog

以 `PRD-MATMUL-PARITY-2026-03-26.md` 為基礎，保留以下仍有效 backlog：

- `G21`: FA2 K/V staging completion
- `G22`: Q4_K f32 path pair kernel redesign
- `G13`: pair matvec redesign
- `G14`: fused SiLU+Down redesign
- `G11`: YaRN / extended RoPE feature-track
- `G3`: small-batch kernel tier

但這些項目全部都降級到主 runtime surface 穩定之後。

原則如下：

- 不先做 ownership / schedule 之前，不再把 kernel backlog 當主戰場
- 沒有穩定 win 的 prototype 不進主線
- 每個 kernel 變更都要配 route / occupancy / barrier evidence

---

## 13. 次主線工作定義

### 13.1 MoE Support

狀態：

- CPU decode working
- GPU fusion analysis 完成
- GPU decode path 尚未接線

結論：

- 保持 active
- 但排在 `Qwen3.5` 主線與 dense prefill 主線之後

### 13.2 Model Dedup

狀態：

- Phase 1-2 完成
- Phase 3-6 未完成
- 但 structural refactor groundwork 已落地：
  - `qwen35`
  - `qwen3`
  - `gemma3`
  - `llama`
  - `shared`
  - `prefill_schedule`

結論：

- 這是可維護性投資
- 應在主 perf work 不被打斷的前提下持續推進

### 13.3 Inference Routing

狀態：

- 完整 PRD 已定義
- 仍是 proposed track

結論：

- 這是產品覆蓋面工作
- 不能代替 native perf 主線

---

## 14. 驗收標準

### 14.1 Qwen3.5 Prefill

- recurrent main prefill path 不再以 per-layer `MetalBuffer::new(...)` 作為主熱路徑
- recurrent route metadata 能明確顯示：
  - `qkv-fast`
  - `qkv-handoff`
  - backend-owned carry-over
  - CPU fallback
- 常見 prefill hot path 不再以 CPU materialization 為 baseline
- `build_qwen35_prefill_schedule(...)` 或等價頂層 schedule builder 已存在且可輸出 route/schedule summary
- `write_all_batch_logits(...)` 已有 GPU path、correctness tests、reusable scratch、runtime gate，且不再是未定義 rollout 狀態
- `Qwen3.5-9B` 在 `64 / 128 / 512` prompt 長度下都有可重跑 artifact

### 14.2 Dense Prefill

- 對代表性 dense model：
  - `Llama3-8B`
  - `Qwen3-8B`
  - `Gemma3-12B`
- 與 `llama.cpp` 的 gap 有穩定 attribution
- route 選擇不再主要依賴 per-model profile

### 14.3 Runtime Truthfulness

- CLI、bench、profile、interactive mode 對 decode mode 選擇一致
- perf run 無 silent fallback
- profile output 不再把主要 GPU decode 時間全部歸到不可用 bucket
- benchmark / profile output 包含 route、decode mode、fallback reason、command-buffer shape

### 14.4 Maintainability / Secondary Tracks

- `LLaMA / Qwen3 / Gemma3 / Qwen3.5` 不再以單一巨檔維護
- shared runtime / schedule code 不再集中在單一雜物檔
- MoE GPU decode 若進行，必須不回歸 dense path
- dedup work 不得破壞 architecture-specific correctness
- routing feature 若進行，必須清楚標明 backend kind，且不能掩蓋 native regressions

---

## 14.1 必要 artifact 與 benchmark matrix

### Qwen3.5

- `Qwen3.5-9B-Q4_K_M`
  - `prefill-profile --prompt-tokens 64`
  - `prefill-profile --prompt-tokens 128`
  - `prefill-profile --prompt-tokens 512`
  - `bench --prompt-tokens 512 --decode-tokens 128`
- `Qwen3.5-27B-*`
  - 至少一組 prefill baseline
  - 至少一組 decode baseline

### Dense representative models

- `Llama3-8B`
- `Qwen3-8B`
- `Gemma3-12B`

每組 artifact 至少必須輸出：

- tok/s
- route label
- recurrent ownership label
- command-buffer shape
- fallback reason
- logits path

### 對照基準

- `llama.cpp` apple-to-apple baseline
- 若適用，保留 `mistral.rs` shape / contract 對照作為設計參考，而不是性能 gate

---

## 14.2 Phase 完成定義

### Phase 0 完成

- 基線 artifact 已更新
- route/ownership/command-buffer metadata 已固定輸出

### Phase 1 完成

- recurrent hot path 預設不再是 CPU-materialized baseline
- backend-owned carry-over 可被觀察且穩定

### Phase 2 完成

- recurrent main prefill path 已以 reusable scratch 為預設
- scratch reuse 以 stable shape key 管理

### Phase 3 完成

- `build_qwen35_prefill_schedule(...)` 或等價頂層 builder 已存在
- Qwen3.5 prefill schedule 可 inspect / benchmark / review
- 若要視為完全完成，仍需補 single flattened full-op graph

### Phase 4 完成

- batched logits correctness matrix 完成
- 路徑已 rollout 並有明確 gate

### Phase 5 完成

- dense prefill gap 有 route-aware attribution
- 不再依賴 profile-first 思維來描述主因

### Phase 6 完成

- CLI / bench / profile / interactive 描述同一條 runtime path
- perf run 無 silent fallback

### Phase 7 完成

- secondary tracks 有明確邊界，且未干擾主線

### Code Management Milestone

- `LLaMA / Qwen3 / Gemma3 / Qwen3.5` 已拆成模組化結構
- `shared.rs` 已拆成 runtime / gpu-decode / gpu-batch
- `prefill_schedule.rs` 已拆成 common / qwen35 / tests
- 後續 dedup 與 runtime work 不再建立在 2k~8k 行單檔之上

---

## 15. 舊文件處置規則

### 15.1 已實作並吸收

- `PRD-QWEN35-RECURRENT-OWNERSHIP-AND-PREFILL-SHAPE-2026-03-29.md`
- `PRD-PREFILL-GAP-CLOSURE-VS-LLAMA-CPP-AND-MISTRAL-RS-2026-03-29.md`

### 15.2 已過時但保留歷史價值

- `PRD-QWEN35-SINGLE-CB-DECODE.md`

原因：

- 它描述的是舊的最急 decode bottleneck
- 但目前主瓶頸已轉向 ownership / schedule / logits

### 15.3 仍可作局部參考

- `PRD-MATMUL-PARITY-2026-03-26.md`
- `PRD-MOE-SUPPORT-2026-03-26.md`
- `PRD-REFACTOR-MODEL-DEDUP.md`
- `PRD-INFERENCE-ROUTING-2026-03-27.md`
- `PRD-AX-ENGINE-PERF-RECOVERY.md`

### 15.4 被本文件完全吸收的主線規劃

- `PRD-AX-ENGINE-v3.md`
- `PRD-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PLAN-AX-ENGINE-v3-QWEN-GAP-CLOSURE-2026-03-27.md`
- `PRD-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`
- `PLAN-QWEN35-GPU-PRIMARY-RECURRENT-AND-PREFILL-SCHEDULING-2026-03-29.md`

---

## 16. 立即執行順序

1. 更新 `Qwen3.5` 與 dense representative models 的 baseline artifacts
2. 拆掉 recurrent main prefill path 的 per-layer temp MetalBuffer allocation
3. 收斂 recurrent ownership contract 到 backend-owned-first
4. 把既有 `build_qwen35_prefill_schedule(...)` 收斂成 full-op builder
   - 目前已是 top-level full-flow phase schedule
   - 剩餘工作是 flatten 成單一 op graph
5. 將穩定的 batched logits 路徑視情況往 shared dense output-head 收斂
6. 將 dense prefill gap closure 轉為 execution-graph convergence
7. 對齊剩餘 CLI / artifact pipeline
8. 最後才回頭處理 kernel redesign、MoE GPU wiring、dedup、routing

---

## 16.1 執行守則

從這份 PRD 起，後續所有實作遵守以下規則：

- 任何性能結論都必須附 route-aware artifact
- 若 change 改變 runtime shape，必須更新 benchmark note
- 若 change 只改善 fallback path，不得包裝成主線 throughput 勝利
- 若 change 只是增加 heuristic / env knob，但未改善 runtime contract，不得視為主線完成
- 若新 kernel prototype 沒有穩定 win，直接回退，不保留半啟用狀態

---

## 17. 成功定義

AX Engine 在這份最終 PRD 下成功，代表下列事情同時成立：

- `Qwen3.5` 不再主要被 recurrent ownership 與 prefill fragmentation 拖住
- dense prefill 與 `llama.cpp` 的差距有清楚、持續收斂的 execution plan
- perf artifact 能描述真實 runtime，而不是另一套 bench-only story
- AX 的 native path 仍然是產品中心，而不是被 fallback feature 取代

---

## 18. 最後結論

AX 下一步不應再問：

- 還要不要多做一輪 profile？
- 還要不要再試一個單點 kernel？
- 還要不要再加一個 env knob？

AX 下一步應直接做：

- recurrent ownership contract
- scratch reuse
- prefill schedule builder
- dense prefill runtime convergence

這就是目前唯一合理的主線。
