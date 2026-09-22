# W2 — Energy-billing perimeter itemization (2026-08-05)

> Blueprint item W2 (`reboot-blueprint-2026-08.md` §3). Source of truth:
> `outputs/energy/gpu_power_phases.json` (160; copy in scratch extract) produced by
> `scripts/measure_gpu_energy.py`; conventions in `make_energy_figures.py`.
> Feeds §III energy accounting prose and §VII.C limitations. Every number below is
> measured or declared — nothing interpolated.

## 1. Measurement setup (as measured, RTX 4060 edge-server proxy)

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4060 (115 W limit), NUC9i7QNX host |
| Power readout | driver 550.144.03; `nvidia-smi -q -d POWER` Power-Samples block (~8 Hz internal sampler, ~14.4 s trailing window); NVML power.draw unsupported on this card |
| Protocol | poll every 4 s; only samples ≥ 20 s after steady-state loop start (≥ one trailing window); average of window-averages |
| Workload | campaign's own degraded test images (5 dB Rician) + real questions; Qwen2-VL-2B, greedy, `max_new_tokens=24`, campaign pixel limits |
| Idle baseline | 52.30 W (25 windows, std 0.10 W); post-run re-check 52.76 W (+0.9 % drift) |

## 2. Itemized billing, per answered question

| Component | Billed as | Value | Perimeter note |
|---|---|---|---|
| **E_tx (any level)** | `channel_uses / B × P_tx`, B = 1 MHz, P_tx = 0.5 W headline | per-level, per-SNR from tidy CSV | P_tx anchored to 3GPP TR 36.777 aerial UE (23 dBm ≈ 0.2 W); sensitivity grid 0.1–1 W reported (`jpa_sensitivity.tex`). Radiated-power model — RF chain overhead not separately billed |
| **VLM forward (s2/s3 route)** | measured, incremental over idle | **32.31 J/item** (107.87 W avg, 0.581 s/item, 413 items) | headline uses incremental (idle not charged to answers); total-power variant 62.72 J/item also reported in `energy_summary.json` (`j_per_answer_totalpower`) |
| **Detector (s1 route, onboard)** | declared band, Jetson Orin Nano class | 7–15 W × 15–50 ms = **0.105–0.750 J**, midpoint 0.4275 J | 4060-measured detector kept as cross-check: 0.066 J/item incremental (58.65 W, 10.4 ms/item, 14,389 items). Band, not point — shown as horizontal band on F5 |
| **Symbolic decode (s1 route)** | declared | **0 J** | CPU-only few-op decode; declared not measured (stated in §III) |
| **Router inference (NEW — predictor-primary)** | declared, bounded | **≈0 J** (< 10⁻⁴ J) | logistic: 2 heads × 45-dim dot ≈ 10² FLOPs; MLP (32,16): ≈ 4×10³ FLOPs/decision — both ≥ 3 orders below the detector band's low end (0.105 J). Billed 0 J, declared with bound in §III |
| **Router training (offline)** | out of perimeter, disclosed | one-time | logistic/MLP fit on train-split logs; amortizes over deployment; disclosed in §VII.C rather than billed per answer |
| **Idle power** | not billed (headline) | 52.3 W | amortization policy: answers charged incremental energy only; total-power variant reported as sensitivity |

## 3. Explicitly outside the perimeter (state in §VII.C)

- UAV platform power (flight, gimbal, camera ISP) — orthogonal to the routing decision.
- Host CPU beyond declared-0 items (symbolic decode, router inference — bounded above).
- Receiver-side radio front-end and decoding energy (LDPC decode billed to neither side; stated).
- Network/protocol stack beyond the radiated-power abstraction E_tx = uses/B × P_tx.

## 4. Sanity anchors

- Idle drift over the campaign window: +0.46 W (+0.9 %) — below the 1 W resolution of any claim.
- W5 self-check: recomputed pooled Rician (acc 0.6798, 8.868 J, f_img 0.2594) reproduces the published λ=0 frontier row exactly.
- The 2.2× (rule) and 3.6× (predictor, W5) headline ratios both survive the total-power variant and every P_tx in the 0.1–1 W grid (compute term dominates: 32.31 J ≫ max E_tx ≈ 2 J at −5 dB).

## 5. Open item

- The Jetson detector band is declared from datasheet + latency band, not measured on a Jetson. If a reviewer demands a measured point, the deferred Jetson datapoint (blueprint §3, deferred list) is the answer — do not silently convert the band into a point.
