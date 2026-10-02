# GAP-01 BPG real-bitstream n=2400 — Colab prelaunch plan

**Steering update:** target is now Colab CLI (L4 preferred; T4 requested alternative). No further lab SSH retries. User-only authentication is required; current status **COLAB_AUTH_NEEDED**. This plan predates local codec smoke and is updated before any semantic/PHY/Colab launch. See `COLAB_LAUNCH.md`.

Session `01a0f50d-f011-72b0-99f7-6466ce324b54`. Plan written before any BPG encode/PHY/VLM launch. Work is confined to `exp/bpg-phy-benchmark-n2400`; backup `backup/pre-gap01-20261002` remains at a22dc52. No Mac paper checkout edits, production-default changes or deferred-gap work.

## Scope and reuse

Use the completed JPEG2400 stack and selection: six question types, label-free per-type SHA256 order with salt `B-smoke-v1`; first1/type for n=6, first400/type for n=2400. Budgets2k/4k with medium Tv; nominal SNR0/5/10/20; seeds1701–1703. The exact same learned baseline packet/frame hashes, receiver adapter, processor geometry/pixel target100352, greedy16-token generation and true DigiSem-PHY LDPC/QPSK/Rician/CRC apply. No analytical SNR cutoff or rounded historical delivery-cache keys.

Full JPEG outputs already collected locally at `/workspace/codex_runs/next_batch_mask_jpeg_calib_20260930_2310/evidence/B_real_2400/`; prior actual runtime was2781.76s (46.36min). These are provenance references, not BPG results. Remote JPEG source output is `/home/qiankun/phd_research/vqa_semcom/outputs/next_batch_mask_jpeg_calib_20260930_2310_B_jpeg2400/`.

BPG native payload is wrapped with one byte `B` (budget cap B−1), parallel to JPEG's one-byte `J`. New wrapper is isolated and is not registered in the production codec router. Encode original RGB via lossless intermediate PNG; decode selected native BPG via `bpgdec`, resize decoded RGB to the same learned receiver geometry, then feed the unchanged medium processor. BPG bitstream is transmitted, not a noise proxy.

Freeze first-fit candidate order before seeing any labels: maximum side384,320,256,224,192,160,128,96,64; QP20,25,30,35,40,45,50,51 (ascendingQP within each descending side). libbpg0.9.8, x265,8bit,YCbCr420, compression level1 (bounded encode cost). QP and JPEG quality are distinct parameters, not claimed equal. This protocol aims at cap alignment and receiver parity, not content/rate-optimal BPG. Keep encoding failures in semantic denominator.

For both codecs use actual `Ns=510*ceil(framed_bytes/48)` and `gamma_eff=gamma+10log10(21420/Ns)` exactly once. Same shared Rician fade/noise identity per image/seed; full exact-byte recovery is required. Preserve actual n/k and packet/source hashes. Rule B rows are composed from fixed2k/medium at≤10 and fixed4k/medium above10; do not fit/tune a BPG policy.

## Counts and launch gates

- Local codec smoke: six **generated fixtures**,12 framed BPG encode/decode cases. No benchmark images, no PHY/VLM; does not count as real n=6 semantic smoke.
- Colab semantic n=6:12 new BPG clean predictions plus1 excluded warmup;144 BPG PHY tasks maximum;144 learned tasks reused from JPEG2400;4 noiseless roundtrips.
- Colab semantic n=2400:4,800 BPG packets/clean predictions, with12 reusable from BPG smoke. New full invocation≤4,788 VLM predictions +1 warmup,≤57,456 BPG noisy PHY tasks after smoke reuse. Learned baseline clean predictions and57,600 delivery outcomes reused from frozen grid/JPEG2400, with matching fingerprint checks.
- Logical full grid is2,400×2 budgets×4 SNR×3 seeds×2 codecs=115,200 evaluations. Byte-identical packets at two caps may share one physical task; report logical evaluations separately from unique transmitted tasks.
- No new training, real-joule measurement or multi-user/deferred-gap work.

Before smoke/full, verify Colab asset scanner Rule B/default AST and current source hashes, installed bpgenc/bpgdec fingerprints, complete JPEG baseline cache and adapter hashes. Refuse missing/changed assets or any incompatible cache; do not silently regenerate baseline PHY. Run smoke and independently validate artifacts before launching full with an active Colab kernel and≤12 PHY workers. Detached children alone do not guarantee VM liveness.

## Cost, persistence and risks

JPEG completed run provides GPU clean-inference scaling of roughly20–30min and PHY10–20min. BPG encode search adds potentially5–90min or more depending on candidate count and host speed; the generated-fixture smoke is not a dataset timing benchmark. Budget45–150min full wall time initially, then refine from remote n=6 before committing resources. libbpg build roughly several minutes on a modern CPU; record actual local timings. The JPEG baseline ran on RTX40608GiB; Colab L4/T4 use a different GPU backend. L4 supports the inherited BF16 path; T4 requires a bounded dtype compatibility smoke and must not silently change receiver precision. Start4 PHY workers on Colab,≤12 only if host CPU/RAM permits. Reserve≥10GiB disk and enough inodes; preserve journals and decoded PNGs.

Durable per-image encoding journal avoids expensive re-encoding on resume. Freeze source/input/binary/protocol hashes, validate cached packet/decoded hashes, append clean and PHY journals with fsync, recover only a torn final line and reject interior corruption. Exclusive output lock prevents duplicate writers. Resume identical argv plus`--resume`; detach via wrapper writing PID/argv/exit status. Do not publish partial conditional Acc as n=2400 results.

Historical proxy files are excluded from both runner and real tables. BPG output/table namespace is separate from JPEG. No main/feature defaults or paper files are changed. Reuse of receiver/PHY is tested structurally and in local fixture smoke; actual remote end-to-end validation remains mandatory.

## Current blockers before launch

Both remote routes failed initial retry (ZeroTier No route to host; LAN timeout). GitHub public read/clone works, but box GitHub SSH authentication is rejected and gh has no login/token. No Mac SSH endpoint/credentials were supplied. Complete local prep/branch commit and an exportable patch/bundle if push remains blocked; never claim pushed or full-grid execution without evidence.

The GitHub a22dc52 scanner is an older snapshot without v2/default_policy_names, whereas the previously audited live remote scanner contains frozen Rule B. Do **not** change the old scanner in this branch to resolve that mismatch. Require the existing remote v2/default AST before a real run; local branch preservation and historical live freeze evidence must be labeled separately from a current remote PASS.

## Steering outcome / completed local checks

Generated-fixture codec smoke passed in5.853s:6 source fixtures,12 capped native BPG encode/decode cases,90 candidate encoder calls including deterministic rechecks,12 decoder calls. Zero semantic benchmark queries, PHY/VLM/training. Local binaries are the executor-provided libbpg0.9.8 scalar build; exact binary hashes and compatibility notes are retained. Actual Colab binaries must be fingerprinted again.

Both default Colab token and session files are absent. No interactive login or VM provisioning was attempted. Authentication is the primary gate; **ASSET_BUNDLE_NEEDED** remains an additional gate because box has completed JPEG journals/metadata but lacks the complete frozen original-image/learned-packet/receiver/adapter stack. No true n=6 or n=2400 result is claimed.
