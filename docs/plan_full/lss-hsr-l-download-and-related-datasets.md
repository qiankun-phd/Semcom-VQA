# LSS-HSR-L Download Locator and Related Dataset Context

Date: 2026-05-30

Purpose: keep the real-data next step concrete while preserving claim
boundaries. This note records what can be verified from public metadata/search
and what still requires the downloaded `数据集及使用说明.zip`.

## Target Dataset

- Name: LSS-HSR-L: Low-Altitude target recognition dataset based on
  Holographic Staring Radar
- CSTR: `31253.11.sciencedb.radars.00063`
- DOI: `10.57760/sciencedb.radars.00063`
- ScienceDB OpenAPI metadata: unrestricted access, accessible for free,
  license `CC BY-NC 4.0`, size `209569478` bytes
- User-provided/public page file metadata: `数据集及使用说明.zip`, `199.86 MB`
- Stated modalities: Doppler waterfall plot and trajectory data
- Stated scenes: Shenzhen, Changsha, Chongqing; urban, airport, suburban
- Stated target classes: DJI Air3, DJI Mini3 Pro, DJI Mavic 3e,
  DJI Phantom 4 RTK, 小麻雀, 鸟群, 大型候鸟, 地面固定旋转目标, 汽车

The direct local blocker remains:

```text
paper/data/lss_hsr_l_dataset_audit.json is missing
```

Therefore the real-data gate reports:

```text
overall_status: missing-audit
```

The page-level metadata gate now reports:

```text
overall_status: scidb-metadata-ready-archive-pending
```

This is enough to treat LSS-HSR-L as a non-commercial research dataset anchor
for the narrative, but not enough to promote real-data experiments before the
downloaded archive's README / usage instructions are audited.

## Practical Download / Placement Path

1. Open the ScienceDB page for DOI `10.57760/sciencedb.radars.00063`.
2. Download `数据集及使用说明.zip`.
3. Put it either locally or on the verification server.

Recommended local path:

```text
/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip
```

Recommended verification-server path:

```text
/home/qiankun/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip
```

Before running any audit, generate the guarded local-to-server import plan:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py
```

The plan is dry-run by default and records the exact locator, `rsync`, remote
locator, and remote guarded pipeline commands. After the local zip exists, use
explicit confirmation to sync and audit on the validation server:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py \
  --execute --confirm SYNC_REMOTE_LSS_HSR_L_DATA
```

For local-only processing, run:

```bash
python3 hppo-uav/run/lss_hsr_l_real_data_pipeline.py \
  --dataset-path paper/data/raw/lss_hsr_l/数据集及使用说明.zip
```

The pipeline will stop before derived labels if README/license/label/modality
checks fail.

Inside the guarded pipeline, the manifest now passes through:

```text
manifest -> semantic calibration -> radar-question labels
```

The calibration stage is explicit about claim boundaries: provided numeric
confidence/risk fields are preserved; missing fields are filled only as
traceable priors and marked `prior-derived-needs-real-calibration`.

Before running the pipeline, check placement:

```bash
python3 hppo-uav/run/lss_hsr_l_data_locator.py
```

To check the validation server as well:

```bash
python3 hppo-uav/run/lss_hsr_l_data_locator.py --check-remote
```

Current checked status:

```text
local: not-found
verification server lab-s2: not-found
```

## Related Low-Altitude Radar Datasets

Public search also surfaces related ScienceDB radar datasets that can help
position LSS-HSR-L without replacing it:

| Dataset | DOI/CSTR signal | Why it matters |
|---|---|---|
| LSS-FMCWR-1.0 | DOI `10.57760/sciencedb.radars.00017`, CSTR `31253.11.sciencedb.radars.00017` | Low-altitude FMCW radar dataset with multi-band drone echo data; useful context for radar-only detection benchmarks |
| LSS-FMCWR-2.0 | DOI `10.57760/sciencedb.radars.00054`, CSTR `31253.11.sciencedb.radars.00054` | Multi-band multi-angle FMCW low-altitude target detection; public metadata indicates CC BY-NC-ND 4.0, but its license does not automatically apply to LSS-HSR-L |
| MTDSP multisource marine targets | DOI `10.57760/sciencedb.radars.00046`, CSTR `31253.11.sciencedb.radars.00046` | Shows ScienceDB radar series has heterogeneous access sizes and modalities; not directly low-altitude UAV VQA |

These datasets support the narrative that radar datasets usually validate
detection/recognition. The LSS-HSR-L opportunity is different: it can trigger
which VQA confirmation question should be asked next.

## Claim Boundary

Do not transfer license terms from related datasets to LSS-HSR-L.

Do not claim commercial use. The current ScienceDB metadata license is
`CC BY-NC 4.0`, and the downloaded README/license may add more constraints.

Do not claim LSS-HSR-L is a camera-VQA dataset. It is a radar event source for
question-conditioned visual confirmation.

Do not claim real experimental evidence until:

```bash
python3 hppo-uav/run/low_altitude_vqa_readiness_gate.py
```

reports:

```text
overall_status: ready-for-real-data-derived-experiments
```
