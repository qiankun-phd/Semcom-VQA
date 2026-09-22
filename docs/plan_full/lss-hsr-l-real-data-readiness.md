# LSS-HSR-L Real-Data Readiness

Date: 2026-05-30

Purpose: record the current real-data state for LSS-HSR-L and the exact next
command once the ScienceDB archive is available.

Download/context note:
`plan/lss-hsr-l-download-and-related-datasets.md` records the recommended
local/server placement path and related ScienceDB radar datasets used only for
context, not as license substitutes.

## Current Data Location Check

Current executable locator:

```bash
python3 hppo-uav/run/lss_hsr_l_data_locator.py
```

Current local result:

```text
overall_status: not-found
```

Current local + verification-server result:

```bash
python3 hppo-uav/run/lss_hsr_l_data_locator.py --check-remote
```

```text
overall_status: not-found
remote host: lab-s2
remote roots seen: /home/qiankun/HPPO-VQA, /home/qiankun
remote files: none
```

Current guarded remote import plan:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py
```

Current dry-run result:

```text
mode: dry-run
local_source_present: false
remote_dataset_path: /home/qiankun/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip
```

This plan writes the exact local locator, remote directory creation, `rsync`,
remote locator, and remote real-data pipeline commands. It does not contact the
server in dry-run mode.

Local workspace search:

```bash
find /Users/zhangqiankun/Documents/mpu/HPPO-VQA -maxdepth 5 -type f \
  \( -name '*LSS*' -o -name '*HSR*' -o -name '*数据集*' -o -name '*.zip' \)
```

Result: no LSS-HSR-L zip or extracted dataset found in the current workspace.

Verification server search:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 lab-s2 \
  "hostname; find /home/qiankun -maxdepth 5 -type f \
  \( -iname '*lss*' -o -iname '*hsr*' -o -name '*数据集*' -o -name '*.zip' \) \
  2>/dev/null | sed -n '1,120p'"
```

Result: server reachable; no LSS-HSR-L / HSR / `数据集及使用说明.zip`
file was found in the searched paths.

## Dataset Anchor

- CSTR: `31253.11.sciencedb.radars.00063`
- DOI: `10.57760/sciencedb.radars.00063`
- ScienceDB page title: `LSS-HSR-L：Low-Altitude target recognition dataset based on Holographic Staring Radar`
- ScienceDB OpenAPI status: `scidb-metadata-ready-archive-pending`
- Page-level access: `unrestricted`, accessible for free
- Page-level license: `CC BY-NC 4.0`
- OpenAPI size: `209569478` bytes, consistent with the displayed `199.86 MB`
- Modalities: Doppler waterfall plot and trajectory data
- Target classes: DJI Air3, DJI Mini3 Pro, DJI Mavic 3e, DJI Phantom 4 RTK,
  小麻雀, 鸟群, 大型候鸟, 地面固定旋转目标, 汽车

The page-level metadata gate is reproducible from:

```bash
python3 hppo-uav/run/lss_hsr_l_scidb_metadata_gate.py
```

This confirms non-commercial research-use suitability at the ScienceDB metadata
level. It does not replace the downloaded archive's README / usage-instruction
audit, which remains the authority for derived-label and redistribution claims.

## Next Command After Download

After the archive is downloaded to the recommended local path, first regenerate
the remote import plan:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py
```

If the plan shows `local_source_present: true`, sync to the verification server
and run the guarded remote audit pipeline only with the explicit confirmation
token:

```bash
python3 hppo-uav/run/lss_hsr_l_remote_plan.py \
  --execute --confirm SYNC_REMOTE_LSS_HSR_L_DATA
```

For a local-only audit, run the enhanced read-only audit:

```bash
python3 hppo-uav/run/lss_hsr_l_dataset_audit.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

The audit now reports:

- DOI/CSTR identity;
- README / usage instruction candidates;
- license files and license content clues;
- 4 target-group coverage;
- expected 9-class path coverage;
- Doppler/waterfall or image-like modality hints;
- trajectory/track hints;
- radar-to-question feasibility status;
- derived patrol-warning question templates.

After the manifest is generated, run the semantic calibration stage if you want
to inspect the radar-to-VQA fields before question generation:

```bash
python3 hppo-uav/run/lss_hsr_l_semantic_calibration.py \
  --input-csv paper/data/lss_hsr_l_sample_manifest.csv
```

This stage preserves provided numeric fields. If `radar_confidence`,
`drone_bird_margin`, or `trajectory_risk_score` are missing, it fills them from
explicit target/scene/modality priors and marks the row as
`prior-derived-needs-real-calibration`. Those rows are valid for interface
testing only, not final TCCN evidence.

## Real-Data Promotion Rule

Do not promote the synthetic LSS-HSR-L demo to real evidence until the audit
returns:

```text
status: patrol-warning-ready-for-label-design
readme_or_usage: pass
license_or_usage_terms: pass
target_group_coverage: pass
doppler_waterfall_or_image: pass
trajectory_data: pass
```

If `nine_class_coverage` is only `review`, inspect README/label files manually;
path names alone may not expose every class even when labels are present.

After the audit, run the promotion gate:

```bash
python3 hppo-uav/run/lss_hsr_l_real_data_gate.py --strict
```

Promotion status meanings:

- `real-data-ready-for-derived-labels`: the archive can move into manifest,
  radar-question, semantic-control feature, and communication-policy stages.
- `real-data-needs-review`: at least one README/license/label/modality gate
  needs manual inspection before claims or experiments.
- `missing-audit`: the dataset audit JSON has not been generated yet.

## One-Command Real-Data Pipeline

After downloading the archive, the safest path is to run the guarded pipeline:

```bash
python3 hppo-uav/run/lss_hsr_l_real_data_pipeline.py \
  --dataset-path /path/to/数据集及使用说明.zip
```

This runs:

```text
audit -> real-data gate -> manifest -> semantic calibration -> radar questions
-> semantic-control features -> communication policy probe
```

If the real-data gate does not pass, the pipeline stops before creating
derived radar-question labels. This prevents README/license/label problems
from becoming accidental experiment claims.
