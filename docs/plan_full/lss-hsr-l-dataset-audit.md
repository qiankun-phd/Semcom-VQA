# LSS-HSR-L Dataset Audit

Dataset path: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/数据集及使用说明`
DOI: `10.57760/sciencedb.radars.00063`
CSTR: `31253.11.sciencedb.radars.00063`
Input kind: `directory`
Status: **patrol-warning-ready-for-label-design**
Files: `1457`
Total uncompressed bytes: `235070249`

## README / License

- README candidates: `['LSS-HSR-L：全息凝视雷达低空目标探测识别数据集使用说明-0507.docx', 'dataset.py']`
- License candidates: `[]`
- License clues: `['LSS-HSR-L：全息凝视雷达低空目标探测识别数据集使用说明-0507.docx: doi', 'ScienceDB metadata license: https://creativecommons.org/licenses/by-nc/4.0/']`

## Real-Data Gates

| Gate | Status | Detail |
|---|---|---|
| `readme_or_usage` | `pass` | 2 README/usage candidates |
| `license_or_usage_terms` | `pass` | 0 license candidates, 2 content clues |
| `target_group_coverage` | `pass` | 4/4 target groups detected |
| `nine_class_coverage` | `pass` | 9/9 expected classes detected from paths |
| `doppler_waterfall_or_image` | `pass` | Doppler/waterfall or image-like files detected |
| `trajectory_data` | `pass` | Trajectory/track hints detected |
| `radar_to_question_feasibility` | `pass` | Enough structure for derived radar-question labels |

## Target Group Hits

- rotary_uav: `22` examples
- biological: `22` examples
- fixed_rotating: `22` examples
- vehicle: `22` examples

## Expected 9-Class Hits

- DJI Air3: `22` examples
- DJI Mini3 Pro: `22` examples
- DJI Mavic 3e: `22` examples
- DJI Phantom 4 RTK: `22` examples
- 小麻雀: `20` examples
- 鸟群: `22` examples
- 大型候鸟: `21` examples
- 地面固定旋转目标: `22` examples
- 汽车: `22` examples

## Modality Hits

- doppler_waterfall: `2` examples
- trajectory: `2` examples
- image_like: `0` examples
- matlab_or_array: `22` examples
- table_like: `0` examples

## Derived Question Templates

- `target_identity`: What type of low-altitude target is present?
- `drone_or_bird`: Is this target a rotary-wing UAV rather than a biological target?
- `intrusion_risk`: Does the target trajectory suggest patrol-warning risk?
- `visual_confirm`: Should a UAV camera be dispatched for visual confirmation?

## Next Checks

- Read README/license before any publication or engineering use.
- Confirm official class-id mapping for the 9 target classes.
- Confirm whether scene labels are per sample or dataset-level only.
- Confirm whether trajectory data contains timestamps and coordinates.
- Build derived radar-question labels only after license allows derivatives.
