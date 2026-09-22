# LSS-HSR-L Sample Manifest

Dataset path: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_demo/lss_hsr_l_demo_dataset`
Input kind: `directory`
Output CSV: `paper/data/lss_hsr_l_demo_manifest.csv`
Sample rows: `4`
Unknown target rows: `0`

## Modality Counts

- doppler_waterfall: `2`
- trajectory: `2`

## Target Group Counts

- biological: `1`
- fixed_rotating: `1`
- rotary_uav: `1`
- vehicle: `1`

## Next Step

Run:

```bash
python3 hppo-uav/run/lss_hsr_l_question_builder.py --input-csv paper/data/lss_hsr_l_demo_manifest.csv
```
