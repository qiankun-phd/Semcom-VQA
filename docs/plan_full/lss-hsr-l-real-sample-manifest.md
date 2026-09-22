# LSS-HSR-L Sample Manifest

Dataset path: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/数据集及使用说明`
Input kind: `directory`
Output CSV: `paper/data/lss_hsr_l_real_sample_manifest.csv`
Sample rows: `1455`
Unknown target rows: `0`

## Modality Counts

- array: `1455`

## Target Group Counts

- biological: `469`
- fixed_rotating: `150`
- rotary_uav: `573`
- vehicle: `263`

## Next Step

Run:

```bash
python3 hppo-uav/run/lss_hsr_l_question_builder.py --input-csv paper/data/lss_hsr_l_real_sample_manifest.csv
```
