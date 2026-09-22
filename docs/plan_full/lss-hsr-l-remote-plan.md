# LSS-HSR-L Remote Import Plan

Mode: `dry-run`
Confirm token: `SYNC_REMOTE_LSS_HSR_L_DATA`
Local source: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip`
Local source present: `False`
Local source size bytes: `None`
Remote host: `lab-s2`
Remote root: `/home/qiankun/HPPO-VQA`
Remote dataset path: `/home/qiankun/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip`

## Steps

### `local-locator`

Record whether the ScienceDB archive is present in the local workspace.

```bash
python3 hppo-uav/run/lss_hsr_l_data_locator.py --extra-path '/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip' --out-json paper/data/lss_hsr_l_data_locator_remote_plan.json --out-md plan/lss-hsr-l-data-locator-remote-plan.md
```

### `remote-mkdir`

Create the remote raw-data directory without deleting or overwriting other files.

```bash
ssh lab-s2 'set -e
mkdir -p /home/qiankun/HPPO-VQA/paper/data/raw/lss_hsr_l'
```

### `rsync-archive`

Copy the local LSS-HSR-L archive to the validation server.

```bash
rsync -az '/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/raw/lss_hsr_l/数据集及使用说明.zip' lab-s2:/home/qiankun/HPPO-VQA/paper/data/raw/lss_hsr_l/
```

### `remote-locator`

Confirm the archive location on the validation server.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
python3 hppo-uav/run/lss_hsr_l_data_locator.py --extra-path '"'"'paper/data/raw/lss_hsr_l/数据集及使用说明.zip'"'"' --out-json paper/data/lss_hsr_l_data_locator_remote_import.json --out-md plan/lss-hsr-l-data-locator-remote-import.md'
```

### `remote-real-data-pipeline`

Run the guarded audit-to-policy pipeline on the validation server.

```bash
ssh lab-s2 'set -e
cd /home/qiankun/HPPO-VQA
python3 hppo-uav/run/lss_hsr_l_real_data_pipeline.py --dataset-path '"'"'paper/data/raw/lss_hsr_l/数据集及使用说明.zip'"'"' --prefix lss_hsr_l_remote_real --out-json paper/data/lss_hsr_l_remote_real_data_pipeline.json --out-md plan/lss-hsr-l-remote-real-data-pipeline.md'
```

## Notes

- Dry-run mode only writes this plan; it does not contact the validation server.
- Execution refuses to start unless the local archive exists and the confirm token is provided.
- The remote real-data pipeline remains guarded by README/license/modality gates.
