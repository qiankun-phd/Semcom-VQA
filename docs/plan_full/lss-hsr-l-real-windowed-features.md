# LSS-HSR-L Windowed Radar Features

Overall: **windowed-features-ready**
Input manifest: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_sample_manifest.csv`
Input feature CSV: `/Users/zhangqiankun/Documents/mpu/HPPO-VQA/paper/data/lss_hsr_l_real_mat_statistics.csv`
Output feature CSV: `paper/data/lss_hsr_l_real_windowed_statistics.csv`
Rows: `1455`
Window count: `6`

## Added Features

- `dpl_window_mean_std`
- `dpl_window_peak_std`
- `dpl_window_contrast_max`
- `speed_window_abs_mean_std`
- `speed_window_std_max`
- `range_window_span_max`
- `height_window_span_max`
- `snr_window_mean_std`
- `snr_window_p90_max`

## Claim Boundary

These features capture coarse temporal variation inside each radar sample.
They are still a lightweight calibration input, not a final radar-recognition model.
