# Colab launch after user authentication

Prepared with installed `google-colab-cli0.7.4` help/source. No auth or VM was
started by this batch. Current blocker: **COLAB_AUTH_NEEDED**; default
`~/.config/colab-cli/token.json` and `sessions.json` are absent. Authentication
must be completed by the user in a one-to-one context. Agents must not run an
interactive login. After user authentication, executor provisions:

```bash
colab new --gpu L4 --session gap01-bpg
# Alternative requested tier:
# colab new --gpu T4 --session gap01-bpg
colab status --session gap01-bpg
```

L4 is preferred for the original BF16/NF4 receiver. T4 does not support native
BF16, so validate the unchanged receiver in the bounded n=6 run first. A float16
override must not silently become the same-stack benchmark. Do not extrapolate
RTX4060 GPU timings to Colab; record actual GPU/library/dtype fingerprints.

## Required assets (additional gate)

Use `/content/vqa_semcom` as an isolated runtime root. `VQA_SEMCOM_ROOT` changes
the benchmark root. Asset JSON is left unchanged; original lab absolute paths
are mapped in memory. Upload the prepared script archive and learned true-PHY
cache archive with actual CLI syntax:

```bash
colab upload --session gap01-bpg gap01_scripts.tar.gz /content/gap01_scripts.tar.gz
colab upload --session gap01-bpg jpeg2400_reuse.tar.gz /content/jpeg2400_reuse.tar.gz
```

Those two local archives **do not contain the complete dataset/model stack**.
The unchanged frozen asset bundle is also required:

- `outputs/rgb_selector_large_20260922/`: test_manifest, test_records,
  test_truth.sealed, grid_test/representations; all selected original images
  and2k/4k learned native packets and decoded PNGs referenced by those JSONs.
- `outputs/prerun_protocol_finalization_20260908/phy.py` and its dependencies.
- `outputs/rgb_rate_visual_budget_20260922/code/runtime.py`;
  `outputs/qwen_compression_joint_stage1_20260921/run_stage1.py`;
  `outputs/tdiuc_qlora_pilot_20260916/run_training.py` and imported local modules.
- `outputs/rgb_joint_selector_20260922/code/route_frame.py` and imports.
- Original `outputs/tdiuc_qlora_pilot_20260916/standard/final_adapter/` including
  adapter config; weights SHA256 must match
  `a2c9476840f0aa56553ff12f7eaec5be4b7b4df32df3c2d7865aea4ebe3c6a21`.
- Original frozen base model/processor/tokenizer assets referenced by receiver
  source, or an explicitly verified source/version. Do not silently switch model.
- Frozen live scanner at `code/vqa_semcom_v0/experiments/rgb_channel_snr_scan/run_snr_scan.py`
  (historical known SHA `d63c62a1518bd2d39fdd335eb4ff2de5530cac53bd8fefb41c74eaa0f0ee6d3b`).
  The GitHub a22dc52 older scanner lacks v2 and cannot pass this gate.

Most dataset/adapter assets are not present on box. Authentication alone does
not remove **ASSET_BUNDLE_NEEDED**. Obtain the original assets via available
Drive/object-store/parent transfer; do not restart the disabled lab SSH loop.
Verify all cache/source/image/packet hashes before inference. Original path
literals in imported frozen modules can be supported by a compatibility symlink
**inside this ephemeral VM only**:

```python
import os, pathlib, subprocess, tarfile
root=pathlib.Path('/content/vqa_semcom'); root.mkdir(parents=True,exist_ok=True)
os.environ['VQA_SEMCOM_ROOT']=str(root)
# Extract trusted prepared archives and unchanged asset bundle to their documented roots.
with tarfile.open('/content/gap01_scripts.tar.gz') as f: f.extractall(root,filter='data')
reuse=root/'outputs/jpeg2400_reuse';reuse.mkdir(parents=True,exist_ok=True)
with tarfile.open('/content/jpeg2400_reuse.tar.gz') as f: f.extractall(reuse,filter='data')
legacy=pathlib.Path('/home/qiankun/phd_research/vqa_semcom')
legacy.parent.mkdir(parents=True,exist_ok=True)
if not legacy.exists():legacy.symlink_to(root,target_is_directory=True)
else:assert legacy.resolve()==root
```

Do not copy the old GitHub scanner over a frozen scanner supplied by the asset
bundle. Do not modify a production checkout to satisfy this path mapping.

## Install and preflight inside VM

Run Python via `colab exec --session gap01-bpg --file vm_setup.py --timeout 600`
after preparing that file with the following commands through subprocess:

```bash
apt-get update
apt-get install -y build-essential libpng-dev libjpeg-dev python3-venv curl patch
bash /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/install_libbpg.sh /content/bpg-tools
# Install pinned receiver/PHY dependencies from the original environment manifest.
# Requires torch/transformers/peft/bitsandbytes/Pillow/numpy/numba/pyldpc.
# Colab's preinstalled versions are not automatically equivalent to the lab stack.
export VQA_SEMCOM_ROOT=/content/vqa_semcom
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/preflight_colab.py --reuse-phy /content/vqa_semcom/outputs/jpeg2400_reuse
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/verify_rule_b.py --repo-root /content/vqa_semcom
```

The installer builds official pinned source with declared GCC header and
non-assembly compatibility changes, records binary hashes and does not install
into a production system prefix. Any new VM binaries are separately fingerprinted;
they are not claimed equal to box binaries without comparison.

## Bounded smoke, then full

In a Python file executed by Colab CLI, use `subprocess.run([...],check=True)` for
these shell-equivalent commands. Use active-kernel execution for liveness and
frequent checkpoints; a detached child alone does not guarantee Colab retention.

```bash
export VQA_SEMCOM_ROOT=/content/vqa_semcom
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/bpg_bitstream_mode2_full_grid.py --n 6 --workers 2 --output /content/vqa_semcom/outputs/gap01_bpg_smoke6 --reuse-phy /content/vqa_semcom/outputs/jpeg2400_reuse --bpgenc /content/bpg-tools/bin/bpgenc --bpgdec /content/bpg-tools/bin/bpgdec --max-new-vlm 12
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/validate_results.py --output /content/vqa_semcom/outputs/gap01_bpg_smoke6
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/bpg_bitstream_mode2_full_grid.py --n 2400 --workers 4 --output /content/vqa_semcom/outputs/gap01_bpg2400 --reuse-phy /content/vqa_semcom/outputs/jpeg2400_reuse --reuse-bpg-smoke /content/vqa_semcom/outputs/gap01_bpg_smoke6 --bpgenc /content/bpg-tools/bin/bpgenc --bpgdec /content/bpg-tools/bin/bpgdec --max-new-vlm 4788
python /content/vqa_semcom/code/vqa_semcom_v0/experiments/bpg_phy_benchmark_n2400/validate_results.py --output /content/vqa_semcom/outputs/gap01_bpg2400
```

Workers4 are a starting Colab CPU budget; set≤12 only after checking assigned
vCPUs/RAM. Resume the same full command plus`--resume`. Preserve the entire
output directory, source/binary files and assets across VM loss; original resume
requires identical protocol fingerprints and paths. Download output archives
before VM disposal; do not delete/stop sessions implicitly in this batch.

Expected n=6≤12 BPG clean VLM+1warmup,≤144 new BPG PHY; learned144 reused.
Full≤4,788 new VLM+1warmup after smoke,≤57,456 new noisy PHY after smoke;
learned57,600 reused. Exact counts depend on encode failures and byte-identical
task deduplication. Historical noise proxy is never an input. Archive final
72 per-seed/24 mean-SD real rows, journals,protocol/report and freeze PASS.
