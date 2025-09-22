<!-- install: -->
# Install
```bash
cd roboverse_learn/algorithms/diffusion_policy

pip install -e .
```

<!-- eval: -->
# Eval
```bash
python roboverse_learn/eval.py --task CloseBox --sim isaaclab --algo_run_name CloseBox_franka_10 --checkpoint_num 400
```

<!-- train: -->
# Train
```bash
cd roboverse_learn/algorithms/diffusion_policy

# process data to zarr
bash data2zarr_dp.sh CloseBox 10 ~/RoboVerse/data_isaaclab/demo/CloseBox/robot-franka

# train
bash train.sh CloseBox_franka 10 42 0 False
```
