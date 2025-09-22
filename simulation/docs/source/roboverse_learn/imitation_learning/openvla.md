# OpenVLA

OpenVLA is the first open-source 7B Vision-Language-Action model, which was built upon Prismatic VLM. RLDS is the mainstream data format for VLAs. To finetune RoboVerse data on OpenVLA, you need to convert the RoboVerse data to RLDS format. The following steps will guide you through the whole finetuning process.

## RLDS Conversion
In RoboVerse, we have provided `rlds_utils` to convert the RoboVerse data to RLDS format. The script is located at `roboverse_learn/rlds_utils`.
First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```
If you want to manually create an environment, the key packages to install are `tensorflow`,
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.

Then, please refer to the `roboverse` folder. All you need to do is create a soft link of `demo` to the `roboverse` folder and run `tfds build --overwrite` in the `roboverse` folder after verifying your installation of the conversion environment. The script will automatically convert all the episodes into RLDS format. The transformed rlds dataset will be stored in `~/tensorflow_datasets/roboverse_dataset`.

## Finetune OpenVLA
Clone the [RoboVerse](https://github.com/wbjsamuel/RoboVerse-OpenVLA) version of OpenVLA and install the environment as required.
  ```bash
  # Create and activate conda environment
  conda create -n openvla python=3.10 -y
  conda activate openvla

  # Install PyTorch. Below is a sample command to do this, but you should check the following link
  # to find installation instructions that are specific to your compute platform:
  # https://pytorch.org/get-started/locally/
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

  # Clone and install the openvla repo
  git clone https://github.com/openvla/openvla.git
  cd openvla
  pip install -e .

  # Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
  #   =>> If you run into difficulty, try `pip cache remove flash_attn` first
  pip install packaging ninja
  ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
  pip install "flash-attn==2.5.5" --no-build-isolation
  ```
Then, create a symbolic link of converted rlds data to your workspace.
```bash
cd openvla
ln -s <path_to_rlds_data> data
```
Launch finetuning with the following command:
```bash
cd openvla
conda activate openvla-tf
pip install -e .
bash launch_finetune.sh
```
`launch_finetune.sh` is as follows. You can modify `launch_finetune.sh` to change the finetuning settings.
```bash
#!/bin/bash

export HF_TOKEN=YOUR_HF_TOKEN
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export HF_HOME=cache

torchrun --standalone --nnodes=1 --nproc-per-node=8 vla-scripts/finetune.py \
    --vla_path=openvla/openvla-7b \
    --data_root_dir=data/ \
    --dataset_name=pickcube \
    --run_root_dir=runs \
    --lora_rank=32 \
    --batch_size=16 \
    --grad_accumulation_steps=1 \
    --learning_rate=5e-4 \
    --image_aug=True \
    --wandb_project=openvla-finetune-lora32-pickcube \
    --wandb_entity=YOUR_WANDB_ENTITY \
    --save_steps=5000 \
    --save_latest_checkpoint_only=True # [Optional] Whether to save only one checkpoint per run and continually overwrite the latest checkpoint(If False, saves all checkpoints)
```

## Evaluation on RoboVerse
After finetuning, you can evaluate the model on RoboVerse. The evaluation script is located at `roboverse_learn/eval_vla.py`. You can run the script with the following command:
```bash

python roboverse_learn/eval_vla.py --task TASK_NAME --algo openvla --ckpt_path PATH_TO_CHECKPOINT
```

## For Developers
- Codebase on Berkeley EM: `/home/ghr/wangbangjun/RoboVerse-OpenVLA`.
- Launch Script: `stackcube_l0_finetune.sh`.
- Dataset transform `RoboVerse-OpenVLA/prismatic/vla/datasets/rlds/oxe/transforms.py`. V2 uses `roboverse_v2_dataset_transform`.
- Data path: `/home/ghr/tensorflow_datasets/stackcube_level0/`
