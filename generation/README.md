# MMAudio Fine-tuning for Liquid Pouring Audio Synthesis

This repository contains a fine-tuned version of [MMAudio](https://github.com/hkchengrex/MMAudio) specialized for generating synchronized audio for liquid pouring actions. This work extends the original MMAudio model to achieve better performance on pouring-specific audio-visual synthesis tasks.

## Overview

This project adapts the MMAudio model for specialized liquid pouring audio generation. The fine-tuned model demonstrates improved quality and realism when generating audio for various pouring scenarios compared to the base MMAudio.

**Supported Pouring Types:**
- Water pouring
- Milk pouring  
- Soda pouring
- Rice pouring

## Dataset

The custom pouring dataset contains:
- **Training samples**: 9,111 video clips
- **Validation samples**: 93 video clips
- **Test samples**: 93 video clips
- **Duration**: 8 seconds per clip
- **Resolution**: 384×384 pixels
- **Audio quality**: 44.1kHz sampling rate

### Dataset Structure
```
data/
├── videos/           # Video files (.mp4)
├── train.tsv        # Training set annotations (id + label format)
├── val.tsv          # Validation set annotations
└── test.tsv         # Test set annotations
```

**Dataset Download**:
- Annotations: https://drive.google.com/file/d/1xV3_jmC7heDxavMLmaLsZhtmrVDntGFb/view?usp=sharing
- Videos: https://drive.google.com/file/d/1FPG6G_fXyI6vDynW05uRkwh715eYuQIi/view?usp=sharing

**Setup dataset**:
```bash
# Download and extract annotations
unzip dataset_annotations.zip -d data/

# Download and extract videos  
mkdir -p data/videos
tar -xzf dataset_videos.tar.gz -C data/videos/
```

## Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.5.1+ with CUDA support
- Ubuntu/Linux (tested environment)
- GPU with 12GB+ VRAM recommended

### Setup

1. **Clone this repository:**
```bash
git clone https://github.com/renwang435/multigen.git
cd generation/
```

2. **Install dependencies:**
```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMAudio package
pip install -e .
```

3. **Obtain required weights:**
   - Download base MMAudio weights from [Hugging Face](https://huggingface.co/hkchengrex/MMAudio)
   - Place `mmaudio_large_44k_v2.pth` in the `weights/` directory
   - External dependencies (VAE, vocoder) will be downloaded automatically

## Training

### Fine-tuning from Pretrained Model

To reproduce the fine-tuning results:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py \
    exp_id=pouring_finetune \
    model=large_44k_v2 \
    weights=weights/mmaudio_large_44k_v2.pth \
    num_iterations=20000
```

### Training from Scratch

For comparison, training from scratch:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py \
    exp_id=pouring_scratch \
    model=large_44k_v2
```

### Configuration

Key training parameters:
- `exp_id`: Experiment identifier for output organization
- `model`: Model architecture (large_44k_v2 for best quality)
- `weights`: Path to pretrained weights for fine-tuning
- `num_iterations`: Total training iterations
- `batch_size`: Per-GPU batch size (default: 4)

Training outputs are saved in `output/{exp_id}/` including:
- Model checkpoints (.pth)
- EMA weights
- Training logs

### Pre-trained Fine-tuned Model

A pre-trained fine-tuned model checkpoint is available:
- **Download**: https://drive.google.com/file/d/14aD_f8aRtmr98fGU6ZZbgCcWSh6vfdSJ/view?usp=sharing
- **Model**: `pouring_large_window_last.pth`
- **Training**: Fine-tuned on 9,111 pouring video samples
- **Performance**: Optimized for 4-12 second audio generation
- **Usage**: Compatible with standard MMAudio inference pipeline

This checkpoint can be directly used for inference without retraining.

## Inference

### Command-line Interface

Generate audio for your videos:

```bash
python demo.py --duration=8 --video=path/to/video.mp4 --prompt="pouring water"
```

### Using Fine-tuned Models

To use the fine-tuned model for inference with a custom checkpoint:

```bash
python demo.py \
    --duration=12 \
    --video=path/to/your/video.mp4 \
    --prompt="pouring water" \
    --negative_prompt="music" \
    --ckpt=output/pouring_large_window/pouring_large_window_last.pth
```

**Parameters:**
- `--duration`: Audio duration in seconds (4-12s work well)
- `--video`: Path to input video file
- `--prompt`: Positive text prompt to guide audio generation
- `--negative_prompt`: Negative prompt to avoid unwanted audio characteristics
- `--ckpt`: Path to fine-tuned model checkpoint



## Technical Details

### Model Architecture
- Based on MMAudio large_44k_v2 architecture
- Flow-matching-based audio generation with synchronization module
- CLIP visual encoder and Synchformer for temporal alignment
- 44.1kHz audio output for high fidelity

### Data Processing
- Video preprocessing: 8 FPS for CLIP, 25 FPS for Synchformer
- Audio processing: 44.1kHz sampling, 8-second duration
- Multimodal dataset loading with memory mapping for efficiency

### Hardware Requirements
- **Minimum**: 12GB GPU memory, 32GB RAM
- **Recommended**: 24GB+ GPU memory, 64GB+ RAM, multiple GPUs
- **Storage**: 50GB+ free space for data and checkpoints

## Customization

To adapt this work for your own dataset:

1. **Prepare your data** in the same TSV format:
   ```
   id    label
   video_001    your_action_description
   ```

2. **Update data configuration** in `config/data/base.yaml`

3. **Modify data loading** in `mmaudio/data/data_setup.py` to use your dataset

## License

This project is based on MMAudio and follows the same licensing terms. Please refer to the original MMAudio repository for complete license information.

## Acknowledgments

This work builds upon the excellent MMAudio framework. We thank the original authors for making their code and models available.

- [MMAudio](https://github.com/hkchengrex/MMAudio) - Base model and framework
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) - VAE architecture and BigVGAN model
- [Synchformer](https://github.com/v-iashin/Synchformer) - Temporal synchronization
- [EDM2](https://github.com/NVlabs/edm2) - Network architecture components
- [PSNN](https://gamma.cs.unc.edu/PSNN/) - Analyzing Liquid Pouring Sequences via Audio-Visual Neural Networks
- [Sound of Water](https://bpiyush.github.io/pouring-water-website/) - Inferring Physical Properties from Pouring Liquids