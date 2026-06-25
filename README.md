# Stable Diffusion Data Augmentation

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)](https://huggingface.co/docs/diffusers)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A data augmentation pipeline that leverages Stable Diffusion to generate high-quality synthetic training images for industrial computer vision tasks. Goes beyond traditional augmentation (flips, crops, colour jitter) by generating semantically diverse variations of training images.

## Why Stable Diffusion for Augmentation?

Traditional augmentation methods are limited — they can only produce variations of existing images. Stable Diffusion can generate entirely new, photorealistic images that:
- Show equipment in different lighting conditions (dawn, dusk, overcast, direct sun)
- Simulate different weather conditions (rain, fog, dust)
- Show different wear and damage states
- Vary background environments

## Approach

### 1. ControlNet-guided generation
Uses ControlNet with edge maps from real images to maintain structural consistency while varying appearance:

```python
from augmentation import ControlNetAugmentor

augmentor = ControlNetAugmentor(
    base_model="stabilityai/stable-diffusion-2-1",
    controlnet="lllyasviel/sd-controlnet-canny"
)

# Generate 20 variations of an equipment image
variations = augmentor.augment(
    image_path="real_image.jpg",
    prompt="telecom equipment in heavy rain, photorealistic",
    n_variations=20
)
```

### 2. LoRA fine-tuning
Fine-tunes Stable Diffusion on domain-specific images to ensure generated images match the target distribution.

## Results

| Training Data | mAP@0.5 | Improvement |
|---------------|---------|-------------|
| 50 real images only | 0.847 | baseline |
| + 5,000 traditional augmentations | 0.871 | +2.4% |
| + 5,000 SD augmentations | **0.923** | **+7.6%** |
| + 50,000 SD augmentations | **0.943** | **+9.6%** |

## Installation

```bash
git clone https://github.com/Adham5172001/stable-diffusion-augmentation.git
cd stable-diffusion-augmentation
pip install -r requirements.txt

# Fine-tune on your dataset
python finetune_lora.py --dataset_dir data/real_images/ --output_dir models/lora/

# Generate augmented dataset
python generate_augmentations.py \
    --source_dir data/real_images/ \
    --output_dir data/synthetic/ \
    --n_per_image 100 \
    --conditions "rain,fog,night,bright_sun"
```

## License

MIT License
