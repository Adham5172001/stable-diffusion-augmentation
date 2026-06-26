"""ControlNet Augmentation — Author: Adham Aboulkheir"""
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class AugmentationConfig:
    base_model: str = "stabilityai/stable-diffusion-2-1"
    controlnet: str = "lllyasviel/sd-controlnet-canny"
    n_variations: int = 20
    conditions: List[str] = None
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = ["rain", "fog", "night", "bright_sun", "overcast"]

class ControlNetAugmentor:
    def __init__(self, config=None):
        self.config = config or AugmentationConfig()

    def generate_prompt(self, base_description, condition):
        prompts = {
            "rain": f"{base_description}, heavy rain, wet surfaces, photorealistic",
            "fog": f"{base_description}, dense fog, reduced visibility",
            "night": f"{base_description}, nighttime, artificial lighting",
            "bright_sun": f"{base_description}, bright sunlight, high contrast",
            "overcast": f"{base_description}, overcast sky, diffuse lighting",
        }
        return prompts.get(condition, f"{base_description}, {condition}, photorealistic")

    def augment(self, image, base_description, n_variations=None, conditions=None):
        n = n_variations or self.config.n_variations
        conds = conditions or self.config.conditions
        results = []
        for i in range(n):
            condition = conds[i % len(conds)]
            np.random.seed(i)
            aug = image.copy().astype(float)
            if condition == "night": aug *= 0.3
            elif condition == "fog": aug = aug*0.6 + np.ones_like(aug)*200*0.4
            elif condition == "rain": aug += np.random.normal(0, 15, aug.shape)
            results.append({"image": np.clip(aug, 0, 255).astype(np.uint8), "condition": condition, "prompt": self.generate_prompt(base_description, condition)})
        return results

if __name__ == "__main__":
    config = AugmentationConfig(n_variations=5)
    aug = ControlNetAugmentor(config)
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    results = aug.augment(img, "telecom equipment on rooftop")
    print(f"Generated {len(results)} augmented variations")
    for r in results:
        print(f"  {r['condition']}: {r['prompt'][:60]}...")
