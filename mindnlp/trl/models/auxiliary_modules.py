'''
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# pylint: disable=C,R

import os
import numpy as np
import mindspore as ms
from mindnlp.core import nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from mindnlp.transformers import CLIPModel,normalize,resize


def vector_norm(x, ord=2, dim=None, keepdim=False):
    if dim is None:
        norm = np.linalg.norm(x, ord=ord)
    else:
        norm = np.linalg.norm(x, ord=ord, axis=dim)
    
    if keepdim:
        norm = np.expand_dims(norm, axis=dim)
    
    return norm

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.normalize = normalize
        self.target_size = 224
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = ms.load_checkpoint(cached_path)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        images = resize(self.target_size)(images)
        images = self.normalize(images, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        embed = self.clip.get_image_features(pixel_values=images)
        # normalize embedding
        embed = embed / vector_norm(embed, dim=-1, keepdim=True)
        reward = self.mlp(embed).squeeze(1)
        return reward


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=ms.Tensor,
    )

    def _fn(images):
        images = (images).clamp(0, 1)
        scores = scorer(images)
        return scores, {}

    return _fn
