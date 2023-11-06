# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test CLIP"""
import unittest
import numpy as np
import mindspore
from mindspore import Tensor

import mindnlp
from mindnlp.transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig
from mindnlp.transformers.models.clip.clip import (CLIPVisionEmbeddings,
                           CLIPTextEmbeddings,
                           CLIPAttention,
                           CLIPMLP,
                           CLIPEncoderLayer,
                           CLIPTextTransformer,
                           CLIPEncoder,
                           CLIPTextModel,
                           CLIPVisionTransformer,
                           CLIPVisionModel,
                           CLIPModel,
                           CLIPTextModelWithProjection,
                           CLIPVisionModelWithProjection,
                           )


class TestModelingCLIP(unittest.TestCase):
    """
    Test for clip model
    """
    def setUp(self):
        """
        Set up.
        """
        super().setUp()
        self.text_config = CLIPTextConfig(vocab_size=1000,
                                          hidden_size=128,
                                          intermediate_size=256,
                                          projection_dim=128,
                                          num_hidden_layers=2)
        self.vision_config = CLIPVisionConfig(hidden_size=128,
                                              intermediate_size=256,
                                              projection_dim=128,
                                              num_hidden_layers=2,
                                              num_attention_heads=8)

    def test_clip_vision_embeddings(self):
        """
        Test CLIP Vision Embeddings
        """
        input_x = Tensor(np.random.randn(3, 3, 224, 224), mindspore.float32)
        vision_embed = CLIPVisionEmbeddings(self.vision_config)
        logits = vision_embed(input_x)

        assert logits.shape == (3, 50, self.vision_config.hidden_size)

    def test_clip_text_embeddings(self):
        """
        Test CLIP Text Embeddings
        """
        input_x = Tensor(np.random.randint(low=0 ,high=10, size=(1,77)), mindspore.int32)
        text_embed = CLIPTextEmbeddings(self.text_config)
        logits = text_embed(input_x)

        assert logits.shape == (1, 77, self.text_config.hidden_size)

    def test_clip_attention(self):
        """
        Test CLIP Attention
        """
        hidden_states = Tensor(np.random.randn(2, 10, self.text_config.hidden_size), mindspore.float32)
        attention_mask = Tensor(np.random.randint(0, 2, (2, 1, 10, 10)), mindspore.int32)
        causal_attention_mask = Tensor(np.random.randint(0, 2, (2, 1, 10, 10)), mindspore.int32)

        text_attention = CLIPAttention(self.text_config)
        attn_output, attn_weights_reshaped = text_attention(hidden_states, attention_mask, causal_attention_mask, output_attentions=True)

        assert attn_output.shape == (2, 10, self.text_config.hidden_size)
        assert attn_weights_reshaped.shape == (2, 8, 10, 10)

    def test_clip_mlp(self):
        """
        Test CLIP Mlp
        """
        hidden_states = Tensor(np.random.randn(2, 5, self.text_config.hidden_size), mindspore.float32)
        mlp_layer = CLIPMLP(self.text_config)
        logits = mlp_layer(hidden_states)

        assert logits.shape == (2, 5, self.text_config.hidden_size)

    def test_clip_encoder_layer(self):
        """
        Test CLIP Encoder Layer
        """
        hidden_states = Tensor(np.random.randn(2, 5, self.text_config.hidden_size), mindspore.float32)
        attention_mask = Tensor(np.random.randint(0, 2, (2, 1, 5, 5)), mindspore.int32)
        causal_attention_mask = Tensor(np.random.randint(0, 2, (2, 1, 5, 5)), mindspore.int32)
        clip_encoder = CLIPEncoderLayer(self.text_config)
        logits = clip_encoder(hidden_states, attention_mask, causal_attention_mask)

        assert np.array(logits).shape == (1, 2, 5, self.text_config.hidden_size)

    def test_clip_text_transformer(self):
        """
        Test CLIP Text Transformer
        """
        input_ids = Tensor(np.random.randint(0, 10, (2, 3)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 3)), mindspore.int32)
        attention_mask = Tensor(np.random.randint(0, 2, (2, 3)), mindspore.int32)

        model = CLIPTextTransformer(self.text_config)



        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        assert outputs[0].shape == (2, 3, self.text_config.hidden_size)
        assert outputs[1].shape == (2, self.text_config.hidden_size)

    def test_clip_encoder(self):
        """
        Test CLIP Encoder
        """
        batch_size = 2
        seq_length = 5
        inputs_embeds = Tensor(np.random.randn(batch_size, seq_length, self.text_config.hidden_size), mindspore.float32)
        attention_mask = Tensor(np.random.randint(0, 2, (2,1,5,5)), mindspore.int32)
        encoder = CLIPEncoder(self.text_config)
        logits = encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        assert np.array(logits).shape == (1, 2, 5, self.text_config.hidden_size)

    def test_clip_text_model(self):
        """
        Test CLIP Text Model
        """
        model = CLIPTextModel(self.text_config)



        input_ids = Tensor(np.random.randint(low=0, high=self.text_config.vocab_size, size=(1, 77)), mindspore.int32)
        outputs = model(input_ids=input_ids)

        assert outputs[0].shape == (1, 77, self.text_config.hidden_size)
        assert outputs[1].shape == (1, self.text_config.hidden_size)

    def test_clip_vision_transformer(self):
        """
        Test CLIP Vision Transformer
        """
        batch_size = 2
        image_height = 224
        image_width = 224
        num_channels = 3
        pixel_values = Tensor(np.random.randn(batch_size, num_channels, image_height, image_width), mindspore.float32)
        model = CLIPVisionTransformer(self.vision_config)



        outputs = model(pixel_values)

        assert outputs[0].shape == (2, 50, self.vision_config.hidden_size)
        assert outputs[1].shape == (2, self.vision_config.hidden_size)

    def test_clip_vision_model(self):
        """
        Test CLIP Vision Model
        """
        model = CLIPVisionModel(self.vision_config)



        pixel_values = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)
        outputs = model(pixel_values=pixel_values)

        assert outputs[0].shape == (1, 50, self.vision_config.hidden_size)
        assert outputs[1].shape == (1, self.vision_config.hidden_size)

    def test_clip_model(self):
        """
        Test CLIP Model
        """
        config = CLIPConfig(self.text_config, self.vision_config)

        model = CLIPModel(config=config)



        input_ids = Tensor(np.random.randint(0, 10, (2, 3)), mindspore.int32)
        pixel_values = Tensor(np.random.randn(self.vision_config.hidden_size, 3, 224, 224), mindspore.float32)
        attention_mask = Tensor(np.random.randint(0, 2, (2, 3)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 3, (2, 3)), mindspore.int32)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        assert outputs[0].shape == (self.vision_config.hidden_size, 2)
        assert outputs[1].shape == (2, self.vision_config.hidden_size)
        assert outputs[2].shape == (2, config.projection_dim)
        assert outputs[3].shape == (self.vision_config.hidden_size, config.projection_dim)
        assert outputs[4][0].shape == (2, 3, self.vision_config.hidden_size)
        assert outputs[4][1].shape == (2, self.vision_config.hidden_size)
        assert outputs[5][0].shape == (self.vision_config.hidden_size, 50, self.vision_config.hidden_size)
        assert outputs[5][1].shape == (self.vision_config.hidden_size, self.vision_config.hidden_size)


    def test_clip_text_model_with_projection(self):
        """
        Test CLIP Text Model with projection
        """
        model = CLIPTextModelWithProjection(self.text_config)



        input_ids = Tensor(np.random.randint(0, 10, (1, 16)), mindspore.int32)
        attention_mask = Tensor(np.random.randint(0, 2, (1, 16)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (1, 16)), mindspore.int32)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

        assert outputs[0].shape == (1, self.text_config.hidden_size)
        assert outputs[1].shape == (1, 16, self.vision_config.hidden_size)

    def test_clip_vision_model_with_projection(self):
        """
        Test CLIP Vision Model with projection
        """
        model = CLIPVisionModelWithProjection(self.vision_config)



        pixel_values = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)
        outputs = model(pixel_values=pixel_values)

        assert outputs[0].shape == (1, self.text_config.hidden_size)
        assert outputs[1].shape == (1, 50, self.vision_config.hidden_size)
