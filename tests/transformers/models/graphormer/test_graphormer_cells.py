# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Test Graphormer Cell
"""
import unittest
import numpy as np
import mindspore as ms


from mindnlp.transformers.models.graphormer.modeling_graphormer import (
    GraphormerMultiheadAttention,
    GraphormerGraphEncoderLayer,
    GraphormerGraphEncoder)
from mindnlp.transformers.models.graphormer.configuration_graphormer import (
    GraphormerConfig)

from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_configuration_common import ConfigTester
from .test_modeling_graphormer import GraphormerModelTester


class GraphormerMultiheadAttentionTest(ModelTesterMixin, unittest.TestCase):
    def setUp(self):
        self.model_tester = GraphormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraphormerConfig, has_text_modality=False)

    def prepare_config_and_inputs(self, batch_size: int = 10):
        config = self.model_tester.get_config()
        mdt = self.model_tester
        query = floats_tensor([mdt.graph_size + 1, mdt.batch_size, mdt.embedding_dim])
        attn_bias = ids_tensor(
            [mdt.batch_size * mdt.num_attention_heads, mdt.graph_size + 1, mdt.graph_size + 1],
            config.num_atoms * 2 + 1)
        key_padding_mask = ms.tensor(np.full((mdt.batch_size, mdt.graph_size + 1), False))
        inputs = dict(query=query,
                      key=query,
                      value=query,
                      attn_bias=attn_bias,
                      key_padding_mask=key_padding_mask,
                      need_weights=False,
                      attn_mask=None)

        return config, inputs

    def test_model(self):
        config, inputs = self.prepare_config_and_inputs()
        model = GraphormerMultiheadAttention(config)
        result = model(**inputs)
        self.assertEqual(
            result[0].shape, (self.model_tester.graph_size + 1,
                              self.model_tester.batch_size,
                              self.model_tester.hidden_size)
        )


class GraphormerGraphEncoderLayerTest(ModelTesterMixin, unittest.TestCase):
    def setUp(self):
        self.model_tester = GraphormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraphormerConfig, has_text_modality=False)

    def prepare_config_and_inputs(self, batch_size: int = 10):
        config = self.model_tester.get_config()
        mdt = self.model_tester
        input_nodes = floats_tensor([mdt.graph_size + 1, mdt.batch_size, mdt.embedding_dim])
        self_attn_bias = ids_tensor([mdt.batch_size,
                                     mdt.num_attention_heads,
                                     mdt.graph_size + 1,
                                     mdt.graph_size + 1],config.num_atoms * 2 + 1)
        self_attn_padding_mask = ms.tensor(np.full((mdt.batch_size, mdt.graph_size + 1), False))
        inputs = dict(input_nodes = input_nodes,
                      self_attn_bias = self_attn_bias,
                      self_attn_mask = None,
                      self_attn_padding_mask = self_attn_padding_mask)
        return config, inputs

    def test_model(self):
        config, inputs = self.prepare_config_and_inputs()
        model = GraphormerGraphEncoderLayer(config)
        result = model(**inputs)
        self.assertEqual(
            result[0].shape, (self.model_tester.graph_size + 1,
                              self.model_tester.batch_size,
                              self.model_tester.hidden_size)
        )


class GraphormerGraphEncoderTest(ModelTesterMixin, unittest.TestCase):
    def setUp(self):
        self.model_tester = GraphormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraphormerConfig,
                                          has_text_modality=False)

    def prepare_config_and_inputs(self, batch_size: int = 10):
        config = self.model_tester.get_config()
        mdt = self.model_tester
        input_nodes = ids_tensor([mdt.batch_size, mdt.graph_size, 1],
                                 mdt.num_atoms)
        input_edges = ids_tensor([mdt.batch_size,
                                  mdt.graph_size,
                                  mdt.graph_size,
                                  mdt.multi_hop_max_dist, 1], mdt.num_edges)
        attn_bias = ids_tensor([mdt.batch_size,
                                mdt.graph_size + 1,
                                mdt.graph_size + 1], mdt.num_atoms)
        in_degree = ids_tensor([mdt.batch_size, mdt.graph_size], mdt.num_in_degree)
        out_degree = ids_tensor([mdt.batch_size, mdt.graph_size], mdt.num_out_degree)
        spatial_pos = ids_tensor([mdt.batch_size, mdt.graph_size, mdt.graph_size], mdt.num_spatial)
        attn_edge_type = ids_tensor([mdt.batch_size, mdt.graph_size, mdt.graph_size, 1], mdt.num_edges)


        self_attn_bias = ids_tensor([mdt.batch_size,
                                     mdt.num_attention_heads,
                                     mdt.graph_size + 1,
                                     mdt.graph_size + 1],config.num_atoms * 2 + 1)
        self_attn_padding_mask = ms.tensor(np.full((mdt.batch_size, mdt.graph_size + 1), False))
        inputs = dict(input_nodes=input_nodes,
                      input_edges=input_edges,
                      attn_bias=attn_bias,
                      in_degree=in_degree,
                      out_degree=out_degree,
                      spatial_pos=spatial_pos,
                      attn_edge_type=attn_edge_type)
        return config, inputs

    def test_model(self):
        config, inputs = self.prepare_config_and_inputs()
        model = GraphormerGraphEncoder(config)
        inner_states, graph_rep = model(**inputs)
        # what about layerdrop?
        self.assertEqual(len(inner_states), self.model_tester.num_hidden_layers+1)
        # difference beteween hidden_size and embedding_dim?
        self.assertEqual(inner_states[0].shape, (self.model_tester.graph_size + 1,
                                               self.model_tester.batch_size,
                                               self.model_tester.embedding_dim))
        self.assertEqual(graph_rep.shape, (self.model_tester.batch_size,
                                           self.model_tester.embedding_dim))
