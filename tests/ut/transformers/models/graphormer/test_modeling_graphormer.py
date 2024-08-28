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
Test Graphormer
"""
import unittest
import copy
import inspect
import os
import tempfile
import unittest
import numpy as np

from mindnlp.utils import is_mindspore_available
from mindnlp.transformers.models.graphormer.configuration_graphormer import GraphormerConfig

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor

from mindnlp.utils.testing_utils import (
    require_mindspore,
    slow
)

if is_mindspore_available():
    from mindspore import tensor
    from mindnlp.transformers.models.graphormer.modeling_graphormer import(
        GraphormerModel,
        GraphormerForGraphClassification,
        GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST)

class GraphormerModelTester:
    def __init__(
        self,
        parent,
        num_classes=1,
        num_atoms=32 * 9,
        num_edges=32 * 3,
        num_in_degree=32,
        num_out_degree=32,
        num_spatial=32,
        num_edge_dis=16,
        multi_hop_max_dist=5,  # sometimes is 20
        spatial_pos_max=32,
        edge_type="multi_hop",
        init_fn=None,
        max_nodes=32,
        share_input_output_embed=False,
        num_hidden_layers=2,
        embedding_dim=32,
        ffn_embedding_dim=32,
        num_attention_heads=4,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        layerdrop=0.0,
        encoder_normalize_before=False,
        pre_layernorm=False,
        apply_graphormer_init=False,
        activation_fn="gelu",
        embed_scale=None,
        freeze_embeddings=False,
        num_trans_layers_to_freeze=0,
        traceable=False,
        q_noise=0.0,
        qn_block_size=8,
        kdim=None,
        vdim=None,
        bias=True,
        self_attention=True,
        batch_size=10,
        graph_size=20,
        is_training=True,
    ):
        self.parent = parent
        self.num_classes = num_classes
        self.num_labels = num_classes
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.max_nodes = max_nodes
        self.num_hidden_layers = num_hidden_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.share_input_output_embed = share_input_output_embed
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.init_fn = init_fn
        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        attn_bias = ids_tensor(
            [self.batch_size, self.graph_size + 1, self.graph_size + 1], self.num_atoms
        )  # Def not sure here
        attn_edge_type = ids_tensor([self.batch_size, self.graph_size, self.graph_size, 1], self.num_edges)
        spatial_pos = ids_tensor([self.batch_size, self.graph_size, self.graph_size], self.num_spatial)
        in_degree = ids_tensor([self.batch_size, self.graph_size], self.num_in_degree)
        out_degree = ids_tensor([self.batch_size, self.graph_size], self.num_out_degree)
        input_nodes = ids_tensor([self.batch_size, self.graph_size, 1], self.num_atoms)
        input_edges = ids_tensor(
            [self.batch_size, self.graph_size, self.graph_size, self.multi_hop_max_dist, 1], self.num_edges
        )
        labels = ids_tensor([self.batch_size], self.num_classes)

        config = self.get_config()
        return config, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, input_nodes, input_edges, labels

    def get_config(self):
        return GraphormerConfig(
            num_atoms=self.num_atoms,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_edges=self.num_edges,
            num_spatial=self.num_spatial,
            num_edge_dis=self.num_edge_dis,
            edge_type=self.edge_type,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
            max_nodes=self.max_nodes,
            num_hidden_layers=self.num_hidden_layers,
            embedding_dim=self.embedding_dim,
            hidden_size=self.embedding_dim,
            ffn_embedding_dim=self.ffn_embedding_dim,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            layerdrop=self.layerdrop,
            encoder_normalize_before=self.encoder_normalize_before,
            pre_layernorm=self.pre_layernorm,
            apply_graphormer_init=self.apply_graphormer_init,
            activation_fn=self.activation_fn,
            embed_scale=self.embed_scale,
            freeze_embeddings=self.freeze_embeddings,
            num_trans_layers_to_freeze=self.num_trans_layers_to_freeze,
            share_input_output_embed=self.share_input_output_embed,
            traceable=self.traceable,
            q_noise=self.q_noise,
            qn_block_size=self.qn_block_size,
            init_fn=self.init_fn,
            kdim=self.kdim,
            vdim=self.vdim,
            self_attention=self.self_attention,
            bias=self.bias,
        )

    def create_and_check_model(
        self, config, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, input_nodes, input_edges, labels
    ):
        model = GraphormerModel(config=config)
        model.set_train(False)
        result = model(
            input_nodes=input_nodes,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            input_edges=input_edges,
            attn_edge_type=attn_edge_type,
            labels=labels,
        )
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.graph_size + 1, self.hidden_size)
        )

    def create_and_check_for_graph_classification(
        self, config, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, input_nodes, input_edges, labels
    ):
        model = GraphormerForGraphClassification(config)
        model.set_train(False)
        result = model(
            input_nodes=input_nodes,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            input_edges=input_edges,
            attn_edge_type=attn_edge_type,
            labels=labels
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            attn_bias,
            attn_edge_type,
            spatial_pos,
            in_degree,
            out_degree,
            input_nodes,
            input_edges,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "attn_bias": attn_bias,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "input_nodes": input_nodes,
            "input_edges": input_edges,
            "labels": labels,
        }
        return config, inputs_dict

@require_mindspore
class GraphormerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GraphormerForGraphClassification,)
    all_generative_model_classes = ()
    pipeline_model_mapping = {"feature-extraction": GraphormerModel}
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"
    has_attentions = False  # does not output attention

    def setUp(self):
        self.model_tester = GraphormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraphormerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Graphormer does not use one single inputs_embedding but three")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Graphormer does not implement feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Graphormer does not share input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Graphormer does not share common arg names")
    def test_forward_signature(self):
        pass

    def test_initialization(self):
        def _config_zero_init(config):
            configs_no_init = copy.deepcopy(config)
            for key in configs_no_init.__dict__.keys():
                if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
                    setattr(configs_no_init, key, 1e-10)
            return configs_no_init

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.parameters_and_names():
                if param.requires_grad:
                    self.assertTrue(
                        -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.set_train(False)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            batch_size = self.model_tester.batch_size

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [batch_size, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Always returns hidden_states
            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="Skip the grad related tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # Inputs are 'input_nodes' and 'input_edges' not 'input_ids'
    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name_nodes = list(model_signature.parameters.keys())[1]
            observed_main_input_name_edges = list(model_signature.parameters.keys())[2]
            self.assertEqual(model_class.main_input_name_nodes, observed_main_input_name_nodes)
            self.assertEqual(model_class.main_input_name_edges, observed_main_input_name_edges)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_nodes", "input_edges"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_graph_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_graph_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GraphormerForGraphClassification.from_pretrained(model_name)
            self.assertIsNotNone(model)

@require_mindspore
class GraphormerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_graph_classification(self):
        model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")

        # Actual real graph data from the MUTAG dataset
        # fmt: off
        model_input = {
            "attn_bias": tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                    ],
                ]
            ),
            "attn_edge_type": tensor(
                [
                    [
                        [[0], [3], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [3], [0], [3], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [3], [0], [3], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[3], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [3], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [3], [0], [0], [0]],
                        [[0], [0], [0], [3], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [3], [3], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [3], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [3], [3]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0]],
                    ],
                    [
                        [[0], [3], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0]],
                        [[3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [3], [0], [3], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [3], [0], [0], [0], [3], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [3], [0], [3], [3], [0], [0], [0], [0], [0], [0]],
                        [[3], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [3], [3], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    ],
                ]
            ),
            # fmt: on
            "spatial_pos": tensor(
                [
                    [
                        [1, 2, 3, 4, 3, 2, 4, 5, 6, 5, 6, 7, 8, 7, 9, 10, 10],
                        [2, 1, 2, 3, 4, 3, 5, 6, 5, 4, 5, 6, 7, 6, 8, 9, 9],
                        [3, 2, 1, 2, 3, 4, 4, 5, 4, 3, 4, 5, 6, 5, 7, 8, 8],
                        [4, 3, 2, 1, 2, 3, 3, 4, 3, 2, 3, 4, 5, 4, 6, 7, 7],
                        [3, 4, 3, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 5, 7, 8, 8],
                        [2, 3, 4, 3, 2, 1, 3, 4, 5, 4, 5, 6, 7, 6, 8, 9, 9],
                        [4, 5, 4, 3, 2, 3, 1, 2, 3, 4, 5, 6, 5, 4, 6, 7, 7],
                        [5, 6, 5, 4, 3, 4, 2, 1, 2, 3, 4, 5, 4, 3, 5, 6, 6],
                        [6, 5, 4, 3, 4, 5, 3, 2, 1, 2, 3, 4, 3, 2, 4, 5, 5],
                        [5, 4, 3, 2, 3, 4, 4, 3, 2, 1, 2, 3, 4, 3, 5, 6, 6],
                        [6, 5, 4, 3, 4, 5, 5, 4, 3, 2, 1, 2, 3, 4, 4, 5, 5],
                        [7, 6, 5, 4, 5, 6, 6, 5, 4, 3, 2, 1, 2, 3, 3, 4, 4],
                        [8, 7, 6, 5, 6, 7, 5, 4, 3, 4, 3, 2, 1, 2, 2, 3, 3],
                        [7, 6, 5, 4, 5, 6, 4, 3, 2, 3, 4, 3, 2, 1, 3, 4, 4],
                        [9, 8, 7, 6, 7, 8, 6, 5, 4, 5, 4, 3, 2, 3, 1, 2, 2],
                        [10, 9, 8, 7, 8, 9, 7, 6, 5, 6, 5, 4, 3, 4, 2, 1, 3],
                        [10, 9, 8, 7, 8, 9, 7, 6, 5, 6, 5, 4, 3, 4, 2, 3, 1],
                    ],
                    [
                        [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 4, 5, 5, 0, 0, 0, 0],
                        [2, 1, 2, 3, 4, 5, 4, 3, 4, 3, 5, 6, 6, 0, 0, 0, 0],
                        [3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 4, 5, 5, 0, 0, 0, 0],
                        [4, 3, 2, 1, 2, 3, 4, 3, 4, 5, 5, 6, 6, 0, 0, 0, 0],
                        [5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 6, 7, 7, 0, 0, 0, 0],
                        [6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 5, 6, 6, 0, 0, 0, 0],
                        [5, 4, 3, 4, 3, 2, 1, 2, 3, 4, 4, 5, 5, 0, 0, 0, 0],
                        [4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 3, 4, 4, 0, 0, 0, 0],
                        [3, 4, 3, 4, 5, 4, 3, 2, 1, 2, 2, 3, 3, 0, 0, 0, 0],
                        [2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 3, 4, 4, 0, 0, 0, 0],
                        [4, 5, 4, 5, 6, 5, 4, 3, 2, 3, 1, 2, 2, 0, 0, 0, 0],
                        [5, 6, 5, 6, 7, 6, 5, 4, 3, 4, 2, 1, 3, 0, 0, 0, 0],
                        [5, 6, 5, 6, 7, 6, 5, 4, 3, 4, 2, 3, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            "in_degree": tensor(
                [
                    [3, 3, 3, 4, 4, 3, 3, 3, 4, 4, 3, 3, 4, 3, 4, 2, 2],
                    [3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 4, 2, 2, 0, 0, 0, 0],
                ]
            ),
            "out_degree": tensor(
                [
                    [3, 3, 3, 4, 4, 3, 3, 3, 4, 4, 3, 3, 4, 3, 4, 2, 2],
                    [3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 4, 2, 2, 0, 0, 0, 0],
                ]
            ),
            "input_nodes": tensor(
                [
                    [[3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3]],
                    [[3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [0], [0], [0], [0]],
                ]
            ),
            "input_edges": tensor(
                [
                    [
                        [
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                    ],
                    [
                        [
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [4]],
                            [[4], [4], [4], [4], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[4], [4], [4], [0], [0]],
                            [[4], [0], [0], [0], [0]],
                            [[4], [4], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                        [
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                            [[0], [0], [0], [0], [0]],
                        ],
                    ],
                ]
            ),
            "labels": tensor([1, 0]),
        }

        output = model(**model_input)["logits"]

        expected_shape = (2, 1)
        self.assertEqual(output.shape, expected_shape)

        expected_logs = tensor(
            [[7.6060], [7.4126]]
        )
        self.assertTrue(np.allclose(output.asnumpy(), expected_logs.asnumpy(), atol=5e-3))

    r"""
    Test Graphormer
    """
    def setUp(self):
        """
        Set up.
        """
        self.config = GraphormerConfig(n_layer=2, vocab_size=1000,
                                       n_embd=128, hidden_size=128,
                                       n_head=8)

    def test_graphormer_model(self):
        r"""
        Test GraphormerModel
        """
        model = GraphormerModel(self.config)
