import unittest
import numpy as np
import mindspore as ms


from mindnlp.transformers.models.graphormer.modeling_graphormer import (
    GraphormerMultiheadAttention)
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
        # query = floats_tensor([mdt.batch_size, mdt.graph_size, mdt.embedding_dim])
        query = floats_tensor([mdt.graph_size + 1, mdt.batch_size, mdt.embedding_dim])
        # attn_bias = ids_tensor(
        #     [mdt.batch_size, mdt.graph_size + 1, mdt.graph_size + 1],
        #     config.num_atoms)
        attn_bias = ids_tensor(
            [mdt.batch_size * mdt.num_attention_heads, mdt.graph_size + 1, mdt.graph_size + 1],
            config.num_atoms)
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
