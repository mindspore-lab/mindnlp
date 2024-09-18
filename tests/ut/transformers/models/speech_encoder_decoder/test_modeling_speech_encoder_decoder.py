# coding=utf-8
# Copyright 2021 HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tempfile
import unittest

# from transformers import is_torch_available
from mindnlp.utils.testing_utils import slow

from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ..bert.test_modeling_bert import BertModelTester
from ..wav2vec2.test_modeling_wav2vec2 import Wav2Vec2ModelTester



import numpy as np
import mindspore
import mindspore.common.dtype as mstype
from mindspore import Tensor

from mindnlp.core import ops
from mindnlp.transformers import (
    BertLMHeadModel,
    SpeechEncoderDecoderConfig,
    SpeechEncoderDecoderModel,
    Wav2Vec2Model,
)
from mindnlp.transformers.modeling_outputs import BaseModelOutput
from mindnlp.transformers.models.speech_to_text.modeling_speech_to_text import Speech2TextEncoder
from mindnlp.transformers.models.speech_to_text.configuration_speech_to_text import Speech2TextConfig

def prepare_speech_to_text_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_features.ne(0)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = ops.ones(config.encoder_layers, config.encoder_attention_heads)
    if decoder_head_mask is None:
        decoder_head_mask = ops.ones(config.decoder_layers, config.decoder_attention_heads)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = ops.ones(config.decoder_layers, config.decoder_attention_heads)
    return {
        # "input_ids": input_features,
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }

class Speech2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        num_conv_layers=2,
        conv_kernel_sizes=(5, 5),
        conv_channels=32,
        input_feat_per_channel=24,
        input_channels=1,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=20,
        max_target_positions=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_features = floats_tensor(
            [self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size
        )
        attention_mask = ops.ones([self.batch_size, self.seq_length], dtype=mstype.int64)
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(2)

        config = self.get_config()
        inputs_dict = prepare_speech_to_text_inputs_dict(
            config,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
        return config, inputs_dict

    def get_config(self):
        return Speech2TextConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            num_conv_layers=self.num_conv_layers,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_channels=self.conv_channels,
            input_feat_per_channel=self.input_feat_per_channel,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_model_forward(self, config, inputs_dict):
        model = Speech2TextModel(config=config).set_train(False)

        input_features = inputs_dict["input_features"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

        self.parent.assertTrue(last_hidden_state.shape, (13, 7, 16))

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = Speech2TextModel(config=config).get_decoder().set_train(False)
        input_ids = inputs_dict["decoder_input_ids"]
        attention_mask = inputs_dict["decoder_attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size).clamp(2)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = ops.cat([input_ids, next_tokens], axis=-1)
        next_attention_mask = ops.cat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(Tensor(np.allclose(output_from_past_slice.as_numpy(), output_from_no_past_slice.as_numpy(), atol=1e-2)))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = Speech2TextModel(config=config).set_train(False)
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = Speech2TextEncoder.from_pretrained(tmpdirname)

        encoder_last_hidden_state_2 = encoder(
            inputs_dict["input_features"], attention_mask=inputs_dict["attention_mask"]
        )[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = Speech2TextDecoder.from_pretrained(tmpdirname)

        encoder_attention_mask = encoder._get_feature_vector_attention_mask(
            encoder_last_hidden_state.shape[1], inputs_dict["attention_mask"]
        )

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)

class EncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_decoder_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = SpeechEncoderDecoderModel(encoder_decoder_config)
        enc_dec_model.set_train(False)

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        self.assertFalse(enc_dec_model.config.tie_word_embeddings)

        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=outputs_encoder_decoder.encoder_hidden_states[-1])
        outputs_encoder_decoder = enc_dec_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_with_inputs(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        inputs = input_values if input_features is None else input_features
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model

        outputs_encoder_decoder = enc_dec_model(
            inputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        outputs_encoder_decoder_kwarg = enc_dec_model(
            inputs=inputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder_kwarg["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_save_and_load(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.set_train(False)
        with mindspore._no_grad():
            outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
        out_2 = outputs[0].numpy()
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            enc_dec_model.save_pretrained(tmpdirname)
            enc_dec_model = SpeechEncoderDecoderModel.from_pretrained(tmpdirname)

            after_outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_1 = after_outputs[0].numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def check_save_and_load_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.set_train(False)
        with mindspore._no_grad():
            outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
        out_2 = outputs[0].numpy()
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:
            enc_dec_model.encoder.save_pretrained(encoder_tmp_dirname)
            enc_dec_model.decoder.save_pretrained(decoder_tmp_dirname)
            SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_pretrained_model_name_or_path=encoder_tmp_dirname,
                decoder_pretrained_model_name_or_path=decoder_tmp_dirname,
            )

            after_outputs = enc_dec_model(
                input_values=input_values,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_1 = after_outputs[0].numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_values=input_values,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        inputs = input_values if input_features is None else input_features

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        seq_len = enc_dec_model.encoder._get_feat_extract_output_lengths(inputs.shape[1])
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads, seq_len, seq_len))

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def check_encoder_decoder_model_generate(
        self, config, decoder_config, input_values=None, input_features=None, **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # make sure EOS token is set to None to prevent early stopping of generation
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None

        inputs = input_values if input_features is None else input_features

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            inputs, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(generated_output.shape, (inputs.shape[0],) + (decoder_config.max_length,))

    def test_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_with_inputs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_with_inputs(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained_configs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained_configs(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=False)

    def test_encoder_decoder_model_from_pretrained_return_dict(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=True)

    def test_save_and_load_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load(**input_ids_dict)

    def test_save_and_load_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def test_training_gradient_checkpointing(self):
        inputs_dict = self.prepare_config_and_inputs()
        encoder_model, decoder_model = self.get_encoder_decoder_model(
            inputs_dict["config"], inputs_dict["decoder_config"]
        )

        model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        model.set_train()
        # model.gradient_checkpointing_enable()
        model.config.decoder_start_token_id = 0
        model.config.pad_token_id = 0

        model_inputs = {
            "attention_mask": inputs_dict["attention_mask"],
            "labels": inputs_dict["labels"],
            "decoder_input_ids": inputs_dict["decoder_input_ids"],
        }
        inputs = inputs_dict["input_features"] if "input_features" in inputs_dict else inputs_dict["input_values"]

        loss = model(inputs, **model_inputs).loss

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()

        with mindspore._no_grad():
            outputs = model_2(**inputs)
            out_2 = outputs[0].numpy()
            out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = SpeechEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(**inputs)
            out_1 = after_outputs[0].numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


class Wav2Vec2BertModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path="facebook/wav2vec2-base-960h", decoder_pretrained_model_name_or_path="google-bert/bert-base-cased"
        )
        batch_size = 13
        input_values = floats_tensor([batch_size, 512], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 512])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = Wav2Vec2Model(config).set_train(False)
        decoder_model = BertLMHeadModel(decoder_config).set_train(False)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        wav2vec2_model_tester = Wav2Vec2ModelTester(self)
        encoder_config_and_inputs = wav2vec2_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_values,
            input_mask,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_values": input_values,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }


class Speech2TextBertModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/s2t-small-librispeech-asr", "google-bert/bert-base-cased"
        )
        batch_size = 13
        input_features = floats_tensor([batch_size, 7, 80], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 7])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = Speech2TextEncoder(config).set_train(False)
        decoder_model = BertLMHeadModel(decoder_config).set_train(False)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        speech2text_model_tester = Speech2TextModelTester(self)
        encoder_config_and_inputs = speech2text_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()

        config, inputs = encoder_config_and_inputs
        input_features = inputs["input_features"]
        input_mask = inputs["attention_mask"]

        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_features": input_features,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }

    # can't save full model for now because Speech2TextModel != Speech2TextEncoder
    def test_encoder_decoder_model_from_pretrained_configs(self):
        pass

    # can't save full model for now because Speech2TextModel != Speech2TextEncoder
    def test_save_and_load_from_pretrained(self):
        pass

    # all published pretrained models are Speech2TextModel != Speech2TextEncoder
    def test_real_model_save_load_from_pretrained(self):
        pass
