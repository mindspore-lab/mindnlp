Modules
===================

:py:class:`~mindnlp.modules` is used to build neural network models, which
can be used with `MindSpore`. :py:class:`~mindnlp.modules` can be classified
into three functional modules: `Embedding`, `Encoder-Decoder` and `Attention`.
We will introduce the three function in the following three sections.

Embedding
-------------------

Encoder-Decoder
-------------------

Encoder-Decoder is a model framework, which is a general term for a class of
algorithms. Various algorithms can be used in this framework to solve different
tasks. Encoder converts the input sequence into a sentiment vector, and decoder
generates the target translation based on the output of the encoder.

We can use encoder and decoder provided by MindNLP to construct model as
the following example of a machine translation model. More information
about this model are shown in
:doc:`Machine Translation Example <../examples/Machine Translation>` .

.. code-block::


    from mindspore import nn
    from mindnlp.abc import Seq2seqModel
    from mindnlp.modules import RNNEncoder, RNNDecoder

    class MachineTranslation(Seq2seqModel):
        def __init__(self, encoder, decoder):
            super().__init__(encoder, decoder)
            self.encoder = encoder
            self.decoder = decoder

        def construct(self, en, de):
            encoder_out = self.encoder(en)
            decoder_out = self.decoder(de, encoder_out=encoder_out)
            output = decoder_out[0]
            return output.swapaxes(1,2)

    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 512
    dec_hid_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5

    # encoder
    en_embedding = nn.Embedding(input_dim, enc_emb_dim)
    en_rnn = nn.RNN(enc_emb_dim, hidden_size=enc_hid_dim, num_layers=2, has_bias=True,
                    batch_first=True, dropout=enc_dropout, bidirectional=False)
    rnn_encoder = RNNEncoder(en_embedding, en_rnn)

    # decoder
    de_embedding = nn.Embedding(output_dim, dec_emb_dim)
    input_feed_size = 0 if enc_hid_dim == 0 else dec_hid_dim
    rnns = [
        nn.RNNCell(
            input_size=dec_emb_dim + input_feed_size
            if layer == 0
                else dec_hid_dim,
            hidden_size=dec_hid_dim
            )
            for layer in range(2)
    ]
    rnn_decoder = RNNDecoder(de_embedding, rnns, dropout_in=enc_dropout, dropout_out = dec_dropout,attention=True, encoder_output_units=enc_hid_dim)

The Encoder-Decoder modules included in MindNLP are shown in the following
table. You can click on the name to see the detailed API, or learn about
them through :doc:`MindNLP.modules.encoder <../api/modules/encoder>` and
:doc:`MindNLP.modules.decoder <../api/modules/decoder>` .

==============  ============================================================
Name            Introduction
==============  ============================================================
``CNNEncoder``  Convolutional encoder consisting of len(convolutions) layers
``RNNEncoder``  Stacked Elman RNN Encoder
``RNNDecoder``  Stacked Elman RNN Decoder
==============  ============================================================

Attention
-------------------
