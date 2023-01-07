Modules
===================

:py:class:`~mindnlp.modules` is used to build neural network models, which
can be used with `MindSpore`. :py:class:`~mindnlp.modules` can be classified
into three functional modules: `Embedding`, `Encoder-Decoder` and `Attention`.
We will introduce the three function in the following three sections.

Embedding
-------------------

embedding is essentially a word embedding technique,
which can represent a character or word as a low-dimensional vector.
mindnlp provides a quick and easy way to construct embeddings through
pre-trained glove,fasttext,word2vec word vectors.
You can also create your own custom embeddings.

Next we demonstrate how to quickly construct an embedding
using glove word vectors via MindNLP.

.. code:: python

    import numpy as np
    from mindspore import Tensor
    from mindspore.dataset.text.utils import Vocab
    from mindnlp.modules.embeddings.glove_embedding import Glove

    # Define your own vocab
    vocab = Vocab.from_list(['default', 'one', 'two', 'three'])

    # Define your own embedding table
    init_embed = Tensor(np.zeros((4, 4)).astype(np.float32))

    # Create your own embedding object
    glove_embed = Glove(vocab, init_embed)

    # You can also use pre-trained word vectors
    glove_embed_pretrained, _ = Glove.from_pretrained()

After creating the embedding, we will use it for lookup next:

.. code:: python

    # The index to query for
    ids = Tensor([1, 2, 3])

    # Computed by the built embedding
    output = glove_embed(ids)

You can get more information about the embedding API from
:doc:`MindNLP.modules.embeddings <../api/modules/embeddings>`.

Encoder-Decoder
-------------------

Encoder-Decoder is a model framework, which is a general term for a class of
algorithms. Various algorithms can be used in this framework to solve different
tasks. Encoder converts the input sequence into a sentiment vector, and decoder
generates the target translation based on the output of the encoder.

We can use encoder and decoder provided by MindNLP to construct model as
the following example of a machine translation model. More information
about this model are shown in
:doc:`Machine Translation Example <../examples/machine_translation>` .

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

========================================================  =====================
Name                                                      Introduction
========================================================  =====================
:class:`~mindnlp.modules.encoder.cnn_encoder.CNNEncoder`  Convolutional encoder
                                                          consisting of
                                                          len(convolutions)
                                                          layers
:class:`~mindnlp.modules.encoder.rnn_encoder.RNNEncoder`  Stacked Elman RNN
                                                          Encoder
:class:`~mindnlp.modules.decoder.rnn_decoder.RNNDecoder`  Stacked Elman RNN
                                                          Decoder
========================================================  =====================

Attention
-------------------

Attention is a simulation of the attention mechanism of the human brain.
When people see something, they tend to focus on important information
and ignore other information. Attention in natural language processing
is to assign attention weights to text, and the essence of Attention is
to change from focusing on all to focusing. In MindNLP, we provide
various modules for attention mechanisms, so you can use them quickly.

Next we will demonstrate how to build a multi-headed attention
module via MindNLP.

.. code:: python

    import mindspore
    import mindspore.numpy as np
    from mindspore import ops
    from mindspore import Tensor
    from mindspore.text.modules.attentions import MutiHeadAttention
    # initialize random number seeds
    standard_normal = ops.StandardNormal(seed=0)

    # query is [batch_size, seq_len_q, hidden_size]
    q = standard_normal((2, 32, 512))

    # key is [batch_size, seq_len_k, hidden_size]
    k = standard_normal((2, 20, 512))

    # value is [batch_size, seq_len_k, hidden_size]
    v = standard_normal((2, 20, 512))

    # now query shape is (2, 32 ,512)->(2, 8, 32, 64)
    # and key shape is (2, 20 ,512)->(2, 8, 20, 64)
    # query * key.transpose(-1, -2):
    # (2, 8, 32, 64) * (2, 8, 64, 20) ->(2, 8, 32, 20)
    # equal with mask shape that is [batch_size, seq_len_q, seq_len_k]
    mask_shape = (2, 32, 20)
    mask = Tensor(np.ones(mask_shape), mindspore.bool_)

    # use additive attention
    net = MutiHeadAttention(heads=8, attention_mode="add")
    # you can also use cosine attention via multi-head attention
    net = MutiHeadAttention(heads=8, attention_mode="cos")
    # you can also use dot-product attention via multi-head attention
    # default dot-product attention mode
    net = MutiHeadAttention(heads=8)

    # x is the output of multi-head attention
    # attn is the attention score
    x, attn = net(query, key, value, mask)


Of course, you can also use the most basic scaled dot-product attention
attention to build the module:

.. code:: python

    import mindspore
    from mindspore import Tensor
    from mindspore.text.modules.attentions import ScaledDotAttention
    model = ScaledDotAttention(dropout=0.9)
    # You can customize the query, key, vlaue vector
    q = Tensor(np.ones((2, 32, 512)), mindspore.float32)
    k = Tensor(np.ones((2, 20, 512)), mindspore.float32)
    v = Tensor(np.ones((2, 20, 400)), mindspore.float32)
    output, att = model(q, k, v)
    # output shape is (2, 1024, 512)
    # att shape is (2, 1024, 32)

Currently mindnlp has implemented 8 attention mechanisms.
You can get more information about the attention API from
:doc:`MindNLP.modules.attentions <../api/modules/attentions>` .
