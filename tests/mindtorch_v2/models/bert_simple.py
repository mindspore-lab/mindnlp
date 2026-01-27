"""Simple BERT model implementation using mindtorch_v2.

This is a minimal BERT-like model for testing forward/backward passes.
Not intended for production use.
"""

import math
import mindtorch_v2 as torch
import mindtorch_v2.nn as nn


class BertConfig:
    """Configuration for BERT model."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps


class BertEmbeddings(nn.Module):
    """Token + position + segment embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length)
            # Expand to batch dimension
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """Reshape from (batch, seq, hidden) to (batch, heads, seq, head_size)."""
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores: (batch, heads, seq, seq)
        # Use contiguous() before view() since permute creates non-contiguous tensors
        attention_scores = torch.bmm(
            query_layer.contiguous().view(-1, query_layer.shape[2], query_layer.shape[3]),
            key_layer.contiguous().view(-1, key_layer.shape[2], key_layer.shape[3]).transpose(1, 2)
        )
        attention_scores = attention_scores.view(
            query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[2]
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # attention_mask: (batch, 1, 1, seq) with 0 for valid, large negative for masked
            attention_scores = attention_scores + attention_mask

        # Softmax and dropout
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context: (batch, heads, seq, head_size)
        context_layer = torch.bmm(
            attention_probs.contiguous().view(-1, attention_probs.shape[2], attention_probs.shape[3]),
            value_layer.contiguous().view(-1, value_layer.shape[2], value_layer.shape[3])
        )
        context_layer = context_layer.view(
            query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], self.attention_head_size
        )

        # Reshape back: (batch, seq, hidden)
        context_layer = context_layer.permute(0, 2, 1, 3)
        batch_size, seq_length, _, _ = context_layer.shape
        context_layer = context_layer.contiguous().view(batch_size, seq_length, self.all_head_size)

        return context_layer


class BertSelfOutput(nn.Module):
    """Self-attention output projection."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Full attention block: self-attention + output projection."""

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    """Feed-forward intermediate layer."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Feed-forward output with residual."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Full transformer layer."""

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """Stack of transformer layers."""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """Pool the first token for classification."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Take [CLS] token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """Full BERT model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def get_extended_attention_mask(self, attention_mask):
        """Create attention mask for self-attention."""
        # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Convert 0/1 mask to large negative values
        # 1 -> 0, 0 -> -10000
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)

        return encoder_output, pooled_output
