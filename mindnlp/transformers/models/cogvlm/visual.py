"""
vit part
"""
from argparse import Namespace
import mindspore
from mindspore import nn,ops
from ...activations import ACT2FN


class PatchEmbedding(nn.Cell):
    """
    patch embedding
    """
    def __init__(self, config):
        """Initializes a PatchEmbedding object with the provided configuration settings.
        
        Args:
            self (PatchEmbedding): The PatchEmbedding instance itself.
            config (object):
                An object containing configuration settings for the PatchEmbedding.

                - in_channels (int): Number of input channels.
                - hidden_size (int): Size of the hidden layer.
                - patch_size (int): Size of the patch/kernel.
                - num_positions (int): Number of positions for position embedding.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the configuration provided is invalid or missing required parameters.
        """
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size,
                              stride=config.patch_size,pad_mode='valid',has_bias=True)
        self.cls_embedding = mindspore.Parameter(ops.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def construct(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        """
        Construct method in the PatchEmbedding class.
        
        This method constructs embeddings from input images.
        
        Args:
            self: The instance of the PatchEmbedding class.
            images (tensor(B, C, H, W)):
                Input images in tensor format with dimensions Batch (B), Channels (C), Height (H), and Width (W).
        
        Returns:
            tensor(B, L, D): Returns a tensor representing embeddings where B is the batch size,
                L is the sequence length, and D is the embedding dimension.
        
        Raises:
            None specified.
        """
        x = self.proj(images)
        x = x.flatten(start_dim=2).swapaxes(1, 2)
        cls_token = self.cls_embedding.broadcast_to((x.shape[0], -1, -1))
        x = ops.cat((cls_token, x), axis=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Cell):
    """
    attention
    """
    def __init__(self, config):
        """
        Initializes an instance of the Attention class.
        
        Args:
            self (Attention): The instance of the Attention class.
            config (object): An object containing configuration parameters for the attention mechanism.
                It must have the following attributes:

                - num_heads (int): The number of attention heads to use.
                - hidden_size (int): The size of the hidden layers in the attention mechanism.

                The config object should adhere to the following restrictions:

                - num_heads: Must be a positive integer.
                - hidden_size: Must be a positive integer.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the config parameter is not an object or does not have the expected attributes.
            ValueError: If the config attributes do not adhere to the specified restrictions.
        """
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Dense(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p = config.dropout_prob)

    def construct(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        """
        This method 'construct' is a part of the 'Attention' class and is used to construct the output tensor based on the input tensor.
        
        Args:
            self: A reference to the current instance of the class.
            x: A tensor of shape (B, L, D) representing the input data, where B is the batch size, L is the sequence length, and D is the input dimension.
        
        Returns:
            Returns a tensor of shape (B, L, D) representing the constructed output based on the input tensor.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = self.attention(q,k,v)
        output = self.dense(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        """
        Performs attention mechanism on the input tensors q, k, and v.
        
        Args:
            self: The instance of the Attention class.
            q: The query tensor of shape (batch_size, query_length, d_model).
            k: The key tensor of shape (batch_size, key_length, d_model).
            v: The value tensor of shape (batch_size, value_length, d_model).
        
        Returns:
            The output tensor of shape (batch_size, query_length, d_model),
                which is the result of applying attention mechanism on the input tensors.
        
        Raises:
            None.
        """
        attn_weights = ops.matmul(q * self.scale, k.swapaxes(-2, -1))
        attn_weights = ops.softmax(attn_weights,axis=-1)
        output = ops.matmul(attn_weights, v)
        output = output.transpose(0,2,1,3)
        return output


class MLP(nn.Cell):
    """
    MLP
    """
    def __init__(self, config):
        """
        Initializes an instance of the MLP class.
        
        Args:
            self (MLP): The instance of the MLP class.
            config (object): An object containing configuration parameters for the MLP model.
                This object should have attributes like 'hidden_act', 'hidden_size', 'intermediate_size'.

                - The 'hidden_act' attribute determines the activation function to be used.
                - The 'hidden_size' attribute specifies the size of the hidden layer.
                - The 'intermediate_size' attribute specifies the size of the intermediate layer.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a multi-layer perceptron (MLP) by applying linear transformations
        and activation functions to the input tensor.
        
        Args:
            self (MLP): An instance of the MLP class.
            x (mindspore.Tensor):
                The input tensor to be processed by the MLP. It should be a tensor compatible with the MLP's architecture.
        
        Returns:
            mindspore.Tensor: The output tensor obtained after passing through the MLP layers.
        
        Raises:
            TypeError: If the input 'x' is not a valid tensor.
            ValueError: If the input 'x' does not meet the expected requirements for the MLP.
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Cell):
    """
    transformer layer
    """
    def __init__(self, config):
        """
        Initializes a TransformerLayer object with the provided configuration.
        
        Args:
            self (TransformerLayer): The instance of the TransformerLayer class.
            config (object): The configuration object containing the settings for the TransformerLayer.
                Expected attributes:

                - hidden_size (int): The size of the hidden layers in the TransformerLayer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
        
        Returns:
            None.
        
        Raises:
            AttributeError: If the required attributes are missing in the config object.
            TypeError: If the config object is not of the expected type.
        """
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        """
        Constructs the TransformerLayer.
        
        Args:
            self (TransformerLayer): An instance of the TransformerLayer class.
            hidden_states (tensor): The input hidden states to the layer. 
        
        Returns:
            None: This method modifies the hidden_states in-place.
        
        Raises:
            None.
        
        Description:
            This method constructs a TransformerLayer by applying various operations on the input hidden states.
            It follows the standard Transformer architecture.
        
        The method performs the following steps:

        1. Apply attention mechanism to the hidden_states using self.attention.
        2. Apply input layer normalization to the attention output using self.input_layernorm.
        3. Add the attention output and the input layer normalized attention output.
        4. Apply multi-layer perceptron (MLP) to the resulting hidden_states using self.mlp.
        5. Apply post-attention layer normalization to the MLP output using self.post_attention_layernorm.
        6. Add the hidden_states and the post-attention layer normalized MLP output.
        
        Note that this method modifies the hidden_states in-place and does not return any value.
        """
        attention_input = hidden_states
        attention_output = self.attention(attention_input)
        layernorm = self.input_layernorm(attention_output)
        hidden_states = attention_input + layernorm
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Cell):

    """
    Represents a Transformer model for sequence-to-sequence tasks.
    
    This class inherits from the nn.Cell class and implements the Transformer architecture, which consists of multiple
    Transformer layers. Each layer module applies self-attention mechanism and position-wise feed-forward networks to
    input hidden states.
    
    The Transformer class initializes with a configuration object and creates a list of Transformer layers based on the
    specified number of hidden layers in the configuration. The construct method applies the series of Transformer
    layers to the input hidden states, resulting in transformed hidden states.
    
    This class provides an efficient and flexible implementation of the Transformer model for various natural language
    processing tasks, such as machine translation and language modeling.
    """
    def __init__(self, config):
        """
        Initializes an instance of the Transformer class.
        
        Args:
            self (object): The instance of the Transformer class.
            config (object): An object containing configuration parameters for the Transformer.
                This object should have the following attributes:

                - num_hidden_layers (int): The number of hidden layers in the Transformer.
                The config object is used to initialize the layers in the Transformer with TransformerLayer instances.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.layers = nn.CellList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states):
        """
        Constructs the output by passing the hidden states through each layer module in the Transformer.
        
        Args:
            self (Transformer): An instance of the Transformer class.
            hidden_states (Tensor): The input hidden states to be processed by the Transformer.
                Expected to be a tensor of shape (batch_size, sequence_length, hidden_size).
        
        Returns:
            None: This method does not return any value directly.
                The final processed hidden states are returned after passing through all layers.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Cell):

    """ 
    This class represents a Gated Linear Unit (GLU) module, which is used in neural networks for processing
    sequential data. It is implemented as a subclass of the nn.Cell class.
    
    GLU applies a gating mechanism to the input data, allowing it to selectively pass through different branches of
    the network. It consists of several layers, including linear projections, layer normalization, activation functions,
    and dense transformations.
    
    Attributes:
        linear_proj (nn.Dense): A linear projection layer that maps the input features to the hidden size.
        norm1 (nn.LayerNorm): A layer normalization module that normalizes the hidden size.
        act1 (nn.GELU): An activation function module that applies the Gaussian Error Linear Unit (GELU) function.
        act2 (ops.silu): An activation function module that applies the Sigmoid Linear Unit (SiLU) function.
        dense_h_to_4h (nn.Dense): A dense transformation layer that maps the hidden size to the intermediate size.
        gate_proj (nn.Dense): A dense transformation layer that maps the hidden size to the intermediate size.
        dense_4h_to_h (nn.Dense): A dense transformation layer that maps the intermediate size back to the hidden size.
    
    Methods:
        construct(x): Performs the forward pass through the GLU module, taking the input data 'x' and returning
            the transformed output.
    
    Example:
        >>> config = Configuration(hidden_size=256, intermediate_size=512)
        >>> in_features = 128
        >>> x = torch.randn(batch_size, in_features)
        >>> glu = GLU(config, in_features)
        >>> output = glu.construct(x)
        ...
        >>> # 'output' now contains the transformed input data after passing through the GLU module.
    
    """
    def __init__(self, config, in_features):
        """
        Initializes an instance of the GLU class.
        
        Args:
            self: The object itself.
            config: An object of class 'Config' containing configuration settings for the GLU layer.
            in_features: An integer representing the size of the input feature vector.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.linear_proj = nn.Dense(in_features, config.hidden_size, has_bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU(approximate=False)
        self.act2 = ops.silu
        self.dense_h_to_4h = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.gate_proj = nn.Dense(config.hidden_size, config.intermediate_size, has_bias=False)
        self.dense_4h_to_h = nn.Dense(config.intermediate_size, config.hidden_size, has_bias=False)

    def construct(self, x):
        """
        Constructs a GLU (Gated Linear Unit) using the given input.
        
        Args:
            self: An instance of the GLU class.
            x: The input tensor of shape (batch_size, input_dim).
        
        Returns:
            None.
        
        Raises:
            None.
        """
        x = self.linear_proj(x)
        x1 = self.act1(self.norm1(x))
        x2 = self.act2(self.gate_proj(x1)) * self.dense_h_to_4h(x1)
        x3 = self.dense_4h_to_h(x2)
        return x3


class EVA2CLIPModel(nn.Cell):

    """
    This class represents a model for EVA2CLIP (Embedding Vision and Audio to Clip) task, which combines vision and
    audio inputs to generate video embeddings.
    It inherits from nn.Cell and contains methods for initializing the model and constructing the forward pass.
    
    Attributes:
        patch_embedding (PatchEmbedding): Instance of PatchEmbedding class for extracting image patches.
        transformer (Transformer): Instance of Transformer class for processing image patches.
        linear_proj (GLU): Instance of GLU class for linear projection.
        boi (Parameter): Beginning of input parameter for the model.
        eoi (Parameter): End of input parameter for the model.
    
    Methods:
        __init__: Initializes the EVA2CLIPModel with the provided configuration.
        construct: Constructs the forward pass of the model using the input images.
    
    Example:
        ```python
        >>> model = EVA2CLIPModel(config)
        >>> output = model.construct(images)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the EVA2CLIPModel class.
        
        Args:
            self: The instance of the class.
            config: A configuration object containing parameters for the model's vision components and hidden size.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        self.boi = mindspore.Parameter(ops.zeros((1, 1, config.hidden_size)))
        self.eoi = mindspore.Parameter(ops.zeros((1, 1, config.hidden_size)))

    def construct(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        """
        Constructs the EVA2CLIP model by processing the input images.
        
        Args:
            self: The instance of the EVA2CLIPModel class.
            images (tensor(B, C, H, W)): The input images to be processed. 
                It should be a tensor with dimensions (B, C, H, W), where B represents 
                the batch size, C represents the number of channels, and H and W represent 
                the height and width of the images, respectively.
        
        Returns:
            tensor(B, L, D): The processed output tensor. It has dimensions (B, L, D), 
            where B represents the batch size, L represents the length, and D represents 
            the dimension of the tensor.
        
        Raises:
            None.
        """
        x = self.patch_embedding(images)
        x1 = self.transformer(x)
        x2 = x1[:, 1:]
        x3 = self.linear_proj(x2)
        boi = self.boi.broadcast_to((x3.shape[0], -1, -1))
        eoi = self.eoi.broadcast_to((x3.shape[0], -1, -1))
        x4 = ops.cat((boi, x3, eoi), axis=1)
        return x4

__all__ = [
    'EVA2CLIPModel',
    'TransformerLayer'
]
