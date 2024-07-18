# coding=utf-8
# Copyright 2018 The Google MS Team Authors and The HuggingFace Inc. team.
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

""" Auto Model class."""


from collections import OrderedDict

from mindnlp.utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "MSAlbertModel"),
        ("bart", "MSBartModel"),
        ("beit", "MSBeitModel"),
        ("bert", "MSBertModel"),
        ("big_bird", "MSBigBirdModel"),
        ("blenderbot", "MSBlenderbotModel"),
        ("blenderbot-small", "MSBlenderbotSmallModel"),
        ("bloom", "MSBloomModel"),
        ("clip", "MSCLIPModel"),
        ("distilbert", "MSDistilBertModel"),
        ("electra", "MSElectraModel"),
        ("gpt-sw3", "MSGPT2Model"),
        ("gpt2", "MSGPT2Model"),
        ("gpt_neo", "MSGPTNeoModel"),
        ("gptj", "MSGPTJModel"),
        ("longt5", "MSLongT5Model"),
        ("marian", "MSMarianModel"),
        ("mbart", "MSMBartModel"),
        ("mt5", "MSMT5Model"),
        ("opt", "MSOPTModel"),
        ("pegasus", "MSPegasusModel"),
        ("regnet", "MSRegNetModel"),
        ("resnet", "MSResNetModel"),
        ("roberta", "MSRobertaModel"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormModel"),
        ("roformer", "MSRoFormerModel"),
        ("t5", "MST5Model"),
        ("vision-text-dual-encoder", "MSVisionTextDualEncoderModel"),
        ("vit", "MSViTModel"),
        ("wav2vec2", "MSWav2Vec2Model"),
        ("whisper", "MSWhisperModel"),
        ("xglm", "MSXGLMModel"),
        ("xlm-roberta", "MSXLMRobertaModel"),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "MSAlbertForPreTraining"),
        ("bart", "MSBartForConditionalGeneration"),
        ("bert", "MSBertForPreTraining"),
        ("big_bird", "MSBigBirdForPreTraining"),
        ("electra", "MSElectraForPreTraining"),
        ("longt5", "MSLongT5ForConditionalGeneration"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("mt5", "MSMT5ForConditionalGeneration"),
        ("roberta", "MSRobertaForMaskedLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMaskedLM"),
        ("roformer", "MSRoFormerForMaskedLM"),
        ("t5", "MST5ForConditionalGeneration"),
        ("wav2vec2", "MSWav2Vec2ForPreTraining"),
        ("whisper", "MSWhisperForConditionalGeneration"),
        ("xlm-roberta", "MSXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "MSAlbertForMaskedLM"),
        ("bart", "MSBartForConditionalGeneration"),
        ("bert", "MSBertForMaskedLM"),
        ("big_bird", "MSBigBirdForMaskedLM"),
        ("distilbert", "MSDistilBertForMaskedLM"),
        ("electra", "MSElectraForMaskedLM"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("roberta", "MSRobertaForMaskedLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMaskedLM"),
        ("roformer", "MSRoFormerForMaskedLM"),
        ("xlm-roberta", "MSXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "MSBartForConditionalGeneration"),
        ("blenderbot", "MSBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "MSBlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "MSEncoderDecoderModel"),
        ("longt5", "MSLongT5ForConditionalGeneration"),
        ("marian", "MSMarianMTModel"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("mt5", "MSMT5ForConditionalGeneration"),
        ("pegasus", "MSPegasusForConditionalGeneration"),
        ("t5", "MST5ForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("beit", "MSBeitForImageClassification"),
        ("regnet", "MSRegNetForImageClassification"),
        ("resnet", "MSResNetForImageClassification"),
        ("vit", "MSViTForImageClassification"),
    ]
)

FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "MSVisionEncoderDecoderModel"),
    ]
)

FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "MSBartForCausalLM"),
        ("bert", "MSBertForCausalLM"),
        ("big_bird", "MSBigBirdForCausalLM"),
        ("bloom", "MSBloomForCausalLM"),
        ("electra", "MSElectraForCausalLM"),
        ("gpt-sw3", "MSGPT2LMHeadModel"),
        ("gpt2", "MSGPT2LMHeadModel"),
        ("gpt_neo", "MSGPTNeoForCausalLM"),
        ("gptj", "MSGPTJForCausalLM"),
        ("opt", "MSOPTForCausalLM"),
        ("roberta", "MSRobertaForCausalLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForCausalLM"),
        ("xglm", "MSXGLMForCausalLM"),
        ("xlm-roberta", "MSXLMRobertaForCausalLM"),
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "MSAlbertForSequenceClassification"),
        ("bart", "MSBartForSequenceClassification"),
        ("bert", "MSBertForSequenceClassification"),
        ("big_bird", "MSBigBirdForSequenceClassification"),
        ("distilbert", "MSDistilBertForSequenceClassification"),
        ("electra", "MSElectraForSequenceClassification"),
        ("mbart", "MSMBartForSequenceClassification"),
        ("roberta", "MSRobertaForSequenceClassification"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForSequenceClassification"),
        ("roformer", "MSRoFormerForSequenceClassification"),
        ("xlm-roberta", "MSXLMRobertaForSequenceClassification"),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "MSAlbertForQuestionAnswering"),
        ("bart", "MSBartForQuestionAnswering"),
        ("bert", "MSBertForQuestionAnswering"),
        ("big_bird", "MSBigBirdForQuestionAnswering"),
        ("distilbert", "MSDistilBertForQuestionAnswering"),
        ("electra", "MSElectraForQuestionAnswering"),
        ("mbart", "MSMBartForQuestionAnswering"),
        ("roberta", "MSRobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForQuestionAnswering"),
        ("roformer", "MSRoFormerForQuestionAnswering"),
        ("xlm-roberta", "MSXLMRobertaForQuestionAnswering"),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "MSAlbertForTokenClassification"),
        ("bert", "MSBertForTokenClassification"),
        ("big_bird", "MSBigBirdForTokenClassification"),
        ("distilbert", "MSDistilBertForTokenClassification"),
        ("electra", "MSElectraForTokenClassification"),
        ("roberta", "MSRobertaForTokenClassification"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForTokenClassification"),
        ("roformer", "MSRoFormerForTokenClassification"),
        ("xlm-roberta", "MSXLMRobertaForTokenClassification"),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "MSAlbertForMultipleChoice"),
        ("bert", "MSBertForMultipleChoice"),
        ("big_bird", "MSBigBirdForMultipleChoice"),
        ("distilbert", "MSDistilBertForMultipleChoice"),
        ("electra", "MSElectraForMultipleChoice"),
        ("roberta", "MSRobertaForMultipleChoice"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMultipleChoice"),
        ("roformer", "MSRoFormerForMultipleChoice"),
        ("xlm-roberta", "MSXLMRobertaForMultipleChoice"),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "MSBertForNextSentencePrediction"),
    ]
)

FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech-encoder-decoder", "MSSpeechEncoderDecoderModel"),
        ("whisper", "MSWhisperForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("whisper", "MSWhisperForAudioClassification"),
    ]
)

FLAX_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_MAPPING_NAMES)
FLAX_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
FLAX_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
FLAX_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)
FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)


class MSAutoModel(_BaseAutoModelClass):

    """
    MSAutoModel is a Python class that represents a model for a Microsoft Azure Machine Learning service.
    It inherits from the _BaseAutoModelClass and provides functionality for creating and managing auto ML
    models on the Azure platform.
    """
    _model_mapping = FLAX_MODEL_MAPPING


class MSAutoModelForPreTraining(_BaseAutoModelClass):

    """
    Represents a model for pre-training in Microsoft's Machine Learning framework.
    This class serves as a base class for different auto models used for pre-training.
    It inherits functionality from the _BaseAutoModelClass class.
    """
    _model_mapping = FLAX_MODEL_FOR_PRETRAINING_MAPPING


class MSAutoModelForCausalLM(_BaseAutoModelClass):

    """
    This class represents an auto-regressive language model for causal language modeling using Microsoft's AutoModel.
    
    The MSAutoModelForCausalLM class inherits from the _BaseAutoModelClass, providing additional functionality and customization options.
    
    Attributes:
        base_model_name_or_path (str): The name or path of the base model to be used for language modeling.
        config (AutoConfig): The configuration object for the auto-regressive language model.
        tokenizer (AutoTokenizer): The tokenizer object for the auto-regressive language model.
        model (AutoModelForCausalLM): The underlying model for the auto-regressive language model.

    Methods:
        __init__:
            Initializes a new instance of the MSAutoModelForCausalLM class.

        forward:
            Performs a forward pass through the auto-regressive language model.

        generate:
            Generates text using the auto-regressive language model.

        save_pretrained:
            Saves the auto-regressive language model and its configuration and tokenizer to the specified directory.

        from_pretrained:
            Instantiates a new instance of the MSAutoModelForCausalLM class from a pretrained model.

        from_config:
            Instantiates a new instance of the MSAutoModelForCausalLM class from a configuration object.

        from_pretrained:
            Instantiates a new instance of the MSAutoModelForCausalLM class from a pretrained model.

        from_pretrained:
            Instantiates a new instance of the MSAutoModelForCausalLM class from a pretrained model.
    """
    _model_mapping = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING


class MSAutoModelForMaskedLM(_BaseAutoModelClass):

    """
    A class representing an auto model for masked language modeling using Microsoft's AutoModel architecture.

    This class, MSAutoModelForMaskedLM, is a subclass of _BaseAutoModelClass and provides an implementation for
    generating predictions for masked tokens in a given input sequence. It utilizes Microsoft's AutoModel architecture
    which combines transformer-based models with language modeling techniques to achieve state-of-the-art performance
    on masked language modeling tasks.

    The MSAutoModelForMaskedLM class inherits the core functionality from the _BaseAutoModelClass, which provides a
    generic interface for auto models. It extends this base class by implementing specific methods and configurations
    that are tailored for masked language modeling tasks.

    The MSAutoModelForMaskedLM class can be instantiated with various parameters to control the architecture, model
    weights, tokenization, and other settings. It supports loading pre-trained models, fine-tuning on custom datasets,
    and generating predictions for masked tokens.

    Example:
        ```python
        >>> model = MSAutoModelForMaskedLM(model_name='bert-base-uncased')
        >>> input_sequence = "The [MASK] is blue."
        ...
        >>> # Generate predictions for masked tokens
        >>> predictions = model.predict_masked_tokens(input_sequence)
        ...
        >>> print(predictions)
        ```
    """
    _model_mapping = FLAX_MODEL_FOR_MASKED_LM_MAPPING


class MSAutoModelForSeq2SeqLM(_BaseAutoModelClass):

    """
    This class represents a pre-trained model for sequence-to-sequence language modeling with automatic architecture selection.
    
    It is a subclass of the '_BaseAutoModelClass' and inherits its methods and attributes.
    The 'MSAutoModelForSeq2SeqLM' class provides an interface for automatically selecting and loading the appropriate
    model architecture for sequence-to-sequence language modeling tasks.

    Attributes:
        config_class (type): The class to use for instantiating the model configuration.
        base_model_prefix (str): The prefix to use for the base model.
        _keys_to_ignore_on_load_missing (List[str]): A list of keys to ignore when loading the model.

    Methods:
        from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs): Class method to instantiate a
            pre-trained 'MSAutoModelForSeq2SeqLM' instance from a pre-trained model.
        forward(**inputs): Performs a forward pass of the model with the given inputs.
        generate(**kwargs): Generates text using the model with the provided inputs.
        prepare_inputs_for_generation(input_ids, **kwargs): Prepares the inputs for text generation.
        save_pretrained(save_directory): Saves the model to the specified directory.
        save_model(save_directory): Deprecated method. Use 'save_pretrained' instead.

    Note:
        This class is designed for sequence-to-sequence language modeling tasks and provides a convenient way
        to select and use the appropriate model architecture.

        Example:
            ```python
            >>> from transformers import MSAutoModelForSeq2SeqLM
            ...
            >>> model = MSAutoModelForSeq2SeqLM.from_pretrained("microsoft/MSDialog-GPT-large-finetuned-turbo")
            >>> input_text = "What is the capital of France?"
            >>> generated_text = model.generate(input_text)
            >>> print(generated_text)
            ```
    
        This example demonstrates how to use the 'MSAutoModelForSeq2SeqLM' class to load a pre-trained model and generate text using the model.
    """
    _model_mapping = FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


class MSAutoModelForSequenceClassification(_BaseAutoModelClass):

    """
    This class represents an auto model for sequence classification tasks in the Microsoft AutoML framework.
    
    The 'MSAutoModelForSequenceClassification' class is a subclass of '_BaseAutoModelClass' and
    provides a convenient interface for training and evaluating sequence classification models.
    It utilizes the power of AutoML to automatically search for the best model architecture
    and hyperparameters for a given sequence classification task.

    The class inherits the functionalities of the '_BaseAutoModelClass' class,
    which includes methods for loading and saving models, as well as performing inference using trained models.

    To use the 'MSAutoModelForSequenceClassification' class, first initialize an instance by providing
    the required parameters such as the number of classes, input dimensions, and other relevant configuration
    options.
    Then, you can call the 'fit' method to start the automatic model search process.
    This method takes in the training data, performs the model search, and returns the best model found.

    Once the model has been trained, you can use the 'evaluate' method to evaluate its performance on a separate validation or test dataset.
    This method calculates various evaluation metrics such as accuracy, precision, recall, and F1-score.

    In addition to these core methods, the 'MSAutoModelForSequenceClassification' class provides various helper methods
    for configuring and fine-tuning the model search process.
    These include methods for setting the search space, defining custom metrics, specifying early stopping criteria, and more.

    Note that the automatic model search process may take some time to complete, depending on the size of the dataset
    and the complexity of the search space.
    However, it helps to alleviate the burden of manually tuning hyperparameters and allows you to focus on
    other aspects of your sequence classification task.
    
    For more information on how to use the 'MSAutoModelForSequenceClassification' class and the Microsoft AutoML framework,
    please refer to the official documentation and examples.
    
    """
    _model_mapping = FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


class MSAutoModelForQuestionAnswering(_BaseAutoModelClass):

    """
    MSAutoModelForQuestionAnswering is a class that represents a pre-trained model for question answering tasks
    using the Microsoft Azure Machine Learning service.
    
    This class inherits from _BaseAutoModelClass, which provides the base functionality for automatic modeling
    in the Microsoft Azure Machine Learning service.
    
    The MSAutoModelForQuestionAnswering class provides methods and attributes specific to question answering tasks,
    enabling users to fine-tune and deploy pre-trained models for question answering in a production environment.

    Attributes:
        model_name_or_path (str): The name or path of the pre-trained model.
        tokenizer_name_or_path (str): The name or path of the tokenizer associated with the pre-trained model.
        config_name_or_path (Optional[str]): The name or path of the model configuration file.
        cache_dir (Optional[str]): The directory where the pre-trained models and related files will be cached.
        revision (Union[str, int]): The revision number of the model to load from the Hugging Face model hub.
        use_auth_token (Union[str, bool]): The authentication token to use for downloading models from the Hugging Face model hub.

    Methods:
        from_pretrained(cls, model_name_or_path, *args, **kwargs):
            Class method to instantiate an instance of MSAutoModelForQuestionAnswering from a pre-trained model.
        forward(self, ...): Method to perform the forward pass of the model, taking inputs and returning the predicted outputs.
        train(self, ...): Method to train the model on a given dataset.
        evaluate(self, ...): Method to evaluate the performance of the model on a given dataset.
        save_pretrained(self, ...): Method to save the model and associated files to a specified directory.
        from_pretrained(cls, ...):
            Class method to load a pre-trained instance of MSAutoModelForQuestionAnswering from a specified directory.
        generate(self, ...): Method to generate text using the model.
        get_named_parameters(self, ...): Method to get named parameters of the model.
        get_input_embeddings(self, ...): Method to get the input embeddings of the model.
        set_input_embeddings(self, ...): Method to set the input embeddings of the model.

    Example:
        ```python
        >>> # Instantiate a pre-trained model for question answering
        >>> model = MSAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
        ...
        >>> # Perform the forward pass of the model
        >>> outputs = model.forward(inputs)
        ...
        >>> # Train the model on a given dataset
        >>> model.train(train_dataset)
        ...
        >>> # Evaluate the performance of the model on a given dataset
        >>> model.evaluate(eval_dataset)
        ...
        >>> # Save the model and associated files to a specified directory
        >>> model.save_pretrained('saved_model')
        ...
        >>> # Load a pre-trained instance of MSAutoModelForQuestionAnswering from a specified directory
        >>> loaded_model = MSAutoModelForQuestionAnswering.from_pretrained('saved_model')
        ...
        >>> # Generate text using the model
        >>> generated_text = model.generate(input_text)
        ...
        >>> # Get named parameters of the model
        >>> params = model.get_named_parameters()
        ...
        >>> # Get the input embeddings of the model
        >>> embeddings = model.get_input_embeddings()
        ...
        >>> # Set the input embeddings of the model
        >>> model.set_input_embeddings(new_embeddings)
        ```
    """
    _model_mapping = FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING


class MSAutoModelForTokenClassification(_BaseAutoModelClass):

    """
    Represents a model for token classification using Microsoft's AutoModel framework.
    
    This class inherits from _BaseAutoModelClass and provides functionality for token classification tasks.
    It encapsulates the architecture and configuration of the model, including loading pre-trained weights
    and performing inference on token sequences.
    The model supports fine-tuning on specific token classification datasets
    and provides a high-level interface for integrating with downstream applications.
    
    Note:
        This docstring is a placeholder and should be updated with specific details about the class attributes, methods, and usage examples.
    """
    _model_mapping = FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


class MSAutoModelForMultipleChoice(_BaseAutoModelClass):

    """
    This class represents an automated model for multiple choice tasks in the Microsoft Azure Machine Learning service.
    
    The 'MSAutoModelForMultipleChoice' class inherits from the '_BaseAutoModelClass' class,
    which provides the foundational functionality for automated model creation.
    
    The 'MSAutoModelForMultipleChoice' class is specifically designed to handle multiple choice tasks in the
    Microsoft Azure Machine Learning service.
    It streamlines the process of creating, training, and evaluating models for multiple choice tasks,
    reducing the amount of manual effort required.

    To use this class, first instantiate an object of the 'MSAutoModelForMultipleChoice' class.
    Then, call the appropriate methods to perform tasks such as loading data, preprocessing, training the model, and
    evaluating its performance.

    This class encapsulates various methods and attributes that are essential for automating the model creation process
    for multiple choice tasks.
    It leverages the power of the Microsoft Azure Machine Learning service to provide a seamless
    and efficient experience for users.

    Note:
        This class requires the Microsoft Azure Machine Learning service to be properly set up and configured in order to function correctly.
    
    For detailed information on how to use this class and its methods, please refer to the documentation and examples provided.
    """
    _model_mapping = FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


class MSAutoModelForNextSentencePrediction(_BaseAutoModelClass):

    """
    This class represents an implementation of a pre-trained model for next sentence prediction using the Microsoft AutoModel framework.
    
    The MSAutoModelForNextSentencePrediction class inherits from the _BaseAutoModelClass,
    which provides basic functionality for automatic model loading and inference.
    
    Next sentence prediction is a task in natural language processing that involves predicting
    whether two given sentences are logically connected, such as being consecutive or having a cause-effect relationship.

    This class encapsulates the architecture and weights of a pre-trained model specifically designed for next sentence prediction.
    It provides methods for loading the model, encoding input sentences, and making predictions.

    To use this class, first initialize an instance by providing the necessary model configuration.
    Then, load the pre-trained weights using the 'load_weights' method.
    After loading, you can encode input sentences using the 'encode' method,
    which converts the sentences into numerical representations suitable for model input.
    Finally, use the 'predict' method to predict the connection between pairs of sentences.

    Note that this class assumes the pre-trained weights have been downloaded and stored in a specific format compatible
    with the Microsoft AutoModel framework.
    If the weights are not available, they can be obtained from the official Microsoft website or other trusted sources.

    For more details on how to use this class and examples of its functionality,
    refer to the documentation and code examples provided in the Microsoft AutoModel repository.

    Attributes:
        model_config (dict): A dictionary containing the configuration of the pre-trained model.
        model_weights (str): The file path or URL to the pre-trained weights of the model.

    Methods:
        load_weights:
            Loads the pre-trained weights of the model from the specified path.

        encode:
            Encodes a list of input sentences into numerical representations suitable for model input.
            Returns a list of encoded representations, where each representation is a list of floats.

        predict:
            Predicts the connection between pairs of input sentences.
            Returns a list of probabilities, where each probability represents the likelihood of the two sentences being logically connected.
    """
    _model_mapping = FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


class MSAutoModelForImageClassification(_BaseAutoModelClass):

    """
    This class represents an auto model for image classification using the Microsoft Azure cognitive services. 
    
    The 'MSAutoModelForImageClassification' class is a Python class that inherits from the '_BaseAutoModelClass'.
    It provides an interface for training and deploying machine learning models specifically
    designed for image classification tasks using the Microsoft Azure cognitive services.

    Features:
        >   - Data preprocessing: The class supports various data preprocessing techniques such as resizing, cropping,
        and normalization to prepare the image data for training and prediction.
        >   - Model training: The class allows users to train the image classification model using their own labeled dataset.
        It supports popular deep learning architectures like CNN (Convolutional Neural Networks) and transfer learning techniques.
        >   - Model evaluation: Users can evaluate the performance of the trained model using standard evaluation metrics
        such as accuracy, precision, recall, and F1-score.
        >   - Model deployment: Once the model is trained, it can be deployed in production environments to
        perform real-time image classification tasks.
        >   - Integration with Microsoft Azure cognitive services: The class seamlessly integrates with the
        Microsoft Azure cognitive services, allowing users to leverage powerful cloud-based functionalities such as
        automatic scaling, high availability, and advanced analytics.

    Usage:
        >   1. Instantiate an object of the 'MSAutoModelForImageClassification' class.
        >   2. Configure the model parameters and hyperparameters.
        >   3. Preprocess the input image data using the provided data preprocessing methods.
        >   4. Train the model using the labeled dataset.
        >   5. Evaluate the model's performance using the evaluation metrics.
        >   6. Deploy the trained model in a production environment.
        >   7. Utilize the model for real-time image classification tasks.

    Note:
        The 'MSAutoModelForImageClassification' class requires a valid Microsoft Azure cognitive services subscription
        and the necessary API keys for authentication and authorization.
    
    For detailed implementation instructions and code examples, refer to the official documentation and examples
    provided by Microsoft Azure cognitive services.
    """
    _model_mapping = FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


class MSAutoModelForVision2Seq(_BaseAutoModelClass):

    """
    The MSAutoModelForVision2Seq class is a Python class that represents a vision-to-sequence auto model
    for multi-modal tasks. This class inherits from the _BaseAutoModelClass and provides functionalities for
    vision to sequence transformation in multi-modal tasks.
    """
    _model_mapping = FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING


class MSAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):

    """
    Represents a speech sequence-to-sequence model for multi-source automatic speech recognition (ASR) and natural language generation (NLG).
    
    This class inherits from _BaseAutoModelClass and provides a pre-trained model for processing speech input
    and generating sequence-to-sequence outputs. It supports multi-source ASR and NLG tasks, making it suitable
    for a wide range of speech-related applications.

    The MSAutoModelForSpeechSeq2Seq class encapsulates the functionality for loading the pre-trained model,
    processing input speech data, and generating corresponding sequence-to-sequence outputs.
    It also provides methods for fine-tuning the model and evaluating its performance on speech-related tasks.

    Users can instantiate an object of this class to leverage the pre-trained speech sequence-to-sequence model
    for ASR and NLG tasks, enabling efficient and accurate processing of speech data with multi-source support.
    """
    _model_mapping = FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
