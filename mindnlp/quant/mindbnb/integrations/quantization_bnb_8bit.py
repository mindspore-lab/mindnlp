from .replace_modules import replace_with_bnb_linear


def quant_8bit(model, modules_to_not_convert=None,quantization_config=None):

    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )
    
    return model

