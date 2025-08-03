from mindnlp.transformers import GenerationConfig
from mindnlp.transformers import Pipeline, pipeline

class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)


class StarCoder(GeneratorBase):
    def __init__(self, pretrained: str, mirror: str = 'huggingface'):
        self.pretrained: str = pretrained
        self.mirror: str = mirror
        self.pipe: Pipeline = pipeline(
            "text-generation", model=pretrained, mirror=mirror)
        self.generation_config = GenerationConfig.from_pretrained(pretrained, mirror=mirror)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text
