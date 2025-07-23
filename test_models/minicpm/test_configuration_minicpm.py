from mindnlp.models.miniCPM.miniCPM_config import MiniCPMConfig

def test_config_fields():
    config = MiniCPMConfig()
    assert config.hidden_size == 4096
    assert config.vocab_size == 128256
    assert config.num_hidden_layers == 32
