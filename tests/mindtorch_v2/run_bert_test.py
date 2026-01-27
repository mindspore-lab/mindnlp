"""Run BERT tests with mindtorch_v2 backend."""

import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Use the torch_proxy system to install mindtorch_v2 as torch
from mindtorch_v2._torch_proxy import install
install()

print("mindtorch_v2 torch_proxy installed")

# Now try to import transformers BERT
if __name__ == "__main__":
    print("\nTrying to import transformers BertModel...")
    try:
        from transformers import BertConfig, BertModel
        print("Import successful!")

        # Try to create a small model
        config = BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
        print(f"\nCreating BertModel with config: hidden_size={config.hidden_size}")
        model = BertModel(config)
        print(f"Model created successfully!")

        # Try forward pass
        import torch
        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones((2, 8))

        print("\nRunning forward pass...")
        outputs = model(input_ids, attention_mask=attention_mask)
        print(f"Output shape: {outputs.last_hidden_state.shape}")

        print("\n=== SUCCESS: transformers BERT works with mindtorch_v2! ===")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
