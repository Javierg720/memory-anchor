"""
Gemma Remember — Export to GGUF

Exports the fine-tuned LoRA model to GGUF format for offline deployment:
  - llama.cpp (desktop)
  - Ollama (desktop)
  - LiteRT / MediaPipe (Android tablet)

Usage:
    python scripts/export_gguf.py
    python scripts/export_gguf.py --quantization q4_k_m
    python scripts/export_gguf.py --quantization q8_0  # higher quality
"""

import argparse

import yaml
from unsloth import FastVisionModel


def load_config(path="configs/training_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# Quantization options ranked by size vs quality
QUANT_OPTIONS = {
    "q4_k_m": "Good balance of size and quality (~4GB for E4B). Recommended for tablets.",
    "q5_k_m": "Slightly better quality (~5GB). Good for laptops.",
    "q8_0": "Near-original quality (~8GB). For machines with more RAM.",
    "f16": "Full 16-bit. Largest, best quality. For GPUs.",
}


def export(config, adapter_path=None, quantization="q4_k_m"):
    """Merge LoRA adapter into base model and export to GGUF."""
    adapter = adapter_path or config["output"]["model_dir"]
    output_dir = config["output"]["gguf_dir"]

    print(f"Loading model + adapter from {adapter}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        adapter,
        load_in_16bit=config["model"]["load_in_16bit"],
        max_seq_length=config["model"]["max_seq_length"],
    )

    print(f"Exporting to GGUF ({quantization})...")
    print(f"  {QUANT_OPTIONS.get(quantization, 'Custom quantization')}")

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization,
    )

    print(f"\nExported to {output_dir}/")
    print(f"\nTo run with Ollama:")
    print(f"  ollama create gemma-remember -f {output_dir}/Modelfile")
    print(f"  ollama run gemma-remember")
    print(f"\nTo run with llama.cpp:")
    print(f"  ./llama-server -m {output_dir}/*.gguf --port 8080")


def main():
    parser = argparse.ArgumentParser(description="Export Gemma Remember to GGUF")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=list(QUANT_OPTIONS.keys()),
                        help="Quantization method")
    args = parser.parse_args()

    config = load_config(args.config)
    export(config, args.adapter, args.quantization)


if __name__ == "__main__":
    main()
