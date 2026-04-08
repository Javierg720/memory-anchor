"""
Gemma Remember — Inference Engine

Load the fine-tuned LoRA adapter and run inference on:
  - A photo (multimodal): "Who is this person?"
  - A text query: "Tell me about Sarah."

Designed to run on consumer hardware with no internet.

Usage:
    python scripts/inference.py --image path/to/photo.jpg
    python scripts/inference.py --text "Who is Sarah?"
    python scripts/inference.py --image photo.jpg --question "Do I know them?"
"""

import argparse

import yaml
from PIL import Image
from unsloth import FastVisionModel


def load_config(path="configs/training_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config, adapter_path=None):
    """Load base model + LoRA adapter for inference."""
    model_name = config["model"]["name"]
    adapter = adapter_path or config["output"]["model_dir"]

    print(f"Loading {model_name} + adapter from {adapter}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        adapter,
        load_in_16bit=config["model"]["load_in_16bit"],
        max_seq_length=config["model"]["max_seq_length"],
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def ask_with_image(model, tokenizer, image_path, question, config):
    """Ask a question about a photo — the core Gemma Remember interaction."""
    system_prompt = config["system_prompt"].strip()
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,     # Low temp = less hallucination
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the model's response (after the last turn)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0].strip()
    return response


def ask_text_only(model, tokenizer, question, config):
    """Ask a text-only question (no photo)."""
    system_prompt = config["system_prompt"].strip()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0].strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="Gemma Remember inference")
    parser.add_argument("--image", type=str, help="Path to a photo")
    parser.add_argument("--text", type=str, help="Text-only question")
    parser.add_argument("--question", type=str, default="Who is this person?",
                        help="Question to ask about the photo")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    if not args.image and not args.text:
        print("Please provide --image or --text (or both)")
        return

    config = load_config(args.config)
    model, tokenizer = load_model(config, args.adapter)

    if args.image:
        print(f"\nShowing photo: {args.image}")
        print(f"Asking: {args.question}\n")
        response = ask_with_image(model, tokenizer, args.image, args.question, config)
    else:
        print(f"\nAsking: {args.text}\n")
        response = ask_text_only(model, tokenizer, args.text, config)

    print("Gemma Remember says:")
    print(f"  {response}")


if __name__ == "__main__":
    main()
