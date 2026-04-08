# Gemma Remember

**An offline dementia care companion powered by Gemma 4.**

Gemma Remember helps people with dementia remember their loved ones. Upload family photos, voice clips, and short stories. When they ask "Who is this?", the app gently reminds them — grounded entirely in the data you provide. No hallucinations. No cloud. No data leaves the device.

## How It Works

1. **You provide memories**: family photos, voice messages, captions like "This is your son Arki — he built you that birdhouse in '98"
2. **We fine-tune Gemma 4** (locally, with LoRA via Unsloth) on those memories
3. **The app runs offline** on a laptop or Android tablet — camera input, voice input, gentle text responses

## Project Structure

```
gemma-remember/
├── scripts/
│   ├── prepare_data.py      # Convert photos + captions + audio → training JSONL
│   ├── train.py              # Fine-tune Gemma 4 E4B with Unsloth LoRA
│   ├── inference.py          # Run inference on a photo or text query
│   └── export_gguf.py        # Export to GGUF for llama.cpp / Ollama
├── app/
│   └── app.py                # Gradio local UI — camera, voice, text
├── configs/
│   └── training_config.yaml  # Hyperparameters and paths
├── data/
│   ├── raw/                  # YOUR private data (never committed)
│   │   ├── photos/           # Family photos (jpg/png)
│   │   ├── audio/            # Voice clips (wav/mp3)
│   │   └── captions/         # Text files matching photo names
│   └── processed/            # Generated JSONL (never committed)
├── mock_data/                # Fake test data (safe to share)
├── models/                   # Saved model weights (gitignored)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate mock data to test the pipeline
python scripts/prepare_data.py --mock

# 3. Fine-tune (needs GPU, ~8GB VRAM with E2B, ~16GB with E4B)
python scripts/train.py

# 4. Run the local UI
python app/app.py

# 5. (Optional) Export for mobile
python scripts/export_gguf.py
```

## Hardware Requirements

| Model | VRAM (Training) | VRAM (Inference) | Notes |
|-------|-----------------|------------------|-------|
| Gemma 4 E2B | ~8 GB | ~4 GB | Fits most laptops with GPU |
| Gemma 4 E4B | ~16 GB | ~8 GB | Better quality, needs more RAM |

## Privacy

- All data stays on your machine. Period.
- No telemetry, no uploads, no cloud calls.
- The .gitignore blocks all personal data from ever being committed.
- Models are trained and run locally.

## For the Gemma 4 Good Hackathon

This project is built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle. We believe the most powerful use of AI is the gentlest — helping someone remember the people they love.

## License

MIT — use it, adapt it, help someone you love.
