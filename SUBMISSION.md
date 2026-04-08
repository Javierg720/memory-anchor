# Hackathon Submission: Gemma Remember

## One-Line Summary

An offline, privacy-first app that uses Gemma 4 to help people with dementia remember their loved ones through personal photos, voice clips, and stories.

## Category

Health & Wellbeing

## What It Does

Gemma Remember lets caregivers upload family photos, voice messages, and short stories about loved ones. The app fine-tunes Gemma 4 (E4B multimodal) locally using LoRA, then runs fully offline. When the person with dementia shows a photo or asks "Who is this?", the app responds warmly and personally — grounded entirely in real family data.

Example: show a photo → "This is your son Arki — he built you that birdhouse in '98, remember? The blue one with the crooked roof."

## How We Built It

1. **Data Pipeline**: Family photos + text captions + optional voice clips → Gemma 4 multimodal chat format (JSONL). Multiple question variants per memory for robustness.

2. **Fine-Tuning**: Unsloth + LoRA (r=16) on Gemma 4 E4B-IT. 16-bit, gradient checkpointing, batch=1 with accumulation=4. Fits in ~16GB VRAM (E2B variant fits in ~8GB).

3. **Inference**: Low temperature (0.3) to minimize hallucination. System prompt instructs the model to ONLY use provided family data — never invent details.

4. **Deployment**: Export to GGUF (q4_k_m) for llama.cpp / Ollama. Runs on an Android tablet with no internet.

5. **UI**: Gradio local web app — camera upload, voice recording, text input. Large buttons, warm colors, designed for elderly users.

## Why Gemma 4

- Multimodal: understands photos natively
- Small enough for consumer hardware (E2B = 8GB, E4B = 16GB)
- Open weights: can fine-tune and deploy fully offline
- GGUF export: runs on tablets without internet

## Impact

- 55M people live with dementia worldwide
- 11M unpaid caregivers in the US alone
- Gemma Remember costs nothing beyond a one-time setup on a consumer device
- Privacy is absolute: no data ever leaves the device

## Team

Built for the Gemma 4 Good Hackathon on Kaggle.

## Links

- GitHub: https://github.com/Javierg720/gemma-remember
- Submission Notebook: notebooks/memory_anchor_submission.ipynb
