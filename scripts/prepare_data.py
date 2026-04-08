"""
Gemma Remember — Data Preparation

Converts a folder of family photos, captions, and audio clips into
Gemma 4 multimodal chat format (JSONL) for fine-tuning with Unsloth.

Folder layout expected:
  data/raw/photos/   → sarah_birthday.jpg, arki_birdhouse.png, ...
  data/raw/captions/ → sarah_birthday.txt, arki_birdhouse.txt, ...
  data/raw/audio/    → sarah_birthday.wav, arki_birdhouse.mp3, ... (optional)

Each caption file is plain text with the warm reminder, e.g.:
  "This is your daughter Sarah. She loved baking cookies with you every Sunday."

The audio files are optional — if present, they get transcribed and the
transcript is woven into the training example.

Output: data/processed/train.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path

import yaml


def load_config(config_path="configs/training_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_matching_files(photos_dir, captions_dir, audio_dir):
    """Match photos to captions and (optionally) audio by filename stem."""
    photos = {p.stem: p for p in Path(photos_dir).glob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    captions = {p.stem: p for p in Path(captions_dir).glob("*.txt")}
    audio_files = {}
    if Path(audio_dir).exists():
        audio_files = {p.stem: p for p in Path(audio_dir).glob("*")
                       if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}}

    matched = []
    for stem, photo_path in sorted(photos.items()):
        if stem not in captions:
            print(f"  SKIP {photo_path.name}: no matching caption file")
            continue
        entry = {
            "stem": stem,
            "photo": str(photo_path),
            "caption": captions[stem].read_text().strip(),
            "audio": str(audio_files[stem]) if stem in audio_files else None,
        }
        matched.append(entry)

    print(f"  Matched {len(matched)} photo-caption pairs "
          f"({sum(1 for m in matched if m['audio'])} with audio)")
    return matched


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper (runs locally, no cloud)."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        # Fallback: if whisper not installed, just note the audio exists
        return f"[Voice clip from {Path(audio_path).name}]"


def build_question_variants(stem):
    """Generate natural question variants a person with dementia might ask."""
    name_guess = stem.replace("_", " ").title()
    # Remove trailing numbers/dates
    name_guess = re.sub(r'\d+', '', name_guess).strip()

    return [
        "Who is this person?",
        "Who is this?",
        "Do I know them?",
        f"Tell me about this photo.",
        "Can you help me remember?",
        "I think I know this person... who are they?",
        "This face looks familiar. Who is it?",
    ]


def build_training_example(entry, system_prompt, question=None):
    """
    Build a single Gemma 4 multimodal chat example.

    Format follows Unsloth's expected structure for vision fine-tuning:
    messages list with role="user" and role="model" (not "assistant").
    Multimodal content (image) comes before text in user message.
    """
    if question is None:
        questions = build_question_variants(entry["stem"])
        question = questions[hash(entry["stem"]) % len(questions)]

    # Build the warm response
    response = entry["caption"]
    if entry.get("audio") and entry.get("transcript"):
        response += f'\n\n(There\'s also a voice message: "{entry["transcript"]}")'

    # User message: image first, then text question
    user_content = [
        {"type": "image", "image": entry["photo"]},
        {"type": "text", "text": question},
    ]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "model",
            "content": [{"type": "text", "text": response}],
        },
    ]

    return {"messages": messages}


def build_text_only_example(entry, system_prompt):
    """Build a text-only variant (for when user asks by name, no photo)."""
    name_guess = entry["stem"].replace("_", " ").title()
    name_guess = re.sub(r'\d+', '', name_guess).strip()

    questions = [
        f"Tell me about {name_guess}.",
        f"Who is {name_guess}?",
        f"What do you know about {name_guess}?",
    ]
    question = questions[hash(entry["stem"] + "text") % len(questions)]
    response = entry["caption"]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
        {"role": "model", "content": [{"type": "text", "text": response}]},
    ]
    return {"messages": messages}


def generate_mock_data(mock_dir):
    """Generate fake family data for testing the pipeline."""
    print("Generating mock data...")
    photos_dir = Path(mock_dir) / "photos"
    captions_dir = Path(mock_dir) / "captions"
    audio_dir = Path(mock_dir) / "audio"

    for d in [photos_dir, captions_dir, audio_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Fake family members
    family = [
        ("sarah_birthday", "This is your daughter Sarah. She turned 35 last March. "
         "She brought that chocolate cake you love — the one with the raspberries on top. "
         "She lives in Portland now but calls you every Sunday."),
        ("arki_birdhouse", "This is your son Arki. He built you that birdhouse in the "
         "backyard in 1998 — remember, the blue one with the crooked roof? He was so "
         "proud of it. He's an engineer now, just like you always said he'd be."),
        ("maria_wedding", "This is your wife Maria on your wedding day, June 14th, 1975. "
         "She wore her mother's dress. You danced to 'Unforgettable' by Nat King Cole. "
         "She said it was the happiest day of her life."),
        ("tommy_graduation", "This is your grandson Tommy at his high school graduation. "
         "Class of 2023. He got a scholarship to study marine biology — he loves the ocean "
         "just like you do. He still wears the watch you gave him."),
        ("bella_dog", "This is Bella, your golden retriever. She's been with you for 8 years. "
         "She loves belly rubs and sleeps at the foot of your bed every night. "
         "Sarah got her for you after you retired."),
        ("family_christmas_2022", "This is Christmas 2022 at your house. Everyone came — "
         "Sarah, Arki, Tommy, little Maya, and Maria made her famous tamales. "
         "Arki played guitar and you all sang carols. Maya sat on your lap the whole time."),
        ("maya_first_steps", "This is your great-granddaughter Maya taking her first steps. "
         "She was 11 months old. She walked straight to you — you were sitting in your "
         "favorite chair. Everyone cheered. You cried happy tears."),
        ("fishing_trip_1995", "This is you and Arki on your fishing trip to Lake Tahoe in 1995. "
         "You caught a 12-pound trout and Arki caught nothing — he still jokes about it. "
         "You both got sunburned and ate hot dogs for dinner."),
    ]

    # Create placeholder images (1x1 pixel PNGs) and captions
    try:
        from PIL import Image
        for stem, caption in family:
            # Create a small colored placeholder image
            color = tuple((hash(stem + str(i)) % 200 + 55) for i in range(3))
            img = Image.new("RGB", (224, 224), color)
            img.save(photos_dir / f"{stem}.jpg")
            (captions_dir / f"{stem}.txt").write_text(caption)
    except ImportError:
        # No PIL — create empty files as placeholders
        for stem, caption in family:
            (photos_dir / f"{stem}.jpg").write_bytes(b"\x00")
            (captions_dir / f"{stem}.txt").write_text(caption)

    print(f"  Created {len(family)} mock entries in {mock_dir}/")
    return mock_dir


def prepare_dataset(raw_dir=None, mock=False, config_path="configs/training_config.yaml"):
    """Main pipeline: raw data → training JSONL."""
    config = load_config(config_path)
    system_prompt = config["system_prompt"].strip()

    if mock:
        mock_dir = config["data"]["mock_dir"]
        generate_mock_data(mock_dir)
        photos_dir = f"{mock_dir}/photos"
        captions_dir = f"{mock_dir}/captions"
        audio_dir = f"{mock_dir}/audio"
    else:
        base = raw_dir or config["data"]["raw_dir"]
        photos_dir = f"{base}/photos"
        captions_dir = f"{base}/captions"
        audio_dir = f"{base}/audio"

    print(f"Reading from: {photos_dir}")
    entries = find_matching_files(photos_dir, captions_dir, audio_dir)

    if not entries:
        print("ERROR: No matched photo-caption pairs found.")
        return

    # Transcribe audio where available
    for entry in entries:
        if entry["audio"]:
            print(f"  Transcribing {Path(entry['audio']).name}...")
            entry["transcript"] = transcribe_audio(entry["audio"])

    # Build training examples
    examples = []
    for entry in entries:
        # Primary: image + question → warm reminder
        examples.append(build_training_example(entry, system_prompt))

        # Add 2 more question variants per entry for diversity
        questions = build_question_variants(entry["stem"])
        for i in range(min(2, len(questions) - 1)):
            q = questions[(hash(entry["stem"]) + i + 1) % len(questions)]
            examples.append(build_training_example(entry, system_prompt, question=q))

        # Text-only variant (no photo)
        examples.append(build_text_only_example(entry, system_prompt))

    # Write JSONL
    out_dir = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} training examples to {out_path}")
    print(f"  ({len(entries)} base entries × ~4 variants each)")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gemma Remember training data")
    parser.add_argument("--mock", action="store_true", help="Generate and use mock data")
    parser.add_argument("--raw-dir", type=str, help="Override raw data directory")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    prepare_dataset(raw_dir=args.raw_dir, mock=args.mock, config_path=args.config)
