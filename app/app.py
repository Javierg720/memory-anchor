"""
Gemma Remember — Local UI

A gentle, simple interface for someone with dementia (or their caregiver).
Three ways to ask: show a photo, record a voice, or type a question.

Runs as a local Gradio web server — no internet, no data leaves the device.

Usage:
    python app/app.py
    python app/app.py --port 7860 --share  # only if you trust the network
"""

import argparse
import os
import sys
import tempfile

import gradio as gr
import yaml

# Add parent dir to path so we can import scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.inference import ask_text_only, ask_with_image, load_model


def load_config(path="configs/training_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# Global model reference (loaded once at startup)
MODEL = None
TOKENIZER = None
CONFIG = None


def init_model(config_path="configs/training_config.yaml"):
    global MODEL, TOKENIZER, CONFIG
    CONFIG = load_config(config_path)
    MODEL, TOKENIZER = load_model(CONFIG)
    print("Model loaded and ready.")


def transcribe_audio_input(audio_path):
    """Transcribe voice input to text using Whisper."""
    if audio_path is None:
        return ""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        return "[Whisper not installed — please type your question instead]"


def respond(image, audio, text_input):
    """
    Handle a query from the user. Priority:
    1. If image is provided → multimodal query
    2. If audio is provided → transcribe and use as question
    3. If text is provided → text-only query
    """
    if MODEL is None:
        return "Model not loaded yet. Please wait..."

    # Determine the question
    question = text_input or ""
    if audio is not None:
        transcribed = transcribe_audio_input(audio)
        if transcribed and not transcribed.startswith("["):
            question = transcribed

    if not question:
        question = "Who is this person?" if image is not None else "Can you help me remember?"

    # Run inference
    if image is not None:
        # Save uploaded image to temp file
        temp_path = os.path.join(tempfile.gettempdir(), "gemma_remember_query.jpg")
        image.save(temp_path)
        response = ask_with_image(MODEL, TOKENIZER, temp_path, question, CONFIG)
    else:
        response = ask_text_only(MODEL, TOKENIZER, question, CONFIG)

    return response


def build_ui():
    """Build the Gradio interface — big buttons, clear text, warm colors."""

    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
        font-family: 'Georgia', serif !important;
    }
    .main-title {
        text-align: center;
        color: #5B4A3F;
        font-size: 2em;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        color: #8B7D6B;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .output-text {
        font-size: 1.3em !important;
        line-height: 1.6 !important;
        color: #3E2F1C !important;
        padding: 20px !important;
        background: #FFF8F0 !important;
        border-radius: 12px !important;
        border: 2px solid #E8D5C4 !important;
    }
    """

    with gr.Blocks(css=css, title="Gemma Remember", theme=gr.themes.Soft()) as app:
        gr.HTML('<div class="main-title">Gemma Remember</div>')
        gr.HTML('<div class="subtitle">Show me a photo, and I\'ll help you remember.</div>')

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Show a photo",
                    sources=["upload", "webcam"],
                    height=300,
                )
                audio_input = gr.Audio(
                    type="filepath",
                    label="Or ask with your voice",
                    sources=["microphone"],
                )
                text_input = gr.Textbox(
                    label="Or type your question",
                    placeholder="Who is this person?",
                    lines=2,
                )
                ask_btn = gr.Button(
                    "Help Me Remember",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="",
                    lines=8,
                    elem_classes=["output-text"],
                    interactive=False,
                    show_label=False,
                )

        ask_btn.click(
            fn=respond,
            inputs=[image_input, audio_input, text_input],
            outputs=output,
        )

        gr.HTML(
            '<div style="text-align:center; color:#aaa; margin-top:2em; font-size:0.85em;">'
            'Everything stays on this device. Your memories are safe.'
            '</div>'
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Gemma Remember UI")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public link (use with caution)")
    args = parser.parse_args()

    print("Loading Gemma Remember...")
    init_model(args.config)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
