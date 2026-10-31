#!/usr/bin/env python3
"""
Generate video-generation prompts from whole videos with a local
Qwen3-VL model, loaded in-process via transformers (no server, no API).

The model is loaded ONCE and used for every video in the manifest.

Usage:
    python scripts/generate_prompt_video.py \
        --manifest runs/batch_01/manifest.txt \
        --output-dir runs/batch_01/prompts \
        --style-reference assets/examples/sample_caption.txt

Output files are named <NNN>_<video-stem>.txt (NNN = 1-based line number in
the manifest), matching the IDs used by the pipeline shell script.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT = """You write detailed prompts for synthetic video generation.

You are given a video. Describe the whole video as one continuous scene,
including how it evolves over time.

Write a single coherent prompt of approximately 120-180 words covering:
- Environment, setting, and background.
- Visible people, their approximate number, clothing, and positions.
- What the people do over time: movements, interactions, entering or
  leaving, changes of posture.
- Objects, furniture, and other scene elements.
- Camera viewpoint, framing, camera motion, lighting, and visual appearance.

Be concrete and visually descriptive. Do not speculate about anything that
cannot be established from the video. Write in present tense, as a direct
visual scene prompt.

Return only the prompt text. No title, commentary, markdown, or explanation."""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def shrink_video(video: Path, dst: Path, max_fps: float, max_frames: int,
                 short_side: int) -> None:
    """Re-encode to a small clip covering the WHOLE video with a capped
    frame count, so that every frame in the file is used by the model."""
    duration = probe_duration(video)
    fps = max(min(max_fps, max_frames / max(duration, 0.1)), 0.25)
    log(f"  duration {duration:.1f}s -> {fps:.2f} fps (~{int(duration * fps)} frames)")

    vf = (f"fps={fps:.4f},"
          f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-an", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )


def build_messages(video_path: Path, style_reference: str) -> list[dict]:
    user_text = (
        "Describe this video as a detailed video-generation prompt. "
        "Cover the whole duration and mention how the people and the scene "
        "change over time. Do not invent unobservable details."
    )
    if style_reference:
        user_text += (
            "\n\nUse the following existing prompt as a STYLE REFERENCE for "
            "the level of detail and writing style. Do not copy its scene "
            f"content:\n\n{style_reference}"
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": user_text},
        ]},
    ]


@torch.inference_mode()
def describe_video(model, processor, video_path: Path, style_reference: str,
                   max_new_tokens: int) -> str:
    messages = build_messages(video_path, style_reference)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    log(f"  input tokens: {inputs['input_ids'].shape[-1]}")

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                               do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, backend="torchcodec"
    )[0]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*```(?:text)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = " ".join(text.split())
    if not text:
        raise RuntimeError("model returned an empty response")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Text file with one video path per line.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--style-reference", type=Path, default=None)
    p.add_argument("--max-fps", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=64)
    p.add_argument("--short-side", type=int, default=360)
    p.add_argument("--max-new-tokens", type=int, default=400)
    args = p.parse_args()

    style_reference = ""
    if args.style_reference is not None:
        if not args.style_reference.is_file():
            p.error(f"Style reference not found: {args.style_reference}")
        style_reference = args.style_reference.read_text(encoding="utf-8").strip()

    videos = [Path(line.strip()) for line in
              args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {MODEL_ID} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    failed = 0
    for index, video in enumerate(videos, start=1):
        vid_id = f"{index:03d}_{video.stem}"
        out_file = args.output_dir / f"{vid_id}.txt"
        log(f"[{vid_id}] {video}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                small = Path(tmp) / "small.mp4"
                shrink_video(video, small, args.max_fps, args.max_frames,
                             args.short_side)
                prompt = describe_video(model, processor, small,
                                        style_reference, args.max_new_tokens)
            out_file.write_text(prompt + "\n", encoding="utf-8")
            log(f"[{vid_id}] OK")
        except Exception:
            failed += 1
            out_file.unlink(missing_ok=True)
            log(f"[{vid_id}] FAILED")
            traceback.print_exc()

    log(f"Done: {len(videos) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()