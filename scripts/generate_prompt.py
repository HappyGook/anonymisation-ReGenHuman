#!/usr/bin/env python3
"""
Generate a video-generation prompt from a whole video using a local
Qwen3-VL model served by vLLM in Docker (OpenAI-compatible HTTP endpoint).

No openai package or external API: only the standard library + ffmpeg.

The video is first shrunk with ffmpeg (low resolution, capped frame count,
no audio), then sent base64-encoded as a `video_url` content part.

Example server (adjust to your setup):

    docker run --runtime nvidia --gpus all --ipc=host -p 8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      vllm/vllm-openai:latest \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --max-model-len 32768 \
      --limit-mm-per-prompt '{"video": 1, "image": 0}'
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MODEL_ID = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.getenv("VLM_API_KEY", "EMPTY")  # vLLM ignores it unless --api-key is set

SYSTEM_PROMPT = """
You write detailed prompts for synthetic video generation.

You are given a video. Describe the whole video as one continuous scene,
including how it evolves over time.

Write a single coherent prompt of approximately 120-160 words covering:
- Environment, setting, and background.
- Visible people, their approximate number, clothing, and positions.
- What the people do over time: movements, interactions, entering or
  leaving, changes of posture.
- Objects, furniture, and other scene elements.
- Camera viewpoint, framing, camera motion, lighting, and visual appearance.

Be concrete and visually descriptive. Do not speculate about anything that
cannot be established from the video. Write present tense, as a direct
visual scene prompt.

Return only the prompt text. No title, commentary, markdown, or explanation.
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def shrink_video(video: Path, dst: Path, max_fps: float, max_frames: int,
                 short_side: int) -> float:
    """Re-encode to a small, frame-capped clip covering the WHOLE video.

    Returns the fps of the resulting clip.
    """
    duration = probe_duration(video)
    fps = min(max_fps, max_frames / max(duration, 0.1))
    fps = max(fps, 0.25)
    log(f"video duration {duration:.1f}s -> sampling at {fps:.2f} fps "
        f"(~{int(duration * fps)} frames)")

    # scale so the short side == short_side, dims divisible by 2
    vf = (
        f"fps={fps:.4f},"
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-an", "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-pix_fmt", "yuv420p",
            str(dst),
        ],
        check=True,
    )
    return fps


def build_user_text(style_reference: str) -> str:
    text = (
        "Describe this video as a detailed video-generation prompt. "
        "Cover the whole duration and mention how the people and the scene "
        "change over time. Do not invent unobservable details."
    )
    if style_reference:
        text += (
            "\n\nUse the following existing prompt as a STYLE REFERENCE for "
            "the level of detail and writing style. Do not copy its scene "
            f"content:\n\n{style_reference}"
        )
    return text


def query_vlm(video_path: Path, fps: float, style_reference: str,
              max_tokens: int, timeout: float) -> str:
    b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_text(style_reference)},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        # forwarded by vLLM to the Qwen3-VL processor
        "mm_processor_kwargs": {"fps": fps},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"VLM server returned HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach VLM server at {BASE_URL}: {e}") from e

    text = body["choices"][0]["message"].get("content")
    if not text:
        raise RuntimeError("VLM returned an empty response.")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*```(?:text)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return " ".join(text.split())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a video-generation prompt from a whole video."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--style-reference", type=Path, default=None)
    p.add_argument("--max-fps", type=float, default=2.0,
                   help="Upper bound for frame sampling rate (default 2).")
    p.add_argument("--max-frames", type=int, default=64,
                   help="Cap on frames sent to the VLM (default 64).")
    p.add_argument("--short-side", type=int, default=360,
                   help="Short side in px after downscaling (default 360).")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    if not args.video.is_file():
        p.error(f"Video does not exist: {args.video}")

    style_reference = ""
    if args.style_reference is not None:
        if not args.style_reference.is_file():
            p.error(f"Style reference not found: {args.style_reference}")
        style_reference = args.style_reference.read_text(encoding="utf-8").strip()

    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "small.mp4"
        fps = shrink_video(args.video, small, args.max_fps,
                           args.max_frames, args.short_side)
        log(f"sending {small.stat().st_size / 1e6:.1f} MB to {BASE_URL}")
        prompt = query_vlm(small, fps, style_reference,
                           args.max_tokens, args.timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt + "\n", encoding="utf-8")
    print(prompt)


if __name__ == "__main__":
    main()