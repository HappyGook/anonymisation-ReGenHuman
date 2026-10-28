#!/usr/bin/env bash

set -Eeuo pipefail

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR="${1:?Usage: $0 INPUT_DIR COUNT [OUTPUT_DIR]}"
COUNT="${2:?Usage: $0 INPUT_DIR COUNT [OUTPUT_DIR]}"

OUTPUT_DIR="${3:-runs/batch_01}"
VARIANT="${VARIANT:-structhuman}"
SEED="${SEED:-}"

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# --------------------------------------------------
# Directories
# --------------------------------------------------

mkdir -p \
    "$OUTPUT_DIR/prompts" \
    "$OUTPUT_DIR/logs"

MANIFEST="$OUTPUT_DIR/manifest.txt"

# --------------------------------------------------
# 1. Sample videos
# --------------------------------------------------

echo "Sampling $COUNT videos from: $INPUT_DIR"

SAMPLER_ARGS=(
    --input-dir "$INPUT_DIR"
    --count "$COUNT"
)

if [[ -n "$SEED" ]]; then
    SAMPLER_ARGS+=(--seed "$SEED")
fi

"$PYTHON" scripts/sample_videos.py \
    "${SAMPLER_ARGS[@]}" > "$MANIFEST"

echo "Selected videos:"
cat "$MANIFEST"

# --------------------------------------------------
# 2. Process each video
# --------------------------------------------------

INDEX=0
FAILED=0

while IFS= read -r VIDEO || [[ -n "$VIDEO" ]]; do
    [[ -z "$VIDEO" ]] && continue

    INDEX=$((INDEX + 1))

    NAME="$(basename "$VIDEO")"
    STEM="${NAME%.*}"
    ID="$(printf '%03d_%s' "$INDEX" "$STEM")"

    PROMPT_FILE="$OUTPUT_DIR/prompts/$ID.txt"
    LOG_FILE="$OUTPUT_DIR/logs/$ID.log"

    echo
    echo "=================================================="
    echo "[$INDEX/$COUNT] Processing: $VIDEO"
    echo "=================================================="

    # ----------------------------------------------
    # 2a. Describe the whole video with the local VLM
    # ----------------------------------------------

    if ! "$PYTHON" scripts/generate_prompt.py \
        --video "$VIDEO" \
        --output "$PROMPT_FILE" \
        --style-reference assets/examples/sample_caption.txt \
        > "$LOG_FILE" 2>&1; then

        echo "ERROR: Prompt generation failed for $VIDEO"
        tail -n 30 "$LOG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Generated prompt:"
    cat "$PROMPT_FILE"

    # ----------------------------------------------
    # 2b. Run ReGenHuman
    # ----------------------------------------------

    if ! "$PYTHON" demo.py \
        --input "$VIDEO" \
        --variant "$VARIANT" \
        --prompt "$(cat "$PROMPT_FILE")" \
        >> "$LOG_FILE" 2>&1; then

        echo "ERROR: ReGenHuman failed for $VIDEO"
        tail -n 50 "$LOG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Successfully processed: $VIDEO"

done < "$MANIFEST"

# --------------------------------------------------
# Summary
# --------------------------------------------------

echo
echo "=================================================="
echo "Batch completed."
echo "Selected videos: $INDEX"
echo "Failures:        $FAILED"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="

if (( FAILED > 0 )); then
    exit 1
fi