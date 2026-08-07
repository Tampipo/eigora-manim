#!/usr/bin/env bash
# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Render a Manim scene and produce a web-ready, compressed MP4 in output/.
#
# Usage: scripts/render.sh scenes/stm.py STMIntro [quality] [crf]
#   quality: l|m|h|k (default h = 1080p60)
#   crf:     libx264 quality, lower = bigger/better (default 32 -- these are
#            flat-shaded diagrams, not high-frequency footage, so crf 32 is
#            visually indistinguishable from 23 at a fraction of the size.
#            Drop to ~23 only for scenes with fine texture/noise.)
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <scene_file.py> <SceneClassName> [quality: l|m|h|k] [crf]" >&2
  exit 1
fi

SCENE_FILE="$1"
SCENE_CLASS="$2"
QUALITY="${3:-h}"
CRF="${4:-32}"

declare -A RES=([l]=480p15 [m]=720p30 [h]=1080p60 [k]=2160p60)
STEM="$(basename "${SCENE_FILE%.py}")"

manim render "-q${QUALITY}" --format=mp4 "$SCENE_FILE" "$SCENE_CLASS"

RAW="media/videos/${STEM}/${RES[$QUALITY]}/${SCENE_CLASS}.mp4"
mkdir -p output
OUT="output/${SCENE_CLASS}.mp4"

ffmpeg -y -i "$RAW" -c:v libx264 -crf "$CRF" -preset slow -movflags +faststart "$OUT"

echo "Web-ready file: $OUT"
echo "Next: copy it into eigora-web/public/videos/ and embed with <Video src=\"/videos/${SCENE_CLASS}.mp4\" />"
