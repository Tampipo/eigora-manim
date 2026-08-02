# physense-manim

Manim scenes for [Physense](https://physense.tampipo.fr) explainer videos (e.g. the scanning tunneling microscope walkthrough).

This is a rendering studio, not a deployed service: only scene sources live here. Rendered videos land in `output/` (gitignored) and the finished, compressed file is copied into `physense-web/public/videos/`, which is where it's actually served from — see [Shipping a video](#shipping-a-video-to-the-website).

Where it makes sense, scenes import [`physense-qm`](https://github.com/Tampipo/physense-qm) directly and animate the real solved wavefunctions instead of re-deriving the physics by hand in Manim.

---

## Setup

System dependencies (already present if you've used Manim before): `ffmpeg`, a LaTeX distribution (`latex`, `pdflatex`, `dvisvgm`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Rendering

```bash
scripts/render.sh scenes/stm.py STMSurfaceZoom        # quality h (1080p60) by default
scripts/render.sh scenes/stm.py STMSurfaceZoom m      # 720p30, good for web
scripts/render.sh scenes/stm.py STMSurfaceZoom k 23   # 4K, higher quality (lower crf)
```

This runs `manim render`, then re-encodes the result with `ffmpeg` (`libx264`, `+faststart`) into `output/<SceneClassName>.mp4` — a compressed, web-ready file. Both `media/` (Manim's cache) and `output/` are gitignored: everything here is reproducible from `scenes/`.

---

## Shipping a video to the website

Once a render in `output/` is final:

1. Copy the file into `../physense-web/public/videos/`.
2. Embed it in the relevant MDX article:

   ```mdx
   <Video src="/videos/STMSurfaceZoom.mp4" caption="…" />
   ```

`physense-web` commits its own copy as a normal static asset, so the site doesn't depend on this repo at build or deploy time.

---

## Structure

```
scenes/     Manim Scene subclasses, one file per topic
output/     Rendered exports (gitignored) — copied to physense-web when final
scripts/    render.sh — render + compress in one step
```
