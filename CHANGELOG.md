# Changelog

All notable changes to this plugin are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-11-14

Faster pages, smarter prompts, and a much better failure mode on big scans.

### Added

- `ocr_batch` — OCR several images in one call, grouped by pixel budget so a
  batch never exceeds what the GPU can hold.
- `mode: "auto"` — a cheap layout heuristic (`/v1/ocr/detect`) picks the prompt:
  tables → `table`, plain pages → `document`, screenshots and photos → `page`.
- `region: "x,y,w,h"` on `ocr_image` — crop first, then upscale small crops, for
  one table or one formula out of a big page.
- Batched inference: pages and images that share a prompt now run in a single
  forward pass (measured 2.6× per-item speed-up on three identical images).
- Settings: `batchSize` (1–8) and `retries` (0–5) on the 识别默认值 card.

### Changed

- `ocr_pdf` groups pages by pixels and batches them, reusing per-page cache hits.
- Default `maxPixels` is now 2.2 MP (an A4 page at 150 dpi). 3.6 MP asked for
  >20 GB in the vision tower on a 16 GB card.
- Replies report real token counts for batched work instead of `0 tok`.

### Fixed

- Out-of-memory on large scans: the service halves the pixel budget and retries
  (up to twice) instead of failing the request.
- A second service process could be spawned when the settings page and a tool
  call raced; spawns are now de-duplicated.
- Mixed-size batches produced garbage because right padding made short rows
  start generating from pad positions; the processor now pads on the left.

## [0.1.0] — 2026-11-14

First public release: local OCR for DeepSeek Harness.

### Added

- Five tools — `ocr_image`, `ocr_pdf`, `ocr_kie`, `ocr_vqa`, `ocr_status` —
  backed by Xiaomi-OCR-0 (0.8B) running on the local GPU.
- Managed runtime: the plugin spawns, health-checks, restarts and idle-stops a
  stdlib-only Python service that speaks OpenAI `/v1/chat/completions`, so the
  same tools also work against Ollama, llama.cpp, SGLang or a remote endpoint.
- Independent **本地 OCR** settings page (`settings.section`) with live service
  state, runtime control, backend selection, recognition defaults and an
  in-page trial run against bundled sample images.
- PDF pipeline: page rasterisation through `pypdfium2` (installed into a
  plugin-private `pylibs` directory), per-page OCR, automatic spill to
  `<pdf>.ocr.md` when the reply would be too long.
- Post-processing: OmniDocBench **OTSL → HTML** tables (wrapped and bare
  forms, empty trailing rows dropped) and Markdown-friendly LaTeX formulas.
- Result cache keyed by image hash + prompt + parameters, with `refresh: true`
  bypass.
- Precision policy: `float16` on ROCm (twice the bf16 decode rate measured on
  an RX 6800 XT), `bfloat16` on CUDA, `float32` on CPU.
- Interpreter discovery: explicit path → `XIAOMI_OCR_PYTHON` → remembered
  interpreter → private venv → user-supplied search roots (ComfyUI layout).
- `tests/smoke.mjs` (dependency-free) and CI on Node 20/22.
