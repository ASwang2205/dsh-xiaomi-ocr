# Changelog

All notable changes to this plugin are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
