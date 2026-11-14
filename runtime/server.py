#!/usr/bin/env python3
"""Xiaomi-OCR-0 local inference service.

A tiny, dependency-light HTTP server that keeps the 0.8B Xiaomi-OCR-0 model
resident and exposes it through an OpenAI-compatible ``/v1/chat/completions``
endpoint (plus ``/health``).  It therefore works as a drop-in local backend for
any OpenAI-compatible client, and the DSH plugin can also point at Ollama /
llama.cpp / SGLang / a remote endpoint with the same code path.

Design notes
------------
* Only stdlib HTTP (`http.server`) — no FastAPI/uvicorn.
* The model is loaded once at startup and guarded by a lock: one generation at
  a time, requests queue up (a 0.8B OCR model is latency-bound on one GPU, and
  batching would complicate the request contract for little gain).
* Images arrive as data URLs (base64), local paths, or http(s) URLs; they are
  decoded with Pillow (``torchvision.io.decode_image`` is unavailable in some
  ROCm wheels because it is built without libjpeg) and optionally downscaled.
* Every reply carries a ``xiaomi_ocr`` extension object with timings and image
  geometry so callers can report real numbers to the user.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("xiaomi-ocr")

MAX_BODY_BYTES = 96 * 1024 * 1024
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def _decode_data_url(url: str):
    from PIL import Image

    header, _, payload = url.partition(",")
    if not payload:
        raise ValueError("malformed data URL")
    raw = base64.b64decode(payload, validate=False)
    return Image.open(io.BytesIO(raw))


def _load_image(source: str):
    """Load a PIL image from a data URL, an http(s) URL, or a local path."""
    from PIL import Image

    if not isinstance(source, str) or not source:
        raise ValueError("image source must be a non-empty string")
    if source.startswith("data:"):
        img = _decode_data_url(source)
    elif source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "xiaomi-ocr/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            img = Image.open(io.BytesIO(resp.read()))
    else:
        path = os.path.expanduser(source)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        img = Image.open(path)
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    except Exception:  # pragma: no cover - optional metadata
        pass
    return img.convert("RGB")


def _fit(img, max_pixels: int):
    """Downscale so that width*height <= max_pixels, preserving aspect ratio."""
    from PIL import Image

    w, h = img.size
    if max_pixels <= 0 or w * h <= max_pixels:
        return img, w, h
    scale = (max_pixels / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS), nw, nh


# --------------------------------------------------------------------------- #
# model runner
# --------------------------------------------------------------------------- #
class OcrEngine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = "float32"
        self.loaded_at = None
        self.load_error = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.queue = 0
        self.started_at = time.time()
        self.last_activity = time.time()
        self.model_dir = args.model

    # -- lifecycle ---------------------------------------------------------
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.time()
        if self.args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.args.device
        if device == "cuda" and not torch.cuda.is_available():
            LOG.warning("CUDA/ROCm requested but unavailable, falling back to CPU")
            device = "cpu"

        dtype_name = self.args.dtype
        if dtype_name == "auto":
            # Measured on this machine (RX 6800 XT / ROCm 7.14): fp16 decodes at
            # ~2x the bf16 rate (26.6 vs 12.9 tok/s), while CUDA GPUs generally
            # prefer bf16.  CPU is always fp32.
            if device == "cpu":
                dtype_name = "float32"
            elif getattr(torch.version, "hip", None):
                dtype_name = "float16"
            else:
                dtype_name = "bfloat16"
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[dtype_name]

        LOG.info("loading model from %s (device=%s dtype=%s)", self.model_dir, device, dtype_name)
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_dir, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype_name
        self.loaded_at = time.time()
        LOG.info("model ready in %.1fs", self.loaded_at - t0)

        if not self.args.no_warmup:
            try:
                self._warmup()
            except Exception as exc:  # pragma: no cover
                LOG.warning("warmup failed: %s", exc)

    def _warmup(self):
        from PIL import Image

        img = Image.new("RGB", (64, 96), "white")
        self.generate(prompt="Extract the text in the image.", images=[img], max_new_tokens=4)

    # -- inference ---------------------------------------------------------
    def _render_text(self, messages):
        import copy

        convo = []
        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    kind = part.get("type")
                    if kind in ("text", "input_text"):
                        parts.append({"type": "text", "text": part.get("text") or part.get("content") or ""})
                    elif kind in ("image", "image_url", "input_image"):
                        parts.append({"type": "image"})
            if not parts:
                parts = [{"type": "text", "text": ""}]
            convo.append({"role": role, "content": parts})
        tpl = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return tpl

    def _collect_images(self, messages):
        sources = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    value = part.get("image_url")
                    if isinstance(value, dict):
                        sources.append(value.get("url"))
                    else:
                        sources.append(value)
                elif part.get("type") in ("image", "input_image"):
                    sources.append(part.get("url") or part.get("image_url") or part.get("image"))
        return [s for s in sources if isinstance(s, str) and s]

    def build_inputs(self, messages, max_pixels):
        import torch

        text = self._render_text(messages)
        sources = self._collect_images(messages)
        images, geo = [], []
        for src in sources:
            img = _load_image(src)
            w0, h0 = img.size
            img, w, h = _fit(img, max_pixels)
            images.append(img)
            geo.append({"width": w0, "height": h0, "sent_width": w, "sent_height": h})
        if images:
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=[text], return_tensors="pt")
        inputs = inputs.to(self.device)
        pv = inputs.get("pixel_values")
        if pv is not None:
            dtype = self.model.dtype if hasattr(self.model, "dtype") else None
            if dtype is not None and pv.dtype != dtype:
                inputs["pixel_values"] = pv.to(dtype)
        del torch
        return inputs, geo, text

    def generate(self, prompt=None, messages=None, images=None, max_new_tokens=None):
        """Run one generation. ``prompt`` + ``images`` is the internal fast path;
        ``messages`` is the OpenAI-shaped one."""
        import torch

        if messages is None:
            parts = []
            if images:
                parts = [{"type": "image"} for _ in images]
            parts.append({"type": "text", "text": prompt or ""})
            messages = [{"role": "user", "content": parts}]

        max_new_tokens = int(max_new_tokens or self.args.max_new_tokens)
        t_start = time.time()
        acquired = False
        self.queue += 1
        try:
            self.lock.acquire()
            acquired = True
            self.queue -= 1
            if images is not None:
                images_in, geo = [], []
                for img in images:
                    w0, h0 = img.size
                    fitted, w, h = _fit(img, self.args.max_pixels)
                    images_in.append(fitted)
                    geo.append({"width": w0, "height": h0, "sent_width": w, "sent_height": h})
                text = self._render_text(messages)
                inputs = self.processor(text=[text], images=images_in, return_tensors="pt").to(self.device)
            else:
                inputs, geo, text = self.build_inputs(messages, self.args.max_pixels)

            pv = inputs.get("pixel_values")
            if pv is not None and pv.dtype != self.model.dtype:
                inputs["pixel_values"] = pv.to(self.model.dtype)

            n_in = int(inputs["input_ids"].shape[1])
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if self.args.repetition_penalty and self.args.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = self.args.repetition_penalty

            t0 = time.time()
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            dt = time.time() - t0
            n_out = int(out.shape[1] - n_in)
            decoded = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
            decoded = THINK_RE.sub("", decoded).strip()

            self.requests += 1
            self.prompt_tokens += n_in
            self.completion_tokens += n_out
            self.latencies.append(dt)
            self.latencies = self.latencies[-40:]
            self.last_activity = time.time()
            return {
                "text": decoded,
                "prompt_tokens": n_in,
                "completion_tokens": n_out,
                "seconds": round(dt, 3),
                "speed": round(n_out / dt, 2) if dt > 0 else None,
                "total_seconds": round(time.time() - t_start, 3),
                "images": geo,
                "prompt_text": text,
            }
        except Exception:
            self.errors += 1
            raise
        finally:
            if acquired:
                self.lock.release()

    # -- stats -------------------------------------------------------------
    def status(self):
        recent = self.latencies[-10:]
        return {
            "ok": self.load_error is None and self.model is not None,
            "model": os.path.basename(self.model_dir.rstrip("\\/")),
            "model_dir": self.model_dir,
            "device": self.device,
            "dtype": self.dtype,
            "loaded": self.model is not None,
            "load_error": self.load_error,
            "uptime_s": round(time.time() - self.started_at, 1),
            "queue": max(self.queue, 0),
            "busy": self.lock.locked(),
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_seconds": round(sum(recent) / len(recent), 2) if recent else None,
            "idle_s": round(time.time() - self.last_activity, 1),
        }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "xiaomi-ocr/1.0"
    engine: OcrEngine = None  # set by serve()

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    # -- plumbing ----------------------------------------------------------
    def _send_json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, code, message, kind="invalid_request_error"):
        self._send_json({"error": {"message": message, "type": kind, "code": code}}, code)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routes ------------------------------------------------------------
    def do_GET(self):  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/healthz", "/"):
            self._send_json(self.engine.status())
        elif path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": self.engine.status()["model"],
                    "object": "model",
                    "owned_by": "local",
                    "created": int(self.engine.started_at),
                }],
            })
        else:
            self._error(404, "not found")

    def do_POST(self):  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path == "/shutdown":
            self._send_json({"ok": True, "bye": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._error(404, "not found")
            return
        if self.engine.model is None:
            self._error(503, self.engine.load_error or "model not loaded", "service_unavailable")
            return
        try:
            body = self._read_body()
        except Exception as exc:
            self._error(400, f"bad request body: {exc}")
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            prompt = body.get("prompt")
            if not prompt:
                self._error(400, "'messages' or 'prompt' is required")
                return
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        max_new_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        try:
            result = self.engine.generate(messages=messages, max_new_tokens=max_new_tokens)
        except FileNotFoundError as exc:
            self._error(400, f"image not found: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface to caller
            LOG.exception("generation failed")
            self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            return

        created = int(time.time())
        self._send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or self.engine.status()["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "xiaomi_ocr": {
                "seconds": result["seconds"],
                "total_seconds": result["total_seconds"],
                "tokens_per_s": result["speed"],
                "images": result["images"],
            },
        })


def _idle_watchdog(server, engine, seconds):
    while True:
        time.sleep(5)
        if engine.lock.locked():
            continue
        if time.time() - engine.last_activity > seconds:
            LOG.info("idle for %ss, shutting down", seconds)
            server.shutdown()
            return


def serve(args):
    engine = OcrEngine(args)
    Handler.engine = engine
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001 - report then keep serving /health
        LOG.exception("model load failed")
        engine.load_error = f"{type(exc).__name__}: {exc}"

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    LOG.info("listening on http://%s:%d", args.host, args.port)
    print(json.dumps({"ready": True, "port": args.port, "pid": os.getpid(), "device": engine.device}), flush=True)
    if args.idle_exit and args.idle_exit > 0:
        threading.Thread(target=_idle_watchdog, args=(httpd, engine, args.idle_exit), daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    finally:
        httpd.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Xiaomi-OCR-0 OpenAI-compatible local server")
    parser.add_argument("--model", required=True, help="path to the local model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096, dest="max_new_tokens")
    parser.add_argument("--max-pixels", type=int, default=3_600_000, dest="max_pixels",
                        help="downscale images larger than this many pixels (0 = never)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, dest="repetition_penalty")
    parser.add_argument("--idle-exit", type=float, default=0, dest="idle_exit",
                        help="exit after N idle seconds (0 = stay resident)")
    parser.add_argument("--no-warmup", action="store_true", dest="no_warmup")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    def _stop(_signum, _frame):
        LOG.info("signal received, stopping")
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except Exception:  # pragma: no cover
            pass

    serve(args)


if __name__ == "__main__":
    main()
