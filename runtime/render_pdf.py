#!/usr/bin/env python3
"""Rasterise PDF pages to PNG files for OCR.

Invoked by the DSH plugin as a short-lived child process:

    python render_pdf.py --pdf in.pdf --out DIR --pages 1-3,7 --dpi 160 --libs DIR

Prints a single JSON object on stdout::

    {"ok": true, "total": 12, "pages": [{"page": 1, "path": "...", "width": 1654, "height": 2339}]}

``pypdfium2`` is installed into a plugin-private library directory (``--libs``)
instead of the host interpreter's site-packages, so this never mutates a shared
Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_pages(spec: str, total: int) -> list[int]:
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(total))
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, _, end_s = chunk.partition("-")
            start = int(start_s) if start_s.strip() else 1
            end = int(end_s) if end_s.strip() else total
        else:
            start = end = int(chunk)
        for page in range(start, end + 1):
            if 1 <= page <= total and (page - 1) not in out:
                out.append(page - 1)
    return sorted(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render PDF pages to PNG")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", default="all")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--max-pages", type=int, default=60, dest="max_pages")
    ap.add_argument("--libs", default="", help="extra sys.path entry holding pypdfium2")
    ap.add_argument("--password", default="")
    args = ap.parse_args(argv)

    if args.libs and os.path.isdir(args.libs):
        sys.path.insert(0, args.libs)

    try:
        import pypdfium2 as pdfium
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"pypdfium2 unavailable: {exc}"}))
        return 2

    if not os.path.isfile(args.pdf):
        print(json.dumps({"ok": False, "error": f"pdf not found: {args.pdf}"}))
        return 2

    os.makedirs(args.out, exist_ok=True)
    pdf = pdfium.PdfDocument(args.pdf, password=args.password or None)
    total = len(pdf)
    wanted = parse_pages(args.pages, total)
    truncated = False
    if args.max_pages and len(wanted) > args.max_pages:
        wanted = wanted[: args.max_pages]
        truncated = True

    scale = max(args.dpi, 36) / 72.0
    rendered = []
    for index in wanted:
        page = pdf[index]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil().convert("RGB")
        path = os.path.join(args.out, f"page-{index + 1:04d}.png")
        pil.save(path, format="PNG", optimize=True)
        rendered.append({"page": index + 1, "path": path, "width": pil.width, "height": pil.height})

    print(json.dumps({
        "ok": True,
        "total": total,
        "selected": len(wanted),
        "truncated": truncated,
        "pages": rendered,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
