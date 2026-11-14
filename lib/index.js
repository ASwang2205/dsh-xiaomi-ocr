/**
 * dsh-xiaomi-ocr — local OCR for DeepSeek Harness.
 *
 * Replaces cloud OCR round-trips with a 0.8B vision model (Xiaomi-OCR-0) that
 * runs on this machine.  The plugin is backend-agnostic: the shipped Python
 * service speaks OpenAI's `/v1/chat/completions`, so the same tools work
 * against Ollama, llama.cpp, SGLang, or a remote endpoint.
 *
 * Host half responsibilities:
 *   1. own the managed runtime (detect Python, spawn/stop the server, health)
 *   2. expose `ocr_image` / `ocr_pdf` / `ocr_kie` / `ocr_vqa` / `ocr_status`
 *   3. serve a tiny JSON RPC for the settings page (`settings.section`)
 */

import { spawn, spawnSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import z from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = path.resolve(HERE, '..')
export const name = 'dsh-xiaomi-ocr'
export const NS = name
export const inject = ['tools']

const DATA_ROOT = path.join(os.homedir(), '.dsh', 'xiaomi-ocr')
const DEFAULT_MODEL_DIR = path.join(DATA_ROOT, 'models', 'Xiaomi-OCR-0')
const CACHE_DIR = path.join(DATA_ROOT, 'cache')
const LOG_DIR = path.join(DATA_ROOT, 'logs')
const TMP_DIR = path.join(DATA_ROOT, 'tmp')
const PYLIBS = path.join(DATA_ROOT, 'pylibs')
const RUNTIME_CACHE = path.join(DATA_ROOT, 'runtime.json')
const SERVER_SCRIPT = path.join(PKG_ROOT, 'runtime', 'server.py')
const RENDER_SCRIPT = path.join(PKG_ROOT, 'runtime', 'render_pdf.py')

/** Prompt text from the upstream Xiaomi-OCR-0 model card — keep verbatim. */
const PROMPTS = {
  document:
    'Extract all information from the main body of the document image and represent it in markdown format, ' +
    'ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should ' +
    'be represented using LATEX format, and the parsing should be organized according to the reading order.',
  text: 'Extract the text in the image.',
  table: 'Parse the table in the image into OTSL.',
  formula: 'Identify the formula in the image and represent it using LATEX format.',
  page:
    'Recognize all text in the image and output it in markdown format, keeping the original reading order and ' +
    'line breaks.',
}

export const Config = z.object({
  backend: z.string().default('managed').volatile(), // managed | openai
  endpoint: z.string().default('http://127.0.0.1:18765/v1').volatile(),
  apiKey: z.string().role('secret').volatile(),
  model: z.string().default('Xiaomi-OCR-0').volatile(),
  timeoutMs: z.number().default(300000).volatile(),
  maxTokens: z.number().default(4096).volatile(),
  maxPixels: z.number().default(3600000).volatile(),
  mode: z.string().default('document').volatile(),
  pdfDpi: z.number().default(160).volatile(),
  pdfMaxPages: z.number().default(20).volatile(),
  tableToHtml: z.boolean().default(true).volatile(),
  cache: z.boolean().default(true).volatile(),
  clipChars: z.number().default(24000).volatile(),
  autoStart: z.boolean().default(true).volatile(),
  idleExitSeconds: z.number().default(1800).volatile(),
  pythonPath: z.string().default('').volatile(),
  pythonRoots: z.array(z.string()).default([]).volatile(),
  modelDir: z.string().default('').volatile(),
  serverPort: z.number().default(18765).volatile(),
})

// --------------------------------------------------------------------------- //
// config helpers
// --------------------------------------------------------------------------- //

/** Unwrap the live `Volatile<T>` refs the loader hands to `apply`. */
function resolveConfig(config) {
  const out = {}
  for (const [key, value] of Object.entries(config ?? {})) {
    out[key] = value !== null && typeof value === 'object' && typeof value.get === 'function' ? value.get() : value
  }
  return out
}

// --------------------------------------------------------------------------- //
// small utilities
// --------------------------------------------------------------------------- //

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function ensureDirs() {
  await Promise.all([DATA_ROOT, CACHE_DIR, LOG_DIR, TMP_DIR].map((dir) => fsp.mkdir(dir, { recursive: true })))
}

function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex')
}

/** Sample images downloaded next to the weights, handy for first-run checks. */
function listSamples(limit = 6) {
  const roots = [
    path.join(DATA_ROOT, 'samples', 'example_pics'),
    path.join(DATA_ROOT, 'samples'),
  ]
  const out = []
  for (const root of roots) {
    try {
      for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        if (out.length >= limit) return out
        if (!entry.isFile()) continue
        if (!/\.(png|jpe?g|webp|bmp)$/i.test(entry.name)) continue
        out.push({ name: entry.name, path: path.join(root, entry.name) })
      }
    } catch {
      /* no samples */
    }
  }
  return out
}

function resolveToolPath(input, exec) {
  if (!input) return ''
  const raw = String(input)
  if (path.isAbsolute(raw)) return path.normalize(raw)
  const candidates = [
    exec?.agent?.cwd,
    exec?.agent?.session?.cwd,
    exec?.cwd,
    process.cwd(),
  ].filter((value) => typeof value === 'string' && value)
  for (const base of candidates) {
    const candidate = path.resolve(base, raw)
    if (fs.existsSync(candidate)) return candidate
  }
  return path.resolve(candidates[0] ?? process.cwd(), raw)
}

function mimeOf(file) {
  switch (path.extname(file).toLowerCase()) {
    case '.png': return 'image/png'
    case '.webp': return 'image/webp'
    case '.gif': return 'image/gif'
    case '.bmp': return 'image/bmp'
    case '.tif':
    case '.tiff': return 'image/tiff'
    default: return 'image/jpeg'
  }
}

/** Read PNG/JPEG/WebP dimensions without a decoder, for result metadata. */
function imageSize(buffer) {
  try {
    if (buffer.length > 24 && buffer.readUInt32BE(0) === 0x89504e47) {
      return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) }
    }
    if (buffer.length > 4 && buffer[0] === 0xff && buffer[1] === 0xd8) {
      let offset = 2
      while (offset + 9 < buffer.length) {
        if (buffer[offset] !== 0xff) { offset += 1; continue }
        const marker = buffer[offset + 1]
        const size = buffer.readUInt16BE(offset + 2)
        if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
          return { height: buffer.readUInt16BE(offset + 5), width: buffer.readUInt16BE(offset + 7) }
        }
        offset += 2 + size
      }
    }
    if (buffer.length > 30 && buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP') {
      const fmt = buffer.toString('ascii', 12, 16)
      if (fmt === 'VP8 ' && buffer.length > 30) {
        return { width: buffer.readUInt16LE(26) & 0x3fff, height: buffer.readUInt16LE(28) & 0x3fff }
      }
    }
  } catch {
    /* dimensions are cosmetic */
  }
  return null
}

// --------------------------------------------------------------------------- //
// OTSL tables -> HTML
// --------------------------------------------------------------------------- //

const OTSL_TOKENS = /<(fcel|ecel|lcel|ucel|xcel|ched|rhed|nl)>/g
const OTSL_ANY = /<(fcel|ecel|lcel|ucel|xcel|ched|rhed|nl)>/

/** Bump when the post-processing shape changes, so cached replies are not reused. */
const CACHE_VERSION = 2

/** Turn one OTSL body (rows separated by `<nl>`) into an HTML table, or null. */
function otslBlockToHtml(body) {
  const rows = []
  for (const rawRow of body.split(/<nl>/i)) {
    const row = rawRow.replace(/^\s*<(tr|td|tbody|thead)>/i, '').replace(/<\/(tr|td|tbody|thead)>\s*$/i, '')
    if (!row.trim()) continue
    // `split` with a capture group yields [before, tag, text, tag, text, …]:
    // cell contents live at the even indices *after* the first tag.
    const cells = row
      .split(OTSL_TOKENS)
      .filter((_, index) => index > 0 && index % 2 === 0)
      .map((cell) => cell.replace(/<\/?(tr|td|table|tbody|thead)>/gi, '').trim())
    // Trailing `<ecel>` runs only repeat an empty grid row; drop them.
    if (cells.every((cell) => cell === '')) continue
    rows.push(cells)
  }
  if (rows.length === 0) return null
  const width = Math.max(...rows.map((row) => row.length))
  const html = rows.map((row) => {
    const padded = row.concat(Array.from({ length: width - row.length }, () => ''))
    return '<tr>' + padded.map((cell) => `<td>${cell}</td>`).join('') + '</tr>'
  })
  return `<table><tbody>${html.join('')}</tbody></table>`
}

/**
 * Convert OmniDocBench OTSL table markup into a plain HTML table.
 *
 * The model emits either a wrapped `<table>…</table>` block or a bare run of
 * `<fcel>/<nl>` cells; both shapes are handled, and anything without OTSL
 * tokens is returned untouched.
 */
function otslToHtml(text) {
  if (!OTSL_ANY.test(text)) return text
  if (/<table>/i.test(text)) {
    return text.replace(/<table>[\s\S]*?<\/table>/gi, (block) => {
      const body = block.replace(/^<table>/i, '').replace(/<\/table>$/i, '')
      return otslBlockToHtml(body) ?? block
    })
  }
  return otslBlockToHtml(text) ?? text
}

// --------------------------------------------------------------------------- //
// Python discovery
// --------------------------------------------------------------------------- //

function candidatesForPythonRoots(roots) {
  const found = []
  for (const root of roots ?? []) {
    if (typeof root !== 'string' || !root) continue
    const direct = path.join(root, 'python.exe')
    if (fs.existsSync(direct)) found.push(direct)
    try {
      for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue
        // ComfyUI Desktop / portable layouts: <root>/<install>/ComfyUI/.venv/Scripts/python.exe
        for (const candidate of [
          path.join(root, entry.name, 'ComfyUI', '.venv', 'Scripts', 'python.exe'),
          path.join(root, entry.name, '.venv', 'Scripts', 'python.exe'),
          path.join(root, entry.name, 'python_embeded', 'python.exe'),
        ]) {
          if (fs.existsSync(candidate)) found.push(candidate)
        }
      }
    } catch {
      /* not a readable directory */
    }
  }
  return found
}

/** Last interpreter that worked, remembered so a restart does not re-probe. */
function readCachedPython() {
  try {
    const raw = JSON.parse(fs.readFileSync(RUNTIME_CACHE, 'utf8'))
    return typeof raw.pythonPath === 'string' ? raw.pythonPath : ''
  } catch {
    return ''
  }
}

function writeCachedPython(pythonPath) {
  try {
    fs.mkdirSync(DATA_ROOT, { recursive: true })
    fs.writeFileSync(RUNTIME_CACHE, `${JSON.stringify({ pythonPath, updatedAt: new Date().toISOString() }, null, 2)}\n`, 'utf8')
  } catch {
    /* a read-only data dir must not break OCR */
  }
}

function pythonCandidates(cfg) {
  const list = []
  const push = (value) => {
    if (typeof value === 'string' && value && !list.includes(value)) list.push(value)
  }
  push(cfg.pythonPath)
  push(process.env.XIAOMI_OCR_PYTHON)
  push(readCachedPython())
  push(path.join(DATA_ROOT, 'venv', 'Scripts', 'python.exe'))
  for (const candidate of candidatesForPythonRoots(cfg.pythonRoots)) push(candidate)
  return list
}

function probePython(exe) {
  try {
    const result = spawnSync(
      exe,
      ['-c', 'import sys, torch, transformers, PIL; print(sys.version.split()[0]); print(torch.__version__); print(transformers.__version__)'],
      { encoding: 'utf8', timeout: 60000, windowsHide: true },
    )
    if (result.status !== 0) {
      return { path: exe, ok: false, error: (result.stderr || result.stdout || '').trim().split('\n').pop() }
    }
    const [pythonVersion, torchVersion, transformersVersion] = result.stdout.trim().split(/\r?\n/)
    return { path: exe, ok: true, pythonVersion, torchVersion, transformersVersion }
  } catch (error) {
    return { path: exe, ok: false, error: String(error?.message ?? error) }
  }
}

// --------------------------------------------------------------------------- //
// managed runtime supervisor
// --------------------------------------------------------------------------- //

class Runtime {
  constructor(getConfig, logger) {
    this.getConfig = getConfig
    this.logger = logger
    this.child = null
    this.lastError = null
    this.readyAt = 0
    this.starting = null
    this.python = null
    this.pythonCheckedAt = 0
  }

  get cfg() {
    return this.getConfig()
  }

  get origin() {
    const endpoint = (this.cfg.endpoint || 'http://127.0.0.1:18765/v1').replace(/\/+$/, '')
    return endpoint.replace(/\/v1$/, '')
  }

  get isManaged() {
    return (this.cfg.backend || 'managed') === 'managed'
  }

  async health(timeoutMs = 2500) {
    try {
      const response = await fetch(`${this.origin}/health`, { signal: AbortSignal.timeout(timeoutMs) })
      if (!response.ok) return null
      return await response.json()
    } catch {
      return null
    }
  }

  async pythonInfo(force = false) {
    if (!force && this.python && Date.now() - this.pythonCheckedAt < 60000) return this.python
    const cfg = this.cfg
    for (const candidate of pythonCandidates(cfg)) {
      const info = probePython(candidate)
      if (info.ok) {
        this.python = info
        this.pythonCheckedAt = Date.now()
        writeCachedPython(info.path)
        return info
      }
      this.lastProbeError = info.error
    }
    this.python = null
    this.pythonCheckedAt = Date.now()
    return null
  }

  async ensure(signal) {
    if (!this.isManaged) return { ok: true, external: true }
    const health = await this.health()
    if (health?.ok) return { ok: true, health }
    if (!this.cfg.autoStart) {
      throw new Error(
        `本地 OCR 服务未运行（${this.origin}）。请在设置 → 本地 OCR 里点「启动」，或把 backend 设为外部 OpenAI 兼容端点。`,
      )
    }
    if (!this.starting) {
      this.starting = this.spawnServer().finally(() => {
        this.starting = null
      })
    }
    const started = await this.starting
    if (signal?.aborted) throw new Error('已取消')
    return started
  }

  async spawnServer() {
    await ensureDirs()
    const cfg = this.cfg
    const modelDir = cfg.modelDir || DEFAULT_MODEL_DIR
    if (!fs.existsSync(path.join(modelDir, 'model.safetensors'))) {
      const error =
        `未找到本地模型：${modelDir}。` +
        `请在设置 → 本地 OCR → 运行时里点「下载模型」，或手动行 \`hf download xiaomi-research/Xiaomi-OCR-0 --local-dir "${modelDir}"\`。`
      this.lastError = error
      throw new Error(error)
    }
    const python = await this.pythonInfo()
    if (!python) {
      const error =
        '没有找到可用的 Python 运行时（需要 torch + transformers）。' +
        '可在设置 → 本地 OCR → 运行时里指定 python.exe，或先按 docs/runtime.md 创建独立环境。'
      this.lastError = error
      throw new Error(error)
    }
    const port = String(cfg.serverPort || 18765)
    const args = [
      SERVER_SCRIPT,
      '--model', modelDir,
      '--port', port,
      '--max-pixels', String(cfg.maxPixels || 3600000),
      '--idle-exit', String(cfg.idleExitSeconds ?? 1800),
      '--log-file', path.join(LOG_DIR, 'server.log'),
    ]
    this.logger?.info?.(`spawning local OCR service: ${python.path} ${args.join(' ')}`)
    const child = spawn(python.path, args, {
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUNBUFFERED: '1',
        HF_HUB_OFFLINE: '1',
        TOKENIZERS_PARALLELISM: 'false',
      },
    })
    child.unref()
    this.child = child
    const startedAt = Date.now()
    while (Date.now() - startedAt < 180000) {
      await sleep(1200)
      const health = await this.health(1500)
      if (health?.ok) {
        this.readyAt = Date.now()
        this.lastError = null
        return { ok: true, health, cold: true }
      }
      if (health && health.loaded === false && health.load_error) {
        this.lastError = health.load_error
        throw new Error(`模型加载失败：${health.load_error}`)
      }
    }
    this.lastError = '本地 OCR 服务启动超时（180s）'
    throw new Error(this.lastError)
  }

  async stop() {
    if (!this.isManaged) return { ok: false, error: '当前为外部端点，无法停止' }
    let stopped = false
    try {
      const response = await fetch(`${this.origin}/shutdown`, { method: 'POST', signal: AbortSignal.timeout(3000) })
      stopped = response.ok
    } catch {
      stopped = false
    }
    if (!stopped && this.child?.pid) {
      try {
        spawnSync('taskkill', ['/PID', String(this.child.pid), '/T', '/F'], { windowsHide: true })
      } catch {
        /* ignore */
      }
    }
    this.child = null
    await sleep(400)
    return { ok: true }
  }

  async restart() {
    await this.stop()
    this.starting = null
    return this.ensure()
  }
}

// --------------------------------------------------------------------------- //
// OCR client
// --------------------------------------------------------------------------- //

class OcrClient {
  constructor(runtime, getConfig) {
    this.runtime = runtime
    this.getConfig = getConfig
  }

  get cfg() {
    return this.getConfig()
  }

  async call({ file, base64, mime, prompt, maxTokens, signal, cacheKey }) {
    const cfg = this.cfg
    await this.runtime.ensure(signal)
    const endpoint = (cfg.endpoint || 'http://127.0.0.1:18765/v1').replace(/\/+$/, '')
    const body = {
      model: cfg.model || 'Xiaomi-OCR-0',
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mime};base64,${base64}` } },
          { type: 'text', text: prompt },
        ],
      }],
      max_tokens: Math.max(64, Number(maxTokens || cfg.maxTokens || 4096)),
      temperature: 0,
    }
    const headers = { 'Content-Type': 'application/json' }
    if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`
    const started = Date.now()
    let response
    try {
      response = await fetch(`${endpoint}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: signal ?? AbortSignal.timeout(Number(cfg.timeoutMs || 300000)),
      })
    } catch (error) {
      if (signal?.aborted) throw error
      throw new Error(
        `无法连接 OCR 后端 ${endpoint}：${String(error?.message ?? error)}。` +
        (this.runtime.isManaged ? '可在设置 → 本地 OCR 里检查服务状态。' : ''),
      )
    }
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new Error(`OCR 后端返回 ${response.status}: ${text.slice(0, 400)}`)
    }
    const payload = await response.json()
    const text = payload?.choices?.[0]?.message?.content ?? ''
    const meta = {
      ms: Date.now() - started,
      promptTokens: payload?.usage?.prompt_tokens ?? null,
      completionTokens: payload?.usage?.completion_tokens ?? null,
      tokensPerSecond: payload?.xiaomi_ocr?.tokens_per_s ?? null,
      images: payload?.xiaomi_ocr?.images ?? null,
      cacheKey,
    }
    return { text, meta }
  }
}

// --------------------------------------------------------------------------- //
// cache
// --------------------------------------------------------------------------- //

class ResultCache {
  constructor(getConfig) {
    this.getConfig = getConfig
    this.memo = new Map()
  }

  get enabled() {
    return this.getConfig().cache !== false
  }

  async load(key) {
    if (!this.enabled) return null
    if (this.memo.has(key)) return this.memo.get(key)
    try {
      const raw = await fsp.readFile(path.join(CACHE_DIR, `${key}.json`), 'utf8')
      const value = JSON.parse(raw)
      this.memo.set(key, value)
      return value
    } catch {
      return null
    }
  }

  async save(key, entry) {
    if (!this.enabled) return
    this.memo.set(key, entry)
    try {
      await ensureDirs()
      await fsp.writeFile(path.join(CACHE_DIR, `${key}.json`), JSON.stringify(entry), 'utf8')
    } catch {
      /* cache is best effort */
    }
  }

  async stats() {
    try {
      const files = await fsp.readdir(CACHE_DIR)
      let bytes = 0
      for (const file of files) {
        try {
          bytes += (await fsp.stat(path.join(CACHE_DIR, file))).size
        } catch {
          /* ignore */
        }
      }
      return { entries: files.filter((file) => file.endsWith('.json')).length, bytes }
    } catch {
      return { entries: 0, bytes: 0 }
    }
  }

  async clear() {
    this.memo.clear()
    try {
      for (const file of await fsp.readdir(CACHE_DIR)) {
        await fsp.rm(path.join(CACHE_DIR, file), { force: true, recursive: true })
      }
    } catch {
      /* ignore */
    }
  }
}

// --------------------------------------------------------------------------- //
// PDF rasterisation
// --------------------------------------------------------------------------- //

async function ensurePdfLibs(python) {
  if (fs.existsSync(path.join(PYLIBS, 'pypdfium2'))) return true
  await ensureDirs()
  const result = spawnSync(python, ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--target', PYLIBS, 'pypdfium2'], {
    encoding: 'utf8',
    timeout: 300000,
    windowsHide: true,
  })
  if (fs.existsSync(path.join(PYLIBS, 'pypdfium2'))) return true
  throw new Error(
    '未能安装 PDF 渲染依赖 pypdfium2。可手动执行：\n' +
    `"${python}" -m pip install --target "${PYLIBS}" pypdfium2\n` +
    `错误：${(result.stderr || result.stdout || '').trim().split('\n').slice(-3).join(' ')}`,
  )
}

async function renderPdf({ pdf, pages, dpi, maxPages, python }) {
  await ensurePdfLibs(python)
  await ensureDirs()
  const outDir = path.join(TMP_DIR, `pdf-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`)
  const args = [
    RENDER_SCRIPT,
    '--pdf', pdf,
    '--out', outDir,
    '--pages', pages || 'all',
    '--dpi', String(dpi || 160),
    '--max-pages', String(maxPages || 20),
    '--libs', PYLIBS,
  ]
  const result = spawnSync(python, args, { encoding: 'utf8', timeout: 600000, windowsHide: true, maxBuffer: 16 * 1024 * 1024 })
  const stdout = (result.stdout || '').trim()
  if (!stdout) {
    throw new Error(`PDF 渲染失败：${(result.stderr || '').slice(-500) || '无输出'}`)
  }
  const parsed = JSON.parse(stdout.split('\n').pop())
  if (!parsed.ok) throw new Error(`PDF 渲染失败：${parsed.error}`)
  return parsed
}

// --------------------------------------------------------------------------- //
// apply
// --------------------------------------------------------------------------- //

export function apply(ctx, config) {
  const logger = ctx.logger ?? console
  const getConfig = () => resolveConfig(config)
  const runtime = new Runtime(getConfig, logger)
  const cache = new ResultCache(getConfig)
  const client = new OcrClient(runtime, getConfig)

  const language = () => (getConfig().toolLang === 'en' ? 'en' : 'zh')

  // -- core OCR of one image --------------------------------------------------
  async function ocrFile(file, { mode, prompt, maxTokens, signal, refresh, exec } = {}) {
    const cfg = getConfig()
    const buffer = await fsp.readFile(file)
    const chosenMode = mode || cfg.mode || 'document'
    const finalPrompt = prompt || PROMPTS[chosenMode] || PROMPTS.document || PROMPTS.document
    const tokens = Math.max(64, Number(maxTokens || cfg.maxTokens || 4096))
    const key = sha256(Buffer.concat([buffer, Buffer.from(`|${finalPrompt}|${tokens}|${cfg.model}|${chosenMode}|v${CACHE_VERSION}`)]))
    if (!refresh) {
      const hit = await cache.load(key)
      if (hit) {
        return { text: hit.text, meta: { ...hit.meta, cached: true }, size: imageSize(buffer) }
      }
    }
    const { text, meta } = await client.call({
      file,
      base64: buffer.toString('base64'),
      mime: mimeOf(file),
      prompt: finalPrompt,
      maxTokens: tokens,
      signal,
      cacheKey: key,
    })
    const cleaned = cfg.tableToHtml === false ? text : otslToHtml(text)
    await cache.save(key, { text: cleaned, meta, at: Date.now() })
    return { text: cleaned, meta: { ...meta, cached: false }, size: imageSize(buffer) }
  }

  // -- text shaping -----------------------------------------------------------
  function decorate(text, file, mode, meta) {
    const cfg = getConfig()
    const limit = Number(cfg.clipChars || 24000)
    let body = text.trim()
    let clipped = false
    if (body.length > limit) {
      body = body.slice(0, limit) + '\n\n…（输出过长已截断，完整结果请用 saveTo 保存到文件）'
      clipped = true
    }
    return { body, clipped }
  }

  async function saveText(target, content) {
    if (!target) return null
    const out = path.isAbsolute(target) ? target : path.resolve(process.cwd(), target)
    await fsp.mkdir(path.dirname(out), { recursive: true })
    await fsp.writeFile(out, content, 'utf8')
    return out
  }

  // -- tools ------------------------------------------------------------------
  const tools = []

  tools.push(defineTool({
    name: 'ocr_image',
    description:
      'OCR an image file with the LOCAL Xiaomi-OCR-0 model (runs on this machine; nothing is uploaded). ' +
      'Returns Markdown: tables as OTSL (optionally converted to HTML), formulas as LaTeX, reading order preserved. ' +
      'Modes: document (full page layout, default), text (plain text lines), table (single table), formula (single formula), ' +
      'page (handwriting / scene text). Use ocr_pdf for PDFs.',
    parameters: {
      path: { type: 'string', required: true, description: 'Absolute path (relative paths resolve against the session workspace).' },
      mode: { type: 'string', description: 'document | text | table | formula | page (default: configured mode, usually document)' },
      prompt: { type: 'string', description: 'Custom instruction that replaces the built-in prompt for this call.' },
      maxTokens: { type: 'number', description: 'Upper bound for generated tokens (default 4096).' },
      saveTo: { type: 'string', description: 'Write the full Markdown to this file as well.' },
      refresh: { type: 'boolean', description: 'Ignore the cached result for identical input.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const file = resolveToolPath(args.path, exec)
      const stat = await fsp.stat(file).catch(() => null)
      if (!stat?.isFile()) throw new Error(`文件不存在：${file}`)
      const mode = args.mode || getConfig().mode || 'document'
      const { text, meta, size } = await ocrFile(file, {
        mode,
        prompt: args.prompt,
        maxTokens: args.maxTokens,
        refresh: args.refresh === true,
        signal: exec?.signal,
        exec,
      })
      const { body } = decorate(text, file, mode, meta)
      const saved = await saveText(args.saveTo, text)
      const header = `<!-- ${path.basename(file)} · ${size ? `${size.width}×${size.height}` : '?'} · ` +
        `${meta.cached ? 'cached' : `${meta.completionTokens ?? '?'} tok / ${((meta.ms ?? 0) / 1000).toFixed(1)}s`} -->`
      const trailer = saved ? `\n\n_已保存到 ${saved}_` : ''
      return `${header}\n\n${body}${trailer}`
    },
  }))

  tools.push(defineTool({
    name: 'ocr_pdf',
    description:
      'OCR a PDF locally: every selected page is rasterised and parsed page by page with Xiaomi-OCR-0. ' +
      'Returns one Markdown block per page. Long documents are clipped in the reply and written to disk — ' +
      'pass saveTo (or read the returned path) for the full text.',
    parameters: {
      path: { type: 'string', required: true, description: 'Absolute path to the PDF.' },
      pages: { type: 'string', description: 'Page range such as "1-5,8" (default: all pages, capped by maxPages).' },
      dpi: { type: 'number', description: 'Rasterisation DPI (default 160; raise to 200 for tiny print).' },
      maxPages: { type: 'number', description: 'Hard cap on rendered pages (default from settings, 20).' },
      mode: { type: 'string', description: 'document | text | table | formula | page' },
      saveTo: { type: 'string', description: 'Path for the combined Markdown (default: <pdf>.ocr.md when output is long).' },
      refresh: { type: 'boolean', description: 'Ignore cached per-page results.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const cfg = getConfig()
      const file = resolveToolPath(args.path, exec)
      if (!fs.existsSync(file)) throw new Error(`文件不存在：${file}`)
      const python = await runtime.pythonInfo()
      if (!python) throw new Error('没有可用的 Python 运行时，无法渲染 PDF。请在设置 → 本地 OCR 里指定 python.exe。')
      const rendered = await renderPdf({
        pdf: file,
        pages: args.pages,
        dpi: args.dpi || cfg.pdfDpi || 160,
        maxPages: args.maxPages || cfg.pdfMaxPages || 20,
        python: python.path,
      })
      const mode = args.mode || 'document'
      const chunks = []
      const started = Date.now()
      let tokens = 0
      for (const page of rendered.pages) {
        if (exec?.signal?.aborted) throw new Error('已取消')
        const pageResult = await ocrFile(page.path, {
          mode,
          maxTokens: args.maxTokens,
          refresh: args.refresh === true,
          signal: exec?.signal,
          exec,
        })
        tokens += pageResult.meta.completionTokens || 0
        chunks.push(`\n\n## 第 ${page.page} 页${pageResult.meta.cached ? '（缓存）' : ''}\n\n${pageResult.text.trim()}`)
        await fsp.rm(page.path, { force: true }).catch(() => {})
      }
      const full = `<!-- ${path.basename(file)} · ${rendered.pages.length}/${rendered.total} 页 · ` +
        `${tokens} tok · ${((Date.now() - started) / 1000).toFixed(1)}s -->${chunks.join('')}`
      const limit = Number(cfg.clipChars || 24000)
      let reply = full
      let savedPath = null
      if (full.length > limit) {
        const target = args.saveTo || `${file}.ocr.md`
        savedPath = await saveText(target, full)
        reply = full.slice(0, limit) + `\n\n…（共 ${full.length} 字符，完整内容已写入 ${savedPath}）`
      } else if (args.saveTo) {
        savedPath = await saveText(args.saveTo, full)
        reply = full + `\n\n_已保存到 ${savedPath}_`
      }
      if (rendered.truncated) {
        reply += `\n\n_注意：文档共 ${rendered.total} 页，按上限只处理了前 ${rendered.pages.length} 页。_`
      }
      return reply
    },
  }))

  tools.push(defineTool({
    name: 'ocr_kie',
    description:
      'Key-information extraction on a local image: give a JSON schema (or a short field list) and get structured JSON back. ' +
      'Useful for receipts, invoices, ID cards, forms, shipping labels.',
    parameters: {
      path: { type: 'string', required: true, description: 'Absolute path to the image.' },
      schema: { type: 'string', required: true, description: 'JSON Schema (as a string) or a plain field list describing what to extract.' },
      maxTokens: { type: 'number', description: 'Upper bound for generated tokens (default 4096).' },
      refresh: { type: 'boolean', description: 'Ignore the cached result.' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const file = resolveToolPath(args.path, exec)
      if (!fs.existsSync(file)) throw new Error(`文件不存在：${file}`)
      const prompt = `Extract key information in the image and return it as JSON matching this schema: ${args.schema}`
      const { text, meta } = await ocrFile(file, {
        mode: 'kie',
        prompt,
        maxTokens: args.maxTokens,
        refresh: args.refresh === true,
        signal: exec?.signal,
        exec,
      })
      const fence = '```'
      return `${fence}json\n${text.trim()}\n${fence}\n\n<!-- ${meta.cached ? 'cached' : `${meta.completionTokens ?? '?'} tok / ${((meta.ms ?? 0) / 1000).toFixed(1)}s`} -->`
    },
  }))

  tools.push(defineTool({
    name: 'ocr_vqa',
    description:
      'Ask a question about a local image (chart reading, screenshots, diagrams, handwriting). ' +
      'Runs on the local Xiaomi-OCR-0 model — no upload.',
    parameters: {
      path: { type: 'string', required: true, description: 'Absolute path to the image.' },
      question: { type: 'string', required: true, description: 'The question to answer about the image.' },
      maxTokens: { type: 'number', description: 'Upper bound for generated tokens (default 2048).' },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const file = resolveToolPath(args.path, exec)
      if (!fs.existsSync(file)) throw new Error(`文件不存在：${file}`)
      const { text, meta } = await ocrFile(file, {
        mode: 'vqa',
        prompt: args.question,
        maxTokens: args.maxTokens || 2048,
        refresh: true,
        signal: exec?.signal,
        exec,
      })
      return `${text.trim()}\n\n<!-- ${meta.completionTokens ?? '?'} tok / ${((meta.ms ?? 0) / 1000).toFixed(1)}s -->`
    },
  }))

  tools.push(defineTool({
    name: 'ocr_status',
    description:
      'Report the local OCR runtime: backend, endpoint, health, model, device, queue, cache size, Python interpreter, ' +
      'and the paths involved. Use it when OCR fails or before promising the user that OCR is available.',
    parameters: {},
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute() {
      return await describeStatus()
    },
  }))

  async function describeStatus() {
    const cfg = getConfig()
    const health = await runtime.health()
    const python = await runtime.pythonInfo()
    const cacheStats = await cache.stats()
    const rows = [
      `后端: ${cfg.backend === 'openai' ? '外部 OpenAI 兼容端点' : '内置本地服务'}`,
      `端点: ${cfg.endpoint}`,
      `模型: ${cfg.model}`,
      `服务: ${health?.ok ? `运行中 · ${health.device} · ${health.dtype} · 队列 ${health.queue ?? 0}` : '未运行'}`,
      health?.ok ? `已处理: ${health.requests} 次 · 平均 ${health.avg_seconds ?? '?'}s/页 · 错误 ${health.errors}` : null,
      `Python: ${python ? `${python.path} (Python ${python.pythonVersion}, torch ${python.torchVersion}, transformers ${python.transformersVersion})` : '未找到'}`,
      `模型目录: ${cfg.modelDir || DEFAULT_MODEL_DIR}${fs.existsSync(path.join(cfg.modelDir || DEFAULT_MODEL_DIR, 'model.safetensors')) ? '' : '（缺失）'}`,
      `数据目录: ${DATA_ROOT}`,
      `缓存: ${cacheStats.entries} 条 / ${(cacheStats.bytes / 1024).toFixed(0)} KB`,
      runtime.lastError ? `最近错误: ${runtime.lastError}` : null,
    ].filter(Boolean)
    return rows.join('\n')
  }

  for (const tool of tools) ctx.tools.register(tool)

  // -- settings page RPC ------------------------------------------------------
  ctx.inject(['webServer'], (sctx) => {
    const webServer = sctx.get('webServer') ?? sctx.webServer
    if (!webServer?.register) return

    const sendJson = (response, code, payload) => {
      const body = Buffer.from(JSON.stringify(payload), 'utf8')
      response.writeHead(code, {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': body.length,
        'Cache-Control': 'no-store',
      })
      response.end(body)
    }

    const readJson = (request) =>
      new Promise((resolve) => {
        let raw = ''
        request.on('data', (chunk) => {
          raw += chunk
          if (raw.length > 4 * 1024 * 1024) request.destroy()
        })
        request.on('end', () => {
          try {
            resolve(raw ? JSON.parse(raw) : {})
          } catch {
            resolve({})
          }
        })
        request.on('error', () => resolve({}))
      })

    const stateHandler = async (_request, response) => {
      const cfg = getConfig()
      const health = await runtime.health(1200)
      const python = runtime.python ?? null
      const cacheStats = await cache.stats()
      sendJson(response, 200, {
        ok: true,
        version: readVersion(),
        config: { ...cfg, apiKey: cfg.apiKey ? '***' : '' },
        health,
        python,
        pythonError: runtime.lastProbeError ?? null,
        paths: {
          data: DATA_ROOT,
          models: path.join(DATA_ROOT, 'models'),
          model: cfg.modelDir || DEFAULT_MODEL_DIR,
          modelPresent: fs.existsSync(path.join(cfg.modelDir || DEFAULT_MODEL_DIR, 'model.safetensors')),
          cache: CACHE_DIR,
          logs: LOG_DIR,
          server: SERVER_SCRIPT,
        },
        cache: cacheStats,
        samples: listSamples(),
        lastError: runtime.lastError,
      })
    }

    const actionHandler = async (request, response) => {
      const body = await readJson(request)
      const action = String(body.action ?? '')
      const cfg = getConfig()
      try {
        switch (action) {
          case 'start': {
            const result = await runtime.ensure()
            return sendJson(response, 200, { ok: true, health: result.health ?? (await runtime.health()) })
          }
          case 'stop': {
            await runtime.stop()
            return sendJson(response, 200, { ok: true })
          }
          case 'restart': {
            await runtime.restart()
            return sendJson(response, 200, { ok: true, health: await runtime.health() })
          }
          case 'probe': {
            const info = await runtime.pythonInfo(true)
            return sendJson(response, 200, { ok: Boolean(info), python: info, error: runtime.lastProbeError ?? null })
          }
          case 'clear-cache': {
            await cache.clear()
            return sendJson(response, 200, { ok: true, cache: await cache.stats() })
          }
          case 'open-data': {
            spawn('explorer.exe', [DATA_ROOT], { detached: true, stdio: 'ignore' }).unref()
            return sendJson(response, 200, { ok: true })
          }
          case 'open-logs': {
            spawn('explorer.exe', [LOG_DIR], { detached: true, stdio: 'ignore' }).unref()
            return sendJson(response, 200, { ok: true })
          }
          case 'test': {
            const sample = String(body.path ?? '').trim()
            if (!sample) return sendJson(response, 400, { ok: false, error: '缺少测试图片路径' })
            const file = path.isAbsolute(sample) ? sample : path.resolve(process.cwd(), sample)
            if (!fs.existsSync(file)) return sendJson(response, 400, { ok: false, error: `文件不存在：${file}` })
            const mode = String(body.mode ?? cfg.mode ?? 'document')
            const started = Date.now()
            const result = await ocrFile(file, { mode, refresh: body.refresh === true, maxTokens: Number(body.maxTokens || cfg.maxTokens) })
            return sendJson(response, 200, {
              ok: true,
              text: result.text,
              ms: Date.now() - started,
              meta: result.meta,
              size: result.size,
            })
          }
          default:
            return sendJson(response, 400, { ok: false, error: `unknown action: ${action}` })
        }
      } catch (error) {
        runtime.lastError = String(error?.message ?? error)
        return sendJson(response, 500, { ok: false, error: runtime.lastError })
      }
    }

    sctx.effect(() =>
      webServer.register({ kind: 'exact', path: '/dsh-xiaomi-ocr/state', handler: stateHandler }),
    )
    sctx.effect(() =>
      webServer.register({ kind: 'exact', path: '/dsh-xiaomi-ocr/action', handler: actionHandler }),
    )
    logger?.info?.('dsh-xiaomi-ocr: settings RPC registered at /dsh-xiaomi-ocr/*')
  })

  // -- keep the settings form from auto-generating a second UI -----------------
  ctx.inject(['settings'], (sctx) => {
    sctx.effect(() => sctx.settings.configure({ auto: false }, ctx.fiber))
  })

  logger?.info?.('dsh-xiaomi-ocr: 5 OCR tools registered (local, offline)')
}

function readVersion() {
  try {
    return JSON.parse(fs.readFileSync(path.join(PKG_ROOT, 'package.json'), 'utf8')).version ?? '0.0.0'
  } catch {
    return '0.0.0'
  }
}
