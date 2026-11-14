// Dependency-free smoke test for both halves of the plugin.
//
//   node tests/smoke.mjs
//
// It parses the host module, exercises the OTSL → HTML converter, and loads the
// browser bundle against a stub module loader to prove that `apply` registers the
// settings section with the expected ids.  No DSH runtime and no React needed.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const hostSource = fs.readFileSync(path.join(root, 'lib', 'index.js'), 'utf8')
const clientSource = fs.readFileSync(path.join(root, 'lib', 'client.js'), 'utf8')

// -- host half: syntax + converter ------------------------------------------------
const strippedHost = hostSource
  .replace(/^import .*$/gm, '')
  .replace(/^export /gm, '')
  .replaceAll('import.meta.url', '""')
new Function(strippedHost) // throws on syntax errors

const start = hostSource.indexOf('const OTSL_TOKENS')
const end = hostSource.indexOf('// Python discovery')
assert.ok(start > 0 && end > start, 'converter section not found')
const { otslToHtml } = new Function(`${hostSource.slice(start, end)}\nreturn { otslToHtml }`)()

assert.equal(
  otslToHtml('<fcel>Model<fcel>Params<nl><fcel>Qwen3-VL<fcel>0.8B<nl>'),
  '<table><tbody><tr><td>Model</td><td>Params</td></tr><tr><td>Qwen3-VL</td><td>0.8B</td></tr></tbody></table>',
)
assert.equal(
  otslToHtml('<table><fcel>A<fcel>B<nl></table>'),
  '<table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>',
)
assert.equal(otslToHtml('plain markdown, no table'), 'plain markdown, no table')
assert.equal(
  otslToHtml('<fcel>only<nl><ecel><ecel><nl>'),
  '<table><tbody><tr><td>only</td></tr></tbody></table>',
  'empty trailing rows are dropped',
)

// -- browser half: lazy-CJS envelope + slot registration ---------------------------
const registrations = []
const slots = {
  inject(name, register) {
    registrations.push(name)
    register()
  },
  register(def) {
    registrations.push(`${def.name}#${def.id}`)
    return () => {}
  },
}
const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  window: null,
  self: null,
}
sandbox.window = sandbox
sandbox.self = sandbox
vm.createContext(sandbox)
sandbox.__ModuleLoader__ = {
  load: ({ id, factory }) => {
    assert.equal(id, 'dsh-xiaomi-ocr')
    sandbox.__module = factory((name) => {
      assert.equal(name, 'react')
      return { createElement: () => null, useState: (v) => [v, () => {}], useEffect: () => {}, useCallback: (f) => f, useMemo: (f) => f(), useRef: (v) => ({ current: v }) }
    })
  },
}
vm.runInContext(clientSource, sandbox, { filename: 'client.js' })

const exported = sandbox.__module
assert.deepEqual(Object.keys(exported).sort(), ['apply', 'inject'])
assert.deepEqual([...exported.inject], ['slots'])

const ctx = { get: (name) => (name === 'slots' ? slots : undefined), slots }
exported.apply(ctx)
assert.ok(registrations.includes('settings.section'), 'registers a settings section')
assert.ok(registrations.includes('settings.section#xiaomi-ocr'), 'section id matches the entry id')
assert.ok(registrations.includes('settings.section.icon#xiaomi-ocr'), 'registers the matching nav icon')

const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))
assert.equal(pkg.dsh.client.platform, 'web')
assert.ok(pkg.dsh.client.inject.includes('dsh-desktop-workbenches'), 'Desktop installer contract')
assert.equal(pkg.dsh.bundle.patch, 'cordis.patch.yml')
assert.ok(fs.existsSync(path.join(root, 'runtime', 'server.py')))
assert.ok(fs.existsSync(path.join(root, 'runtime', 'render_pdf.py')))

console.log('smoke: ok — host parses, OTSL converter correct, client registers the section')
