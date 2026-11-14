/* dsh-xiaomi-ocr — settings page (browser half).
 *
 * Hand-written lazy-CJS bundle: the Client module loader hands the factory a
 * `require` that resolves the shell's shared modules (react only, here).
 *
 * The page owns an independent settings section (`settings.section`), talks to
 * the Host half over `/dsh-xiaomi-ocr/*` for runtime actions, and persists every
 * option through the shell's configuration forms (revision-fenced writes).
 */
window.__ModuleLoader__.load({
	id: "dsh-xiaomi-ocr",
	factory: (require) => {
		var module = { exports: {} };
		var exports = module.exports;

		const React = require("react");
		const h = React.createElement;
		const { useCallback, useEffect, useMemo, useRef, useState } = React;

		const NS = "dsh-xiaomi-ocr";
		/** Profile composition entry id — the settings namespace the Host serves. */
		const ENTRY_ID = "xiaomi-ocr";
		const API = "/dsh-xiaomi-ocr";
		const SECTION_ID = "xiaomi-ocr";

		const TEXT = {
			nav: "本地 OCR",
			title: "本地 OCR",
			subtitle: "Xiaomi-OCR-0 · 0.8B · 完全离线运行，图片不出本机",
			runtime: "运行时",
			backend: "后端",
			recognition: "识别默认值",
			trial: "试运行",
			save: "写入配置",
			reload: "刷新状态",
		};

		const MODE_OPTIONS = [
			{ value: "document", label: "文档（版面 + 阅读顺序，默认）" },
			{ value: "text", label: "文本行（纯文字）" },
			{ value: "table", label: "表格（OTSL / HTML）" },
			{ value: "formula", label: "公式（LaTeX）" },
			{ value: "page", label: "手写 / 场景文字" },
		];

		// ------------------------------------------------------------------ //
		// tiny UI primitives
		// ------------------------------------------------------------------ //
		const styles = {
			wrap: { display: "flex", flexDirection: "column", gap: "18px", padding: "4px 2px 40px" },
			hero: { display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "16px", flexWrap: "wrap" },
			h1: { fontSize: "20px", fontWeight: 600, margin: 0, lineHeight: 1.3 },
			sub: { margin: "4px 0 0", fontSize: "12px", opacity: 0.65, lineHeight: 1.5 },
			card: { border: "1px solid color-mix(in srgb, currentColor 14%, transparent)", borderRadius: "12px", padding: "16px 18px", display: "flex", flexDirection: "column", gap: "14px" },
			cardTitle: { fontSize: "13px", fontWeight: 600, letterSpacing: "0.02em", opacity: 0.85, margin: 0, textTransform: "none" },
			row: { display: "grid", gridTemplateColumns: "minmax(150px, 220px) 1fr", gap: "12px 16px", alignItems: "start" },
			label: { fontSize: "13px", lineHeight: "30px", opacity: 0.9 },
			hint: { fontSize: "11.5px", opacity: 0.55, marginTop: "4px", lineHeight: 1.5 },
			input: { width: "100%", boxSizing: "border-box", padding: "5px 9px", fontSize: "13px", font: "inherit", color: "inherit", background: "transparent", border: "1px solid color-mix(in srgb, currentColor 22%, transparent)", borderRadius: "7px", outline: "none" },
			select: { width: "100%", boxSizing: "border-box", padding: "5px 9px", fontSize: "13px", font: "inherit", color: "inherit", background: "transparent", border: "1px solid color-mix(in srgb, currentColor 22%, transparent)", borderRadius: "7px" },
			btn: { font: "inherit", fontSize: "12.5px", padding: "5px 12px", borderRadius: "7px", border: "1px solid color-mix(in srgb, currentColor 24%, transparent)", background: "transparent", color: "inherit", cursor: "pointer", lineHeight: 1.5 },
			btnPrimary: { font: "inherit", fontSize: "12.5px", padding: "5px 12px", borderRadius: "7px", border: "1px solid color-mix(in srgb, currentColor 34%, transparent)", background: "color-mix(in srgb, currentColor 10%, transparent)", color: "inherit", cursor: "pointer", fontWeight: 600, lineHeight: 1.5 },
			btnSmall: { font: "inherit", fontSize: "11px", padding: "2px 8px", borderRadius: "6px", border: "1px solid color-mix(in srgb, currentColor 20%, transparent)", background: "transparent", color: "inherit", cursor: "pointer", opacity: 0.7 },
			pill: (ok, warn) => ({
				display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "12px", padding: "3px 10px", borderRadius: "999px",
				border: "1px solid color-mix(in srgb, currentColor 18%, transparent)",
				background: ok ? "color-mix(in srgb, #22c55e 16%, transparent)" : warn ? "color-mix(in srgb, #f59e0b 16%, transparent)" : "color-mix(in srgb, currentColor 8%, transparent)",
				color: "inherit",
			}),
			dot: (ok, warn) => ({ width: "7px", height: "7px", borderRadius: "999px", background: ok ? "#22c55e" : warn ? "#f59e0b" : "#9ca3af" }),
			mono: { fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: "11.5px", opacity: 0.8, wordBreak: "break-all", lineHeight: 1.6 },
			pre: { margin: 0, padding: "12px", borderRadius: "8px", background: "color-mix(in srgb, currentColor 6%, transparent)", fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: "12px", lineHeight: 1.55, maxHeight: "320px", overflow: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word" },
			bars: { display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" },
			grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "10px 18px", fontSize: "12px" },
			kv: { display: "flex", flexDirection: "column", gap: "2px" },
			kvK: { opacity: 0.55, fontSize: "11px" },
			kvV: { fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: "12px" },
			error: { fontSize: "12px", color: "#ef4444", lineHeight: 1.5, whiteSpace: "pre-wrap" },
			ok: { fontSize: "12px", color: "#22c55e", lineHeight: 1.5, whiteSpace: "pre-wrap" },
		};

		function Btn({ children, onClick, kind, disabled, title }) {
			const style = kind === "primary" ? styles.btnPrimary : kind === "small" ? styles.btnSmall : styles.btn;
			return h("button", { type: "button", style: { ...style, opacity: disabled ? 0.45 : undefined, cursor: disabled ? "default" : "pointer" }, onClick: disabled ? undefined : onClick, title, disabled }, children);
		}

		function Card({ title, aside, children }) {
			return h("section", { style: styles.card },
				title || aside
					? h("div", { style: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px" } },
						h("h3", { style: styles.cardTitle }, title),
						aside ? h("div", { style: styles.bars }, aside) : null)
					: null,
				children);
		}

		function Row({ label, hint, children, action }) {
			return h("div", { style: styles.row },
				h("div", null,
					h("div", { style: { fontSize: "13px", lineHeight: "30px", opacity: 0.9 } }, label),
					hint ? h("div", { style: styles.hint }, hint) : null),
				h("div", { style: { display: "flex", alignItems: "center", gap: "8px" } },
					h("div", { style: { flex: 1, minWidth: 0 } }, children),
					action ?? null));
		}

		function Field({ label, hint, value, onCommit, type = "text", placeholder, disabled, unit }) {
			const [draft, setDraft] = useState(value === undefined || value === null ? "" : String(value));
			const focused = useRef(false);
			useEffect(() => {
				if (!focused.current) setDraft(value === undefined || value === null ? "" : String(value));
			}, [value]);
			const commit = () => {
				if (type === "number") {
					const parsed = Number(draft);
					if (!Number.isFinite(parsed)) { setDraft(String(value ?? "")); return; }
					if (parsed !== value) onCommit(parsed);
				} else if (draft !== String(value ?? "")) {
					onCommit(draft);
				}
			};
			return h(Row, { label, hint },
				h("div", { style: { display: "flex", alignItems: "center", gap: "6px" } },
					h("input", {
						style: styles.input,
						type: type === "password" ? "password" : "text",
						inputMode: type === "number" ? "decimal" : undefined,
						value: draft,
						placeholder,
						disabled,
						onFocus: () => { focused.current = true; },
						onChange: (event) => setDraft(event.target.value),
						onBlur: () => { focused.current = false; commit(); },
						onKeyDown: (event) => { if (event.key === "Enter") event.currentTarget.blur(); },
					}),
					unit ? h("span", { style: { fontSize: "11px", opacity: 0.5, whiteSpace: "nowrap", flex: "none" } }, unit) : null));
		}

		function Toggle({ label, hint, value, onChange }) {
			return h(Row, { label, hint },
				h("label", { style: { display: "inline-flex", alignItems: "center", gap: "8px", cursor: "pointer", fontSize: "12.5px" } },
					h("input", { type: "checkbox", checked: Boolean(value), onChange: (event) => onChange(event.target.checked) }),
					h("span", { style: { opacity: 0.7 } }, value ? "已启用" : "已关闭")));
		}

		function Select({ label, hint, value, options, onChange }) {
			return h(Row, { label, hint },
				h("select", { style: styles.select, value: value ?? "", onChange: (event) => onChange(event.target.value) },
					...options.map((option) => h("option", { key: option.value, value: option.value }, option.label))));
		}

		function KV({ k, v }) {
			return h("div", { style: styles.kv }, h("span", { style: styles.kvK }, k), h("span", { style: styles.kvV }, v));
		}

		// ------------------------------------------------------------------ //
		// host helpers
		// ------------------------------------------------------------------ //
		async function postAction(action, extra) {
			const response = await fetch(`${API}/action`, {
				method: "POST",
				headers: { "Content-Type": "application/json", Accept: "application/json" },
				body: JSON.stringify({ action, ...(extra ?? {}) }),
			});
			const payload = await response.json().catch(() => ({}));
			if (!response.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${response.status}`);
			return payload;
		}

		function unwrapSnap(snapshot) {
			if (!snapshot) return { value: {}, writable: false, revision: undefined, status: "loading", user: {} };
			return snapshot;
		}

		/** Locate this plugin's configuration form, tolerating a renamed entry id. */
		function locateForm(ctx) {
			const configForms = ctx.get?.("configForms") ?? ctx.configForms;
			if (!configForms?.get) return null;
			// The settings namespace is the profile composition entry id from
			// cordis.patch.yml (`xiaomi-ocr`); the package name is only a fallback.
			for (const id of [ENTRY_ID, NS]) {
				try {
					const form = configForms.get(id);
					const snapshot = unwrapSnap(form?.getSnapshot?.());
					if (snapshot.status === "ready") return { form, snapshot };
				} catch {
					/* unknown namespace */
				}
			}
			try {
				const view = configForms.describe?.().getSnapshot?.()?.view;
				const match = (view?.namespaces ?? []).find((entry) => entry?.value && typeof entry.value === "object" && "maxPixels" in entry.value);
				if (match?.ns) {
					const form = configForms.get(match.ns);
					return { form, snapshot: unwrapSnap(form?.getSnapshot?.()) };
				}
			} catch {
				/* describe mirror not ready yet */
			}
			const form = configForms.get(ENTRY_ID);
			return { form, snapshot: unwrapSnap(form?.getSnapshot?.()) };
		}

		function useHostState(ctx) {
			const [formState, setFormState] = useState(() => locateForm(ctx) ?? { form: null, snapshot: unwrapSnap(null) });
			const [status, setStatus] = useState(null);
			const [error, setError] = useState("");

			const refreshStatus = useCallback(async () => {
				try {
					const response = await fetch(`${API}/state`, { headers: { Accept: "application/json" } });
					const payload = await response.json();
					if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
					setStatus(payload);
					setError("");
				} catch (cause) {
					setError(String(cause?.message ?? cause));
				}
			}, []);

			useEffect(() => {
				let live = true;
				const pull = () => {
					if (!live) return;
					const next = locateForm(ctx);
					if (next) setFormState(next);
				};
				pull();
				const timer = setInterval(pull, 2000);
				void refreshStatus();
				const statusTimer = setInterval(() => { void refreshStatus(); }, 6000);
				return () => { live = false; clearInterval(timer); clearInterval(statusTimer); };
			}, [ctx, refreshStatus]);

			const write = useCallback(async (field, value) => {
				const { form } = locateForm(ctx) ?? {};
				if (!form?.set) throw new Error("配置表单尚未就绪，请稍后重试");
				const accepted = await form.set(field, value);
				if (!accepted) throw new Error(`字段 ${field} 写入被拒绝（可能被更高优先级的配置覆盖）`);
				const next = locateForm(ctx);
				if (next) setFormState(next);
			}, [ctx]);

			const clear = useCallback(async (field) => {
				const { form } = locateForm(ctx) ?? {};
				if (!form?.unset) return;
				await form.unset(field);
				const next = locateForm(ctx);
				if (next) setFormState(next);
			}, [ctx]);

			return { formState, status, error, setError, refreshStatus, write, clear };
		}

		// ------------------------------------------------------------------ //
		// page
		// ------------------------------------------------------------------ //
		function SectionPage() {
			const ctx = hostCtx;
			const { formState, status, error, setError, refreshStatus, write, clear } = useHostState(ctx);
			const snapshot = formState.snapshot ?? {};
			const value = snapshot.value ?? {};
			const user = snapshot.user ?? {};
			const writable = Boolean(snapshot.writable);
			const [busy, setBusy] = useState("");
			const [message, setMessage] = useState("");
			const [trialPath, setTrialPath] = useState("");
			const [trialMode, setTrialMode] = useState("document");
			const [trial, setTrial] = useState(null);

			const health = status?.health ?? null;
			const running = Boolean(health?.ok);
			const python = status?.python ?? null;
			const paths = status?.paths ?? {};
			const cache = status?.cache ?? { entries: 0, bytes: 0 };

			const act = async (action, extra, label) => {
				setBusy(label ?? action);
				setMessage("");
				setError("");
				try {
					const result = await postAction(action, extra);
					setMessage(`${label ?? action} 完成`);
					await refreshStatus();
					return result;
				} catch (cause) {
					setError(String(cause?.message ?? cause));
					return null;
				} finally {
					setBusy("");
				}
			};

			const commit = async (field, next) => {
				try {
					await write(field, next);
					setMessage("已保存");
					setError("");
					await refreshStatus();
				} catch (cause) {
					setError(String(cause?.message ?? cause));
				}
			};

			const reset = async (field) => {
				try {
					await clear(field);
					setMessage("已恢复默认");
				} catch (cause) {
					setError(String(cause?.message ?? cause));
				}
			};

			const overridden = (field) => user && Object.prototype.hasOwnProperty.call(user, field);

			const field = (key, label, opts = {}) => {
				const current = value[key];
				return h(Field, {
					key,
					label,
					hint: opts.hint,
					value: current,
					type: opts.type ?? "text",
					placeholder: opts.placeholder,
					disabled: !writable,
					unit: opts.unit,
					onCommit: (next) => commit(key, next),
				});
			};

			const resetBtn = (key) => (overridden(key)
				? h(Btn, { kind: "small", onClick: () => reset(key), title: "恢复默认值" }, "重置")
				: null);

			// --- status card -------------------------------------------------
			const statusCard = h(Card, {
				title: "服务状态",
				aside: [
					h("span", { key: "pill", style: styles.pill(running, Boolean(status) && !running) },
						h("span", { style: styles.dot(running, Boolean(status) && !running) }),
						running ? "运行中" : status ? "未运行" : "查询中"),
					h(Btn, { key: "refresh", kind: "small", onClick: () => void refreshStatus() }, TEXT.reload),
				],
			},
				h("div", { style: styles.grid },
					h(KV, { k: "设备", v: health?.device ?? "—" }),
					h(KV, { k: "精度", v: health?.dtype ?? "—" }),
					h(KV, { k: "模型", v: status?.config?.model ?? "Xiaomi-OCR-0" }),
					h(KV, { k: "已处理", v: health ? `${health.requests} 次` : "—" }),
					h(KV, { k: "平均耗时", v: health?.avg_seconds ? `${health.avg_seconds}s` : "—" }),
					h(KV, { k: "队列", v: health ? `${health.queue ?? 0}` : "—" }),
					h(KV, { k: "缓存条目", v: `${cache.entries ?? 0}` }),
					h(KV, { k: "版本", v: status?.version ?? "—" }),
				),
				h("div", { style: styles.bars },
					h(Btn, { kind: "primary", onClick: () => void act("start", {}, "启动服务"), disabled: Boolean(busy) || running }, busy === "启动服务" ? "启动中…" : "启动"),
					h(Btn, { onClick: () => void act("stop", {}, "停止服务"), disabled: Boolean(busy) || !running }, "停止"),
					h(Btn, { onClick: () => void act("restart", {}, "重启服务"), disabled: Boolean(busy) }, "重启"),
					h(Btn, { onClick: () => void act("probe", {}, "探测 Python"), disabled: Boolean(busy) }, "探测 Python"),
				),
				status?.lastError ? h("div", { style: styles.error }, status.lastError) : null,
				!status ? h("div", { style: styles.hint }, "无法读取宿主状态：请确认 DSH Desktop 正在运行。") : null,
			);

			// --- runtime card ------------------------------------------------
			const runtimeCard = h(Card, { title: TEXT.runtime },
				h(Row, { label: "Python 运行时", hint: "需带 torch + transformers；留空则自动探测（ComfyUI 的 ROCm 环境会被找到）" },
					h("div", null,
						h("div", { style: styles.mono }, python ? python.path : status?.pythonError ? `未找到：${status.pythonError}` : "未探测"),
						python ? h("div", { style: styles.mono }, `Python ${python.pythonVersion} · torch ${python.torchVersion} · transformers ${python.transformersVersion}`) : null),
					resetBtn("pythonPath")),
				h(Row, { label: "模型目录", hint: "包含 model.safetensors 的目录" },
					h("div", null,
						h(Field, { value: value.modelDir ?? "", disabled: !writable, onCommit: (next) => commit("modelDir", next), placeholder: paths.model ?? "" }),
						h("div", { style: styles.mono }, `${paths.modelPresent ? "已就绪" : "缺失"} · ${paths.model ?? ""}`)),
					resetBtn("modelDir")),
				h(Field, { label: "服务端口", hint: "内置服务监听端口（改后需重启服务）", value: value.serverPort, type: "number", onCommit: (next) => commit("serverPort", next), unit: "端口" }),
				h(Toggle, { label: "自动启动", hint: "首次调用 OCR 工具时自动拉起本地服务", value: value.autoStart, onChange: (next) => commit("autoStart", next) }),
				h(Field, { label: "空闲退出", hint: "无请求多少秒后释放显存（0 = 常驻）", value: value.idleExitSeconds, type: "number", onCommit: (next) => commit("idleExitSeconds", next), unit: "秒" }),
				h("div", { style: styles.bars },
					h(Btn, { onClick: () => void act("open-data", {}, "打开数据目录") }, "打开数据目录"),
					h(Btn, { onClick: () => void act("open-logs", {}, "打开日志") }, "打开日志"),
					h(Btn, { onClick: () => void act("clear-cache", {}, "清空缓存") }, "清空缓存"),
				),
			);

			// --- backend card ------------------------------------------------
			const backendCard = h(Card, { title: TEXT.backend },
				h(Select, {
					label: "后端类型",
					hint: "内置服务由本插件管理；外部端点可为 Ollama / llama.cpp / SGLang / 云端",
					value: value.backend ?? "managed",
					options: [{ value: "managed", label: "内置本地服务（推荐）" }, { value: "openai", label: "外部 OpenAI 兼容端点" }],
					onChange: (next) => commit("backend", next),
				}),
				h(Field, { label: "端点", hint: "需以 /v1 结尾；内置服务默认 http://127.0.0.1:18765/v1", value: value.endpoint, onCommit: (next) => commit("endpoint", next) }),
				h(Field, { label: "模型名", hint: "发送给后端的 model 字段", value: value.model, onCommit: (next) => commit("model", next) }),
				h(Field, { label: "API Key", hint: "仅外部端点需要，留空表示无鉴权", value: value.apiKey, type: "password", onCommit: (next) => commit("apiKey", next) }),
				h(Field, { label: "超时", hint: "单次请求超时（毫秒）", value: value.timeoutMs, type: "number", onCommit: (next) => commit("timeoutMs", next), unit: "ms" }),
			);

			// --- recognition card --------------------------------------------
			const recognitionCard = h(Card, { title: TEXT.recognition },
				h(Select, { label: "默认模式", hint: "工具未显式指定 mode 时使用", value: value.mode ?? "document", options: MODE_OPTIONS, onChange: (next) => commit("mode", next) }),
				h(Field, { label: "最大输出 tokens", hint: "单页上限，长文档可调大", value: value.maxTokens, type: "number", onCommit: (next) => commit("maxTokens", next) }),
				h(Field, { label: "图片最大像素", hint: "超出会等比缩小，控制显存与延迟", value: value.maxPixels, type: "number", onCommit: (next) => commit("maxPixels", next), unit: "px" }),
				h(Field, { label: "PDF DPI", hint: "160 适合正常排版，小字建议 200", value: value.pdfDpi, type: "number", onCommit: (next) => commit("pdfDpi", next) }),
				h(Field, { label: "PDF 每页上限", hint: "单次调用最多渲染的页数", value: value.pdfMaxPages, type: "number", onCommit: (next) => commit("pdfMaxPages", next), unit: "页" }),
				h(Field, { label: "回复截断阈值", hint: "超过该字符数时自动落盘并只回摘要", value: value.clipChars, type: "number", onCommit: (next) => commit("clipChars", next), unit: "字符" }),
				h(Toggle, { label: "表格转 HTML", value: value.tableToHtml, hint: "把模型输出的 OTSL 表格标记转成 <table>", onChange: (next) => commit("tableToHtml", next) }),
				h(Toggle, { label: "结果缓存", value: value.cache, hint: "相同图片与提示词直接复用，重复识别零成本", onChange: (next) => commit("cache", next) }),
			);

			// --- trial card ---------------------------------------------------
			const runTrial = async () => {
				if (!trialPath.trim()) { setError("请填写一张图片的路径"); return; }
				setBusy("试运行");
				setTrial(null);
				setError("");
				try {
					const result = await postAction("test", { path: trialPath.trim(), mode: trialMode, refresh: true });
					setTrial(result);
					await refreshStatus();
				} catch (cause) {
					setError(String(cause?.message ?? cause));
				} finally {
					setBusy("");
				}
			};

			const trialCard = h(Card, { title: TEXT.trial, aside: h("span", { style: styles.hint }, "用本机模型跑一次识别，验证链路") },
				h(Row, { label: "图片路径", hint: "绝对路径；也可填项目内的相对路径" },
					h("input", { style: styles.input, value: trialPath, placeholder: "C:\\path\\to\\page.png", onChange: (event) => setTrialPath(event.target.value) })),
				h(Select, { label: "模式", value: trialMode, options: MODE_OPTIONS, onChange: setTrialMode }),
				h("div", { style: styles.bars },
					h(Btn, { kind: "primary", onClick: () => void runTrial(), disabled: Boolean(busy) }, busy === "试运行" ? "识别中…" : "开始识别"),
					trial ? h("span", { style: styles.hint }, `${trial.meta?.completionTokens ?? "?"} tok · ${(trial.ms / 1000).toFixed(1)}s`) : null),
				trial ? h("pre", { style: styles.pre }, trial.text) : null,
			);

			return h("div", { style: styles.wrap },
				h("div", { style: styles.hero },
					h("div", null,
						h("h1", { style: styles.h1 }, TEXT.title),
						h("p", { style: styles.sub }, TEXT.subtitle)),
					h("div", { style: styles.bars },
						message ? h("span", { style: styles.ok }, message) : null)),
				error ? h("div", { style: styles.error }, error) : null,
				statusCard,
				runtimeCard,
				backendCard,
				recognitionCard,
				trialCard,
			);
		}

		function SectionIcon({ size = 16, active }) {
			return h("svg", {
				width: size, height: size, viewBox: "0 0 24 24", fill: "none",
				stroke: "currentColor", strokeWidth: active ? 2 : 1.7, strokeLinecap: "round", strokeLinejoin: "round",
				"aria-hidden": true,
			},
				h("path", { d: "M4 8V6a2 2 0 0 1 2-2h2" }),
				h("path", { d: "M16 4h2a2 2 0 0 1 2 2v2" }),
				h("path", { d: "M20 16v2a2 2 0 0 1-2 2h-2" }),
				h("path", { d: "M8 20H6a2 2 0 0 1-2-2v-2" }),
				h("path", { d: "M8 12h8" }),
				h("path", { d: "M8 9h8" }),
				h("path", { d: "M8 15h5" }));
		}

		// ------------------------------------------------------------------ //
		// plugin wiring
		// ------------------------------------------------------------------ //
		let hostCtx = null;

		function apply(ctx) {
			hostCtx = ctx;
			const slots = ctx.get?.("slots") ?? ctx.slots;
			if (!slots?.inject) return;
			slots.inject("settings.section", () =>
				slots.register({
					name: "settings.section",
					id: SECTION_ID,
					order: 42,
					label: () => TEXT.nav,
				}, SectionPage)
			);
			slots.inject("settings.section.icon", () =>
				slots.register({
					name: "settings.section.icon",
					id: SECTION_ID,
				}, SectionIcon)
			);
		}

		const inject = ["slots"];

		exports.apply = apply;
		exports.inject = inject;
		return module.exports;
	}
});
