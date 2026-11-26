// Native DOM inputs only. Never call Gradio trigger/upload/server or fetch.
// Gradio 6.9.0 HTML has no watch API: observe the separate, server-to-browser identity output.
const key = "geniac-eval-viewer-notes-v1";
const textarea = element.querySelector("[data-note-text]");
const fields = element.querySelector("[data-note-fields]");
const checks = [...element.querySelectorAll('input[type="checkbox"]')];
const status = element.querySelector("[data-note-status]");
let entries = new Map();
let active = null;
let storageError = false;
const entryKey = (e) => JSON.stringify([e.evaluation_run, e.trial_id]);
function validate(data) {
    if (!data || data.schema_version !== 1 || !Array.isArray(data.entries)) {
        throw new Error("対応するスキーマバージョン1のJSONではありません");
    }
    const seen = new Set();
    for (const e of data.entries) {
        if (!e || typeof e.evaluation_run !== "string" || !e.evaluation_run ||
            typeof e.trial_id !== "string" || !/^[a-f0-9]{64}$/.test(e.trial_id) ||
            !Array.isArray(e.tags) || !e.tags.every(t => typeof t === "string") ||
            typeof e.note !== "string" || typeof e.updated_at !== "string" ||
            !Number.isFinite(Date.parse(e.updated_at))) {
            throw new Error("メモ項目の形式が不正です");
        }
        if (seen.has(entryKey(e))) throw new Error("JSON内に試行IDの重複があります");
        seen.add(entryKey(e));
    }
    return data.entries;
}
function documentData() { return {schema_version: 1, entries: [...entries.values()]}; }
function persist() {
    try {
        localStorage.setItem(key, JSON.stringify(documentData()));
        storageError = false;
        status.textContent = "ブラウザ内に保存しました";
    } catch (_) {
        storageError = true;
        status.textContent = "ブラウザ保存に失敗しました。ページを閉じる前にJSONを書き出してください。";
    }
}
try {
    const saved = localStorage.getItem(key);
    if (saved) entries = new Map(validate(JSON.parse(saved)).map(e => [entryKey(e), e]));
} catch (_) {
    storageError = true;
    status.textContent = "ブラウザ保存を読み取れません。既存データは上書きせず、JSONで保存してください。";
}
function select() {
    const marker = document.querySelector("[data-note-context]");
    active = marker ? JSON.parse(marker.textContent) : null;
    fields.disabled = !active;
    const e = active ? entries.get(entryKey(active)) : null;
    textarea.value = e?.note || "";
    checks.forEach(c => { c.checked = (e?.tags || []).includes(c.value); });
    element.querySelector("[data-note-identity]").textContent = active
        ? `${active.evaluation_run} / ${active.trial_id}` : "試行を選択してください";
    if (!storageError) status.textContent = e ? "保存済みのメモを復元しました" : "";
}
function edit() {
    if (!active) return;
    const previous = entries.get(entryKey(active));
    const customTags = (previous?.tags || []).filter(t => !checks.some(c => c.value === t));
    entries.set(entryKey(active), {
        evaluation_run: active.evaluation_run, trial_id: active.trial_id,
        tags: [...customTags, ...checks.filter(c => c.checked).map(c => c.value)],
        note: textarea.value, updated_at: new Date().toISOString()
    });
    // If reading storage failed, preserve the original bytes until the user exports the in-memory notes.
    if (!storageError) persist();
    else status.textContent = "ブラウザ保存が利用できません。JSONを書き出してください。";
}
textarea.addEventListener("input", edit);
checks.forEach(c => c.addEventListener("change", edit));
element.querySelector("[data-note-export]").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(documentData(), null, 2)], {type: "application/json"}));
    const a = document.createElement("a");
    a.href = url; a.download = "eval-viewer-notes.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
element.querySelector("[data-note-import]").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
        const incoming = validate(JSON.parse(await file.text()));
        // Validate every conflict before changing anything. Existing notes are never silently replaced.
        for (const e of incoming) {
            const old = entries.get(entryKey(e));
            if (old && JSON.stringify(old) !== JSON.stringify(e)) {
                throw new Error("試行IDが既存メモと衝突しました。既存メモを保持し、読み込みを中止しました");
            }
        }
        incoming.forEach(e => entries.set(entryKey(e), e));
        select();
        if (!storageError) persist();
        else status.textContent = "メモをメモリに読み込みました。JSONを書き出してください。";
    } catch (error) {
        status.textContent = `読み込みエラー: ${error.message}（既存メモを保持）`;
    } finally { event.target.value = ""; }
});
new MutationObserver(select).observe(document.getElementById("viewer-note-context"), {
    childList: true, subtree: true, characterData: true
});
select();
