"""Gradio UI and a narrowly scoped native media route (no transcoding or media cache).

HTML media bypasses Gradio Video/Image preprocessing entirely. Source files are never
handed to its conversion pipeline; only registered raster/video paths can be served.
See Gradio 6.9.0 HTML js_on_load and mount_gradio_app blocked_paths APIs:
https://www.gradio.app/docs/gradio/html
https://www.gradio.app/docs/gradio/mount_gradio_app
"""

from __future__ import annotations

import html
import json
import mimetypes
import secrets
import threading
from pathlib import Path

from geniac_baseline.viewer.reader import (
    MEDIA_SUFFIXES,
    STATES,
    VIDEO_SUFFIXES,
    Reader,
    call_detail,
    discover,
    filter_trials,
    load_detail,
)


class MediaRegistry:
    def __init__(self, reader: Reader):
        self.reader = reader
        self.paths: dict[str, Path] = {}
        self.tokens: dict[Path, str] = {}
        self.lock = threading.Lock()

    def register(self, path: Path) -> str:
        path = self.reader.check(path)
        if path.suffix.lower() not in MEDIA_SUFFIXES or not path.is_file():
            raise ValueError("配信対象の動画・画像ではありません")
        with self.lock:
            if path not in self.tokens:
                token = secrets.token_urlsafe(32)
                self.tokens[path] = token
                self.paths[token] = path
            return self.tokens[path]

    def resolve(self, token: str) -> Path:
        path = self.reader.check(self.paths[token])
        if not path.is_file():
            raise ValueError("メディアを読み取れません")
        return path


def create_app(reader: Reader, state_dir: Path):
    import gradio as gr
    from fastapi import FastAPI, HTTPException
    from starlette.responses import FileResponse

    assets = Path(__file__).parent
    registry = MediaRegistry(reader)
    runs, warnings = discover(reader)
    lock = threading.Lock()

    def find_run(key):
        return next((r for r in runs if r.key == key), None)

    def find_trial(run_key, trial_id):
        run = find_run(run_key)
        return next((t for t in run.trials if t.id == trial_id), None) if run else None

    with gr.Blocks(title="LIBERO評価結果ビューワー", analytics_enabled=False) as demo:
        gr.Markdown("# LIBERO評価結果ビューワー\n保存済み評価の動画・コード・応答・ログを照合します。")
        gr.Textbox(str(reader.root), label="探索ルート（起動引数で指定）", interactive=False)
        refresh = gr.Button("実行一覧を再読み込み")
        run_list = gr.JSON([r.stats() for r in runs], label="評価実行一覧")
        discovery_warning = gr.Textbox("\n".join(warnings), label="探索時の警告", interactive=False)
        run_select = gr.Dropdown([r.key for r in runs], label="評価実行", elem_id="viewer-run")
        stats = gr.JSON(label="選択した実行の集計")
        warning = gr.Textbox(label="実行の警告", interactive=False)
        with gr.Row():
            model = gr.Dropdown(["すべて"], value="すべて", label="モデル")
            task = gr.Dropdown(["すべて"], value="すべて", label="タスク")
            seed = gr.Textbox(label="seed（完全一致）")
            status = gr.Dropdown(["すべて", *STATES], value="すべて", label="状態")
            error = gr.Dropdown(["すべて", "あり", "なし", "不明"], value="すべて", label="実行エラー")
        apply = gr.Button("試行を絞り込む")
        table = gr.Dataframe(
            headers=["モデル", "タスク", "seed", "状態", "実行エラー", "part", "重複", "試行ID"],
            datatype=["str", "str", "number", "str", "str", "str", "str", "str"],
            interactive=False,
            label="試行一覧（下の試行選択で詳細を開く）",
        )
        trial_select = gr.Dropdown(label="試行", elem_id="viewer-trial")
        metadata = gr.JSON(label="試行の識別情報・元パス")
        save_select = gr.Dropdown(label="保存フォルダ（曖昧な最終保存・途中保存も個別に確認）", elem_id="viewer-save")
        detail_warning = gr.Textbox(label="詳細の警告", interactive=False)
        with gr.Row():
            with gr.Column():
                media_select = gr.Dropdown(label="動画・画像（ファイル名で選択）", elem_id="viewer-media")
                media = gr.HTML(
                    value="メディアを選択してください",
                    js_on_load="""
                    function bind() {
                        const v = element.querySelector('video');
                        if (v && !v.dataset.bound) {
                          v.dataset.bound = 'true';
                          v.addEventListener('error', () => {
                            element.querySelector('[data-media-status]').textContent =
                                '再生できません。codec非対応・破損・読み取りエラーの可能性があります。';
                          });
                        }
                    }
                    new MutationObserver(bind).observe(element, {childList: true, subtree: true});
                    bind();
                """,
                )
            with gr.Column():
                call_select = gr.Dropdown(label="モデル呼び出し", elem_id="viewer-call")
                turn_code = gr.Code(label="この呼び出しで生成したコード", language="python", interactive=False)
                response = gr.Textbox(label="保存されたモデル応答（全フィールド）", lines=8, interactive=False)
                prompt = gr.Textbox(
                    label="保存プロンプト（FINISH等の未保存は再構成しません）", lines=8, interactive=False
                )
                feedback = gr.Textbox(label="この判断の前のコード実行結果（保存プロンプト由来）", interactive=False)
                vdm = gr.Textbox(label="保存プロンプト内のVDM観察（判断の前の実行）", interactive=False)
        observation_select = gr.Dropdown(label="VDMファイル（呼び出しとの対応は未確定）")
        observation = gr.Textbox(label="VDMファイル原文", lines=5, interactive=False)
        final_code = gr.Code(label="最終コード code.py（選択した保存時点）", language="python", interactive=False)
        summary = gr.Textbox(
            label="summary.txt 原文（最終・集約情報。ターン別ではありません）", lines=8, interactive=False
        )
        final_feedback = gr.Textbox(label="最終Stdout・集約Stderr", lines=5, interactive=False)
        segment_log = gr.Textbox(label="segmentログ原文（trial専用ログではありません）", lines=8, interactive=False)
        note_context = gr.HTML(value="", elem_id="viewer-note-context", js_on_load="")
        gr.HTML(
            value=None,
            html_template=(assets / "notes.html").read_text(),
            js_on_load=(assets / "notes.js").read_text(),
            elem_id="viewer-notes",
        )

        def reload_runs():
            nonlocal runs
            new_runs, new_warnings = discover(reader)
            with lock:
                runs = new_runs
            return (
                [r.stats() for r in runs],
                "\n".join(new_warnings),
                gr.Dropdown(choices=[r.key for r in runs], value=None),
            )

        def choose_run(key):
            run = find_run(key)
            models = sorted({t.model for t in run.trials}) if run else []
            tasks = sorted({t.task for t in run.trials}) if run else []
            return (
                run.stats() if run else None,
                "\n".join(run.warnings) if run else "",
                gr.Dropdown(choices=["すべて", *models], value="すべて"),
                gr.Dropdown(choices=["すべて", *tasks], value="すべて"),
                [],
                gr.Dropdown(choices=[], value=None),
            )

        def filtered(key, m, t, s, st, er):
            run = find_run(key)
            trials = filter_trials(run, m, t, s, st, er) if run else []
            rows = [
                [t.model, t.task, t.seed, t.state, t.error, t.part, "あり" if t.duplicate else "", t.id] for t in trials
            ]
            return rows, gr.Dropdown(choices=[(t.label, t.id) for t in trials], value=None)

        def choose_trial(key, trial_id):
            t = find_trial(key, trial_id)
            if not t:
                return None, gr.Dropdown(choices=[], value=None), None
            meta = {
                "モデル": t.model,
                "タスク": t.task,
                "seed": t.seed,
                "状態": t.state,
                "重複": t.duplicate,
                "part": t.part or "分割なし",
                "試行ID": t.id,
                "保存": [
                    {
                        "元パス": str(s.path),
                        "segment": s.segment,
                        "最終保存": s.final,
                        "報酬": s.name.reward,
                        "sandbox終了コード": s.name.sandbox_rc,
                    }
                    for s in t.saves
                ],
            }
            selected = str(t.selected.path) if t.selected else None
            return (
                meta,
                gr.Dropdown(choices=[str(s.path) for s in t.saves], value=selected),
                '<span data-note-context style="display:none">'
                + html.escape(json.dumps({"evaluation_run": t.run, "trial_id": t.id}))
                + "</span>",
            )

        def get_detail(key, trial_id, path):
            t = find_trial(key, trial_id)
            save = next((s for s in t.saves if str(s.path) == path), None) if t else None
            if not save:
                return None
            try:
                return load_detail(reader, save)
            except (OSError, ValueError) as e:
                return {
                    "code": "",
                    "summary": "",
                    "feedback": None,
                    "calls": [],
                    "media": [],
                    "observations": {},
                    "warnings": [f"{path}: {e}"],
                }

        def choose_save(key, trial_id, path):
            d = get_detail(key, trial_id, path)
            if not d:
                return (
                    "",
                    "",
                    "",
                    "",
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    "",
                    gr.Dropdown(choices=[], value=None),
                )
            t = find_trial(key, trial_id)
            save = next(s for s in t.saves if str(s.path) == path)
            logs = "未保存 / segmentなし"
            if save.segment:
                # collect/plan.py writes seg_<task>_<seed>/<task>.log.
                base = find_run(key).path / t.part / save.segment
                logs = reader.optional_text(base / f"{t.task}.log", d["warnings"])
            calls = [(f"呼び出し {i}: {c.get('decision', '不明')}", str(i)) for i, c in enumerate(d["calls"])]
            fb = d["feedback"]
            return (
                d["code"],
                d["summary"],
                f"Stdout:\n{fb[0]}\nStderr:\n{fb[1]}" if fb else "最終出力の保存なし",
                logs,
                gr.Dropdown(choices=calls, value="0" if calls else None),
                gr.Dropdown(choices=[(p.relative_to(save.path).as_posix(), str(p)) for p in d["media"]], value=None),
                "\n".join(d["warnings"]),
                gr.Dropdown(choices=list(d["observations"]), value=None),
            )

        def choose_call(key, trial_id, path, number):
            d = get_detail(key, trial_id, path)
            if d and number is not None and str(number).isdigit() and int(number) < len(d["calls"]):
                return call_detail(d["calls"][int(number)])
            return ("",) * 5

        def choose_media(key, trial_id, path, selected):
            d = get_detail(key, trial_id, path)
            if not d or selected not in [str(p) for p in d["media"]]:
                return "メディアなし / 未選択"
            p = Path(selected)
            try:
                token = registry.register(p)
            except (OSError, ValueError) as e:
                return html.escape(str(e))
            url = f"/viewer-media/{token}"
            name = html.escape(p.name)
            if p.suffix.lower() in VIDEO_SUFFIXES:
                return (
                    f"<p>{name}（codecは未確認。保存動画を直接再生）</p>"
                    f'<video controls preload="metadata" src="{url}" style="width:100%"></video>'
                    "<p data-media-status></p>"
                )
            return f'<p>{name}</p><img src="{url}" alt="{name}" style="max-width:100%">'

        def choose_observation(key, trial_id, path, name):
            d = get_detail(key, trial_id, path)
            return d["observations"].get(name, "") if d else ""

        refresh.click(reload_runs, outputs=[run_list, discovery_warning, run_select])
        run_select.change(choose_run, run_select, [stats, warning, model, task, table, trial_select])
        apply.click(filtered, [run_select, model, task, seed, status, error], [table, trial_select])
        trial_select.change(choose_trial, [run_select, trial_select], [metadata, save_select, note_context])
        save_inputs = [run_select, trial_select, save_select]
        save_select.change(
            choose_save,
            save_inputs,
            [
                final_code,
                summary,
                final_feedback,
                segment_log,
                call_select,
                media_select,
                detail_warning,
                observation_select,
            ],
        )
        call_select.change(choose_call, [*save_inputs, call_select], [turn_code, response, prompt, feedback, vdm])
        media_select.change(choose_media, [*save_inputs, media_select], media)
        observation_select.change(choose_observation, [*save_inputs, observation_select], observation)

    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/viewer-media/{token}")
    def get_media(token: str):
        try:
            path = registry.resolve(token)
            return FileResponse(
                path,
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                headers={"X-Content-Type-Options": "nosniff"},
            )
        except (KeyError, OSError, ValueError):
            raise HTTPException(404, "メディアがありません") from None

    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        server_name="127.0.0.1",
        allowed_paths=[],
        blocked_paths=[str(reader.root), str(state_dir), str(Path(__file__).resolve().parents[2])],
        ssr_mode=False,
        enable_monitoring=False,
    )
    return app
