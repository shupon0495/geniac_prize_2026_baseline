"""Bounded, read-only discovery and lazy detail loading of saved CaP-X runs.

CaP-X references were checked in capx/envs/trial.py (the available checkout):
753-759 initial response, 833-859 regenerate/finish, 869-915 partial/final saves,
155-207 video names. VDM history is conditional, not indexed by model call:
patches/geniac_prize/geniac-prize.patch, trial.py hunks at 700 and launch_utils.py at 360.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geniac_baseline.collect.preflight import PreflightError, parse_segment
from geniac_baseline.data.prompts import split_saved, summary_feedback, text_of
from geniac_baseline.data.trials import TrialName, is_final_save, parse_trial_dir_name

STATES = ("成功", "失敗", "途中終了", "結果欠損", "判定不能")
MEDIA_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
PART = re.compile(r"part\d+-of-\d+$")
MAX_TEXT = 32 * 1024 * 1024


class BoundaryError(ValueError):
    """A requested read or state location crosses the data boundary."""


def separate_state(root: Path, state: Path) -> tuple[Path, Path]:
    root, state = root.expanduser().resolve(), state.expanduser().resolve()
    if root == state or root in state.parents or state in root.parents:
        raise BoundaryError("状態領域と探索ルートは互いに包含しない場所を指定してください")
    if not root.is_dir():
        raise BoundaryError(f"探索ルートがディレクトリではありません: {root}")
    return root, state


class Reader:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise BoundaryError("探索ルートがディレクトリではありません")

    def check(self, path: Path) -> Path:
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.root):
            raise BoundaryError(f"探索ルート外: {path}")
        # Reject all symlinks, including in-root aliases, to avoid cycles and duplicate discovery.
        for p in (path, *path.parents):
            if p == self.root:
                break
            if p.is_symlink():
                raise BoundaryError(f"シンボリックリンクは読みません: {p}")
        if not path.resolve().is_relative_to(self.root):
            raise BoundaryError(f"探索ルート外: {path}")
        return path

    def children(self, path: Path, warnings: list[str]) -> list[Path]:
        try:
            return sorted(self.check(path).iterdir())
        except (OSError, ValueError) as e:
            warnings.append(f"{path}: {e}")
            return []

    def directory(self, path: Path, warnings: list[str]) -> bool:
        try:
            return self.check(path).is_dir()
        except (OSError, ValueError) as e:
            warnings.append(f"{path}: {e}")
            return False

    def text(self, path: Path) -> str:
        path = self.check(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise ValueError(f"通常のファイルではありません: {path}")
            data = f.read(MAX_TEXT + 1)
        if len(data) > MAX_TEXT:
            raise ValueError(f"テキスト上限 {MAX_TEXT} bytes を超えています: {path}")
        return data.decode("utf-8", errors="replace")

    def optional_text(self, path: Path, warnings: list[str]) -> str:
        try:
            return self.text(path)
        except (OSError, ValueError) as e:
            warnings.append(f"{path}: {e}")
            return "未保存 / 読み取り不可"


@dataclass
class Save:
    path: Path
    segment: str
    name: TrialName
    final: bool | None
    error: bool | None


@dataclass
class Trial:
    run: str
    part: str
    model: str
    task: str
    seed: int
    saves: list[Save] = field(default_factory=list)
    state: str = "結果欠損"
    duplicate: bool = False

    @property
    def id(self) -> str:
        # Independent of mount point and changing result-folder suffix; segments disambiguate provenance.
        identity = [self.run, self.part, self.model, self.task, self.seed, sorted({s.segment for s in self.saves})]
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()

    @property
    def selected(self) -> Save | None:
        finals = [s for s in self.saves if s.final]
        return finals[0] if len(finals) == 1 else (self.saves[0] if len(self.saves) == 1 else None)

    @property
    def error(self) -> str:
        selected = self.selected
        if selected is None or selected.error is None:
            return "不明"
        return "あり" if selected.error else "なし"

    @property
    def label(self) -> str:
        return f"{self.model} / {self.task} / seed {self.seed} / {self.part or '分割なし'} / {self.state}"


@dataclass
class Run:
    key: str
    path: Path
    trials: list[Trial] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        # Exclude ALL copies of duplicated keys: neither folder order nor outcome chooses a winner.
        unique = [t for t in self.trials if not t.duplicate]
        counts = {state: sum(t.state == state for t in unique) for state in STATES}
        duplicate_keys = {(t.model, t.task, t.seed) for t in self.trials if t.duplicate}
        return {
            "実行": self.key,
            "モデル": sorted({t.model for t in self.trials}),
            "part": self.parts,
            "タスク数": len({t.task for t in self.trials}),
            "試行数（重複を除外）": len(unique),
            **counts,
            "重複キー数": len(duplicate_keys),
            "重複コピー数": sum(t.duplicate for t in self.trials),
            "成功率": f"{counts['成功']} / {len(unique)}" if unique else "算出不可",
            "分母": "重複キー全体を除外。途中終了・結果欠損・判定不能を含む。計画なしでは未保存seedを推定しない。",
        }


def discover(reader: Reader) -> tuple[list[Run], list[str]]:
    """Find structural run roots without opening responses, code, logs or media."""
    runs: list[Run] = []
    warnings: list[str] = []

    def walk(path: Path) -> None:
        children = reader.children(path, warnings)
        dirs = [p for p in children if not p.name.startswith(".") and reader.directory(p, warnings)]
        if any(p.name == "outputs" or p.name.startswith("seg_") or PART.fullmatch(p.name) for p in dirs) or any(
            p.name == "plan.json" for p in children
        ):
            runs.append(read_run(reader, path))
            return
        for p in dirs:
            walk(p)

    walk(reader.root)
    return runs, warnings


def read_run(reader: Reader, path: Path) -> Run:
    reader.check(path)
    relative = path.relative_to(reader.root).as_posix()
    run = Run(path.name if relative == "." else relative, path)
    children = reader.children(path, run.warnings)
    parts = [p for p in children if PART.fullmatch(p.name) and reader.directory(p, run.warnings)]
    run.parts = [p.name for p in parts]
    # Also support unsplit outputs alongside parts without discarding either source.
    locations = [path, *parts]
    groups: dict[tuple[str, str, str, int], Trial] = {}
    expected: set[tuple[str, str, str, int]] = set()
    for location in locations:
        part = "" if location == path else location.name
        entries = reader.children(location, run.warnings)
        segments = [p for p in entries if p.name.startswith("seg_") and reader.directory(p, run.warnings)]
        for segment in [location, *segments]:
            output = segment / "outputs"
            if not reader.directory(output, run.warnings):
                continue
            for model in reader.children(output, run.warnings):
                if not reader.directory(model, run.warnings):
                    continue
                for task in reader.children(model, run.warnings):
                    if not task.name.startswith("libero_") or not reader.directory(task, run.warnings):
                        continue
                    for folder in reader.children(task, run.warnings):
                        if not reader.directory(folder, run.warnings):
                            continue
                        try:
                            name = parse_trial_dir_name(folder.name)
                        except ValueError:
                            name = None
                        if name is None:
                            run.warnings.append(f"{folder}: 不明なtrial名")
                            continue
                        key = part, model.name, task.name, name.trial
                        trial = groups.setdefault(key, Trial(run.key, *key))
                        final, error = None, None
                        try:
                            # Guard the file BEFORE calling the existing final-save predicate.
                            summary = reader.text(folder / "summary.txt")
                            final = is_final_save(folder)
                            feedback = summary_feedback(summary)
                            error = name.sandbox_rc != 0 or bool(feedback and feedback[1].strip())
                        except (OSError, ValueError) as e:
                            run.warnings.append(f"{folder / 'summary.txt'}: {e}")
                        trial.saves.append(
                            Save(folder, "" if segment == location else segment.name, name, final, error)
                        )
        plan = location / "plan.json"
        if plan in entries:
            try:
                raw = json.loads(reader.text(plan))
                # Plan model is the served LoRA name when an adapter was used (collect/plan.py).
                model = raw.get("lora_name") if raw.get("adapter") else raw.get("model")
                if not isinstance(model, str) or not model:
                    raise ValueError("計画のモデルが不明なため欠損seedをモデルに割り当てられません")
                model = model.replace("/", "_")
                pending = []
                for task in raw["tasks"]:
                    if not isinstance(task["task"], str) or not task["task"].startswith("libero_"):
                        continue
                    for seed_range in task["seeds"]:
                        a, b = parse_segment(str(seed_range))
                        if b - a > 100_000:
                            raise ValueError("計画のseed区間が大きすぎます")
                        pending.extend((part, model, task["task"], seed) for seed in range(a, b + 1))
                expected.update(pending)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, PreflightError) as e:
                run.warnings.append(f"{plan}: {e}")
        elif location != path or not parts:
            run.warnings.append(f"{plan}: 計画なし。未保存seed数は不明")

    for key in sorted(expected, key=lambda k: (not bool(k[0]), k)):
        part, model, task, seed = key
        # A parent-level plan covers the entire split run, rather than an extra unsplit copy.
        if not part and parts and any(k[1:] == (model, task, seed) for k in groups):
            continue
        groups.setdefault(key, Trial(run.key, *key))

    for trial in groups.values():
        if not trial.saves:
            continue
        finals = [s for s in trial.saves if s.final]
        if len(finals) > 1 or any(s.final is None for s in trial.saves):
            trial.state = "判定不能"
            run.warnings.append(f"{trial.label}: 最終保存が複数、またはsummaryを確認できません")
        elif not finals:
            trial.state = "途中終了"
        else:
            name = finals[0].name
            trial.state = "成功" if name.task_completed and name.sandbox_rc == 0 else "失敗"
    by_key: dict[tuple[str, str, int], list[Trial]] = collections.defaultdict(list)
    for t in groups.values():
        by_key[t.model, t.task, t.seed].append(t)
    for key, copies in by_key.items():
        if len(copies) > 1:
            for t in copies:
                t.duplicate = True
            run.warnings.append(f"重複 {key}: {[t.part for t in copies]}。合計から全コピーを除外")
    run.trials = sorted(groups.values(), key=lambda t: (t.model, t.task, t.seed, t.part))
    return run


def filter_trials(
    run: Run, model: str = "すべて", task: str = "すべて", seed: str = "", state: str = "すべて", error: str = "すべて"
) -> list[Trial]:
    return [
        t
        for t in run.trials
        if (model == "すべて" or t.model == model)
        and (task == "すべて" or t.task == task)
        and (not seed.strip() or str(t.seed) == seed.strip())
        and (state == "すべて" or t.state == state)
        and (error == "すべて" or t.error == error)
    ]


def load_detail(reader: Reader, save: Save) -> dict[str, Any]:
    warnings: list[str] = []
    folder = reader.check(save.path)
    code = reader.optional_text(folder / "code.py", warnings)
    summary = reader.optional_text(folder / "summary.txt", warnings)
    calls = []
    try:
        raw = reader.text(folder / "all_responses.json").lstrip()
        data, end = json.JSONDecoder().raw_decode(raw)
        if raw[end:].strip():
            warnings.append(f"{folder / 'all_responses.json'}: 末尾データあり。既存採点と同じく先頭JSONを表示")
        if not isinstance(data, list) or any(not isinstance(c, dict) for c in data):
            raise ValueError("モデル呼び出しの配列ではありません")
        calls = data
    except (OSError, ValueError) as e:
        warnings.append(f"{folder / 'all_responses.json'}: {e}")
    media: list[Path] = []
    observations: dict[str, str] = {}

    def walk(path: Path) -> None:
        for p in reader.children(path, warnings):
            try:
                reader.check(p)
                if p.is_dir():
                    walk(p)
                elif p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES:
                    media.append(p)
                elif p.is_file() and p.match("differencing_feedback_*.txt"):
                    observations[p.relative_to(folder).as_posix()] = reader.optional_text(p, warnings)
            except (OSError, ValueError) as e:
                warnings.append(f"{p}: {e}")

    walk(folder)
    return {
        "code": code,
        "summary": summary,
        "feedback": summary_feedback(summary),
        "calls": calls,
        "media": media,
        "observations": observations,
        "warnings": warnings,
    }


def call_detail(call: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Display saved evidence only; a decision prompt describes the PRECEDING execution.

    capx/envs/trial.py:541-546 formats preceding code/stdout/stderr; 833-859 stores
    the regenerate prompt but not the finish prompt. VDM files can omit null feedback,
    so no file-number to model-call-number correspondence is asserted.
    """
    prompt = call.get("initial_prompt") or call.get("multi_turn_prompt")
    prompt_text = json.dumps(prompt, ensure_ascii=False, indent=2) if prompt else "未保存"
    stdout, stderr = "対応する保存プロンプトなし", "対応する保存プロンプトなし"
    vdm = "保存プロンプトで対応を確認できません。下のVDMファイルを個別に確認してください。"
    if isinstance(prompt, list):
        try:
            users = [text_of(m["content"]) for m in prompt if m.get("role") == "user"]
            saved = split_saved(users[-1])
            stdout, stderr = saved["stdout"], saved["stderr"]
            vdm = saved["vdm"] or "このプロンプトにVDM観察はありません"
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            pass
    # JSON preserves every saved field; no claim is made that raw response text was saved in full.
    blocks = call.get("code_blocks")
    code = "\n\n".join(blocks) if isinstance(blocks, list) and all(isinstance(b, str) for b in blocks) else "未保存"
    return (
        code,
        json.dumps(call, ensure_ascii=False, indent=2),
        prompt_text,
        f"Stdout:\n{stdout}\nStderr:\n{stderr}",
        vdm,
    )
