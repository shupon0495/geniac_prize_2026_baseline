from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from geniac_baseline.viewer.reader import (
    BoundaryError,
    Reader,
    call_detail,
    discover,
    filter_trials,
    load_detail,
    separate_state,
)

TASK = "libero_goal_example"


def make_save(base, seed=441, *, model="model", rc=0, done=1, final=True, segment="seg_example"):
    folder = (
        base / segment / "outputs" / model / TASK / f"trial_{seed}_sandboxrc_{rc}_reward_1.000_taskcompleted_{done}"
    )
    folder.mkdir(parents=True)
    (folder / "summary.txt").write_text("program\n  Stdout: ok\n  Stderr: \n  Reward: 1" if final else "program")
    (folder / "code.py").write_text("raise RuntimeError('must never execute')")
    (folder / "all_responses.json").write_text(
        json.dumps(
            [
                {
                    "decision": "initial",
                    "code_blocks": ["print('hello')"],
                    "initial_prompt": [{"role": "user", "content": "task"}],
                },
                {"decision": "finish", "reasoning": "FINISH <script>window.UNSAFE=true</script>"},
            ]
        )
    )
    return folder


def plan(base, seeds, *, model="model"):
    base.mkdir(parents=True, exist_ok=True)
    (base / "plan.json").write_text(json.dumps({"model": model, "tasks": [{"task": TASK, "seeds": seeds}]}))


def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in [root, *root.rglob("*")]
    }


def test_unsplit_partial_final_missing_models_and_lazy_reads(tmp_path, monkeypatch):
    root = tmp_path / "results"
    run_path = root / "run"
    make_save(run_path, final=False, rc=1, done=0)
    final = make_save(run_path)
    make_save(run_path, 442, final=False)
    make_save(run_path, 443, model="other")
    plan(run_path, ["441-444"])
    reader = Reader(root)
    original = reader.text

    def metadata_only(path):
        assert path.name in {"summary.txt", "plan.json"}
        return original(path)

    monkeypatch.setattr(reader, "text", metadata_only)
    runs, warnings = discover(reader)
    assert not warnings
    run = runs[0]
    assert run.key == "run"
    assert len(run.trials) == 5
    success = filter_trials(run, model="model", seed="441")[0]
    assert success.state == "成功" and success.selected.path == final
    assert len(success.saves) == 2
    assert filter_trials(run, seed="442")[0].state == "途中終了"
    assert filter_trials(run, model="model", seed="443")[0].state == "結果欠損"
    assert filter_trials(run, model="other")[0].state == "成功"
    assert not any(t.duplicate for t in run.trials)
    assert filter_trials(run, state="成功", error="なし")


def test_parts_duplicates_ambiguous_and_no_plan(tmp_path):
    root = tmp_path / "results"
    a = root / "run" / "part1-of-2"
    b = root / "run" / "part2-of-2"
    make_save(a)
    make_save(b)
    make_save(a, 442, done=0, rc=1)
    make_save(a, 442, done=1)
    make_save(b, 443, final=False, done=0)
    runs, _ = discover(Reader(root))
    assert len(runs) == 1
    run = runs[0]
    assert run.parts == ["part1-of-2", "part2-of-2"]
    assert len(run.trials) == 4
    assert len({t.id for t in run.trials}) == 4
    assert len(filter_trials(run, seed="441")) == 2
    assert all(t.duplicate for t in filter_trials(run, seed="441"))
    ambiguous = filter_trials(run, seed="442")[0]
    assert ambiguous.state == "判定不能" and ambiguous.selected is None
    stats = run.stats()
    assert stats["成功"] == 0
    assert stats["重複キー数"] == 1
    assert stats["試行数（重複を除外）"] == 2
    assert any("計画なし" in w for w in run.warnings)


def test_plain_outputs_and_same_part_segments(tmp_path):
    root = tmp_path / "results"
    make_save(root / "run", segment="", final=False, rc=1)
    make_save(root / "run", segment="seg_next")
    run = discover(Reader(root))[0][0]
    assert len(run.trials) == 1
    assert run.trials[0].state == "成功"
    assert not run.trials[0].duplicate


def test_parent_plan_covers_parts_and_invalid_plan_is_atomic(tmp_path):
    root = tmp_path / "results"
    make_save(root / "run" / "part1-of-2")
    make_save(root / "run" / "part2-of-2", 442)
    plan(root / "run", ["441-443"])
    run = discover(Reader(root))[0][0]
    assert len(run.trials) == 3
    assert not any(t.duplicate for t in run.trials)
    assert filter_trials(run, seed="443")[0].state == "結果欠損"
    plan(root / "run" / "part1-of-2", ["441-441", "443-443"])
    run = discover(Reader(root))[0][0]
    assert len(run.trials) == 3 and not any(t.duplicate for t in run.trials)
    (root / "run" / "part1-of-2" / "plan.json").unlink()
    plan(root / "run", ["441-450", "bad"])
    run = discover(Reader(root))[0][0]
    assert len(run.trials) == 2  # A partly parsed broken plan must not invent missing trials.
    assert any("plan.json" in w for w in run.warnings)


def test_boundary_symlinks_and_state_rejected_before_write(tmp_path):
    root = tmp_path / "results"
    folder = make_save(root / "good")
    outside = tmp_path / "private"
    make_save(outside / "not_visible")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (folder / "leak.mp4").symlink_to(outside / "secret.mp4")
    (folder / "summary.txt").unlink()
    (folder / "summary.txt").symlink_to(outside / "secret.txt")
    reader = Reader(root)
    runs, warnings = discover(reader)
    assert [r.key for r in runs] == ["good"]
    assert warnings and runs[0].trials[0].state == "判定不能"
    assert not load_detail(reader, runs[0].trials[0].saves[0])["media"]
    with pytest.raises(BoundaryError):
        reader.text(outside / "secret.txt")
    for state in (root, root / "new", tmp_path):
        with pytest.raises(BoundaryError):
            separate_state(root, state)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(BoundaryError):
        separate_state(root, alias / "new")
    assert not (root / "new").exists()


def test_readonly_integrity_details_malformed_json_and_permissions(tmp_path, monkeypatch):
    root = tmp_path / "results"
    folder = make_save(root / "good")
    other = make_save(root / "broken")
    (folder / "code.py").write_text("")  # Evaluation success does not require converter's nonempty code.
    (other / "all_responses.json").write_text("{broken")
    plan(root / "broken", ["441-442"])
    (root / "broken" / "plan.json").write_text("null")
    for p in root.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)
    before = fingerprint(root)
    reader = Reader(root)
    runs, _ = discover(reader)
    good = next(r for r in runs if r.key == "good")
    broken = next(r for r in runs if r.key == "broken")
    assert good.trials[0].state == "成功"
    assert load_detail(reader, good.trials[0].selected)["code"] == ""
    assert load_detail(reader, broken.trials[0].selected)["warnings"]
    assert broken.warnings
    assert before == fingerprint(root)
    original = reader.text

    def denied(path):
        if path == folder / "summary.txt":
            raise PermissionError("test denied")
        return original(path)

    monkeypatch.setattr(reader, "text", denied)
    again, _ = discover(reader)
    assert next(r for r in again if r.key == "good").trials[0].state == "判定不能"
    assert next(r for r in again if r.key == "broken").trials[0].state == "成功"
    for p in [root, *root.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)


def test_stable_id_relocation_and_response_evidence(tmp_path):
    root = tmp_path / "original"
    folder = make_save(root / "run")
    (folder / "all_responses.json").write_text('[{"decision":"finish","reasoning":"FINISH"}] trailing')
    original = discover(Reader(root))[0][0].trials[0]
    standalone = discover(Reader(root / "run"))[0][0].trials[0]
    assert standalone.id == original.id
    other = tmp_path / "relocated"
    shutil.copytree(root, other)
    relocated = discover(Reader(other))[0][0].trials[0]
    assert original.id == relocated.id
    detail = load_detail(Reader(root), original.selected)
    assert detail["calls"][0]["decision"] == "finish"
    assert any("末尾データ" in w for w in detail["warnings"])
    assert call_detail(detail["calls"][0])[2] == "未保存"
    prompt = (
        "The following code blocks were just executed:\ncode\nThe current console stdout is:\n```\nhello"
        "\n```\nThe current console stderr is:\n```\nerror\n```\nIncluded below is the observed differences\nVDM"
    )
    fields = call_detail({"decision": "regenerate", "multi_turn_prompt": [{"role": "user", "content": prompt}]})
    assert "hello" in fields[3] and "error" in fields[3]
    assert "VDM" in fields[4]
