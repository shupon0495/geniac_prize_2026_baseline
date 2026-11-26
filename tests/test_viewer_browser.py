"""Optional real-browser acceptance checks: install .[viewer] and playwright to run.

VIEWER_TEST_BROWSER can select an existing Chromium executable. Browser dependencies
are intentionally not part of the core/dev environment. Only synthetic, temporary data is used.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from tests.test_viewer_reader import fingerprint, make_save

pytest.importorskip("gradio")
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def viewer_server(tmp_path):
    root = tmp_path / "results"
    save = make_save(root / "run")
    make_save(root / "run", 442)
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    # libx264 fixture, never a simulator or generated trial code.
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(save / "video_combined.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    for p in [root, *root.rglob("*")]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    before = fingerprint(root)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = (tmp_path / "server.log").open("w+")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "geniac_baseline.viewer",
            "--root",
            str(root),
            "--state-dir",
            str(tmp_path / "state"),
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=log,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"},
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(log.read())
            try:
                with urlopen(url, timeout=1):
                    break
            except (URLError, TimeoutError):
                time.sleep(0.1)
        else:
            pytest.fail("viewer did not become ready")
        yield url, root, log, tmp_path
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()
        assert before == fingerprint(root)
        for p in [root, *root.rglob("*")]:
            p.chmod(0o755 if p.is_dir() else 0o644)


def dropdown(page, element_id, text):
    widget = page.locator(f"#{element_id}")
    widget.locator("input").click()
    page.get_by_role("option", name=text, exact=False).first.click()


def open_trial(page, seed="441"):
    dropdown(page, "viewer-run", "run")
    page.get_by_role("button", name="試行を絞り込む", exact=True).click()
    dropdown(page, "viewer-trial", f"seed {seed}")
    page.locator("[data-note-fields]:not([disabled])").wait_for()


def test_browser_media_notes_and_no_note_network(viewer_server):
    url, root, log, tmp_path = viewer_server
    with playwright.sync_playwright() as pw:
        executable = os.environ.get("VIEWER_TEST_BROWSER")
        try:
            browser = pw.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        except playwright.Error as e:
            pytest.skip(f"Chromium unavailable: {e}")
        context = browser.new_context()
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "js_on_load" in m.text else None)
        page.goto(url)
        open_trial(page)
        # Saved code and script-like model text are displayed without execution.
        dropdown(page, "viewer-call", "呼び出し 1: finish")
        assert page.evaluate("window.UNSAFE") is None
        dropdown(page, "viewer-media", "video_combined.mp4")
        video = page.locator("video")
        video.wait_for()
        page.wait_for_function("document.querySelector('video')?.readyState >= 2")
        video.evaluate("v => v.play()")
        page.wait_for_function("document.querySelector('video').currentTime > 0.1")
        assert video.evaluate("v => v.videoWidth") == 64
        video.evaluate("v => v.pause()")
        media_url = video.get_attribute("src")
        response = context.request.get(url + media_url, headers={"Range": "bytes=0-15"})
        assert response.status == 206 and len(response.body()) == 16
        assert context.request.get(url + "/viewer-media/not-registered").status == 404
        secret = next(root.rglob("code.py"))
        assert context.request.get(url + "/gradio_api/file=" + str(secret)).status in {403, 404}

        requests = []
        page.on("request", lambda r: requests.append((r.url, r.post_data or "")))
        marker = "PRIVATE_NOTE_BROWSER_ONLY_72319"
        textarea = page.locator("[data-note-text]")
        textarea.fill(marker)
        page.locator('input[type="checkbox"][value="把持失敗"]').check()
        assert "保存しました" in page.locator("[data-note-status]").inner_text()
        with page.expect_download() as download:
            page.get_by_role("button", name="メモJSONを書き出す").click()
        download_path = tmp_path / "notes.json"
        download.value.save_as(download_path)
        exported = json.loads(download_path.read_text())
        assert exported["schema_version"] == 1 and exported["entries"][0]["note"] == marker
        assert exported["entries"][0]["tags"] == ["把持失敗"]
        dropdown(page, "viewer-trial", "seed 442")
        page.wait_for_function("document.querySelector('[data-note-text]').value === ''")
        page.locator("[data-note-text]").fill("second private note")
        dropdown(page, "viewer-trial", "seed 441")
        page.wait_for_function("document.querySelector('[data-note-text]').value.includes('72319')")
        page.reload()
        open_trial(page)
        assert page.locator("[data-note-text]").input_value() == marker
        # Broken JSON and conflicts leave existing entries intact.
        upload = page.locator("[data-note-import]")
        upload.set_input_files({"name": "broken.json", "mimeType": "application/json", "buffer": b"{broken"})
        page.wait_for_function("document.querySelector('[data-note-status]').textContent.includes('エラー')")
        assert page.locator("[data-note-text]").input_value() == marker
        conflict = json.loads(download_path.read_text())
        conflict["entries"][0]["note"] = "conflicting text"
        upload.set_input_files(
            {"name": "conflict.json", "mimeType": "application/json", "buffer": json.dumps(conflict).encode()}
        )
        page.wait_for_function("document.querySelector('[data-note-status]').textContent.includes('衝突')")
        assert page.locator("[data-note-text]").input_value() == marker
        # Input, autosave, export and imports did not attach private content to ANY request.
        assert not any(marker in u + body or "把持失敗" in body or "conflicting text" in body for u, body in requests)
        log.flush()
        log.seek(0)
        assert marker not in log.read()
        # Import on a fresh browser context (different localStorage) restores JSON entirely locally.
        fresh = browser.new_context()
        other_page = fresh.new_page()
        imported_requests = []
        other_page.on("request", lambda r: imported_requests.append(r.post_data or ""))
        other_page.goto(url)
        open_trial(other_page)
        other_page.locator("[data-note-import]").set_input_files(str(download_path))
        other_page.wait_for_function("document.querySelector('[data-note-text]').value.includes('72319')")
        assert not any(marker in body for body in imported_requests)
        # Disabled/quota-exceeded storage must never be reported as saved.
        unavailable = browser.new_context()
        unavailable.add_init_script("Storage.prototype.setItem = function() { throw new Error('quota'); }")
        quota_page = unavailable.new_page()
        quota_page.goto(url)
        open_trial(quota_page)
        quota_page.locator("[data-note-text]").fill("in-memory fallback")
        assert "失敗" in quota_page.locator("[data-note-status]").inner_text()
        with quota_page.expect_download() as quota_download:
            quota_page.get_by_role("button", name="メモJSONを書き出す").click()
        quota_file = tmp_path / "quota-notes.json"
        quota_download.value.save_as(quota_file)
        assert json.loads(quota_file.read_text())["entries"][0]["note"] == "in-memory fallback"
        assert not errors
        # An occupied port fails without silently trying another port or creating state in the data.
        occupied = subprocess.run(
            [
                sys.executable,
                "-m",
                "geniac_baseline.viewer",
                "--root",
                str(root),
                "--state-dir",
                str(tmp_path / "occupied-state"),
                "--port",
                url.rsplit(":", 1)[1],
            ],
            capture_output=True,
            text=True,
        )
        assert occupied.returncode != 0
        assert not (tmp_path / "occupied-state").exists()
        browser.close()
