"""CPU-only local entry point. Configure writable paths before importing Gradio."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

from geniac_baseline.viewer.reader import Reader, separate_state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="保存済みLIBERO評価を読み取り専用で閲覧")
    parser.add_argument("--root", type=Path, required=True, help="評価結果の探索ルート")
    parser.add_argument("--state-dir", type=Path, required=True, help="自分の状態領域（結果配下は拒否）")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    root, state = separate_state(args.root, args.state_dir)
    if not 1 <= args.port <= 65535:
        parser.error("ポートは1〜65535を指定してください")
    # Fail before dependency setup/app discovery; uvicorn also fails if a later bind races us.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    state.mkdir(parents=True, exist_ok=True)
    for name, sub in {
        "GRADIO_TEMP_DIR": "gradio",
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "cache",
        "HF_HOME": "huggingface",
        "UV_CACHE_DIR": "uv-cache",
    }.items():
        target = state / sub
        if not target.resolve().is_relative_to(state):
            parser.error(f"状態領域内のリンクは使用できません: {target}")
        target.mkdir(exist_ok=True)
        os.environ[name] = str(target)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_ALLOWED_PATHS"] = ""
    os.environ["GRADIO_BLOCKED_PATHS"] = str(root)
    os.chdir(state)
    import uvicorn

    from geniac_baseline.viewer.app import create_app

    print(f"閲覧: http://127.0.0.1:{args.port}  探索ルート: {root}", flush=True)
    uvicorn.run(create_app(Reader(root), state), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
