# 保存済みLIBERO評価結果の閲覧

Gradio 6.9.0で保存済みCaP-X / LIBERO-PRO評価を閲覧する。モデル・シミュレータ・GPUは不要。
生成コードはテキストとして表示する。既存の採点・データ変換処理は変更しない。

## 手元またはCPUノードで起動

自分が書き込める場所に、このリポジトリのcheckoutと専用環境を用意する。
探索ルートと状態領域は互いに包含しない場所を指定する。

```bash
uv venv /path/to/own-viewer-venv --python 3.12
UV_CACHE_DIR=/path/to/own-uv-cache uv pip install \
  --python /path/to/own-viewer-venv/bin/python '.[viewer]'
/path/to/own-viewer-venv/bin/geniac-eval-viewer \
  --root /path/to/shared-results --state-dir /path/to/own-viewer-state --port 7860
```

ブラウザで `http://127.0.0.1:7860` を開く。別のホストで実行する場合はSSHで転送する。
`--root` は評価実行の親ディレクトリ、または評価実行そのものを指定できる。
探索ルートの変更はプロセスを再起動して行う。

## SlurmでGPUなしの起動

ログインノードで実行する。

```bash
bash jobs/start_eval_viewer.sh \
  --root /path/to/shared-results \
  --state-dir /path/to/own-viewer-state \
  --partition CPU_PARTITION --cpus 2 --memory 8G --time 04:00:00 --port 7860
```

partitionを省略した場合はクラスタの既定値を使う。CPU用partition名は利用環境で確認する。
既定ではGPUに関するSlurmオプションを付けない。必要な場合だけ `--gpus 1` などを指定する。
1ノード・1タスク・1プロセス・1ポートで動作する。
スクリプトにはBash、Python、uvとLinuxの `setsid`（util-linux）が必要。

同じ状態領域を同時に複数ジョブで初期構築しない。ジョブごとに別の状態領域を指定するか、事前構築する。
状態領域に `venv/`、`logs/`、`tmp/`、`cache/`、`uv-cache/`、`gradio/`、`huggingface/`、`runtime/` を作る。
既存の学習・評価用venvにはインストールしない。状態領域の子ディレクトリのシンボリックリンクは拒否する。

引数は環境変数 `VIEWER_ROOT`、`VIEWER_STATE_DIR`、`VIEWER_PARTITION`、`VIEWER_CPUS`、`VIEWER_MEMORY`、
`VIEWER_TIME`、`VIEWER_PORT`、`VIEWER_GPUS`、`VIEWER_PYTHON` でも指定できる。引数が優先される。
Pythonバージョンは `--python 3.12` などで指定できる。最初のパス検証には `python3` を使用する。
別名のインタープリタが必要なら `VIEWER_BOOTSTRAP_PYTHON` に実行可能ファイルを指定する。

計算ノードから依存を取得できない場合は、同じOS・アーキテクチャの、取得可能なノードで事前構築する。
checkoutと状態領域は計算ノードからも同じパスで見える場所に置く。

```bash
bash jobs/start_eval_viewer.sh --prepare-only --python 3.12 \
  --root /path/to/shared-results --state-dir /path/to/own-viewer-state
bash jobs/start_eval_viewer.sh --reuse-env --partition CPU_PARTITION \
  --root /path/to/shared-results --state-dir /path/to/own-viewer-state
```

`--reuse-env` はインストール・更新を行わない。コードを更新した後は再度 `--prepare-only` で環境を更新する。
投入時にジョブID・ポート・ログ・停止方法を表示する。workerログには割当ノード名とSSH転送例も出る。

```bash
squeue -h -j JOB_ID -o %N
tail -f /path/to/own-viewer-state/logs/eval-viewer-JOB_ID.out
scancel JOB_ID
```

ポートが使用中なら異常終了し、ログにエラーを残す。別ポートへの自動変更は行わない。
TERM・INT・ジョブキャンセル時はアプリや環境準備中の子プロセス群を終了して待ち合わせる。

## 手元PCへの転送

アプリは計算ノードの `127.0.0.1` で待ち受ける。Gradioの公開共有リンクは使用しない。
計算ノードにSSHで接続できる環境の例は次のとおり。自分のSSH設定とサイトの接続規則に合わせて置き換える。

```bash
ssh -N -J LOGIN_ALIAS \
  -L 127.0.0.1:7860:127.0.0.1:7860 COMPUTE_ALIAS
```

転送先は**計算ノード自身のlocalhost**である。ログインノードから計算ノードの外部IPへ転送するだけでは到達しない。
計算ノードへのSSHが許可されないサイトでは、そのサイトで認められた転送方法を確認する。
ここでSSH経路やホスト名を自動推定することはない。

## 調査の流れと集計

1. 「評価実行」を選択し、警告と集計を見る。
2. モデル・タスク・seed・状態・実行エラーを指定し、「試行を絞り込む」を押す。
3. 「試行」を選び、保存フォルダ・モデル呼び出し・動画や画像を選ぶ。
4. 原文のコード・応答・プロンプト・VDM・最終出力・segmentログを照合する。
5. 手元PCのブラウザでタグとメモを記録する。

`partN-of-M/` は親の1実行として表示する。分割なしの `seg_*/outputs/` と `outputs/` にも対応する。
同じpart・モデル・タスク・seedの保存フォルダを1試行にまとめ、既存の `is_final_save` で最終保存を判定する。
成功は最終保存の `taskcompleted_1` かつ `sandboxrc_0`。学習変換で要求するコードの非空条件は課さない。
最終保存がなければ「途中終了」、summaryの読み取りに失敗した場合や最終保存が複数なら「判定不能」と表示する。
途中終了の理由は推定しない。

`plan.json` の予定モデル・タスク・seedに保存がなければ「結果欠損」とする。
LoRAを使った計画では `lora_name` をモデル名として照合する。計画の欠損・不正やモデル不明は警告する。
計画なしでは未保存seedの数を推定しない。
part間の同じモデル・タスク・seedは警告し、各partを個別に選択できる。
**合計と成功率からは重複キーの全コピーを除外**する。分母は画面に示し、途中終了・欠損・判定不能を含む。
複数の最終保存は更新時刻で選ばず、保存フォルダを明示的に選ぶ。

`code.py` は選択した保存時点の最終プログラム。モデル呼び出しごとのコードとは別に表示する。
モデル応答は保存された全フィールドを表示する。保存形式によって生の回答全文が残っているとは限らない。
FINISHの未保存プロンプトは「未保存」とする。
判断プロンプトに含まれるStdout / Stderr / VDMは、その判断**より前**のコード実行の情報として表示する。
summaryのStdoutは最後のブロック、Stderrは集約情報で、ターン別出力ではない。
VDMの履歴は条件付きで追加されるため、ファイル番号をモデル呼び出し番号へ自動対応させない。

利用可能なCaP-X checkoutでは `capx/envs/trial.py:155-207` に
`video_combined.mp4`、`video_turn_00.mp4`、`*_wrist.mp4` などの保存を確認した。
同ファイル `780-791` の動画turnはコードブロックの実行順であり、モデル呼び出し番号との同期は行わない。
`capx/utils/video_utils.py:38-49` ではH.264保存を確認したが、実共有データのcodecは未確認。
未知の名前もファイル名で選べる。画像はPNG・JPEG・GIF・WebP、動画はMP4・WebM・MOV・AVI・MKVを探索する。
ブラウザが非対応のcodec、破損、読み取りエラーでは再生エラーを表示する。自動変換は行わない。

## 手元PCのメモ保存とJSON移行

「分類タグ」「自由記述」はネイティブのブラウザ入力で、GradioのPythonコールバックと接続していない。
本文とタグはサーバーへ送らず、手元PCのlocalStorageに自動保存する。
試行IDは実行の相対位置・part・モデル・タスク・seed・segmentを基にし、マウント先や結果フォルダの接尾辞に依存しない。
実行の相対名やsegmentを変更すると別の識別子になる。

「メモJSONを書き出す」でブラウザ内の全メモを保存する。
別ポート・別URL・別ブラウザへ移る場合は、このJSONを移行先の「メモJSONを読み込む」で読み込む。
JSONにはスキーマバージョン、評価実行、試行ID、タグ、本文、更新日時が含まれる。
File APIで読み込み、サーバーへのアップロードは行わない。
破損JSONやID衝突は読み込み全体を中止し、既存メモを保持する。
衝突する場合は移行元と移行先のJSONをそれぞれ保存して内容を確認する。
保存が無効・容量超過の場合は画面にエラーを表示する。ページを閉じる前にJSONを書き出す。

## 読み取り境界と検証

結果配下にはファイル作成・更新・削除・chmodを行わない。集計と索引はメモリのみ。
外側を含めシンボリックリンクは探索しない。大きなテキスト（32 MiB超）は警告し、ほかの閲覧を続ける。
メディアは選択済みの必要なファイルだけを専用URLで配信する。GradioのVideo/Image変換処理に元パスを渡さず、
元ファイル隣への変換ファイル作成を避ける。共有ルート全体の `allowed_paths` 登録は行わない。
Gradioの汎用ファイル配信からは結果・状態・checkoutをブロックする。

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

ブラウザの受け入れ検証は追加依存がある場合に実行する（コアのテスト環境ではskip）。
専用の検証環境に `.[dev,viewer]` と `playwright imageio-ffmpeg` を入れ、Chromiumを用意する。

```bash
playwright install chromium
pytest -q tests/test_viewer_browser.py
```

既存Chromiumを使う場合は `VIEWER_TEST_BROWSER` に実行可能ファイルを指定する。
fixtureのH.264動画再生、Range配信、ブラウザメモ保存・復元・JSON移行・破損/衝突時の保持、
本文・タグがネットワーク要求やサーバーログに入らないこと、読み取り前後のデータ不変を確認する。
Slurmがない環境ではsbatch stubで投入引数を確認する。
実共有データの動画形式、実クラスタのCPU partition・SSH経路・ジョブ投入は現地での確認が必要。

GradioのAPIは[HTML公式資料](https://www.gradio.app/docs/gradio/html)と
[mount_gradio_app公式資料](https://www.gradio.app/docs/gradio/mount_gradio_app)、インストールした6.9.0のAPIで確認した。
