# maze_all.py コマンド引数ガイド

`maze_all.py` は、迷路画像（PNG）と印刷用PDF（A4）を生成するCLIです。  
基本実行:

```powershell
python maze_all.py [オプション]
```

## 1. 共通引数

- `--shape` (`rect` / `circle`, デフォルト: `rect`)
  - 迷路の形状を指定します。
- `--seed` (int, デフォルト: `None`)
  - 乱数シード。指定すると同じ迷路を再現できます。
- `--out` (str, デフォルト: `maze.png`)
  - 出力PNGファイル名。
- `--pdf` (str, デフォルト: `None`)
  - 出力PDFファイル名。未指定時は `maze_<shape>_YYYYMMDD_HHMMSS.pdf`。
- `--start-icon` (str, デフォルト: `image/start.png`)
  - STARTアイコン画像パス。
- `--goal-icon` (str, デフォルト: `image/goal.png`)
  - GOALアイコン画像パス。
- `--icon-mm` (float, デフォルト: `10.0`)
  - PDF上のSTART/GOALアイコンサイズ（mm）。
- `--pdf-margin-mm` (float, デフォルト: `20.0`)
  - PDF余白（mm）。

## 2. 矩形迷路 (`--shape rect`) 用

- `--width` (int, デフォルト: `30`)
  - 横セル数。
- `--height` (int, デフォルト: `30`)
  - 縦セル数。
- `--cell` (int, デフォルト: `28`)
  - 1セルの描画サイズ（px）。
- `--margin` (int, デフォルト: `24`)
  - PNGの外側余白（px）。
- `--wall` (int, デフォルト: `4`)
  - 壁の太さ（px）。
- `--aa` (int, デフォルト: `1`)
  - アンチエイリアス用倍率（1=無効）。

## 3. 円形迷路 (`--shape circle`) 用

- `--rings` (int, デフォルト: `15`)
  - 同心円の段数。
- `--sectors` (int, デフォルト: `48`)
  - 扇形分割数。偶数推奨。
- `--blank-ratio` (float, デフォルト: `0.33`)
  - 中央空白の比率（0.25〜0.45目安）。
- `--ring-thickness` (int, デフォルト: `24`)
  - リングの厚み（px）。
- `--circle-margin` (int, デフォルト: `30`)
  - 円形PNGの余白（px）。
- `--line-width` (int, デフォルト: `5`)
  - 壁線の太さ（px）。
- `--circle-aa` (int, デフォルト: `3`)
  - アンチエイリアス用倍率。
- `--blank-extra` (int, デフォルト: `18`)
  - 中央空白の追加余白（px）。
- `--entrance-sector` (int, デフォルト: `24`)
  - 入口セクタ番号。
- `--exit-sector` (int, デフォルト: `None`)
  - 出口セクタ番号。未指定なら入口の反対側。

## 4. 実行例

矩形迷路を固定シードで生成:

```powershell
python maze_all.py --shape rect --width 40 --height 30 --seed 1234 --out rect.png --pdf rect.pdf
```

円形迷路を生成（出口は自動で反対側）:

```powershell
python maze_all.py --shape circle --rings 18 --sectors 60 --seed 2026 --out circle.png --pdf circle.pdf
```

アイコンと余白を調整:

```powershell
python maze_all.py --icon-mm 14 --pdf-margin-mm 15 --start-icon image/start.png --goal-icon image/goal.png
```

## 5. 注意点

- `--shape rect` でも `circle` 用引数は受け取れますが、実質使われません（逆も同様）。
- 画像パス（`--start-icon`, `--goal-icon` など）が存在しない場合は `FileNotFoundError` になります。
- `--seed` を未指定にすると、実行ごとに異なる迷路になります。
