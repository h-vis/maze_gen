import random
import argparse
from dataclasses import dataclass
from PIL import Image, ImageDraw

# ---- 迷路表現（通路ビット） ----
N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S: 0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {N: S, S: N, E: W, W: E}

@dataclass
class Maze:
    width: int
    height: int
    cells: list  # cells[y][x]

def generate_maze(width: int, height: int, seed: int | None = None) -> Maze:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    rnd = random.Random(seed)
    cells = [[0 for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]

    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        x, y = stack[-1]
        neighbors = []
        for d in (N, S, E, W):
            nx, ny = x + DX[d], y + DY[d]
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                neighbors.append((d, nx, ny))

        if not neighbors:
            stack.pop()
            continue

        d, nx, ny = rnd.choice(neighbors)
        cells[y][x] |= d
        cells[ny][nx] |= OPPOSITE[d]
        visited[ny][nx] = True
        stack.append((nx, ny))

    return Maze(width, height, cells)

def render_maze_image_rect_walls(
    maze: Maze,
    cell_size: int = 28,
    margin: int = 24,
    wall: int = 4,          # 壁の太さ(px)
    aa_scale: int = 1,      # 2〜4にすると縮小AAでさらに綺麗
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    entrance=(0, 0),
    entrance_side="left",   # left/top
    exit=None,
    exit_side="right",      # right/bottom
) -> Image.Image:
    w, h = maze.width, maze.height
    if exit is None:
        exit = (w - 1, h - 1)

    # AA用に全体をスケールして描画 → 最後に縮小
    cs = cell_size * aa_scale
    mg = margin * aa_scale
    wt = wall * aa_scale

    img_w = mg * 2 + w * cs
    img_h = mg * 2 + h * cs
    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    def cell_origin(x, y):
        return mg + x * cs, mg + y * cs

    def rect(x0, y0, x1, y1):
        # PILは右下含むので、細い隙間を防ぐために整数のまま描く
        draw.rectangle([x0, y0, x1, y1], fill=fg)

    # 外枠：4辺を「塗り長方形」で描く（角が綺麗）
    left = mg
    top = mg
    right = mg + w * cs
    bottom = mg + h * cs
    # 上
    rect(left, top, right, top + wt - 1)
    # 下
    rect(left, bottom - wt, right, bottom - 1)
    # 左
    rect(left, top, left + wt - 1, bottom - 1)
    # 右
    rect(right - wt, top, right - 1, bottom - 1)

    # 内壁：各セルの「東」「南」だけ描く（重複回避）
    for y in range(h):
        for x in range(w):
            x0, y0 = cell_origin(x, y)
            x1, y1 = x0 + cs, y0 + cs

            # 東壁（x1 位置に縦壁）
            if not (maze.cells[y][x] & E):
                rect(x1 - wt, y0, x1 - 1, y1 - 1)

            # 南壁（y1 位置に横壁）
            if not (maze.cells[y][x] & S):
                rect(x0, y1 - wt, x1 - 1, y1 - 1)

    # 入口/出口の穴を「背景色の長方形」で開ける（細線が残りにくい）
    def carve_hole(cell, side):
        cx, cy = cell
        x0, y0 = cell_origin(cx, cy)
        x1, y1 = x0 + cs, y0 + cs

        # 穴は壁厚より少し大きめに消す（ゴール付近の細線対策）
        pad = wt + 1

        if side == "left":
            draw.rectangle([x0 - pad, y0 + wt, x0 + pad, y1 - wt], fill=bg)
        elif side == "right":
            draw.rectangle([x1 - pad, y0 + wt, x1 + pad, y1 - wt], fill=bg)
        elif side == "top":
            draw.rectangle([x0 + wt, y0 - pad, x1 - wt, y0 + pad], fill=bg)
        elif side == "bottom":
            draw.rectangle([x0 + wt, y1 - pad, x1 - wt, y1 + pad], fill=bg)
        else:
            raise ValueError("side must be left/right/top/bottom")

    carve_hole(entrance, entrance_side)
    carve_hole(exit, exit_side)

    # AA縮小
    if aa_scale != 1:
        img = img.resize((img_w // aa_scale, img_h // aa_scale), resample=Image.Resampling.LANCZOS)

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--wall", type=int, default=4)
    ap.add_argument("--aa", type=int, default=1, help="anti-alias scale (2-4 recommended for nicer corners)")
    ap.add_argument("--out", type=str, default="maze.png")
    args = ap.parse_args()

    m = generate_maze(args.width, args.height, seed=args.seed)
    img = render_maze_image_rect_walls(
        m,
        cell_size=args.cell,
        margin=args.margin,
        wall=args.wall,
        aa_scale=args.aa,
        entrance=(0, 0),
        entrance_side="left",
        exit=(args.width - 1, args.height - 1),
        exit_side="right",
    )
    img.save(args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
