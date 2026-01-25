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

def render_maze_image_joined(
    maze,
    cell_size: int = 28,
    margin: int = 24,
    wall: int = 4,
    aa_scale: int = 1,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    entrance=(0, 0),
    entrance_side="left",
    exit=None,
    exit_side="right",
):
    w, h = maze.width, maze.height
    if exit is None:
        exit = (w - 1, h - 1)

    cs = cell_size * aa_scale
    mg = margin * aa_scale
    wt = wall * aa_scale

    img_w = mg * 2 + w * cs
    img_h = mg * 2 + h * cs
    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # 壁の矩形を引く（伸ばさない）
    def fill_rect(x0, y0, x1, y1, color=fg):
        # x1,y1 は「含む」座標として扱う（Pillowのrectangle仕様に合わせる）
        draw.rectangle([x0, y0, x1, y1], fill=color)

    # グリッド線（セル境界）のピクセル座標
    # grid_x[i] = 左端から iセル境界のx
    grid_x = [mg + i * cs for i in range(w + 1)]
    grid_y = [mg + j * cs for j in range(h + 1)]

    # 壁の存在を「格子線上のセグメント」として保持
    # v[y][x] : x番目の縦グリッド線上で、y→y+1 の縦壁があるか (0<=y<h, 0<=x<=w)
    # hseg[y][x] : y番目の横グリッド線上で、x→x+1 の横壁があるか (0<=y<=h, 0<=x<w)
    v = [[False for _ in range(w + 1)] for __ in range(h)]
    hs = [[False for _ in range(w)] for __ in range(h + 1)]

    # 外周は全部壁
    for y in range(h):
        v[y][0] = True
        v[y][w] = True
    for x in range(w):
        hs[0][x] = True
        hs[h][x] = True

    # 内壁：通路が無いところが壁
    # 東壁（セル(x,y)の右境界）：通路Eが無ければ縦壁
    # 南壁（セル(x,y)の下境界）：通路Sが無ければ横壁
    # 西/北は隣セルの東/南で表現されるのでここでは不要
    for y in range(h):
        for x in range(w):
            if not (maze.cells[y][x] & 4):  # E
                v[y][x + 1] = True
            if not (maze.cells[y][x] & 2):  # S
                hs[y + 1][x] = True

    # 入口/出口の穴：外周壁フラグを落とす（描画そのものを出さない）
    ex, ey = entrance
    ox, oy = exit

    def open_hole(cell, side):
        cx, cy = cell
        if side == "left":
            v[cy][0] = False
        elif side == "right":
            v[cy][w] = False
        elif side == "top":
            hs[0][cx] = False
        elif side == "bottom":
            hs[h][cx] = False
        else:
            raise ValueError("side must be left/right/top/bottom")

    open_hole(entrance, entrance_side)
    open_hole(exit, exit_side)

    # まずセグメントを描く（縦壁・横壁）
    # 縦壁：x=grid_x[xi] 上に太さwtで、y0..y1
    half_l = wt // 2
    half_r = wt - half_l  # 左右で合計wtになるように

    # vertical segments
    for y in range(h):
        y0 = grid_y[y]
        y1 = grid_y[y + 1]
        for xi in range(w + 1):
            if not v[y][xi]:
                continue
            x = grid_x[xi]
            fill_rect(x - half_l, y0, x + half_r - 1, y1 - 1, fg)

    # horizontal segments
    for yi in range(h + 1):
        y = grid_y[yi]
        for x in range(w):
            if not hs[yi][x]:
                continue
            x0 = grid_x[x]
            x1 = grid_x[x + 1]
            fill_rect(x0, y - half_l, x1 - 1, y + half_r - 1, fg)

    # ここが肝：交点（格子点）に「角埋めスクエア」を追加して欠けを完全に潰す
    # 「上下左右のどれかに壁がある交点」だけ塗る
    for yi in range(h + 1):
        for xi in range(w + 1):
            touches = False

            # この交点に接する縦壁（上側/下側）
            if yi > 0 and v[yi - 1][xi]:
                touches = True
            if yi < h and v[yi][xi]:
                touches = True

            # この交点に接する横壁（左側/右側）
            if xi > 0 and hs[yi][xi - 1]:
                touches = True
            if xi < w and hs[yi][xi]:
                touches = True

            if not touches:
                continue

            x = grid_x[xi]
            y = grid_y[yi]
            fill_rect(x - half_l, y - half_l, x + half_r - 1, y + half_r - 1, fg)

    # AA縮小（任意）
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
    img = render_maze_image_joined(
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
