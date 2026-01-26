import random
import argparse
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


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
    maze: Maze,
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
    """
    角欠け対策済みの四角迷路描画。
    重要：START/GOALアイコンはここでは貼らない（PDF側で固定サイズ描画する）。
    戻り値： (img, start_center_px, goal_center_px)
      - start_center_px / goal_center_px は img のピクセル座標（左上原点）。
    """
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

    def fill_rect(x0, y0, x1, y1, color=fg):
        draw.rectangle([x0, y0, x1, y1], fill=color)

    grid_x = [mg + i * cs for i in range(w + 1)]
    grid_y = [mg + j * cs for j in range(h + 1)]

    # 壁セグメント表
    v = [[False for _ in range(w + 1)] for __ in range(h)]
    hs = [[False for _ in range(w)] for __ in range(h + 1)]

    # 外周は全部壁
    for y in range(h):
        v[y][0] = True
        v[y][w] = True
    for x in range(w):
        hs[0][x] = True
        hs[h][x] = True

    # 内壁：通路が無いところが壁（E,Sだけで表現）
    for y in range(h):
        for x in range(w):
            if not (maze.cells[y][x] & E):
                v[y][x + 1] = True
            if not (maze.cells[y][x] & S):
                hs[y + 1][x] = True

    # 入口/出口の穴：外周壁フラグを落とす
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

    # セグメント描画
    half_l = wt // 2
    half_r = wt - half_l

    # vertical
    for y in range(h):
        y0 = grid_y[y]
        y1 = grid_y[y + 1]
        for xi in range(w + 1):
            if not v[y][xi]:
                continue
            x = grid_x[xi]
            fill_rect(x - half_l, y0, x + half_r - 1, y1 - 1, fg)

    # horizontal
    for yi in range(h + 1):
        y = grid_y[yi]
        for x in range(w):
            if not hs[yi][x]:
                continue
            x0 = grid_x[x]
            x1 = grid_x[x + 1]
            fill_rect(x0, y - half_l, x1 - 1, y + half_r - 1, fg)

    # 交点埋め（角欠け防止）
    for yi in range(h + 1):
        for xi in range(w + 1):
            touches = False
            if yi > 0 and v[yi - 1][xi]:
                touches = True
            if yi < h and v[yi][xi]:
                touches = True
            if xi > 0 and hs[yi][xi - 1]:
                touches = True
            if xi < w and hs[yi][xi]:
                touches = True

            if not touches:
                continue

            x = grid_x[xi]
            y = grid_y[yi]
            fill_rect(x - half_l, y - half_l, x + half_r - 1, y + half_r - 1, fg)

    # --- START/GOAL の「中心座標(px)」を計算（PDF側で固定サイズ描画するため） ---
    def icon_center(cell, side):
        cx_cell, cy_cell = cell
        x0 = grid_x[cx_cell]
        x1 = grid_x[cx_cell + 1]
        y0 = grid_y[cy_cell]
        y1 = grid_y[cy_cell + 1]

        offset = int(wt * 3)  # 迷路の外へ出す量（AAスケール済み）

        if side == "left":
            return (x0 - offset, (y0 + y1) / 2)
        elif side == "right":
            return (x1 + offset, (y0 + y1) / 2)
        elif side == "top":
            return ((x0 + x1) / 2, y0 - offset)
        elif side == "bottom":
            return ((x0 + x1) / 2, y1 + offset)
        else:
            raise ValueError("side must be left/right/top/bottom")

    start_center = icon_center(entrance, entrance_side)
    goal_center = icon_center(exit, exit_side)

    # AA縮小
    if aa_scale != 1:
        img = img.resize((img_w // aa_scale, img_h // aa_scale), resample=Image.Resampling.LANCZOS)
        start_center = (start_center[0] / aa_scale, start_center[1] / aa_scale)
        goal_center = (goal_center[0] / aa_scale, goal_center[1] / aa_scale)

    return img, start_center, goal_center


def save_image_as_a4_pdf(
    img: Image.Image,
    out_pdf_path: str,
    margin_mm: float = 15.0,
    start_icon_path: str | None = None,
    goal_icon_path: str | None = None,
    start_center_px: tuple[float, float] | None = None,  # (x,y) image px, origin top-left
    goal_center_px: tuple[float, float] | None = None,
    icon_mm: float = 12.0,  # A4で読みやすい固定サイズ（10〜16mm推奨）
):
    """
    PIL.Image を A4 1ページのPDFとして保存（縦横比維持で中央配置）。
    START/GOALアイコンはPDF上に「固定mm」で重ね描き（迷路画像の縮尺と独立）。
    """
    page_w, page_h = A4
    margin = margin_mm * mm

    c = canvas.Canvas(out_pdf_path, pagesize=A4)

    img_w_px, img_h_px = img.size

    max_w = page_w - margin * 2
    max_h = page_h - margin * 2
    scale = min(max_w / img_w_px, max_h / img_h_px)

    draw_w = img_w_px * scale
    draw_h = img_h_px * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c.drawImage(
        ImageReader(img),
        x, y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    # --- START/GOAL アイコンを固定mmで描画 ---
    def draw_icon_on_pdf(icon_path: str | None, center_px: tuple[float, float] | None):
        if not icon_path or center_px is None:
            return
        icon = ImageReader(icon_path)
        icon_size_pt = icon_mm * mm

        px, py = center_px  # image座標（左上原点）
        # 画像→PDF座標へ（PDFは左下原点）
        pdf_x = x + px * scale
        pdf_y = y + (img_h_px - py) * scale

        c.drawImage(
            icon,
            pdf_x - icon_size_pt / 2,
            pdf_y - icon_size_pt / 2,
            width=icon_size_pt,
            height=icon_size_pt,
            mask="auto",
        )

    draw_icon_on_pdf(start_icon_path, start_center_px)
    draw_icon_on_pdf(goal_icon_path, goal_center_px)

    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--wall", type=int, default=4)
    ap.add_argument("--aa", type=int, default=1, help="anti-alias scale (2-4 recommended)")
    ap.add_argument("--out", type=str, default="maze.png")
    ap.add_argument("--pdf", type=str, default=None, help="output A4 PDF path (auto if omitted)")

    # PDF上の固定アイコンサイズ(mm)
    ap.add_argument("--icon-mm", type=float, default=8.0, help="START/GOAL icon size on A4 in mm (10-16 recommended)")
    ap.add_argument("--start-icon", type=str, default="image/start.png")
    ap.add_argument("--goal-icon", type=str, default="image/goal.png")

    args = ap.parse_args()

    m = generate_maze(args.width, args.height, seed=args.seed)
    img, start_center, goal_center = render_maze_image_joined(
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
    print(f"Saved PNG: {args.out}")

    # PDF保存（指定がなければ日時で自動生成）
    pdf_path = args.pdf
    if pdf_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"maze_{ts}.pdf"

    save_image_as_a4_pdf(
        img,
        pdf_path,
        margin_mm=15.0,
        start_icon_path=args.start_icon,
        goal_icon_path=args.goal_icon,
        start_center_px=start_center,
        goal_center_px=goal_center,
        icon_mm=args.icon_mm,
    )
    print(f"Saved A4 PDF: {pdf_path}")


if __name__ == "__main__":
    main()
