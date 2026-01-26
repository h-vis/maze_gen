import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from math import cos, sin, pi

from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# =========================================================
# PDF: A4に迷路画像をフィット配置し、START/GOALアイコンは固定mmで重ね描き
# =========================================================
def save_image_as_a4_pdf(
    img: Image.Image,
    out_pdf_path: str,
    margin_mm: float = 15.0,
    start_icon_path: str | None = None,
    goal_icon_path: str | None = None,
    start_center_px: tuple[float, float] | None = None,  # img上(px) 左上原点
    goal_center_px: tuple[float, float] | None = None,
    icon_mm: float = 10.0,  # A4上で読みやすい固定サイズ（10〜16mm推奨）
    logo_path: str | None = None,
    logo_width_mm: float = 35.0,   # A4で見やすい横幅
    maze_center_px: tuple[float, float] | None = None,  # 円形迷路の中心(px) 画像座標
    icon_gap_mm: float = -10.0,                           # 迷路とアイコンの隙間
    start_outward_unit: tuple[float, float] | None = None,
    goal_outward_unit: tuple[float, float] | None = None,
):
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

    def draw_icon_on_pdf(
        icon_path: str | None,
        anchor_px: tuple[float, float] | None,
        outward_unit: tuple[float, float] | None,
    ):
        if not icon_path or anchor_px is None:
            return

        ax, ay = anchor_px  # img座標（左上原点）

        # アスペクト比維持：高さを固定
        iw, ih = Image.open(icon_path).size
        target_h = icon_mm * mm
        target_w = target_h * (iw / ih)

        # 外向きベクトルを決める（rectは引数、circleは中心から計算）
        ux = uy = 0.0
        if outward_unit is not None:
            ux, uy = outward_unit
        elif maze_center_px is not None:
            cx_img, cy_img = maze_center_px
            vx = ax - cx_img
            vy = ay - cy_img
            norm = (vx * vx + vy * vy) ** 0.5
            if norm > 1e-6:
                ux = vx / norm
                uy = vy / norm

        # 押し出し量（pt→pxへ）
        half_ext_pt = max(target_w, target_h) / 2
        push_px = (half_ext_pt + icon_gap_mm * mm) / scale

        # outwardが無い（ux=uy=0）のときは押し出しゼロ
        px = ax + ux * push_px
        py = ay + uy * push_px

        pdf_x = x + px * scale
        pdf_y = y + (img_h_px - py) * scale

        c.drawImage(
            ImageReader(icon_path),
            pdf_x - target_w / 2,
            pdf_y - target_h / 2,
            width=target_w,
            height=target_h,
            mask="auto",
        )
    

    if logo_path:
        logo = ImageReader(logo_path)
        logo_w_pt = logo_width_mm * mm

        # 画像のアスペクト比を保つ
        iw, ih = Image.open(logo_path).size
        logo_h_pt = logo_w_pt * (ih / iw)

        # 上中央（余白内）
        logo_x = (page_w - logo_w_pt) / 2
        logo_y = page_h - margin - logo_h_pt

        c.drawImage(
            logo,
            logo_x,
            logo_y,
            width=logo_w_pt,
            height=logo_h_pt,
            mask="auto",
        )

    draw_icon_on_pdf(start_icon_path, start_center_px, start_outward_unit)
    draw_icon_on_pdf(goal_icon_path,  goal_center_px,  goal_outward_unit)

    c.showPage()
    c.save()


# =========================================================
# 四角迷路（グリッド）
# =========================================================
N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S: 0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {N: S, S: N, E: W, W: E}


@dataclass
class RectMaze:
    width: int
    height: int
    cells: list  # cells[y][x] : 通路ビット


def generate_rect_maze(width: int, height: int, seed: int | None = None) -> RectMaze:
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

    return RectMaze(width, height, cells)


def render_rect_maze_image_joined(
    maze: RectMaze,
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
    PNG内にアイコンは貼らない。代わりにアイコン中心座標(px)を返す。
    return: (img, start_center_px, goal_center_px)
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

    v = [[False for _ in range(w + 1)] for __ in range(h)]
    hs = [[False for _ in range(w)] for __ in range(h + 1)]

    # 外周
    for y in range(h):
        v[y][0] = True
        v[y][w] = True
    for x in range(w):
        hs[0][x] = True
        hs[h][x] = True

    # 内壁（E,Sだけで表現）
    for y in range(h):
        for x in range(w):
            if not (maze.cells[y][x] & E):
                v[y][x + 1] = True
            if not (maze.cells[y][x] & S):
                hs[y + 1][x] = True

    # 入口/出口の穴
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

    half_l = wt // 2
    half_r = wt - half_l

    # 縦壁
    for y in range(h):
        y0 = grid_y[y]
        y1 = grid_y[y + 1]
        for xi in range(w + 1):
            if not v[y][xi]:
                continue
            x = grid_x[xi]
            fill_rect(x - half_l, y0, x + half_r - 1, y1 - 1, fg)

    # 横壁
    for yi in range(h + 1):
        y = grid_y[yi]
        for x in range(w):
            if not hs[yi][x]:
                continue
            x0 = grid_x[x]
            x1 = grid_x[x + 1]
            fill_rect(x0, y - half_l, x1 - 1, y + half_r - 1, fg)

    # 交点埋め
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

    # アイコン中心(px)
    def icon_center(cell, side):
        cx_cell, cy_cell = cell
        x0 = grid_x[cx_cell]
        x1 = grid_x[cx_cell + 1]
        y0 = grid_y[cy_cell]
        y1 = grid_y[cy_cell + 1]
        offset = int(wt * 3)  # 外に出す

        if side == "left":
            return (x0 - offset, (y0 + y1) / 2)
        if side == "right":
            return (x1 + offset, (y0 + y1) / 2)
        if side == "top":
            return ((x0 + x1) / 2, y0 - offset)
        if side == "bottom":
            return ((x0 + x1) / 2, y1 + offset)
        raise ValueError("side must be left/right/top/bottom")

    start_center = icon_center(entrance, entrance_side)
    goal_center = icon_center(exit, exit_side)

    if aa_scale != 1:
        img = img.resize((img_w // aa_scale, img_h // aa_scale), resample=Image.Resampling.LANCZOS)
        start_center = (start_center[0] / aa_scale, start_center[1] / aa_scale)
        goal_center = (goal_center[0] / aa_scale, goal_center[1] / aa_scale)

    return img, start_center, goal_center


# =========================================================
# 円形迷路（リング×セクタ）＋中心空白
# =========================================================
PIN, POUT, PCCW, PCW = 1, 2, 4, 8
POPP = {PIN: POUT, POUT: PIN, PCCW: PCW, PCW: PCCW}


@dataclass
class PolarMaze:
    rings: int
    sectors: int
    start_ring: int   # ここより内側は空白
    cells: list       # cells[r][t] : 通れる方向ビット（r<start_ringは未使用）


def generate_polar_maze_with_blank_center(
    rings: int,
    sectors: int,
    start_ring: int,
    seed: int | None = None,
    start_sector: int = 0,
) -> PolarMaze:
    if rings <= 0:
        raise ValueError("rings must be > 0")
    if sectors < 3:
        raise ValueError("sectors must be >= 3")
    if not (0 <= start_ring < rings):
        raise ValueError("start_ring must be in [0, rings-1]")

    rnd = random.Random(seed)
    cells = [[0 for _ in range(sectors)] for _ in range(rings)]
    visited = [[False for _ in range(sectors)] for _ in range(rings)]

    start_sector %= sectors
    sr, st = rings - 1, start_sector  # 外周から開始
    visited[sr][st] = True
    stack = [(sr, st)]

    def neighbors(r, t):
        out = []
        if r - 1 >= start_ring:
            out.append((PIN, r - 1, t))
        if r + 1 <= rings - 1:
            out.append((POUT, r + 1, t))
        out.append((PCCW, r, (t - 1) % sectors))
        out.append((PCW, r, (t + 1) % sectors))
        return out

    while stack:
        r, t = stack[-1]
        cand = [(d, nr, nt) for (d, nr, nt) in neighbors(r, t) if not visited[nr][nt]]
        if not cand:
            stack.pop()
            continue

        d, nr, nt = rnd.choice(cand)
        cells[r][t] |= d
        cells[nr][nt] |= POPP[d]
        visited[nr][nt] = True
        stack.append((nr, nt))

    return PolarMaze(rings=rings, sectors=sectors, start_ring=start_ring, cells=cells)


def _draw_rotated_square(draw: ImageDraw.ImageDraw, x: float, y: float, ang: float, half: float, color):
    # 放射方向(ang)と接線方向に平行な回転正方形（ジョイント）
    urx, ury = cos(ang), sin(ang)
    utx, uty = -sin(ang), cos(ang)
    p1 = (x + half*urx + half*utx, y + half*ury + half*uty)
    p2 = (x + half*urx - half*utx, y + half*ury - half*uty)
    p3 = (x - half*urx - half*utx, y - half*ury - half*uty)
    p4 = (x - half*urx + half*utx, y - half*ury + half*uty)
    draw.polygon([p1, p2, p3, p4], fill=color)


def render_circular_maze_png(
    maze: PolarMaze,
    *,
    ring_thickness_px: int = 24,
    margin_px: int = 30,
    line_width_px: int = 5,
    aa_scale: int = 3,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    entrance_sector: int = 0,
    exit_sector: int | None = None,   # Noneなら直径対向（sectors//2）
    blank_extra_px: int = 18,         # 中心空白の見た目余白
):
    """
    入口/出口は外周（直径対向がデフォルト）。
    中心は start_ring より内側を空白。
    PNG内にアイコンは貼らない。代わりにアイコン中心座標(px)を返す（PDFで固定mm描画）。
    return: (img, start_center_px, goal_center_px)
    """
    R, K, s0 = maze.rings, maze.sectors, maze.start_ring
    if exit_sector is None:
        exit_sector = (entrance_sector + (K // 2)) % K

    entrance_sector %= K
    exit_sector %= K

    rt = ring_thickness_px * aa_scale
    mg = margin_px * aa_scale
    lw = line_width_px * aa_scale
    half = lw / 2.0

    blank_extra = max(0, blank_extra_px) * aa_scale
    outer_r = blank_extra + R * rt

    W = int(mg * 2 + outer_r * 2)
    H = W
    cx = mg + outer_r
    cy = mg + outer_r

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    def p(radius: float, ang: float):
        return (cx + radius * cos(ang), cy + radius * sin(ang))

    def rad_to_deg(a: float) -> float:
        return a * 180.0 / pi

    # 壁セグメント
    arc_wall = [[False for _ in range(K)] for __ in range(R + 1)]
    radial_wall = [[False for _ in range(K)] for __ in range(R)]

    # 内周（start_ring境界）は仕切りとして全部壁
    for t in range(K):
        arc_wall[s0][t] = True

    # 中間境界：内側セルのOUTが閉じていれば壁
    for rb in range(s0 + 1, R + 1):
        r = rb - 1
        for t in range(K):
            arc_wall[rb][t] = not (maze.cells[r][t] & POUT)

    # 外周：入口/出口だけ穴
    arc_wall[R][entrance_sector] = False
    arc_wall[R][exit_sector] = False

    # 放射壁：CWが閉じていれば壁
    for r in range(s0, R):
        for t in range(K):
            tb = (t + 1) % K
            if not (maze.cells[r][t] & PCW):
                radial_wall[r][tb] = True

    # 円弧端の隙間対策（微小延長）
    def arc_delta_deg(radius: float) -> float:
        if radius <= 1e-6:
            return 0.0
        delta_rad = (half * 1.05) / radius
        return rad_to_deg(delta_rad)

    # 円弧描画
    for rb in range(s0, R + 1):
        radius = blank_extra + rb * rt
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        ddeg = arc_delta_deg(radius)
        for t in range(K):
            if not arc_wall[rb][t]:
                continue
            a0 = 2 * pi * (t / K)
            a1 = 2 * pi * ((t + 1) / K)
            draw.arc(bbox, start=rad_to_deg(a0) - ddeg, end=rad_to_deg(a1) + ddeg, fill=fg, width=int(lw))

    # 放射描画（端のはみ出しを抑える：最外周は外側に伸ばさない）
    for r in range(s0, R):
        r0 = blank_extra + r * rt
        r1 = blank_extra + (r + 1) * rt
        for tb in range(K):
            if not radial_wall[r][tb]:
                continue
            ang = 2 * pi * (tb / K)

            r0_draw = r0
            r1_draw = r1

            p0 = p(max(0.0, r0_draw), ang)
            p1 = p(r1_draw, ang)
            draw.line([p0, p1], fill=fg, width=int(lw))

    # # ジョイント（回転正方形）
    # arc_touch = [[False for _ in range(K)] for __ in range(R + 1)]
    # rad_touch = [[False for _ in range(K)] for __ in range(R + 1)]

    # for rb in range(s0, R + 1):
    #     for t in range(K):
    #         if arc_wall[rb][t]:
    #             arc_touch[rb][t] = True
    #             arc_touch[rb][(t + 1) % K] = True

    # for r in range(s0, R):
    #     for tb in range(K):
    #         if radial_wall[r][tb]:
    #             rad_touch[r][tb] = True
    #             rad_touch[r + 1][tb] = True

    # jhalf = (lw / 2.0) * 0.78
    # for rb in range(s0, R + 1):
    #     radius = blank_extra + rb * rt
    #     for tb in range(K):
    #         if not (arc_touch[rb][tb] and rad_touch[rb][tb]):
    #             continue
    #         ang = 2 * pi * (tb / K)
    #         x, y = p(radius, ang)
    #         _draw_rotated_square(draw, x, y, ang, jhalf, fg)

    # --- START/GOAL アイコン中心(px)（PNGに貼らず、PDFで固定mm描画） ---
    icon_offset = lw * 3  # 外周の少し外
    start_ang = 2 * pi * ((entrance_sector + 0.5) / K)
    goal_ang  = 2 * pi * ((exit_sector + 0.5) / K)

    start_center = (cx + (outer_r + icon_offset) * cos(start_ang),
                    cy + (outer_r + icon_offset) * sin(start_ang))
    goal_center  = (cx + (outer_r + icon_offset) * cos(goal_ang),
                    cy + (outer_r + icon_offset) * sin(goal_ang))

    if aa_scale != 1:
        img = img.resize((W // aa_scale, H // aa_scale), resample=Image.Resampling.LANCZOS)
        start_center = (start_center[0] / aa_scale, start_center[1] / aa_scale)
        goal_center  = (goal_center[0] / aa_scale,  goal_center[1]  / aa_scale)

    return img, start_center, goal_center, (cx/aa_scale, cy/aa_scale)



# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shape", choices=["rect", "circle"], default="rect")
    ap.add_argument("--seed", type=int, default=None)

    # 共通出力
    ap.add_argument("--out", type=str, default="maze.png")
    ap.add_argument("--pdf", type=str, default=None, help="output A4 PDF path (auto if omitted)")
    ap.add_argument("--start-icon", type=str, default="image/start.png")
    ap.add_argument("--goal-icon", type=str, default="image/goal.png")
    ap.add_argument("--icon-mm", type=float, default=12.0, help="START/GOAL icon size on A4 in mm (10-16 recommended)")
    ap.add_argument("--pdf-margin-mm", type=float, default=15.0)

    # 四角迷路パラメータ
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--wall", type=int, default=4)
    ap.add_argument("--aa", type=int, default=1)

    # 円迷路パラメータ
    ap.add_argument("--rings", type=int, default=15)
    ap.add_argument("--sectors", type=int, default=48, help="EVEN recommended for diametric entrance/exit")
    ap.add_argument("--blank-ratio", type=float, default=0.33, help="center blank ratio (0.25-0.45 typical)")
    ap.add_argument("--ring-thickness", type=int, default=24)
    ap.add_argument("--circle-margin", type=int, default=30)
    ap.add_argument("--line-width", type=int, default=5)
    ap.add_argument("--circle-aa", type=int, default=3)
    ap.add_argument("--blank-extra", type=int, default=18)
    ap.add_argument("--entrance-sector", type=int, default=24)
    ap.add_argument("--exit-sector", type=int, default=None)

    args = ap.parse_args()
    maze_center = None

    if args.shape == "rect":
        m = generate_rect_maze(args.width, args.height, seed=args.seed)
        img, start_center, goal_center = render_rect_maze_image_joined(
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
    else:
        blank_ratio = max(0.0, min(0.9, args.blank_ratio))
        start_ring = int(round(args.rings * blank_ratio))
        start_ring = min(max(0, start_ring), args.rings - 1)

        maze = generate_polar_maze_with_blank_center(
            args.rings,
            args.sectors,
            start_ring=start_ring,
            seed=args.seed,
            start_sector=args.entrance_sector,
        )
        img, start_center, goal_center, maze_center = render_circular_maze_png(
            maze,
            ring_thickness_px=args.ring_thickness,
            margin_px=args.circle_margin,
            line_width_px=args.line_width,
            aa_scale=args.circle_aa,
            entrance_sector=args.entrance_sector,
            exit_sector=args.exit_sector,
            blank_extra_px=args.blank_extra,
        )

    # PNG保存
    img.save(args.out)
    print(f"Saved PNG: {args.out}")

    # PDF保存（指定がなければ日時で自動生成）
    pdf_path = args.pdf
    if pdf_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"maze_{args.shape}_{ts}.pdf"

    save_image_as_a4_pdf(
        img,
        pdf_path,
        margin_mm=args.pdf_margin_mm,
        start_icon_path=args.start_icon,
        goal_icon_path=args.goal_icon,
        start_center_px=start_center,
        goal_center_px=goal_center,
        maze_center_px=maze_center,
        icon_mm=args.icon_mm,
        logo_path="image/logo.png",
        logo_width_mm=120.0, 
        start_outward_unit=(-1.0, 0.0),  # rect: 左外
        goal_outward_unit=( 1.0, 0.0),  # rect: 右外
    )
    print(f"Saved A4 PDF: {pdf_path}")


if __name__ == "__main__":
    main()
