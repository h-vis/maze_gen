import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from math import cos, sin, pi
from pathlib import Path

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
    margin_mm: float = 20.0,
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

    title_rules: bool = True,
    title_rule_width_pt: float = 0.7,     # 線の太さ（0.5〜1.0くらいが上品）
    title_rule_inset_mm: float = 8.0,     # 余白からさらに内側に入れる量
    title_rule_gap_mm: float = 2.0,       # ロゴと線の間隔

    bottom_art_path: str | None = None,
    bottom_art_width_mm: float = 170.0,   # 余白内に収まる横幅（A4幅210-左右余白40=170）
    bottom_art_y_mm: float = 18.0,        # ページ下からの距離
    name_path: str | None = None,
    name_width_mm: float = 160.0,     # 余白内に収まる幅（170mmより少し小さめが上品）
    name_gap_mm: float = 3.0,         # ロゴ（下線）から名前欄までの間隔
):
    page_w, page_h = A4
    margin = margin_mm * mm
    c = canvas.Canvas(out_pdf_path, pagesize=A4)

    def validate_asset(path_str: str | None, label: str) -> str | None:
        if path_str is None:
            return None
        p = Path(path_str)
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {path_str}")
        return str(p)

    start_icon_path = validate_asset(start_icon_path, "start icon")
    goal_icon_path = validate_asset(goal_icon_path, "goal icon")
    logo_path = validate_asset(logo_path, "logo image")
    bottom_art_path = validate_asset(bottom_art_path, "bottom art image")
    name_path = validate_asset(name_path, "name image")

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
        with Image.open(icon_path) as icon_img:
            iw, ih = icon_img.size
        target_h = icon_mm * mm
        target_w = target_h * (iw / ih)

        # 外向きベクトルを決める（rect/triは引数、circleは中心から計算）
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

    if bottom_art_path:
        bw_pt = bottom_art_width_mm * mm

        # アスペクト比保持
        with Image.open(bottom_art_path) as art_img:
            iw, ih = art_img.size
        bh_pt = bw_pt * (ih / iw)

        # 横中央、下から bottom_art_y_mm の位置
        bx = (page_w - bw_pt) / 2
        by = bottom_art_y_mm * mm

        c.drawImage(
            ImageReader(bottom_art_path),
            bx,
            by,
            width=bw_pt,
            height=bh_pt,
            mask="auto",
        )

    if logo_path:
        logo = ImageReader(logo_path)
        logo_w_pt = logo_width_mm * mm

        # 画像のアスペクト比を保つ
        with Image.open(logo_path) as logo_img:
            iw, ih = logo_img.size
        logo_h_pt = logo_w_pt * (ih / iw)

        # 上中央（余白内）
        logo_x = (page_w - logo_w_pt) / 2
        logo_y = page_h - margin - logo_h_pt

        if title_rules:
            inset = title_rule_inset_mm * mm
            gap = title_rule_gap_mm * mm

            x0 = margin + inset
            x1 = page_w - margin - inset

            y_top = logo_y + logo_h_pt + gap
            y_bot = logo_y - gap

            c.saveState()
            c.setLineWidth(title_rule_width_pt)
            c.line(x0, y_top, x1, y_top)
            c.line(x0, y_bot, x1, y_bot)
            c.restoreState()

        # ロゴ本体
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

    if name_path and logo_path:
        nw_pt = name_width_mm * mm

        with Image.open(name_path) as name_img:
            niw, nih = name_img.size
        nh_pt = nw_pt * (nih / niw)

        gap_pt = float(name_gap_mm) * mm

        # ロゴ位置を再計算（上のロゴ描画と一致させる）
        with Image.open(logo_path) as logo_img:
            liw, lih = logo_img.size
        logo_w_pt = logo_width_mm * mm
        logo_h_pt = logo_w_pt * (lih / liw)
        logo_y = page_h - margin - logo_h_pt

        if title_rules:
            gap_rule = float(title_rule_gap_mm) * mm
            y_bot_local = logo_y - gap_rule
            top_y = y_bot_local - gap_pt
        else:
            top_y = logo_y - gap_pt

        right_inset_mm = 2.0
        name_x = page_w - margin - (right_inset_mm * mm) - nw_pt
        name_y = top_y - nh_pt

        c.drawImage(
            ImageReader(name_path),
            name_x,
            name_y,
            width=nw_pt,
            height=nh_pt,
            mask="auto",
        )

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
    return: (img, start_center_px, goal_center_px, (cx,cy))
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
            draw.arc(
                bbox,
                start=rad_to_deg(a0) - ddeg,
                end=rad_to_deg(a1) + ddeg,
                fill=fg,
                width=int(lw),
            )

    # 放射描画
    def p(radius: float, ang: float):
        return (cx + radius * cos(ang), cy + radius * sin(ang))

    for r in range(s0, R):
        r0 = blank_extra + r * rt
        r1 = blank_extra + (r + 1) * rt
        for tb in range(K):
            if not radial_wall[r][tb]:
                continue
            ang = 2 * pi * (tb / K)
            p0 = p(max(0.0, r0), ang)
            p1 = p(r1, ang)
            draw.line([p0, p1], fill=fg, width=int(lw))

    # START/GOAL アイコン中心(px)
    icon_offset = lw * 3  # 外周の少し外
    start_ang = 2 * pi * ((entrance_sector + 0.5) / K)
    goal_ang  = 2 * pi * ((exit_sector + 0.5) / K)

    start_center = (
        cx + (outer_r + icon_offset) * cos(start_ang),
        cy + (outer_r + icon_offset) * sin(start_ang),
    )
    goal_center = (
        cx + (outer_r + icon_offset) * cos(goal_ang),
        cy + (outer_r + icon_offset) * sin(goal_ang),
    )

    if aa_scale != 1:
        img = img.resize((W // aa_scale, H // aa_scale), resample=Image.Resampling.LANCZOS)
        start_center = (start_center[0] / aa_scale, start_center[1] / aa_scale)
        goal_center  = (goal_center[0]  / aa_scale, goal_center[1]  / aa_scale)

    return img, start_center, goal_center, (cx / aa_scale, cy / aa_scale)


# =========================================================
# 三角形迷路（正三角形タイルグリッド）
#   全体：上向き正三角形
#   行 r (0..levels-1) に 2r+1 個（k=0..2r）
#   k偶数: 上向き△, k奇数: 下向き▽
#   壁は3方向のみ（左右 + 垂直）
# =========================================================
TL, TR, TV = 1, 2, 4  # Left / Right / Vertical
TOPP = {TL: TR, TR: TL, TV: TV}


@dataclass
class TriMaze:
    levels: int
    cells: list  # cells[r][k] : 通路ビット（TL/TR/TV）


def generate_tri_maze(levels: int, seed: int | None = None) -> TriMaze:
    if levels <= 0:
        raise ValueError("levels must be > 0")

    rnd = random.Random(seed)
    cells = [[0 for _ in range(2 * r + 1)] for r in range(levels)]
    visited = [[False for _ in range(2 * r + 1)] for r in range(levels)]

    def is_up(r: int, k: int) -> bool:
        return (k % 2) == 0

    def neighbors(r: int, k: int):
        out = []
        # 左右（同じ行）
        if k - 1 >= 0:
            out.append((TL, r, k - 1))
        if k + 1 <= 2 * r:
            out.append((TR, r, k + 1))

        # 垂直（上向きは下へ、下向きは上へ）
        if is_up(r, k):
            nr, nk = r + 1, k + 1
            if nr < levels and 0 <= nk <= 2 * nr:
                out.append((TV, nr, nk))
        else:
            nr, nk = r - 1, k - 1
            if nr >= 0 and 0 <= nk <= 2 * nr:
                out.append((TV, nr, nk))
        return out

    stack = [(0, 0)]  # 頂点セルから開始
    visited[0][0] = True

    while stack:
        r, k = stack[-1]
        cand = [(d, nr, nk) for (d, nr, nk) in neighbors(r, k) if not visited[nr][nk]]
        if not cand:
            stack.pop()
            continue

        d, nr, nk = rnd.choice(cand)
        cells[r][k] |= d
        cells[nr][nk] |= TOPP[d]
        visited[nr][nk] = True
        stack.append((nr, nk))

    return TriMaze(levels=levels, cells=cells)


def render_tri_maze_png(
    maze: TriMaze,
    *,
    cell_size: int = 36,     # 一辺(px)
    margin: int = 30,
    line_width: int = 5,
    aa_scale: int = 3,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
):
    """
    return: (img, start_center_px, goal_center_px, start_outward_unit, goal_outward_unit)
    """
    import math

    levels = maze.levels
    s = cell_size * aa_scale
    mg = margin * aa_scale
    lw = line_width * aa_scale
    h = s * math.sqrt(3) / 2.0  # 正三角形の高さ

    total_w = mg * 2 + levels * s
    total_h = mg * 2 + levels * h

    W = int(math.ceil(total_w))
    H = int(math.ceil(total_h))

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    def is_up(r: int, k: int) -> bool:
        return (k % 2) == 0

    def tri_vertices(r: int, k: int):
        y_top = mg + r * h
        x_left = mg + (levels - 1 - r) * (s / 2.0) + k * (s / 2.0)

        if is_up(r, k):
            # 上向き△: 左下, 上, 右下
            v0 = (x_left,         y_top + h)
            v1 = (x_left + s/2.0, y_top)
            v2 = (x_left + s,     y_top + h)
        else:
            # 下向き▽: 左上, 右上, 下
            v0 = (x_left,         y_top)
            v1 = (x_left + s,     y_top)
            v2 = (x_left + s/2.0, y_top + h)
        return v0, v1, v2

    # 入口/出口：最下段の左端/右端（どちらも上向き△）
    start_cell = (1, 0)
    goal_cell  = (levels - 1, 2 * (levels - 1))

    # 迷路壁描画（通路があればその辺は描かない）
    for r in range(levels):
        for k in range(2 * r + 1):
            v0, v1, v2 = tri_vertices(r, k)
            bits = maze.cells[r][k]

            if is_up(r, k):
                # TL: v1-v0, TR: v1-v2, TV(下): v0-v2
                if not (bits & TL):
                    draw.line([v1, v0], fill=fg, width=int(lw))
                if not (bits & TR):
                    draw.line([v1, v2], fill=fg, width=int(lw))
                if not (bits & TV):
                    draw.line([v0, v2], fill=fg, width=int(lw))
            else:
                # TL: v0-v2, TR: v1-v2, TV(上): v0-v1
                if not (bits & TL):
                    draw.line([v0, v2], fill=fg, width=int(lw))
                if not (bits & TR):
                    draw.line([v1, v2], fill=fg, width=int(lw))
                if not (bits & TV):
                    draw.line([v0, v1], fill=fg, width=int(lw))

    # 入口/出口の穴：境界辺を「白で消す」（見た目穴）
    hole_w = int(lw * 3.0)  # ← ここを 2.5〜4.0 で調整（大きくしたいなら増やす）

    def boundary_edge_and_out(cell, side: str):
        r, k = cell
        v0, v1, v2 = tri_vertices(r, k)

        if side == "left":
            pts = sorted([v0, v1, v2], key=lambda p: p[0])
            a, b = pts[0], pts[1]  # 左側の辺
            out = (-1.0, 0.0)
        elif side == "right":
            pts = sorted([v0, v1, v2], key=lambda p: p[0], reverse=True)
            a, b = pts[0], pts[1]  # 右側の辺
            out = (1.0, 0.0)
        else:
            raise ValueError("side must be 'left' or 'right'")

        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        return (a, b), (mx, my), out
    
    # start: 左辺の辺を消す
    (start_a, start_b), (smx, smy), start_out = boundary_edge_and_out(start_cell, "left")
    draw.line([start_a, start_b], fill=bg, width=hole_w)

    # goal: 右辺の辺を消す
    (goal_a, goal_b), (gmx, gmy), goal_out = boundary_edge_and_out(goal_cell, "right")
    draw.line([goal_a, goal_b], fill=bg, width=hole_w)

    # アイコン中心（穴の中点から外へ）
    icon_offset = lw * 6
    start_center = (smx + start_out[0] * icon_offset, smy + start_out[1] * icon_offset)
    goal_center  = (gmx + goal_out[0]  * icon_offset, gmy + goal_out[1]  * icon_offset)


    if aa_scale != 1:
        img = img.resize((W // aa_scale, H // aa_scale), resample=Image.Resampling.LANCZOS)
        start_center = (start_center[0] / aa_scale, start_center[1] / aa_scale)
        goal_center  = (goal_center[0]  / aa_scale, goal_center[1]  / aa_scale)

    return img, start_center, goal_center, start_out, goal_out


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shape", choices=["rect", "circle", "tri"], default="rect")
    ap.add_argument("--seed", type=int, default=None)

    # 共通出力
    ap.add_argument("--out", type=str, default="maze.png")
    ap.add_argument("--pdf", type=str, default=None, help="output A4 PDF path (auto if omitted)")
    ap.add_argument("--start-icon", type=str, default="image/start.png")
    ap.add_argument("--goal-icon", type=str, default="image/goal.png")
    ap.add_argument("--icon-mm", type=float, default=10.0, help="START/GOAL icon size on A4 in mm (10-16 recommended)")
    ap.add_argument("--pdf-margin-mm", type=float, default=20.0)

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

    # 三角迷路パラメータ
    ap.add_argument("--tri-levels", type=int, default=18)
    ap.add_argument("--tri-cell", type=int, default=64)
    ap.add_argument("--tri-margin", type=int, default=30)
    ap.add_argument("--tri-line-width", type=int, default=5)
    ap.add_argument("--tri-aa", type=int, default=3)

    args = ap.parse_args()

    maze_center = None
    start_outward_unit = None
    goal_outward_unit = None

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
        # rectは左側へ押し出す（既存方針）
        start_outward_unit = (-4.0, 0.0)
        goal_outward_unit = (-4.0, 0.0)

    elif args.shape == "circle":
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
        # circleはcenterからの外向きをPDF側で計算
        start_outward_unit = None
        goal_outward_unit = None

    else:  # tri
        tm = generate_tri_maze(args.tri_levels, seed=args.seed)
        img, start_center, goal_center, start_outward_unit, goal_outward_unit = render_tri_maze_png(
            tm,
            cell_size=args.tri_cell,
            margin=args.tri_margin,
            line_width=args.tri_line_width,
            aa_scale=args.tri_aa,
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
        maze_center_px=maze_center,          # circleのみ有効
        icon_mm=args.icon_mm,
        logo_path="image/logo.png",
        logo_width_mm=120.0,
        start_outward_unit=start_outward_unit,  # rect/tri: unit指定, circle: None
        goal_outward_unit=goal_outward_unit,
        bottom_art_path="image/animal.png",
        bottom_art_width_mm=170.0,
        bottom_art_y_mm=10.0,
        name_path="image/name.png",
        name_width_mm=120.0,
        name_gap_mm=10.0,
    )
    print(f"Saved A4 PDF: {pdf_path}")


if __name__ == "__main__":
    main()
