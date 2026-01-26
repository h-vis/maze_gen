import argparse
import random
from dataclasses import dataclass
from collections import deque
from math import cos, sin, pi
from PIL import Image, ImageDraw

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from datetime import datetime

# 通路ビット（通れる方向）
IN, OUT, CCW, CW = 1, 2, 4, 8
OPP = {IN: OUT, OUT: IN, CCW: CW, CW: CCW}

@dataclass
class PolarMaze:
    rings: int
    sectors: int
    start_ring: int   # ここより内側は空白（迷路なし）
    cells: list       # cells[r][t] : 通れる方向ビット（r<start_ringは未使用）

def generate_polar_maze_with_blank_center(
    rings: int,
    sectors: int,
    start_ring: int,
    seed: int | None = None,
    start_sector: int = 0,
) -> PolarMaze:
    """
    start_ring より内側（r < start_ring）は迷路を作らない（空白）。
    DFSで完全迷路を生成。開始点は外周(rings-1, start_sector)。
    """
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
    sr, st = rings - 1, start_sector
    visited[sr][st] = True
    stack = [(sr, st)]

    def neighbors(r, t):
        out = []
        if r - 1 >= start_ring:
            out.append((IN, r - 1, t))
        if r + 1 <= rings - 1:
            out.append((OUT, r + 1, t))
        out.append((CCW, r, (t - 1) % sectors))
        out.append((CW, r, (t + 1) % sectors))
        return out

    while stack:
        r, t = stack[-1]
        cand = [(d, nr, nt) for (d, nr, nt) in neighbors(r, t) if not visited[nr][nt]]
        if not cand:
            stack.pop()
            continue

        d, nr, nt = rnd.choice(cand)
        cells[r][t] |= d
        cells[nr][nt] |= OPP[d]
        visited[nr][nt] = True
        stack.append((nr, nt))

    return PolarMaze(rings=rings, sectors=sectors, start_ring=start_ring, cells=cells)

def solve_polar_maze_bfs(
    maze: PolarMaze,
    entrance_sector: int,
    exit_sector: int,
) -> list[tuple[int, int]]:
    """
    入口(外周)→出口(外周)の最短経路をBFSで返す。
    経路は [(r,t), (r,t), ...] のセル列。
    """
    R, K, s0 = maze.rings, maze.sectors, maze.start_ring
    start = (R - 1, entrance_sector % K)
    goal = (R - 1, exit_sector % K)

    q = deque([start])
    prev = {start: None}

    def iter_neighbors(r, t):
        bits = maze.cells[r][t]
        if bits & IN and r - 1 >= s0:
            yield (r - 1, t)
        if bits & OUT and r + 1 < R:
            yield (r + 1, t)
        if bits & CCW:
            yield (r, (t - 1) % K)
        if bits & CW:
            yield (r, (t + 1) % K)

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, t = cur
        for nxt in iter_neighbors(r, t):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)

    if goal not in prev:
        # 完全迷路なので通常あり得ないが、start_ring制限などであり得るので保険
        return []

    # 復元
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _draw_rotated_square(draw: ImageDraw.ImageDraw, x: float, y: float, ang: float, half: float, color):
    urx, ury = cos(ang), sin(ang)        # radial
    utx, uty = -sin(ang), cos(ang)       # tangential
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
    exit_sector: int | None = None,
    blank_extra_px: int = 18,     # 中央の見た目余白
    overlay_solution: list[tuple[int, int]] | None = None,
    overlay_width_mul: float = 1.2,  # 解の線幅倍率（色は変えない）
) -> Image.Image:
    """
    円迷路PNGを描画。入口/出口は外周。中心は start_ring より内側を空白。
    overlay_solution を渡すと、解を通路中心線として重ね描きする（同色）。
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

    # start_ring境界（内周壁）
    for t in range(K):
        arc_wall[s0][t] = True

    # 中間境界
    for rb in range(s0 + 1, R + 1):
        r = rb - 1
        for t in range(K):
            arc_wall[rb][t] = not (maze.cells[r][t] & OUT)

    # 外周：入口/出口を穴に
    arc_wall[R][entrance_sector] = False
    arc_wall[R][exit_sector] = False

    # 放射壁
    for r in range(s0, R):
        for t in range(K):
            tb = (t + 1) % K
            if not (maze.cells[r][t] & CW):
                radial_wall[r][tb] = True

    # 円弧端の隙間対策：角度微小延長
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
            s = rad_to_deg(a0) - ddeg
            e = rad_to_deg(a1) + ddeg
            draw.arc(bbox, start=s, end=e, fill=fg, width=int(lw))

    # 放射描画（半径方向に延長）
    for r in range(s0, R):
        r0 = blank_extra + r * rt
        r1 = blank_extra + (r + 1) * rt
        for tb in range(K):
            if not radial_wall[r][tb]:
                continue
            ang = 2 * pi * (tb / K)
            # 内側端：最内リング（start_ring境界）では内側に伸ばさない
            r0_draw = r0 if (r == s0) else (r0 - half)

            # 外側端：最外リングでは外側に伸ばさない（←はみ出し対策）
            r1_draw = r1 if (r == R - 1) else (r1 + half)

            p0 = p(max(0.0, r0_draw), ang)
            p1 = p(r1_draw, ang)
            draw.line([p0, p1], fill=fg, width=int(lw))

    # ジョイント（回転正方形）
    arc_touch = [[False for _ in range(K)] for __ in range(R + 1)]
    rad_touch = [[False for _ in range(K)] for __ in range(R + 1)]

    for rb in range(s0, R + 1):
        for t in range(K):
            if arc_wall[rb][t]:
                arc_touch[rb][t] = True
                arc_touch[rb][(t + 1) % K] = True

    for r in range(s0, R):
        for tb in range(K):
            if radial_wall[r][tb]:
                rad_touch[r][tb] = True
                rad_touch[r + 1][tb] = True

    jhalf = (lw / 2.0) * 0.78
    for rb in range(s0, R + 1):
        radius = blank_extra + rb * rt
        for tb in range(K):
            if not (arc_touch[rb][tb] and rad_touch[rb][tb]):
                continue
            ang = 2 * pi * (tb / K)
            x, y = p(radius, ang)
            _draw_rotated_square(draw, x, y, ang, jhalf, fg)

    # 解のオーバーレイ（通路中心線）
    if overlay_solution:
        # セル中心（半径はリング中央、角度はセクタ中央）
        def cell_center(r, t):
            radius = blank_extra + (r + 0.5) * rt
            ang = 2 * pi * ((t + 0.5) / K)
            return p(radius, ang)

        pts = [cell_center(r, t) for (r, t) in overlay_solution]
        sol_w = max(1, int(lw * overlay_width_mul))
        # 線を可視化
        solution_color = (0, 102, 204) 
        draw.line(pts, fill=solution_color, width=lw)

    # AA縮小
    if aa_scale != 1:
        img = img.resize((W // aa_scale, H // aa_scale), resample=Image.Resampling.LANCZOS)

    return img

def save_image_as_a4_pdf(
    img,                 # PIL.Image
    out_pdf_path: str,
    margin_mm: float = 15.0,   # A4余白
):
    """
    PIL.Image を A4 1ページのPDFとして保存する。
    画像は縦横比を保って中央配置。
    """
    page_w, page_h = A4  # pt
    margin = margin_mm * mm

    c = canvas.Canvas(out_pdf_path, pagesize=A4)

    img_w_px, img_h_px = img.size

    # PDF上で使える最大サイズ（pt）
    max_w = page_w - margin * 2
    max_h = page_h - margin * 2

    # px → pt のスケール（縦横比維持）
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
        mask='auto'
    )

    c.showPage()
    c.save()

def main():
    ap = argparse.ArgumentParser(description="Circular maze + solution (outer ports, blank center)")
    ap.add_argument("--rings", type=int, default=15)
    ap.add_argument("--sectors", type=int, default=48, help="EVEN recommended for diametric entrance/exit")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ring-thickness", type=int, default=24)
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--line-width", type=int, default=5)
    ap.add_argument("--aa", type=int, default=3)
    ap.add_argument("--entrance-sector", type=int, default=0)
    ap.add_argument("--exit-sector", type=int, default=None, help="default: entrance + sectors//2")
    ap.add_argument("--blank-ratio", type=float, default=0.33, help="center blank ratio (e.g., 0.33)")
    ap.add_argument("--blank-extra", type=int, default=18, help="extra empty radius in px (visual padding)")
    ap.add_argument("--out", type=str, default="maze.png")
    ap.add_argument("--solution-out", type=str, default="solution.txt")
    ap.add_argument("--overlay-solution", action="store_true", help="also output maze_with_solution.png")
    ap.add_argument("--pdf", type=str, default=None, help="output A4 PDF path")
    args = ap.parse_args()

    # 何リング分を空白にするか（中心からの比率）
    blank_ratio = max(0.0, min(0.9, args.blank_ratio))
    start_ring = int(round(args.rings * blank_ratio))
    start_ring = min(max(0, start_ring), args.rings - 1)

    entrance = args.entrance_sector
    exit_sector = args.exit_sector
    if exit_sector is None:
        exit_sector = (entrance + (args.sectors // 2)) % args.sectors

    maze = generate_polar_maze_with_blank_center(
        args.rings, args.sectors, start_ring=start_ring, seed=args.seed, start_sector=entrance
    )

    solution = solve_polar_maze_bfs(maze, entrance_sector=entrance, exit_sector=exit_sector)

    # 迷路画像
    img = render_circular_maze_png(
        maze,
        ring_thickness_px=args.ring_thickness,
        margin_px=args.margin,
        line_width_px=args.line_width,
        aa_scale=args.aa,
        entrance_sector=entrance,
        exit_sector=exit_sector,
        blank_extra_px=args.blank_extra,
    )
    img.save(args.out)
    print(f"Saved maze: {args.out}")

    # 解のテキスト出力
    with open(args.solution_out, "w", encoding="utf-8") as f:
        f.write(f"# rings={args.rings}, sectors={args.sectors}, start_ring={start_ring}\n")
        f.write(f"# entrance_sector={entrance}, exit_sector={exit_sector}\n")
        f.write(f"# path_len={len(solution)}\n")
        for r, t in solution:
            f.write(f"{r},{t}\n")
    print(f"Saved solution: {args.solution_out} (len={len(solution)})")

    # 解を重ねた画像（任意）
    if args.overlay_solution:
        img2 = render_circular_maze_png(
            maze,
            ring_thickness_px=args.ring_thickness,
            margin_px=args.margin,
            line_width_px=args.line_width,
            aa_scale=args.aa,
            entrance_sector=entrance,
            exit_sector=exit_sector,
            blank_extra_px=args.blank_extra,
            overlay_solution=solution,
            overlay_width_mul=1.35,
        )
        out2 = args.out.rsplit(".", 1)
        out2 = (out2[0] + "_with_solution.png") if len(out2) == 1 else (out2[0] + "_with_solution." + out2[1])
        img2.save(out2)
        print(f"Saved maze+solution: {out2}")

    
    # PDF保存（指定がなければ日時で自動生成）
    pdf_path = args.pdf
    if pdf_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"maze_polar_{ts}.pdf"

    save_image_as_a4_pdf(
        img,
        pdf_path,
        margin_mm=15.0
    )
    print(f"Saved A4 PDF: {pdf_path}")

if __name__ == "__main__":
    main()
