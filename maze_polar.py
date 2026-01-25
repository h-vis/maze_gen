import argparse
import random
from dataclasses import dataclass
from math import cos, sin, pi
from PIL import Image, ImageDraw

# 通路ビット（通れる方向）
IN, OUT, CCW, CW = 1, 2, 4, 8
OPP = {IN: OUT, OUT: IN, CCW: CW, CW: CCW}

@dataclass
class PolarMaze:
    rings: int
    sectors: int
    start_ring: int        # ここより内側は空白（迷路なし）
    cells: list            # cells[r][t]  (0..rings-1, 0..sectors-1) ただし r<start_ring は未使用

def generate_polar_maze_with_blank_center(
    rings: int,
    sectors: int,
    start_ring: int,
    seed: int | None = None,
    start_sector: int = 0,
) -> PolarMaze:
    """
    start_ring より内側（r < start_ring）は迷路を作らない（空白）。
    迷路生成は r in [start_ring .. rings-1] だけでDFSを行う。
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
    # 開始点は外周（入口に自然）
    sr = rings - 1
    st = start_sector
    stack = [(sr, st)]
    visited[sr][st] = True

    def neighbors(r, t):
        out = []
        # 内側に行けるのは start_ring より外側の範囲だけ
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

def _draw_rotated_square(draw: ImageDraw.ImageDraw, x: float, y: float, ang: float, half: float, color):
    """放射方向（ang）と接線方向に平行な回転正方形（ジョイント）"""
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
    exit_sector: int | None = None,   # Noneなら直径対向（sectors//2）
    inner_blank_radius_px: int = 18,  # 中央空白の“見た目”半径（start_ring境界の内側に余白を足す）
):
    """
    入口・出口とも外周（直径対向）。
    start_ring より内側は迷路を描かず、start_ring 境界に円を描いて仕切る。
    """
    R = maze.rings
    K = maze.sectors
    s0 = maze.start_ring

    if exit_sector is None:
        exit_sector = (entrance_sector + (K // 2)) % K

    entrance_sector %= K
    exit_sector %= K

    rt = ring_thickness_px * aa_scale
    mg = margin_px * aa_scale
    lw = line_width_px * aa_scale
    half = lw / 2.0

    # 中央空白の見た目調整：start_ring境界の半径に加算する
    blank_extra = max(0, inner_blank_radius_px) * aa_scale

    # start_ring 境界の半径（ここが“迷路の内周”になる）
    inner_r = blank_extra + s0 * rt
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
    # arc_wall[rb][t] : 半径境界rb（s0..R）上の円弧壁（t..t+1）
    # radial_wall[r][tb] : リングr（s0..R-1）内の放射壁（角度境界tb）
    arc_wall = [[False for _ in range(K)] for __ in range(R + 1)]
    radial_wall = [[False for _ in range(K)] for __ in range(R)]

    # start_ring 境界（rb=s0）は「内周壁」として全面描く（中心空白との仕切り）
    for t in range(K):
        arc_wall[s0][t] = True

    # rb=s0+1..R の円弧壁：内側セル（r=rb-1）がOUTで開いていなければ壁
    for rb in range(s0 + 1, R + 1):
        r = rb - 1
        for t in range(K):
            arc_wall[rb][t] = not (maze.cells[r][t] & OUT)

    # 外周（rb=R）は入口・出口を穴に
    arc_wall[R][entrance_sector] = False
    arc_wall[R][exit_sector] = False

    # 放射壁：セル(r,t) のCWが閉じていれば境界tb=(t+1)が壁
    for r in range(s0, R):
        for t in range(K):
            tb = (t + 1) % K
            if not (maze.cells[r][t] & CW):
                radial_wall[r][tb] = True

    # 円弧の端隙間対策：角度微小延長
    def arc_delta_deg(radius: float) -> float:
        if radius <= 1e-6:
            return 0.0
        delta_rad = (half * 1.05) / radius
        return rad_to_deg(delta_rad)

    # 円弧壁描画：rb=s0..R
    for rb in range(s0, R + 1):
        radius = blank_extra + rb * rt
        if radius <= 0:
            continue
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

    # 放射壁描画：r=s0..R-1（半径方向に少し延長）
    for r in range(s0, R):
        r0 = blank_extra + r * rt
        r1 = blank_extra + (r + 1) * rt
        for tb in range(K):
            if not radial_wall[r][tb]:
                continue
            ang = 2 * pi * (tb / K)
            p0 = p(max(0.0, r0 - half * 1.05), ang)
            p1 = p(r1 + half * 1.05, ang)
            draw.line([p0, p1], fill=fg, width=int(lw))

    # ジョイント（回転正方形）：円弧×放射が両方接する交点だけ
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

    jhalf = (lw / 2.0) * 0.78  # 0.65〜0.9調整

    for rb in range(s0, R + 1):
        radius = blank_extra + rb * rt
        for tb in range(K):
            if not (arc_touch[rb][tb] and rad_touch[rb][tb]):
                continue
            ang = 2 * pi * (tb / K)
            x, y = p(radius, ang)
            _draw_rotated_square(draw, x, y, ang, jhalf, fg)

    # AA縮小
    if aa_scale != 1:
        img = img.resize((W // aa_scale, H // aa_scale), resample=Image.Resampling.LANCZOS)

    return img

def main():
    ap = argparse.ArgumentParser(description="Circular maze with blank center (outer entrance/exit, diametric)")
    ap.add_argument("--rings", type=int, default=15)
    ap.add_argument("--sectors", type=int, default=48, help="EVEN recommended for perfect diametric ports")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ring-thickness", type=int, default=24)
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--line-width", type=int, default=5)
    ap.add_argument("--aa", type=int, default=3)
    ap.add_argument("--entrance-sector", type=int, default=0)
    ap.add_argument("--exit-sector", type=int, default=None)
    ap.add_argument("--blank-ratio", type=float, default=0.33, help="center blank ratio of radius (e.g., 0.33)")
    ap.add_argument("--blank-extra", type=int, default=18, help="extra empty radius (px) inside start_ring boundary")
    ap.add_argument("--out", type=str, default="circular_maze.png")
    args = ap.parse_args()

    # 中心から何リング分を空白にするか（半径比で決める）
    # 例: 0.33 → start_ring ≈ rings*0.33
    start_ring = int(round(args.rings * max(0.0, min(0.9, args.blank_ratio))))
    start_ring = min(max(0, start_ring), args.rings - 1)

    maze = generate_polar_maze_with_blank_center(
        args.rings,
        args.sectors,
        start_ring=start_ring,
        seed=args.seed,
        start_sector=args.entrance_sector,
    )
    img = render_circular_maze_png(
        maze,
        ring_thickness_px=args.ring_thickness,
        margin_px=args.margin,
        line_width_px=args.line_width,
        aa_scale=args.aa,
        entrance_sector=args.entrance_sector,
        exit_sector=args.exit_sector,
        inner_blank_radius_px=args.blank_extra,
    )
    img.save(args.out)
    print(f"Saved: {args.out}  (start_ring={start_ring}/{args.rings})")

if __name__ == "__main__":
    main()
