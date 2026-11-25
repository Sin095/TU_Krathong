#!/usr/bin/env python3
"""
Loy Krathong ASCII animation v2  (ลอยกระทง)
--------------------------------------------
Libraries : numpy, Pillow   (+ ffmpeg เพื่อสร้าง .mp4 ไม่งั้นจะได้ .gif)
Usage     : python krathong_ascii.py              -> krathong.mp4 (หรือ .gif)
            python krathong_ascii.py out.gif
            python krathong_ascii.py --preview    -> preview.png (เฟรมเดียว)

ทุกอย่างคำนวณใน "พิกัดดีไซน์" 540x960 แล้วขยาย S เท่าตอนวาดจริง (1080x1920)
กระทงตัวเอกอยู่ตรงกลาง กระทงเล็กประกอบเบื้องหลัง วนลูปได้พอดี
"""
import glob
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- config
DW, DH = 540, 960            # พิกัดดีไซน์
S = 2                        # ตัวคูณขยาย -> 1080x1920
W, H = DW * S, DH * S
CWd, CHd = 4, 7              # ขนาดช่องตัวอักษร (พิกัดดีไซน์) -> 8x14 px
COLS, ROWS = DW // CWd, DH // CHd
FPS, SECONDS = 30, 10
LOOP_FRAMES = FPS * 5         # ฉากหลักวนทุก 5 วิ, โคมลอยวนทุก 10 วิ
FRAMES = FPS * SECONDS
TAU = 2 * math.pi
HORIZON = 0.50 * DH
BG = (2, 3, 8)

HERE = os.path.dirname(os.path.abspath(__file__))


def find_font(size, prefer=()):
    """หาฟอนต์ตัวหนา: โฟลเดอร์เดียวกับสคริปต์ -> Linux -> macOS -> Windows
    (ถ้าไม่เจอเลยจะใช้ฟอนต์จิ๋วของ Pillow ซึ่งภาพจะดูจางและเล็ก — แนะนำให้วาง DejaVuSansMono-Bold.ttf ไว้ข้างสคริปต์)"""
    cands = []
    for name, idx in list(prefer) + [("DejaVuSansMono-Bold.ttf", 0), ("DejaVuSansMono.ttf", 0)]:
        cands.append((os.path.join(HERE, name), idx))
        cands += [(q, idx) for q in glob.glob("/usr/share/fonts/**/" + name, recursive=True)]
    cands += [("/System/Library/Fonts/Menlo.ttc", 1), ("/System/Library/Fonts/Monaco.ttf", 0),
              ("/System/Library/Fonts/Supplemental/Courier New Bold.ttf", 0),
              ("C:/Windows/Fonts/consolab.ttf", 0), ("C:/Windows/Fonts/lucon.ttf", 0)]
    cands += [(q, 0) for q in glob.glob("/usr/share/fonts/**/*Mono*.ttf", recursive=True)]
    for path, idx in cands:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except OSError:
                continue
    print("WARNING: ไม่พบฟอนต์ monospace ตัวหนา — ภาพจะดูจางกว่าที่ควร (วาง DejaVuSansMono-Bold.ttf ไว้ข้างสคริปต์)")
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


xs = (np.arange(COLS) + 0.5) * CWd
ys = (np.arange(ROWS) + 0.5) * CHd
X, Y = np.meshgrid(xs, ys)
ROWIDX = (Y / CHd).astype(int)
COLIDX = (X / CWd).astype(int)

rng = np.random.default_rng(11)
RND = rng.random((ROWS, COLS)).astype(np.float32)
RND2 = rng.random((ROWS, COLS)).astype(np.float32)
STARS = [(rng.uniform(0, DW), rng.uniform(0, HORIZON * 0.96),
          int(rng.integers(1, 4)), rng.uniform(0, TAU), rng.random())
         for _ in range(260)]
# โคมลอยอยู่เหนือโดมทั้งหมด (ยอดโดมอยู่ที่ y≈215): (x, y เริ่ม, ขนาด, เฟส, รอบส่ายซ้ายขวา, รอบโยกขึ้นลง ต่อ 10 วินาที)
LANTERNS = [(0.12 * DW, 186, 1.10, 0.4, 1, 1), (0.30 * DW, 150, 0.90, 2.2, 2, 1),
            (0.58 * DW, 178, 1.00, 4.1, 1, 1), (0.945 * DW, 196, 1.15, 5.3, 2, 1)]
SPARKS = [(rng.uniform(-1, 1), float(rng.choice([0.5, 0.75, 1.0, 1.25])), int(rng.integers(0, 3))) for _ in range(22)]


# ---------------------------------------------------------------- canvas
class Canvas:
    def __init__(self):
        self.ch = np.full((ROWS, COLS), " ", dtype="<U1")
        self.rgb = np.zeros((ROWS, COLS, 3), np.float32)

    def put(self, mask, chars, lum, color, gain=1.0):
        lum = np.broadcast_to(np.asarray(lum, np.float32), mask.shape)
        m = mask & (chars != " ")
        bright = np.clip(0.45 + 0.65 * lum, 0, 1.25) * gain
        col = np.asarray(color, np.float32) * bright[..., None]
        self.ch[m] = chars[m]
        self.rgb[m] = col[m]

    def paint(self, mask, lum, color, ramp, gain=1.0):
        lum = np.broadcast_to(np.asarray(lum, np.float32), mask.shape)
        n = len(ramp)
        idx = np.clip((lum * n).astype(np.int32), 0, n - 1)
        self.put(mask, np.array(list(ramp))[idx], lum, color, gain)

    def clear(self, mask):
        self.ch[mask] = " "
        self.rgb[mask] = 0

    def dot(self, x, y, ch, color):
        r, c = int(y // CHd), int(x // CWd)
        if 0 <= r < ROWS and 0 <= c < COLS:
            self.ch[r, c] = ch
            self.rgb[r, c] = color


def tri(cx, ay, by, hw):
    return (Y >= ay) & (Y <= by) & (np.abs(X - cx) <= hw * (Y - ay) / (by - ay))


# ---------------------------------------------------------------- sky
def draw_haze(c, t):
    """ฟ้าเรืองส้มใกล้ขอบฟ้า (จุดตัวอักษรเบาๆ)"""
    k = np.clip((HORIZON - Y) / 330.0, 0, 1)
    haze = np.exp(-3.0 * k) * (Y < HORIZON)
    haze = haze + 0.32 * np.exp(-((X - DW / 2) / 240) ** 2) * np.exp(-np.clip(HORIZON - Y, 0, None) / 110) * (Y < HORIZON)
    tw = 0.7 + 0.3 * np.sin(TAU * t + RND * TAU)
    vis = (RND2 < haze * 0.75) & (haze > 0.08)
    lum = haze * tw
    c.paint(vis, lum * 1.3, (230, 120, 45), "..:'`", gain=0.9)
    # ม่วงจางๆ ด้านบน
    top = (Y < HORIZON * 0.5) & (RND2 > 0.985)
    c.put(top, np.full(top.shape, "."), 0.3 * tw, (120, 80, 200))


def draw_stars(c, t):
    for x, y, f, ph, k in STARS:
        tw = 0.5 + 0.5 * math.sin(TAU * f * t + ph)
        if tw < 0.2:
            continue
        ch = "." if tw < 0.5 else ("+" if tw < 0.82 else "*")
        if k > 0.93 and tw > 0.6:
            ch = "o"
        tint = (255, 238, 190) if k < 0.6 else ((255, 150, 70) if k < 0.85 else (255, 170, 220))
        c.dot(x, y, ch, np.array(tint) * (0.45 + 0.6 * tw))


def draw_moon(c, t):
    mx, my, mr = 0.80 * DW, 0.14 * DH, 54
    d = np.hypot(X - mx, Y - my)
    tex = 0.78 + 0.22 * np.sin(X * 0.21 + Y * 0.17) * np.cos(Y * 0.31 - X * 0.07 + TAU * t)
    lum = np.clip(tex * (1.08 - 0.25 * (d / mr) ** 3), 0, 1)
    c.paint(d < mr, lum, (255, 246, 215), "o08&%#@@", gain=1.15)
    hl = np.clip(1 - (d - mr) / (mr * 0.95), 0, 1) ** 2
    hl = hl * (0.7 + 0.3 * np.sin(X * 0.5 + Y * 0.3 + TAU * t))
    c.paint((d >= mr) & (hl > 0.1), hl, (255, 225, 150), " .:'`")


# ---------------------------------------------------------------- stadium dome & nagas
def catmull(pts, n=14):
    p = np.array(pts, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for tt in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * tt ** 3))
    out.append(p[-2])
    return np.array(out)


NAGA_K = 1.1          # ขนาดพญานาค (ยิ่งมากยิ่งใหญ่)
NAGA_R = 1.9          # ความอ้วนของลำตัว (ตัวอ้วนกลมแบบในภาพตัวอย่าง)
NAGA_SHIFT = 26       # ดันพญานาคออกจากกึ่งกลางภาพ (พิกัดดีไซน์)
# คอโค้งตัว S -> ลงถึงพื้น -> ลำตัวขดนูนเลื้อยออกไปทางขอบภาพ -> ปลายหางเชิด (จุดแรกคือโคนคอใต้หัว)
NAGA_PTS = [(240, 396), (244, 409), (240, 423), (231, 436), (226, 449), (216, 462), (204, 470),
            (192, 462), (180, 447), (168, 434), (154, 430), (142, 437), (133, 451), (124, 464),
            (110, 473), (96, 474)]
HEAD_DY = -10         # ยกหัวขึ้นจากโคนคอ
HEAD_S = 1.15         # ขนาดหัวเทียบกับลำตัว
HEAD_TILT = math.radians(26)   # ก้มหน้าลงเล็กน้อยเข้าหากึ่งกลาง


def _ell(u, v, cu, cv, ru, rv):
    return ((u - cu) / ru) ** 2 + ((v - cv) / rv) ** 2 < 1


def build_naga(mirror):
    """รูปทรงพญานาค 1 ตัว: ลำตัวจากเส้นโค้ง + หัว (กะโหลก จมูก ปากอ้า ขากรรไกร ตา) + หงอนกระหนก + เครา"""
    K = NAGA_K
    P = catmull(NAGA_PTS)
    P[:, 0] -= NAGA_SHIFT
    if mirror:
        P[:, 0] = DW - P[:, 0]
    P[:, 0] = DW / 2 + (P[:, 0] - DW / 2) * K
    P[:, 1] = 482 - (482 - P[:, 1]) * K
    n = len(P)
    sp_ = np.linspace(0, 1, n)
    r = np.interp(sp_, [0, 0.1, 0.35, 0.55, 0.75, 0.9, 1.0],
                  [5.2, 4.7, 4.6, 6.0, 5.0, 2.6, 1.0]) * K * NAGA_R
    best = np.full((ROWS, COLS), 9.0, np.float32)
    sarr = np.zeros((ROWS, COLS), np.float32)
    for i in range(n):
        d = np.hypot(X - P[i, 0], Y - P[i, 1]) / r[i]
        upd = d < best
        best = np.where(upd, d, best)
        sarr = np.where(upd, sp_[i], sarr)

    # --- หัว: พิกัดท้องถิ่น u (ไปทางที่หันหน้า = เข้าหากึ่งกลาง), v (ลงล่าง)
    sign = -1 if mirror else 1
    ox, oy = P[0][0], P[0][1] + HEAD_DY * K
    u0, v0 = sign * (X - ox) / (K * HEAD_S), (Y - oy) / (K * HEAD_S)
    u = u0 * math.cos(HEAD_TILT) + v0 * math.sin(HEAD_TILT)
    v = -u0 * math.sin(HEAD_TILT) + v0 * math.cos(HEAD_TILT)
    head = (_ell(u, v, 0, -1, 11, 12)           # กะโหลก
            | _ell(u, v, 14, 2, 14, 5.5)        # จมูก/ปากบน
            | _ell(u, v, 3, 12, 8, 9)           # แก้ม
            | _ell(u, v, 11, 19, 12, 4)         # ขากรรไกรล่าง
            | _ell(u, v, 29, -1.5, 3.2, 3.2))   # ปุ่มจมูก
    mouth = (u > 6) & (u < 31) & (v > 7.5) & (v < 15)     # ปากอ้า (ช่องมืด)
    eye = _ell(u, v, 5, -2, 3.2, 4.2) & head

    # หงอนกระหนก: ชี้ขึ้นและโค้งไปข้างหน้า (ทิศเดียวกับใบหน้า) ยาวสุดเป็นเปลวเรียวแหลม + ครีบเล็กที่แก้ม
    crown = np.zeros((ROWS, COLS), bool)
    for ou, ov, au, av, L, r0 in ((-3, -8, 0.05, -1, 40, 6.0), (2, -9, 0.5, -1, 28, 4.6),
                                  (-7, -5, -0.25, -1, 26, 4.6), (-7, 7, -1, 0.45, 14, 3.4),
                                  (-5, 13, -0.9, 0.9, 12, 3.0)):
        nr = math.hypot(au, av)
        for k in np.linspace(0.05, 1.0, 14):
            fwd = 0.25 * L * k * k if av < 0 else 0.0       # โค้งไปข้างหน้า
            cu = ou + au / nr * L * k + fwd
            cv = ov + av / nr * L * k
            crown |= np.hypot(u - cu, v - cv) < (r0 - 0.9 * r0 * k)
    for k in np.linspace(0, 1, 7):                            # เคราใต้คาง
        crown |= np.hypot(u - (8 - 5 * k), v - (22 + 13 * k)) < (2.6 - 1.2 * k)

    body = ((best < 1) | head) & ~mouth
    cross = np.where(head, 0.30, np.clip(best, 0, 1))
    return dict(mask=body & ~crown & ~eye, crown=crown & ~mouth, eye=eye & ~mouth, s=sarr, cross=cross)


NAGAS = [build_naga(False), build_naga(True)]
NAG_ALL = np.zeros((ROWS, COLS), bool)
for _nb in NAGAS:
    NAG_ALL |= _nb["mask"] | _nb["crown"] | _nb["eye"]
REF_ROW = np.clip(((2 * HORIZON - Y) / CHd).astype(int), 0, ROWS - 1)

DOME_HW, DOME_HT = 290, 265      # โดมสูงขึ้น เหลือพื้นที่มืดโล่งด้านบนไว้ใส่ข้อความ


def draw_stadium(c, t):
    flick = 0.9 + 0.1 * math.sin(TAU * 3 * t)
    cx = DW / 2
    shape = (np.abs(X - cx) / DOME_HW) ** 3 + ((HORIZON - Y) / DOME_HT) ** 2
    dome = (shape < 1) & (Y <= HORIZON)
    c.clear(dome)

    # เสาไฟสูง 2 ข้าง (เงามืด)
    for xt in (34, DW - 34):
        pole = (np.abs(X - xt) < 2.2) & (Y > 262) & (Y <= HORIZON)
        c.put(pole, np.full(pole.shape, "|"), 0.28, (50, 42, 78))
        head = (np.abs(X - xt) < 13) & (Y > 244) & (Y < 264)
        c.clear(head)
        c.put(head, np.full(head.shape, "#"), 0.2, (44, 36, 68))
        lamps = head & (COLIDX % 2 == 0) & (ROWIDX % 2 == 0)
        c.put(lamps, np.full(lamps.shape, "o"), 0.35 * flick, (255, 170, 70))

    # ตัวโดม: มืดครึ้ม มีเมฆหม่นลอยเอื่อยๆ (วนรอบละ 10 วินาที)
    tt = t / 2
    cl = (0.5 + 0.28 * np.sin(X * 0.040 + 1.7 * np.sin(Y * 0.028 + TAU * tt) - TAU * tt)
          + 0.22 * np.sin(X * 0.095 - Y * 0.050 + TAU * 2 * tt + RND * 1.2))
    cl = np.clip(cl, 0, 1) ** 1.4
    zone = np.exp(-(((X - cx) / 190.0) ** 4 + ((Y - 300) / 62.0) ** 2))   # พื้นที่ว่างสำหรับข้อความ
    cl = cl * (1 - 0.8 * zone)
    base_c, cloud_c = np.array([9, 8, 20], np.float32), np.array([50, 44, 74], np.float32)
    dcol = base_c * (1 - cl)[..., None] + cloud_c * cl[..., None]
    dch = np.array(list(":;%#"))[np.clip((cl * 4).astype(int), 0, 3)]
    c.put(dome, dch, 0.20, dcol)
    ribs = dome & (zone < 0.12) & ((COLIDX % 7 == 0) | (np.abs(shape - 0.55) < 0.03) | (np.abs(shape - 0.8) < 0.03))
    c.put(ribs, np.where(COLIDX % 7 == 0, "|", "-"), 0.14, (34, 29, 56))

    # เสาโครงหลังคาโผล่เหนือขอบโดม
    for xm in np.linspace(cx - DOME_HW + 25, cx + DOME_HW - 25, 9):
        u = abs(xm - cx) / DOME_HW
        top = HORIZON - DOME_HT * math.sqrt(max(0.0, 1 - u ** 3))
        m = (np.abs(X - xm) < 1.5) & (Y > top - 26) & (Y <= top + 2)
        c.put(m, np.where(ROWIDX % 2 == 0, "|", "!"), 0.22, (40, 34, 64))

    # แสงส้มจากฟ้าด้านหลังเกาะขอบโดม
    rim = dome & (shape > 0.86)
    rl = (0.50 + 0.30 * np.clip((Y - 330) / 150, 0, 1)) * flick
    c.paint(rim, rl, (235, 125, 40), "-=+%", gain=0.72)

    # ฐานสนามสะท้อนแสงอุ่น + ไฟเล็กๆ ระยิบ
    low = dome & (Y > HORIZON - 26)
    c.put(low, np.where(COLIDX % 2 == 0, "|", "I"), 0.04 + 0.05 * (Y - (HORIZON - 26)) / 26, (130, 72, 30), gain=0.7)
    twk = low & (RND > 0.9) & (np.sin(TAU * 2 * t + RND2 * TAU) > -0.2)
    c.put(twk, np.full(twk.shape, "."), 0.9, (255, 190, 90))

    # ทางเข้าเรืองแสง
    door = (np.abs(X - cx) < 13) & (Y > 458) & (Y <= HORIZON)
    c.paint(door, 0.85 * flick, (255, 205, 100), "o0@@")

    # พญานาคทอง (เงาแสงวิ่งตามลำตัว)
    for nb in NAGAS:
        sa, cr = nb["s"], nb["cross"]
        glint = (0.5 + 0.5 * np.sin(TAU * (3 * sa - 2 * t))) ** 3
        lum = (0.58 + 0.34 * (1 - cr ** 2) + 0.22 * glint) * (0.86 + 0.14 * np.cos(sa * 160 + cr * 4))
        c.paint(nb["mask"], lum, (255, 96, 10), "-=o%#&8@", gain=1.0)
        c.paint(nb["crown"], np.full(nb["crown"].shape, 0.72), (255, 175, 40), "*#@8", gain=1.0)
        c.put(nb["eye"], np.full(nb["eye"].shape, "@"), 1.0, (255, 250, 215), gain=1.1)


# ---------------------------------------------------------------- lettering on the dome
TEXT_STR, TEXT_CY, TEXT_CAP, TEXT_GAP = "TU", 288, 58, 20      # ข้อความ / กึ่งกลางแนวตั้ง / ความสูงตัวอักษร / ช่องไฟ


def build_text():
    f = find_font(int(TEXT_CAP / 0.73), prefer=[("DejaVuSans-Bold.ttf", 0), ("/System/Library/Fonts/Helvetica.ttc", 1)])
    img = Image.new("L", (DW, DH), 0)
    d = ImageDraw.Draw(img)
    if f is None:
        return np.zeros((ROWS, COLS), bool), np.zeros((ROWS, COLS), np.float32)
    ws = [d.textlength(ch, font=f) for ch in TEXT_STR]
    x = DW / 2 - (sum(ws) + TEXT_GAP * (len(ws) - 1)) / 2
    for ch, w in zip(TEXT_STR, ws):
        d.text((x, TEXT_CY + TEXT_CAP / 2), ch, font=f, fill=255, anchor="ls")
        x += w + TEXT_GAP
    yi, xi = np.clip(Y.astype(int), 0, DH - 1), np.clip(X.astype(int), 0, DW - 1)
    sharp = np.asarray(img, np.float32)[yi, xi] / 255
    soft = np.asarray(img.filter(ImageFilter.GaussianBlur(9)), np.float32)[yi, xi] / 255
    return sharp > 0.5, soft / max(float(soft.max()), 1e-6)


TEXT_MASK, TEXT_HALO = build_text()


def draw_text(c, t):
    pulse = 0.85 + 0.15 * math.sin(TAU * t)
    halo = (TEXT_HALO > 0.12) & ~TEXT_MASK
    c.paint(halo, TEXT_HALO * 0.8 * pulse, (255, 190, 110), " .:'", gain=0.55)
    c.paint(TEXT_MASK, np.full(TEXT_MASK.shape, 0.8 * pulse), (255, 226, 172), "&8@#", gain=0.72)


# ---------------------------------------------------------------- krathong data
HERO = (0.5 * DW, 0.815 * DH, 1.55)
SMALL = [(0.16, 0.640, 0.30), (0.84, 0.655, 0.34), (0.31, 0.585, 0.20),
         (0.70, 0.598, 0.23), (0.07, 0.775, 0.42)]


def krathong_positions(t):
    hx, hy, hs = HERO
    out = [(hx + 6 * math.sin(TAU * t), hy + 3 * math.sin(TAU * 2 * t), hs, 0.0)]
    for i, (fx, fy, s) in enumerate(SMALL):
        ph = i * 1.3
        out.append((fx * DW + 6 * math.sin(TAU * t + ph),
                    fy * DH + 2 * math.sin(TAU * 2 * t + ph), s, ph))
    return sorted(out, key=lambda k: k[1])


def glow_field(t, kr):
    g = np.zeros((ROWS, COLS), np.float32)
    for x, y, s, ph in kr:
        k = 0.88 + 0.12 * math.sin(TAU * 3 * t + ph)
        sg = (100 * s + 25) if s > 1 else (60 * s + 12)
        amp = 1.0 if s > 1 else 0.55 + 0.4 * s
        g += amp * k * np.exp(-((X - x) ** 2 + (Y - (y - 85 * s)) ** 2) / (2 * sg * sg))
    for nx_ in (DW / 2 - 100, DW / 2 + 100):
        g += 0.12 * (0.92 + 0.08 * math.sin(TAU * 3 * t)) * np.exp(-((X - nx_) ** 2 + (Y - 440) ** 2) / (2 * 48 * 48))
    for x, y, s, a, ph in lantern_state(t):
        sg = 38 * s + 8
        g += 0.45 * a * (0.9 + 0.1 * math.sin(TAU * 3 * t + ph)) * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sg * sg))
    return g


def draw_water(c, t, kr, g):
    wm = Y >= HORIZON
    depth = np.clip((Y - HORIZON) / (DH - HORIZON), 0, 1)
    r1 = np.sin(X * 0.045 + Y * 0.10 - TAU * 2 * t)
    r2 = np.sin(X * 0.11 - Y * 0.07 + TAU * 3 * t)
    rip = 0.5 + 0.25 * r1 + 0.25 * r2
    chars = np.where(r1 > 0.6, "~", np.where(r1 > 0.25, "/", np.where(r1 < -0.6, "\\",
            np.where(r2 > 0.3, "-", np.where(r2 < -0.3, "_", ".")))))
    top, bot = np.array([30, 135, 135], np.float32), np.array([6, 30, 78], np.float32)
    col = top * (1 - depth)[..., None] + bot * depth[..., None]
    warm = np.array([235, 130, 40], np.float32)
    mix = np.clip((1 - depth) ** 4 * 0.45 + g * 0.30, 0, 0.7)[..., None]
    col = col * (1 - mix) + warm * mix
    c.put(wm, chars, rip * (0.55 - 0.25 * depth), col)

    # เงาพระจันทร์
    mx = 0.80 * DW
    wx = np.exp(-((X - mx) / (9 + 36 * depth)) ** 2)
    sh = 0.5 + 0.5 * np.sin(Y * 0.35 - TAU * 2 * t + X * 0.05)
    val = wx * sh * (1 - 0.7 * depth) * 0.8
    c.paint(wm & (val > 0.25), val, (255, 235, 180), "..-:=", gain=0.95)

    # เงาสะท้อนพญานาค: พลิกภาพจากเหนือเส้นขอบน้ำ + คลื่นทำให้เงาสั่นและขาดเป็นช่วงๆ
    wob = np.rint(1.4 * np.sin(Y * 0.17 - TAU * 2 * t)).astype(int)
    refm = NAG_ALL[REF_ROW, np.clip(COLIDX + wob, 0, COLS - 1)] & wm
    shim = 0.5 + 0.5 * np.sin(Y * 0.85 - TAU * 3 * t + X * 0.03)
    nl = np.exp(-(Y - HORIZON) / 260.0) * (0.62 + 0.38 * shim)
    c.paint(refm & (shim > 0.06) & (nl > 0.1), nl * 1.25, (255, 120, 24), "-=+o%#&", gain=1.1)

    # แสงสะท้อนใต้กระทง
    for x, y, s, ph in kr:
        w = 55 * s + 8
        shim = 0.5 + 0.5 * np.sin(Y * 0.5 - TAU * 3 * t + ph + X * 0.04)
        rf = np.exp(-((X - x) / w) ** 2) * np.exp(-np.clip(Y - (y + 30 * s), 0, None) / (70 * s + 12)) * (Y > y + 30 * s)
        rf = rf * shim * (0.9 if s > 1 else 0.6)
        c.paint(wm & (rf > 0.2), rf, (255, 170, 60), ".-=+*", gain=1.1)

    # ประกายบนผิวน้ำ
    gl = wm & (RND > 0.975) & (np.sin(TAU * 2 * t + RND2 * TAU * 3) > 0.55)
    gl_col = np.where((g > 0.15)[..., None], np.array([255, 215, 120]), np.array([170, 235, 220]))
    c.ch[gl] = "+"
    c.rgb[gl] = gl_col[gl] * (0.7 + 0.5 * np.clip(g, 0, 1))[gl][:, None]


def apply_glow(c, g):
    c.rgb *= (1 + 1.5 * g)[..., None]
    c.rgb += (np.array([60, 28, 6]) * g[..., None]) * (c.ch != " ")[..., None]
    empty = (c.ch == " ") & (g > 0.1)
    idx = np.clip(((g - 0.1) / 0.5 * 4).astype(int), 0, 3)
    c.ch[empty] = np.array(list(".:;i"))[idx][empty]
    c.rgb[empty] = (np.array([255, 150, 50]) * np.clip(g, 0, 1)[..., None])[empty]


# ---------------------------------------------------------------- lanterns
RISE = 110  # โคมลอยสูงขึ้นกี่พิกเซล (พิกัดดีไซน์) ตลอด 10 วินาที


def lantern_state(t):
    """โคมลอยขึ้นช้าๆ พร้อมโยกซ้ายขวา/ขึ้นลงเล็กน้อย รอบละ 10 วินาที (t วิ่ง 0..2)
    ช่วงท้ายคลิปมีโคมอีกชุดจางเข้ามาจากตำแหน่งต่ำกว่า แทนชุดที่ลอยสูงไปแล้ว ให้วนลูปเนียน"""
    u = t / 2
    w = np.clip((u - 0.78) / 0.22, 0, 1)
    w = float(w * w * (3 - 2 * w))
    out = []
    for x0, y0, sc, ph, nx, ny in LANTERNS:
        x = x0 + 13 * math.sin(TAU * nx * u + ph)
        bob = 6 * math.sin(TAU * ny * u + ph * 1.7 + 1.0)
        out.append((x, y0 + bob - RISE * u, sc, 1.0 - w, ph))
        out.append((x, y0 + bob + RISE * (1 - u), sc, w, ph))
    return out


def draw_lanterns(c, t):
    for x, y, s, a, ph in lantern_state(t):
        if a <= 0.01:
            continue
        dx, dy = (X - x) / s, (Y - y) / s
        r = np.hypot(dx / 14, dy / 21)
        fl = (0.88 + 0.12 * math.sin(TAU * 3 * t + ph)) * a
        c.paint((r < 2.6) & (r >= 1), 0.28 * (2.6 - r) * a, (255, 140, 40), " .:'")
        lum = (1.0 - 0.5 * r + 0.25 * np.clip(dy / 21, 0, 1)) * fl
        c.paint(r < 1, lum, (255, 105, 28), "o#&8@", gain=0.95)
        edge = (r < 1) & (r > 0.78)
        c.put(edge, np.full(edge.shape, "#"), 0.9 * fl, (255, 150, 55), gain=0.95)

# ---------------------------------------------------------------- the krathong
def draw_krathong(c, cx, cy, s, t, ph, hero=False):
    dx, dy = (X - cx) / s, (Y - cy) / s

    # วงแสงนุ่มรอบกระทง: หรี่น้ำเบาๆ ขอบไล่ระดับ แล้วเติมสีส้มอุ่นเหมือนแสงเทียนตกกระทบน้ำ
    hr = np.minimum(np.hypot(dx / 126, dy / 38), np.hypot(dx / 84, (dy + 52) / 60))
    soft = np.clip((1.35 - hr) / 0.8, 0, 1)
    soft = soft * soft * (3 - 2 * soft)                      # smoothstep
    wfl = 0.85 + 0.15 * math.sin(TAU * 3 * t + ph)
    warm = np.array([255, 140, 45], np.float32)
    c.rgb *= (1 - 0.55 * soft)[..., None]
    c.rgb += warm * (0.30 * soft * wfl)[..., None] * (c.ch != " ")[..., None]
    fill = (c.ch == " ") & (soft > 0.25) & (RND2 > 0.5)
    c.ch[fill] = np.where(RND[fill] > 0.5, ".", ":")
    c.rgb[fill] = warm * (0.55 * soft[fill] * wfl)[:, None]

    # ฐานใบตอง
    r = np.hypot(dx / 120, dy / 32)
    ang = np.arctan2(dy / 32, dx / 120)
    base = r < 1
    rim = base & (r > 0.84)
    leaf = np.where(np.cos(ang * 14) > 0, "/", "\\")
    c.put(rim, leaf, 0.75 + 0.25 * np.cos(ang * 7 + 1), (95, 235, 115), gain=1.1)
    inner = base & ~rim
    c.paint(inner, 0.28 + 0.22 * np.cos(ang * 5), (40, 150, 70), ".:;'`")

    # ดาวเรืองส้ม-เหลืองเป็นวงล้อมกลีบ
    mar = base & (r > 0.60) & (r <= 0.84)
    mlum = 0.55 + 0.4 * np.sin(ang * 22) * np.cos(ang * 9)
    c.paint(mar, mlum, (255, 175, 35), "o0O@&8", gain=1.1)

    # กลีบบัวซ้อนชั้น (นอก -> ใน)
    layers = [(112, 22, -4, 10, (255, 90, 165), False),
              (88, 31, -15, 8, (255, 145, 205), True),
              (60, 36, -27, 7, (255, 205, 232), True),
              (30, 26, -40, 5, (255, 244, 250), True)]
    for rx, ry, ocy, n, color, upper in layers:
        ex, ey = dx / rx, (dy - ocy) / ry
        rr = np.hypot(ex, ey)
        th = np.arctan2(ey, ex)
        cosn = np.abs(np.cos(n * th))
        shape = 0.55 + 0.45 * cosn
        m = rr < shape
        if upper:
            m &= ey < 0.25
        lum = 0.40 + 0.55 * cosn * (0.4 + 0.6 * rr)
        gap = m & (cosn < 0.2)
        c.paint(m & ~gap, lum, color, "o0O@&8", gain=1.1)
        c.paint(gap, np.full(gap.shape, 0.25), np.array(color) * 0.55, ".:")

    # ธูปเทียน
    for sx in (-30, 30):
        px = sx + (-dy - 15) * 0.10 * np.sign(sx)
        stick = (np.abs(dx - px) < 1.6) & (dy > -74) & (dy < -22)
        c.put(stick, np.full(stick.shape, "|"), 0.55, (205, 150, 85))
        tip = np.hypot(dx - (sx + 5.5 * np.sign(sx)), dy + 76) < 3.2
        c.put(tip, np.full(tip.shape, "o"), 1.0, (255, 85, 40), gain=1.3)
    candle = (np.abs(dx) < 4.4) & (dy > -82) & (dy < -34)
    c.put(candle, np.where(ROWIDX % 2 == 0, "I", "l"), 0.9, (252, 238, 205))

    # เปลวเทียน
    fl = 0.9 + 0.1 * math.sin(TAU * 3 * t + ph)
    fx = dx - 2.8 * math.sin(TAU * 2 * t + ph) * (-(dy + 94) / 20 + 0.5)
    r4 = np.hypot(fx / 7.5, (dy + 94) / (13 * fl))
    c.paint(r4 < 1.5, 1.15 - 0.55 * r4, (255, 200, 70), ".*#&8@", gain=1.3)
    c.paint(r4 < 0.55, np.full(r4.shape, 1.0), (255, 250, 200), "@@8&", gain=1.3)

    if hero:
        # ประกายไฟลอยขึ้นจากเปลวเทียน (วนลูปได้)
        for i, (ox, sp, kind) in enumerate(SPARKS):
            phs = (t * 2 * sp + i / len(SPARKS)) % 1.0
            if phs > 0.95:
                continue
            sxp = cx + s * (ox * 55 * phs + 7 * math.sin(TAU * (phs * 2 + i)))
            syp = cy + s * (-96 - 190 * phs)
            fade = 1 - phs
            ch = ("*", "+", ".")[kind] if fade > 0.35 else "."
            c.dot(sxp, syp, ch, np.array([255, 200 - 90 * phs, 70 - 40 * phs]) * (0.5 + 0.6 * fade))


# ---------------------------------------------------------------- frame
def scene(t):
    c = Canvas()
    kr = krathong_positions(t)
    g = glow_field(t, kr)
    draw_haze(c, t)
    draw_stars(c, t)
    draw_moon(c, t)
    draw_lanterns(c, t)
    draw_stadium(c, t)
    draw_text(c, t)
    draw_water(c, t, kr, g)
    apply_glow(c, g)
    for x, y, s, ph in kr:
        draw_krathong(c, x, y, s, t, ph, hero=(s > 1))
    return c


def load_font():
    return find_font(13)


FONT = load_font()
PX_W, PX_H = CWd * S, CHd * S


def render(c):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    rgb = np.clip(c.rgb, 0, 255).astype(np.uint8)
    rows, cols = np.nonzero(c.ch != " ")
    for r, cc in zip(rows, cols):
        d.text((cc * PX_W, r * PX_H - 1), c.ch[r, cc], font=FONT, fill=tuple(int(v) for v in rgb[r, cc]))
    # bloom: เบลอภาพย่อแล้วบวกกลับ ให้แสงฟุ้งเรือง
    small = img.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
    bloom = small.resize((W, H), Image.BILINEAR).point(lambda v: min(255, int(v * 1.5)))
    return ImageChops.screen(img, bloom)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--preview":
        render(scene(float(args[1]) if len(args) > 1 else 0.9)).save("preview.png")
        print("saved preview.png")
        return
    out = args[0] if args else ("krathong.mp4" if shutil.which("ffmpeg") else "krathong.gif")
    frames_iter = (render(scene(f / LOOP_FRAMES)) for f in range(FRAMES))
    if out.endswith(".mp4"):
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-vf", "scale=out_color_matrix=bt709:out_range=tv",      # แปลงสีแบบ HD ให้ตรงกับที่ QuickTime/มือถือเล่น
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-pix_fmt", "yuv420p",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", out], stdin=subprocess.PIPE)
        for i, im in enumerate(frames_iter):
            p.stdin.write(im.tobytes())
            print(f"\rframe {i + 1}/{FRAMES}", end="", flush=True)
        p.stdin.close()
        p.wait()
    else:
        imgs = []
        for i, im in enumerate(frames_iter):
            imgs.append(im.resize((W // 2, H // 2)).quantize(128))
            print(f"\rframe {i + 1}/{FRAMES}", end="", flush=True)
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=1000 // FPS, loop=0)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
