"""
make_dataset.py  -  Session 30 assignment
Generates a small, self-contained image dataset (224x224 JPEGs) so the notebook runs
out-of-the-box.  The images are procedurally drawn (NOT real photographs) with random
colours, positions, rotation, lighting and noise.

To use REAL photos instead: just replace the images inside the folders under ./dataset
(keep the same folder names) - the notebook does not need any code changes.
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 448          # drawing canvas (drawn big, then down-sampled => anti-aliasing)
OUT = 224        # final image size
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

rng = random.Random(30)
nrng = np.random.default_rng(30)


# ----------------------------------------------------------------- helpers
def rcolor(lo=40, hi=230):
    return tuple(rng.randint(lo, hi) for _ in range(3))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def background():
    c1, c2 = rcolor(170, 250), rcolor(150, 250)
    t = np.linspace(0, 1, S)[:, None, None]
    if rng.random() < 0.5:
        t = np.broadcast_to(t, (S, S, 1))
    else:
        t = np.broadcast_to(t.transpose(1, 0, 2), (S, S, 1))
    arr = (np.array(c1) * (1 - t) + np.array(c2) * t).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def finish(img, rot=12, flip=True):
    """global jitter: rotate, flip, brightness, blur, noise -> 224x224"""
    if rot:
        img = img.rotate(rng.uniform(-rot, rot), resample=Image.BICUBIC, fillcolor=(235, 235, 235))
    if flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((OUT, OUT), Image.LANCZOS)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    a = np.asarray(img).astype(np.float32)
    a = a * rng.uniform(0.82, 1.15) + rng.uniform(-12, 12)
    a += nrng.normal(0, rng.uniform(2, 9), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def bezier(p0, p1, p2, p3, n=40):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts


def jit(v, a):
    return v + rng.uniform(-a, a)


def scale_pts(pts, sc, cx=224, cy=224, dx=0, dy=0):
    return [((x - cx) * sc + cx + dx, (y - cy) * sc + cy + dy) for x, y in pts]


# ----------------------------------------------------------------- FOOD
def food(kind):
    img = background()
    d = ImageDraw.Draw(img)
    cx, cy = 224 + rng.randint(-30, 30), 230 + rng.randint(-25, 25)
    sc = rng.uniform(0.8, 1.15)
    if kind == "pizza":
        r = 165 * sc
        crust = (rng.randint(200, 225), rng.randint(150, 175), rng.randint(80, 105))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=crust)
        r2 = r * 0.88
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=(rng.randint(190, 220), rng.randint(50, 80), 40))
        r3 = r * 0.82
        d.ellipse([cx - r3, cy - r3, cx + r3, cy + r3], fill=(250, rng.randint(205, 225), rng.randint(90, 120)))
        for _ in range(rng.randint(6, 10)):
            a, rr = rng.uniform(0, 6.28), rng.uniform(0, r3 * 0.8)
            px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
            pr = rng.randint(18, 28) * sc
            d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(rng.randint(150, 185), 35, 35))
        for _ in range(rng.randint(4, 8)):
            a, rr = rng.uniform(0, 6.28), rng.uniform(0, r3 * 0.85)
            px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(40, rng.randint(120, 170), 50))
        for k in range(4):
            a = k * math.pi / 4 + rng.uniform(-0.05, 0.05)
            d.line([cx - r3 * math.cos(a), cy - r3 * math.sin(a), cx + r3 * math.cos(a), cy + r3 * math.sin(a)],
                   fill=(210, 150, 60), width=4)
    elif kind == "burger":
        w = 150 * sc
        bun = (rng.randint(190, 215), rng.randint(120, 150), rng.randint(50, 75))
        d.pieslice([cx - w, cy - 190 * sc, cx + w, cy + 30 * sc], 180, 360, fill=bun)
        for _ in range(12):
            sx, sy = cx + rng.uniform(-w * 0.7, w * 0.7), cy - rng.uniform(30, 120) * sc
            d.ellipse([sx - 4, sy - 2, sx + 4, sy + 2], fill=(250, 235, 200))
        y = cy + 25 * sc
        d.rounded_rectangle([cx - w - 10, y, cx + w + 10, y + 28 * sc], 14, fill=(60, rng.randint(150, 190), 50))
        y += 24 * sc
        d.polygon([(cx - w, y), (cx + w, y), (cx + w - 25, y + 30 * sc), (cx - w + 25, y + 30 * sc)], fill=(250, 200, 40))
        y += 26 * sc
        d.rounded_rectangle([cx - w, y, cx + w, y + 45 * sc], 20, fill=(95, 50, 30))
        y += 42 * sc
        d.rounded_rectangle([cx - w, y, cx + w, y + 42 * sc], 18, fill=shade(bun, 0.95))
    elif kind == "donut":
        r = 160 * sc
        dough = (rng.randint(205, 230), rng.randint(155, 180), rng.randint(95, 120))
        d.ellipse([cx - r, cy - r * 0.9, cx + r, cy + r * 0.9], fill=dough)
        ic = rng.choice([(235, 110, 170), (110, 70, 45), (245, 235, 220), (140, 200, 235)])
        r2 = r * 0.9
        d.ellipse([cx - r2, cy - r2 * 0.88 - 6, cx + r2, cy + r2 * 0.88 - 6], fill=ic)
        rh = r * 0.32
        d.ellipse([cx - rh, cy - rh * 0.9 - 6, cx + rh, cy + rh * 0.9 - 6], fill=img.getpixel((cx, min(S - 1, max(0, int(cy - 6))))))
        for _ in range(26):
            a, rr = rng.uniform(0, 6.28), rng.uniform(r * 0.42, r * 0.82)
            px, py = cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.86 - 6
            ang = rng.uniform(0, 3.14)
            d.line([px, py, px + 12 * math.cos(ang), py + 12 * math.sin(ang)], fill=rcolor(60, 255), width=5)
    else:  # cupcake
        wrap = rcolor(60, 220)
        d.polygon([(cx - 85 * sc, cy + 15), (cx + 85 * sc, cy + 15), (cx + 62 * sc, cy + 150 * sc), (cx - 62 * sc, cy + 150 * sc)], fill=wrap)
        for k in range(-3, 4):
            d.line([cx + k * 22 * sc, cy + 15, cx + k * 16 * sc, cy + 150 * sc], fill=shade(wrap, 0.7), width=4)
        fc = rng.choice([(245, 170, 200), (250, 245, 235), (150, 100, 70), (190, 150, 235)])
        for i, (w, h) in enumerate([(115, 55), (90, 50), (62, 45)]):
            yy = cy - i * 42 * sc
            d.ellipse([cx - w * sc, yy - h * sc * 0.5, cx + w * sc, yy + h * sc * 0.9], fill=shade(fc, 1 - 0.04 * i))
        d.ellipse([cx - 14, cy - 135 * sc, cx + 14, cy - 107 * sc], fill=(200, 20, 40))
    return finish(img, rot=10)


# ----------------------------------------------------------------- SNEAKER
def sneaker():
    img = background()
    d = ImageDraw.Draw(img)
    main, accent = rcolor(30, 235), rcolor(30, 235)
    sole = rng.choice([(245, 245, 245), (225, 225, 215), (40, 40, 40)])
    sc, dx, dy = rng.uniform(0.85, 1.1), rng.randint(-25, 25), rng.randint(-20, 20)
    P = lambda pts: scale_pts(pts, sc, dx=dx, dy=dy)
    upper = [(70, 305), (68, 195), (110, 172), (185, 178), (225, 220), (300, 238), (372, 262), (392, 305)]
    d.polygon(P(upper), fill=main)
    d.polygon(P([(110, 172), (185, 178), (170, 130), (115, 125), (95, 150)]), fill=shade(main, 0.8))    # collar
    d.polygon(P([(185, 178), (225, 220), (250, 205), (210, 160)]), fill=shade(main, 1.15 if sum(main) < 600 else 0.9))  # tongue
    d.chord(P([(330, 235), (400, 335)])[0] + P([(330, 235), (400, 335)])[1], 180, 360, fill=shade(main, 0.88))
    for k in range(5):
        t = k / 4
        x1, y1 = 205 + 38 * t, 205 + 18 * t
        d.line(P([(x1 - 14, y1 + 14), (x1 + 20, y1 - 12)]), fill=(250, 250, 250), width=7)
    d.line(P([(100, 285), (160, 240), (260, 270), (330, 290)]), fill=accent, width=16, joint="curve")
    d.rounded_rectangle(P([(58, 296), (398, 342)])[0] + P([(58, 296), (398, 342)])[1], 18, fill=sole)
    d.line(P([(62, 330), (394, 330)]), fill=shade(sole, 0.75), width=5)
    return finish(img, rot=8)


# ----------------------------------------------------------------- SELFIE (glasses / no glasses)
def selfie(glasses):
    img = background()
    d = ImageDraw.Draw(img, "RGBA")
    skin = rng.choice([(255, 224, 196), (240, 200, 165), (215, 165, 125), (180, 125, 90), (125, 85, 60)])
    hair = rng.choice([(25, 20, 20), (70, 45, 25), (150, 100, 45), (200, 170, 90), (110, 110, 115), (150, 40, 30)])
    shirt = rcolor(40, 220)
    cx, cy = 224 + rng.randint(-25, 25), 215 + rng.randint(-15, 15)
    rx, ry = rng.randint(100, 118), rng.randint(128, 146)
    d.ellipse([cx - 170, 360, cx + 170, 560], fill=shirt)                         # shoulders
    d.rectangle([cx - 38, cy + ry - 40, cx + 38, 395], fill=shade(skin, 0.92))     # neck
    d.ellipse([cx - rx - 20, cy - ry - 24, cx + rx + 20, cy + 40], fill=hair)      # hair back
    d.ellipse([cx - rx - 14, cy - 28, cx - rx + 18, cy + 40], fill=shade(skin, 0.95))  # ears
    d.ellipse([cx + rx - 18, cy - 28, cx + rx + 14, cy + 40], fill=shade(skin, 0.95))
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=skin)                     # face
    d.pieslice([cx - rx - 6, cy - ry - 8, cx + rx + 6, cy + 20], 180, 360, fill=hair)  # fringe
    ex = rng.randint(46, 56)
    ey = cy - 12 + rng.randint(-6, 6)
    for sx in (-1, 1):
        x = cx + sx * ex
        d.ellipse([x - 24, ey - 13, x + 24, ey + 13], fill=(250, 250, 250))
        d.ellipse([x - 11, ey - 11, x + 11, ey + 11], fill=rng.choice([(60, 40, 25), (40, 90, 140), (50, 110, 70), (30, 30, 30)]))
        d.ellipse([x - 4, ey - 5, x + 4, ey + 3], fill=(10, 10, 10))
        d.line([x - 28, ey - 34, x + 26, ey - 38 + rng.randint(-5, 5)], fill=shade(hair, 0.8), width=7)
    d.line([cx, ey + 6, cx - 8, ey + 62, cx + 8, ey + 66], fill=shade(skin, 0.7), width=5)
    mw = rng.randint(34, 52)
    d.arc([cx - mw, ey + 70, cx + mw, ey + 120], 15, 165, fill=(150, 50, 60), width=7)
    if glasses:
        fc = rng.choice([(20, 20, 20), (30, 30, 30), (120, 60, 20), (190, 30, 40), (30, 60, 150), (200, 170, 40)])
        lw = rng.randint(8, 12)
        shape = rng.choice(["round", "rect"])
        gw, gh = rng.randint(34, 40), rng.randint(28, 36)
        lens = (150, 200, 235, rng.randint(40, 110))
        for sx in (-1, 1):
            x = cx + sx * ex
            box = [x - gw, ey - gh, x + gw, ey + gh]
            if shape == "round":
                d.ellipse(box, fill=lens, outline=fc, width=lw)
            else:
                d.rounded_rectangle(box, 10, fill=lens, outline=fc, width=lw)
        d.line([cx - ex + gw, ey - 4, cx + ex - gw, ey - 4], fill=fc, width=lw - 2)
        d.line([cx - ex - gw, ey - 6, cx - rx - 4, ey - 14], fill=fc, width=lw - 3)
        d.line([cx + ex + gw, ey - 6, cx + rx + 4, ey - 14], fill=fc, width=lw - 3)
    return finish(img, rot=6)


# ----------------------------------------------------------------- T-SHIRT
SHIRT = [(165, 80), (128, 96), (48, 168), (92, 218), (135, 186), (135, 385), (313, 385),
         (313, 186), (356, 218), (400, 168), (320, 96), (283, 80)]


def tshirt(kind):
    img = background()
    sc, dx, dy = rng.uniform(0.85, 1.08), rng.randint(-20, 20), rng.randint(-15, 15)
    pts = scale_pts(SHIRT, sc, dx=dx, dy=dy)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    layer = Image.new("RGB", (S, S), rcolor(30, 235))
    ld = ImageDraw.Draw(layer)
    base = layer.getpixel((5, 5))
    if kind == "striped":
        c2 = rcolor(30, 245)
        sw = rng.randint(14, 30)
        vertical = rng.random() < 0.2
        for k in range(0, S, sw * 2):
            if vertical:
                ld.rectangle([k, 0, k + sw, S], fill=c2)
            else:
                ld.rectangle([0, k, S, k + sw], fill=c2)
    elif kind == "printed":
        pc, pc2 = rcolor(30, 245), rcolor(30, 245)
        px, py = 224 + dx + rng.randint(-15, 15), 240 + dy + rng.randint(-15, 15)
        shape = rng.choice(["circle", "star", "bars", "heart"])
        if shape == "circle":
            r = rng.randint(45, 70)
            ld.ellipse([px - r, py - r, px + r, py + r], fill=pc)
            ld.ellipse([px - r * .5, py - r * .5, px + r * .5, py + r * .5], fill=pc2)
        elif shape == "star":
            r = rng.randint(55, 80)
            st = [(px + (r if i % 2 == 0 else r * .45) * math.cos(i * math.pi / 5 - math.pi / 2),
                   py + (r if i % 2 == 0 else r * .45) * math.sin(i * math.pi / 5 - math.pi / 2)) for i in range(10)]
            ld.polygon(st, fill=pc)
        elif shape == "bars":
            for k in range(4):
                ld.rectangle([px - 70, py - 55 + k * 30, px + 70 - k * 12, py - 38 + k * 30], fill=pc if k % 2 == 0 else pc2)
        else:
            r = rng.randint(35, 50)
            ld.ellipse([px - r, py - r, px, py + 5], fill=pc)
            ld.ellipse([px, py - r, px + r, py + 5], fill=pc)
            ld.polygon([(px - r, py - 14), (px + r, py - 14), (px, py + r + 20)], fill=pc)
    # collar
    ld.ellipse([224 + dx - 52 * sc, 62 + dy, 224 + dx + 52 * sc, 125 + dy], fill=shade(base if kind != "striped" else base, 0.78))
    out = img.copy()
    out.paste(layer, (0, 0), mask)
    d = ImageDraw.Draw(out)
    d.line(pts + [pts[0]], fill=(60, 60, 60), width=3)
    d.ellipse([224 + dx - 50 * sc, 66 + dy, 224 + dx + 50 * sc, 118 + dy], fill=img.getpixel((224, 70)), outline=(60, 60, 60), width=3)
    return finish(out, rot=10)


# ----------------------------------------------------------------- HEADPHONES
def headphones(kind):
    img = background()
    d = ImageDraw.Draw(img)
    c = rng.choice([(25, 25, 25), (110, 110, 118), (180, 30, 40), (30, 70, 160), (200, 160, 60), (90, 90, 95)])
    c2 = shade(c, 0.6 if sum(c) > 300 else 1.8)
    cx, cy = 224 + rng.randint(-15, 15), 240 + rng.randint(-15, 15)
    if kind == "over_ear":
        w = rng.randint(120, 135)
        d.arc([cx - w, cy - 190, cx + w, cy + 110], 180, 360, fill=c, width=24)
        d.arc([cx - w + 28, cy - 160, cx + w - 28, cy + 90], 190, 350, fill=c2, width=10)
        for sx in (-1, 1):
            x = cx + sx * w
            d.ellipse([x - 58, cy - 45, x + 58, cy + 105], fill=c)
            d.ellipse([x - 40, cy - 25, x + 40, cy + 85], fill=c2)
            d.ellipse([x - 22, cy + 0, x + 22, cy + 60], fill=shade(c, 0.85))
    elif kind == "earbuds":
        bx = rng.randint(70, 95)
        by = cy + rng.randint(-10, 30)
        for sx in (-1, 1):
            x = cx + sx * bx
            d.ellipse([x - 24, by - 24, x + 24, by + 24], fill=c)
            d.ellipse([x - 12, by - 12, x + 12, by + 12], fill=c2)
            d.line([x, by - 22, x + sx * 6, by - 70], fill=c, width=12)
            cable = bezier((x + sx * 6, by - 70), (x + sx * 20, by - 170), (cx + sx * 40, cy - 160), (cx, cy - 90))
            d.line(cable, fill=c, width=5)
        d.line([cx, cy - 90, cx, cy + 120], fill=c, width=6)
        d.rounded_rectangle([cx - 12, cy + 120, cx + 12, cy + 175], 6, fill=c2)
    else:  # neckband
        cy -= 55
        w = rng.randint(135, 150)
        d.arc([cx - w, cy - 130, cx + w, cy + 130], 20, 160, fill=c, width=34)
        d.arc([cx - w + 16, cy - 114, cx + w - 16, cy + 114], 35, 145, fill=c2, width=8)
        for sx in (-1, 1):
            ex = cx + sx * w * 0.96
            ey = cy + 60
            d.ellipse([ex - 26, ey - 26, ex + 26, ey + 26], fill=c)
            cable = bezier((ex, ey + 20), (ex + sx * 20, ey + 80), (ex - sx * 10, ey + 110), (ex + sx * 5, ey + 130))
            d.line(cable, fill=c, width=6)
            d.ellipse([ex + sx * 5 - 18, ey + 125, ex + sx * 5 + 18, ey + 160], fill=c2)
    return finish(img, rot=14)


# ----------------------------------------------------------------- build
def save(img, folder, name):
    os.makedirs(folder, exist_ok=True)
    img.save(os.path.join(folder, name), quality=92)


def main():
    # Task 1 : 20 food images (single class folder so flow_from_directory works)
    kinds = ["pizza", "burger", "donut", "cupcake"]
    for i in range(20):
        save(food(kinds[i % 4]), f"{ROOT}/food_images/food", f"food_{i+1:02d}.jpg")
    # Task 2 : 30 sneakers
    for i in range(30):
        save(sneaker(), f"{ROOT}/sneakers/sneaker", f"sneaker_{i+1:02d}.jpg")
    # Task 3 : glasses vs no glasses (25 train + 10 val per class = 70 images)
    for cls, flag in (("glasses", True), ("no_glasses", False)):
        for split, n in (("train", 25), ("val", 10)):
            for i in range(n):
                save(selfie(flag), f"{ROOT}/glasses/{split}/{cls}", f"{cls}_{i+1:02d}.jpg")
    # Task 4 : t-shirts (3 classes, 30 train + 10 val)
    for cls in ("plain", "striped", "printed"):
        for split, n in (("train", 30), ("val", 10)):
            for i in range(n):
                save(tshirt(cls), f"{ROOT}/tshirts/{split}/{cls}", f"{cls}_{i+1:02d}.jpg")
    # Task 5 : headphones (3 classes, 30 train + 10 val)
    for cls in ("over_ear", "earbuds", "neckband"):
        for split, n in (("train", 30), ("val", 10)):
            for i in range(n):
                save(headphones(cls), f"{ROOT}/headphones/{split}/{cls}", f"{cls}_{i+1:02d}.jpg")
    total = sum(len(f) for _, _, f in os.walk(ROOT))
    print("dataset created:", total, "images in", ROOT)


if __name__ == "__main__":
    main()
