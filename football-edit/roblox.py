"""Sprite rendering for the edit: the treasure chest, its glow and sparkles.

The chest is drawn with flat-shaded, extruded polygons so it reads as a 3D block
without needing a 3D engine. Sprites are rendered supersampled and downscaled for
clean edges. (The Roblox avatars come from avatar3d.py.)
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 3  # supersampling factor


# Depth offset (screen px per stud of depth) as a fraction of one stud: up and to the right.
DEPTH = (0.38, -0.30)
LIGHT = np.array([-0.3, -1.0]) / math.hypot(0.3, 1.0)


def hex_to_rgb(c):
    if isinstance(c, (tuple, list)):
        return tuple(int(v) for v in c[:3])
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def shade(rgb, k):
    return tuple(int(max(0, min(255, v * k))) for v in rgb)


def draw_extruded(draw, front, color, depth_px, outline=True, front_color=None):
    """Draw a convex polygon extruded into the screen along depth_px.

    Visible side faces are shaded by how much they face the light; then the front
    face is drawn on top.
    """
    n = len(front)
    # Signed area tells us the winding, so outward normals point the right way.
    area = sum(front[i][0] * front[(i + 1) % n][1] - front[(i + 1) % n][0] * front[i][1] for i in range(n))
    sign = 1 if area > 0 else -1
    d = np.array(depth_px)
    edge_col = shade(color, 0.55)
    for i in range(n):
        p0, p1 = np.array(front[i]), np.array(front[(i + 1) % n])
        e = p1 - p0
        nrm = sign * np.array([e[1], -e[0]])
        ln = np.linalg.norm(nrm)
        if ln == 0:
            continue
        nrm = nrm / ln
        if nrm @ d <= 0:
            continue
        k = 0.70 + 0.45 * max(0.0, float(nrm @ LIGHT))
        quad = [tuple(p0), tuple(p1), tuple(p1 + d), tuple(p0 + d)]
        draw.polygon(quad, fill=shade(color, k))
    draw.polygon(front, fill=front_color or color)
    if outline:
        w = max(1, int(round(np.linalg.norm(d) * 0.12)))
        draw.line(front + [front[0]], fill=edge_col, width=w, joint="curve")


def render_chest(width_px):
    """Treasure chest sprite. Returns (RGBA image, bottom-centre anchor)."""
    u = width_px * SS / 4.0  # chest is 4 units wide
    W, H = int(u * 7), int(u * 7)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    depth = (DEPTH[0] * u * 1.6, DEPTH[1] * u * 1.6)
    wood = (150, 86, 40)
    wood_dark = (105, 56, 24)
    gold = (255, 196, 40)
    gold_dark = (196, 132, 18)
    ox, oy = W / 2 - depth[0] / 2, H - u * 0.8

    def P(x, y):
        return (ox + x * u, oy - y * u)

    # Body
    body = [P(-2, 2.0), P(2, 2.0), P(2, 0), P(-2, 0)]
    draw_extruded(dr, body, wood, depth)
    # planks
    for yy in (0.66, 1.33):
        a, b = P(-2, yy), P(2, yy)
        dr.line([a, b], fill=wood_dark, width=max(2, int(u * 0.06)))
    # Lid: half-cylinder-ish front profile, hinged at the back top edge
    lid = [P(2, 2.0)] + [P(2 * math.cos(math.radians(a)), 2.0 + 1.05 * math.sin(math.radians(a)))
                         for a in range(0, 181, 10)] + [P(-2, 2.0)]
    draw_extruded(dr, lid, wood, depth)
    # Gold bands: vertical on body and lid edges, horizontal rim
    bw = 0.34
    for x in (-2 + 0.05, 2 - bw - 0.05):
        draw_extruded(dr, [P(x, 2.0), P(x + bw, 2.0), P(x + bw, 0), P(x, 0)], gold, (0, 0), outline=True)
        lid_band = [P(x, 2.0), P(x + bw, 2.0)]
        top_pts = [P(xx, 2.0 + 1.05 * math.sqrt(max(0, 1 - (xx / 2) ** 2))) for xx in (x + bw, x)]
        dr.polygon(lid_band + top_pts, fill=gold, outline=gold_dark)
    rim = [P(-2, 2.12), P(2, 2.12), P(2, 1.88), P(-2, 1.88)]
    dr.polygon(rim, fill=gold, outline=gold_dark)
    # Lock plate + keyhole
    lc = P(0, 1.75)
    lw, lh = 0.55 * u, 0.75 * u
    dr.rounded_rectangle([lc[0] - lw / 2, lc[1] - lh / 2, lc[0] + lw / 2, lc[1] + lh / 2], radius=int(u * 0.12),
                         fill=gold, outline=gold_dark, width=max(2, int(u * 0.05)))
    k = u * 0.09
    dr.ellipse([lc[0] - k, lc[1] - k * 1.6, lc[0] + k, lc[1] + k * 0.4], fill=(60, 30, 10))
    dr.polygon([(lc[0] - k * 0.6, lc[1]), (lc[0] + k * 0.6, lc[1]), (lc[0], lc[1] + k * 2.4)], fill=(60, 30, 10))
    # highlight on the lid
    hl = [P(-1.5, 2.35), P(1.2, 2.35)]
    dr.line(hl, fill=(190, 120, 64), width=max(2, int(u * 0.08)))

    bbox = img.getbbox()
    img = img.crop(bbox)
    anchor = ((ox + depth[0] / 2) - bbox[0], oy - bbox[1])
    out = img.resize((max(1, img.width // SS), max(1, img.height // SS)), Image.LANCZOS)
    return out, (anchor[0] / SS, anchor[1] / SS)


def glow_sprite(radius, color=(255, 214, 80)):
    """Soft radial glow with rays, RGBA."""
    s = int(radius * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    c = (s - 1) / 2
    r = np.hypot(xx - c, yy - c) / (s / 2)
    ang = np.arctan2(yy - c, xx - c)
    rays = 0.5 + 0.5 * np.cos(ang * 12)
    a = np.clip(1 - r, 0, 1) ** 1.6 * (0.55 + 0.45 * rays)
    arr = np.zeros((s, s, 4), np.uint8)
    arr[..., :3] = color
    arr[..., 3] = (a * 220).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def sparkle(size, color=(255, 255, 220)):
    s = max(4, int(size))
    img = Image.new("RGBA", (s * SS, s * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s * SS / 2
    r, w = c, c * 0.22
    d.polygon([(c, c - r), (c + w, c - w), (c + r, c), (c + w, c + w), (c, c + r), (c - w, c + w), (c - r, c),
               (c - w, c - w)], fill=color + (255,))
    return img.resize((s, s), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.4))
