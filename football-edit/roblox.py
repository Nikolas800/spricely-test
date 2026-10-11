"""Sprite rendering for the edit: blocky Roblox (R6) avatars and a treasure chest.

Everything is drawn with flat-shaded, extruded polygons so it reads as "3D blocks"
without needing a 3D engine. Sprites are rendered supersampled and downscaled for
clean edges.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 3  # supersampling factor

ROBLOX_YELLOW = (245, 205, 48)  # "Bright yellow", the classic head/arm colour

# Depth offset (screen px per stud of depth) as a fraction of one stud: up and to the right.
DEPTH = (0.38, -0.30)
LIGHT = np.array([-0.3, -1.0]) / math.hypot(0.3, 1.0)

POSES = {
    # (left arm, right arm, left leg, right leg) rotation in degrees, positive = outward
    "idle": (4, 4, 0, 0),
    "tpose": (90, 90, 0, 0),
    "cheer": (160, 25, 4, 4),
    "wave": (8, 140, 0, 6),
    "dab": (125, 55, 6, 6),
    "spread": (35, 35, 8, 8),
}
POSE_CYCLE = ["tpose", "cheer", "spread", "wave", "dab", "idle"]


def hex_to_rgb(c):
    if isinstance(c, (tuple, list)):
        return tuple(int(v) for v in c[:3])
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def shade(rgb, k):
    return tuple(int(max(0, min(255, v * k))) for v in rgb)


def _rot(points, pivot, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    px, py = pivot
    return [(px + (x - px) * ca - (y - py) * sa, py + (x - px) * sa + (y - py) * ca) for x, y in points]


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


def _box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rounded_rect_poly(cx, cy, w, h, r, steps=6):
    pts = []
    corners = [(cx + w / 2 - r, cy - h / 2 + r, -90), (cx + w / 2 - r, cy + h / 2 - r, 0),
               (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)]
    for ox, oy, start in corners:
        for s in range(steps + 1):
            a = math.radians(start + 90 * s / steps)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return pts


def render_avatar(stud_px, shirt, pants, skin=ROBLOX_YELLOW, pose="tpose", face=True, stripes=None):
    """Render a classic R6 avatar at `stud_px` pixels per stud.

    Returns (RGBA image, foot_anchor_xy) where the anchor is the point between the
    feet in image pixels. The sprite is unstretched; callers squash/stretch it.
    `stripes` is an optional second shirt colour drawn as vertical stripes on the torso.
    """
    shirt, pants, skin = hex_to_rgb(shirt), hex_to_rgb(pants), hex_to_rgb(skin)
    la, ra, ll, rl = POSES.get(pose, POSES["idle"])
    u = stud_px * SS
    W, H = int(u * 10), int(u * 10)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    depth = (DEPTH[0] * u, DEPTH[1] * u)

    ox, oy = W / 2, H - u * 2.0  # ground point, room for raised arms above

    def P(x, y):  # studs -> px (y up)
        return (ox + x * u, oy - y * u)

    def limb(x0, x1, y_top, y_bot, pivot, deg, segments):
        """segments: list of (fraction_from_top, color) bands, top to bottom."""
        polys, prev = [], y_top
        for frac, col in segments:
            yb = y_top - (y_top - y_bot) * frac
            polys.append((_rot([P(x0, prev), P(x1, prev), P(x1, yb), P(x0, yb)], P(*pivot), deg), col))
            prev = yb
        # If the limb's "up" faces the extrusion direction, the lower band's top face
        # would show through the upper band, so paint lower bands first.
        a = math.radians(deg)
        up = (math.sin(a), -math.cos(a))
        if up[0] * depth[0] + up[1] * depth[1] > 0:
            polys.reverse()
        for poly, col in polys:
            draw_extruded(dr, poly, col, depth)

    # Legs (shorts on top, then sock/leg colour), left leg rotates outward = clockwise on screen
    sock = shade(pants, 0.8) if pants != shirt else shade(pants, 0.7)
    limb(-1.0, 0.0, 2.0, 0.0, (-0.5, 2.0), ll, [(0.45, pants), (1.0, sock)])
    limb(0.0, 1.0, 2.0, 0.0, (0.5, 2.0), -rl, [(0.45, pants), (1.0, sock)])

    # Left arm (screen-left). Raising it outward = clockwise on screen, i.e. positive angle.
    limb(-2.0, -1.0, 4.0, 2.0, (-1.5, 3.55), la, [(0.32, shirt), (1.0, skin)])

    # Torso
    draw_extruded(dr, [P(-1, 4), P(1, 4), P(1, 2), P(-1, 2)], shirt, depth)
    if stripes is not None:
        for x0 in (-0.6, 0.2):
            dr.polygon([P(x0, 4), P(x0 + 0.4, 4), P(x0 + 0.4, 2), P(x0, 2)], fill=hex_to_rgb(stripes))
    # little collar + number-ish stripe so it reads as a jersey
    cx, cy = P(0, 3.95)
    dr.polygon([(cx - 0.32 * u, cy), (cx + 0.32 * u, cy), (cx, cy + 0.32 * u)], fill=shade(shirt, 0.6))

    # Right arm (screen-right): outward = counter-clockwise on screen
    limb(1.0, 2.0, 4.0, 2.0, (1.5, 3.55), -ra, [(0.32, shirt), (1.0, skin)])

    # Head: rounded block with cylinder-style shading on the front.
    hw, hh = 1.25, 1.2
    hc = P(0, 4.0 + hh / 2 + 0.02)
    head = _rounded_rect_poly(hc[0], hc[1], hw * u, hh * u, 0.32 * u, steps=8)
    draw_extruded(dr, head, skin, depth, outline=False)
    # cylinder gradient on the front face
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(head, fill=255)
    xs = np.linspace(-1, 1, int(hw * u))
    prof = 1.08 - 0.22 * (xs + 0.35) ** 2
    grad = np.zeros((H, W, 3), np.float32)
    x0 = int(hc[0] - hw * u / 2)
    grad[:, :, :] = np.array(skin, np.float32)
    seg = np.clip(np.array(skin, np.float32)[None, :] * prof[:, None], 0, 255)
    grad[:, x0:x0 + len(xs), :] = seg[None, :, :]
    img.paste(Image.fromarray(grad.astype(np.uint8), "RGB"), (0, 0), mask)
    dr = ImageDraw.Draw(img)
    dr.line(head + [head[0]], fill=shade(skin, 0.6), width=max(1, int(u * 0.045)), joint="curve")

    # Top stud on the head (the little cylinder)
    tx, ty = hc[0] + depth[0] * 0.5, hc[1] - hh * u / 2 + depth[1] * 0.5
    dr.ellipse([tx - 0.3 * u, ty - 0.1 * u, tx + 0.3 * u, ty + 0.1 * u], fill=shade(skin, 1.12),
               outline=shade(skin, 0.75), width=max(1, int(u * 0.03)))

    if face:
        ink = (24, 24, 24)
        ex, ey = 0.2 * u, hc[1] - 0.12 * u
        for sx in (-1, 1):
            cxp = hc[0] + sx * ex
            dr.ellipse([cxp - 0.07 * u, ey - 0.15 * u, cxp + 0.07 * u, ey + 0.15 * u], fill=ink)
        sw = 0.62 * u
        dr.arc([hc[0] - sw / 2, hc[1] - 0.32 * u, hc[0] + sw / 2, hc[1] + 0.33 * u], 25, 155, fill=ink,
               width=max(2, int(u * 0.07)))

    bbox = img.getbbox()
    img = img.crop(bbox)
    anchor = (ox - bbox[0], oy - bbox[1])
    out = img.resize((max(1, img.width // SS), max(1, img.height // SS)), Image.LANCZOS)
    return out, (anchor[0] / SS, anchor[1] / SS)


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
