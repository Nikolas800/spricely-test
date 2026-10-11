#!/usr/bin/env python3
"""The prop the ball turns into: a holographic "secret prototype vault".

Remodelled from a Roblox vault: a rounded cube with nested rainbow frames on every
face, an iridescent finish, a glowing magenta screen, a raised rainbow keypad, a
blue dial bar and status LEDs on the front, a magenta handle on top and a glowing
green pad underneath. Rendered once with the avatar3d raymarcher (assets/vault.png);
the neon glow, sheen and sparkles are animated per frame in 2D.

  python vault.py        # re-render assets/vault.png
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import avatar3d as av

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "assets", "vault.png")
SS = 3

BODY_C = np.float32([0, 1.24, 0])  # body centre; the cube has half-size 1
BUTTONS = [  # (x, y) on the front face relative to the body centre, colour
    (-0.2, 0.08, "#6ef3ff"), (0.15, 0.08, "#ff7ad9"), (0.5, 0.08, "#ffe066"),
    (-0.2, -0.22, "#7dff9a"), (0.15, -0.22, "#b38cff"), (0.5, -0.22, "#6ef3ff"),
    (-0.2, -0.52, "#ff7ad9"), (0.15, -0.52, "#ffe066"), (0.5, -0.52, "#7dff9a"),
]
BANDS = ["#ffe94d", "#6dff7a", "#4fd8ff", "#ff7ae0"]  # outer frame -> inner frame
PANEL = "#c6f6ff"


def _rainbow(h):
    h = np.asarray(h, np.float32)[..., None]
    return 0.5 + 0.5 * np.cos(2 * np.pi * (h + np.float32([0, 1 / 3, 2 / 3])))


class Vault(av.Avatar):
    unit = 1.0

    def __init__(self):
        # (name, kind, centre, half size, rounding)
        self.parts = [
            ("base", "cyl", (0, 0.1, 0), (1.5, 0.1), 0.06),
            ("body", "box", tuple(BODY_C), (1.0, 1.0, 1.0), 0.14),
            ("post", "box", (-0.45, 2.36, 0), (0.1, 0.16, 0.12), 0.05),
            ("post", "box", (0.45, 2.36, 0), (0.1, 0.16, 0.12), 0.05),
            ("handle", "box", (0, 2.56, 0), (0.62, 0.09, 0.15), 0.07),
            ("dial", "box", (-0.62, BODY_C[1] - 0.07, 1.0), (0.08, 0.52, 0.06), 0.05),
            ("led_y", "box", (-0.62, BODY_C[1] + 0.66, 1.0), (0.06, 0.06, 0.05), 0.05),
            ("led_g", "box", (-0.62, BODY_C[1] + 0.5, 1.0), (0.06, 0.06, 0.05), 0.05),
        ] + [(f"key{i}", "box", (x, BODY_C[1] + y, 1.0), (0.14, 0.11, 0.06), 0.04)
             for i, (x, y, _) in enumerate(BUTTONS)]

    def sdf(self, p, with_id=False):
        ds = []
        for _, kind, c, size, r in self.parts:
            q = p - np.float32(c)
            ds.append(av.sd_round_cyl(q, size[0], r, size[1]) if kind == "cyl" else av.sd_round_box(q, size, r))
        ds = np.stack(ds, -1)
        d = ds.min(-1)
        return (d, ds.argmin(-1)) if with_id else d

    def _body(self, p):
        """Colour + glow of the cube's faces: rainbow frames, iridescent panel, screen."""
        q = p - BODY_C
        axis = np.argmax(np.abs(q), 1)
        # in-plane coordinates of whichever face each point is on
        u = np.where(axis == 0, q[:, 2], q[:, 0])
        v = np.where(axis == 1, q[:, 2], q[:, 1])
        edge = 1 - np.maximum(np.abs(u), np.abs(v))  # 0 at the face edge
        col = np.empty((len(p), 3), np.float32)
        glow = np.zeros(len(p), np.float32)
        band = np.clip((edge / 0.075).astype(int), 0, len(BANDS))
        for k, c in enumerate(BANDS):
            col[band == k] = av._lin(c)
        inner = band == len(BANDS)
        col[inner] = av._lin(PANEL)
        glow[~inner] = 0.35
        # holographic shimmer: hue drifts with position
        shimmer = _rainbow(0.35 * (u + v) + 0.6 * q[:, 1]) ** 2.2
        col = 0.78 * col + 0.22 * shimmer
        front = (axis == 2) & (q[:, 2] > 0)
        screen = front & (u > -0.45) & (u < 0.7) & (v > 0.36) & (v < 0.72)
        col[screen] = av._lin("#ff2bd6")
        glow[screen] = 0.9
        lines = screen & (np.abs(((v - 0.36) / 0.09) % 1 - 0.5) < 0.08) & (u < 0.4)
        col[lines] = av._lin("#ffb3f2")
        return col, glow

    def albedo(self, p, ids):
        return self.material(p, ids)[0]

    def material(self, p, ids):
        out = np.zeros(p.shape, np.float32)
        glow = np.zeros(len(p), np.float32)
        for k, (name, *_rest) in enumerate(self.parts):
            sel = ids == k
            if not sel.any():
                continue
            if name == "body":
                out[sel], glow[sel] = self._body(p[sel])
            elif name == "base":
                r = np.hypot(p[sel, 0], p[sel, 2])
                out[sel] = av._lin("#2fe07a")
                glow[sel] = np.where(r > 1.3, 1.0, 0.35)
            elif name in ("post", "handle"):
                out[sel] = av._lin("#e93ad6")
                glow[sel] = 0.25
            elif name == "dial":
                out[sel] = av._lin("#3a6dff")
                glow[sel] = 0.4
            elif name.startswith("led"):
                out[sel] = av._lin("#ffe94d" if name == "led_y" else "#5dff7a")
                glow[sel] = 1.2
            else:
                out[sel] = av._lin(BUTTONS[int(name[3:])][2])
                glow[sel] = 0.45
        return out, glow

    def render(self, height_px=600, yaw=30, pitch=-22, ss=2):
        C = av.rot_y(yaw) @ av.rot_x(pitch)
        right, up, fwd = C @ [1, 0, 0], C @ [0, 1, 0], C @ [0, 0, -1]
        half = 2.3
        H = W = height_px * ss
        s = np.linspace(-half, half, W, dtype=np.float32)
        gx, gy = np.meshgrid(s, -s)
        target = np.float32([0, 1.25, 0])
        o = (target + gx[..., None] * right + gy[..., None] * up - fwd * 10).reshape(-1, 3).astype(np.float32)
        d = np.broadcast_to(fwd.astype(np.float32), o.shape).copy()
        t, hit = self.march(o, d, 20.0)
        rgb = np.zeros((len(o), 3), np.float32)
        idx = np.where(hit)[0]
        pts = o[idx] + d[idx] * t[idx, None]
        _, ids = self.sdf(pts, with_id=True)
        lit = self.shade(pts, ids, right, up, fwd)
        col, glow = self.material(pts, ids)
        rgb[idx] = lit + col * glow[:, None]  # self-lit neon parts
        img, _ = av.finish(rgb, hit, H, W, ss)
        return img


def sprite(width):
    """The vault scaled to `width` px, and its anchor (bottom centre of the pad)."""
    if not os.path.exists(PNG):
        print("  rendering assets/vault.png ...")
        os.makedirs(os.path.dirname(PNG), exist_ok=True)
        Vault().render().save(PNG)
    img = Image.open(PNG).convert("RGBA")
    h = max(1, int(img.height * width / img.width))
    img = img.resize((int(width), h), Image.LANCZOS)
    return img, (img.width / 2, img.height * 0.97)


class Glow:
    """Rainbow neon glow around the vault's outline, colour-cycling over time."""

    def __init__(self, spr):
        self.pad = pad = int(spr.width * 0.35)
        a = Image.new("L", (spr.width + 2 * pad, spr.height + 2 * pad), 0)
        a.paste(spr.getchannel("A"), (pad, pad))
        self.alpha = np.asarray(a.filter(ImageFilter.GaussianBlur(pad * 0.45)), np.float32) / 255
        self.alpha = np.clip(self.alpha * 1.6, 0, 1)
        h, w = self.alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        self.angle = (np.arctan2(yy - h / 2, xx - w / 2) / (2 * np.pi)).astype(np.float32)

    def frame(self, t, strength=1.0):
        rgb = _rainbow(self.angle + 0.35 * t) * 0.85 + 0.15
        a = self.alpha * strength * (0.8 + 0.2 * math.sin(t * 5))
        arr = np.concatenate([rgb, a[..., None]], 2)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")


def sheen(spr, t, period=1.6):
    """A bright diagonal glint sweeping across the vault every `period` seconds."""
    p = (t % period) / period * 1.8 - 0.4
    w, h = spr.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = np.exp(-(((xx / w + yy / h) / 2 - p) / 0.06) ** 2) * 0.55
    arr = np.asarray(spr, np.float32)
    arr[..., :3] += (255 - arr[..., :3]) * band[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def sparkle(size, color=(255, 255, 220)):
    s = max(4, int(size))
    img = Image.new("RGBA", (s * SS, s * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s * SS / 2
    r, w = c, c * 0.22
    d.polygon([(c, c - r), (c + w, c - w), (c + r, c), (c + w, c + w), (c, c + r), (c - w, c + w), (c - r, c),
               (c - w, c - w)], fill=color + (255,))
    return img.resize((s, s), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.4))


if __name__ == "__main__":
    if os.path.exists(PNG):
        os.remove(PNG)
    sprite(100)
    print("wrote", PNG)
