#!/usr/bin/env python3
"""Render smooth, plastic-looking Roblox (R6) avatars to transparent PNGs.

A tiny numpy SDF raymarcher: rounded body parts, soft shadows, ambient occlusion,
the classic smile decal and team kits (stripes, sleeves, shorts). The PNGs are what
edit.py pastes over the players.

  python avatar3d.py out.png --shirt "#1b3f94" --stripes "#a50044" --pants "#1b3f94" --pose run
"""

import argparse
import math

import numpy as np
from PIL import Image

SKIN = "#f5cd30"  # classic Roblox "Bright yellow"

# pose -> {part: (rot_x, rot_z) degrees}. rot_x < 0 swings a limb forward, rot_z moves it sideways.
POSES = {
    "idle": {},
    "run": {"larm": (-45, -8), "rarm": (40, 8), "lleg": (32, 0), "rleg": (-32, 0)},
    "star": {"larm": (0, -55), "rarm": (0, 55), "lleg": (0, -18), "rleg": (0, 18)},
    "tpose": {"larm": (0, -90), "rarm": (0, 90)},
    "cheer": {"larm": (0, -160), "rarm": (0, 160), "lleg": (0, -8), "rleg": (0, 8)},
}


def _rgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _lin(c):
    return _rgb(c) ** 2.2


def rot_x(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_y(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_z(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def sd_round_box(p, b, r):
    q = np.abs(p) - (np.asarray(b, np.float32) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r


def sd_round_cyl(p, ra, rb, h):
    """Rounded cylinder around the y axis: radius ra, edge rounding rb, half height h."""
    dx = np.hypot(p[..., 0], p[..., 2]) - ra + rb
    dy = np.abs(p[..., 1]) - h + rb
    return (np.minimum(np.maximum(dx, dy), 0) + np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) - rb)


class Avatar:
    def __init__(self, shirt, pants, stripes=None, skin=SKIN, socks=None, pose="idle"):
        self.col = {
            "skin": _lin(skin), "shirt": _lin(shirt), "pants": _lin(pants),
            "stripes": _lin(stripes) if stripes else None,
            "socks": _lin(socks) if socks else _lin(pants) * 0.85,
            "shoe": _lin("#1c1c1c"), "ink": _lin("#141414"),
        }
        rots = POSES[pose]

        def R(name):
            rx, rz = rots.get(name, (0, 0))
            return rot_z(rz) @ rot_x(rx)

        # (name, kind, pivot, rotation, centre offset in the part's own frame, size)
        self.parts = [
            ("lleg", "box", (-0.505, 2.0, 0), R("lleg"), (0, -1.0, 0), (0.49, 1.0, 0.5)),
            ("rleg", "box", (0.505, 2.0, 0), R("rleg"), (0, -1.0, 0), (0.49, 1.0, 0.5)),
            ("torso", "box", (0, 3.0, 0), np.eye(3, dtype=np.float32), (0, 0, 0), (1.0, 1.0, 0.5)),
            ("larm", "box", (-1.53, 3.5, 0), R("larm"), (0, -0.5, 0), (0.5, 1.0, 0.5)),
            ("rarm", "box", (1.53, 3.5, 0), R("rarm"), (0, -0.5, 0), (0.5, 1.0, 0.5)),
            ("head", "head", (0, 4.03, 0), np.eye(3, dtype=np.float32), (0, 0.6, 0), None),
        ]

    def local(self, p, part):
        _, _, pivot, Rm, off, _ = part
        return (p - np.asarray(pivot, np.float32)) @ Rm - np.asarray(off, np.float32)

    def sdf(self, p, with_id=False):
        ds = []
        for part in self.parts:
            q = self.local(p, part)
            if part[1] == "box":
                ds.append(sd_round_box(q, part[5], 0.13))
            else:
                ds.append(sd_round_cyl(q, 0.62, 0.24, 0.6))
        ds = np.stack(ds, -1)
        d = ds.min(-1)
        return (d, ds.argmin(-1)) if with_id else d

    def albedo(self, p, ids):
        out = np.zeros(p.shape, np.float32)
        c = self.col
        for k, part in enumerate(self.parts):
            sel = ids == k
            if not sel.any():
                continue
            q = self.local(p[sel], part)
            name = part[0]
            if name == "torso":
                col = np.broadcast_to(c["shirt"], q.shape).copy()
                if c["stripes"] is not None:
                    band = np.floor((q[:, 0] + 1.0) / 0.4).astype(int)
                    col[(band == 1) | (band == 3)] = c["stripes"]
            elif name in ("larm", "rarm"):
                col = np.where((q[:, 1] > 0.4)[:, None], c["shirt"], c["skin"])
            elif name in ("lleg", "rleg"):
                col = np.where((q[:, 1] > 0.05)[:, None], c["pants"], c["socks"])
                col = np.where((q[:, 1] < -0.82)[:, None], c["shoe"], col)
            else:  # head + face decal on the front
                col = np.broadcast_to(c["skin"], q.shape).copy()
                u, v, front = q[:, 0], q[:, 1], q[:, 2] > 0.3
                eyes = (((np.abs(u) - 0.2) / 0.08) ** 2 + ((v - 0.04) / 0.16) ** 2) < 1
                rr = np.hypot(u, v - 0.06)
                ang = np.degrees(np.arctan2(u, -(v - 0.06)))
                smile = (rr > 0.27) & (rr < 0.335) & (np.abs(ang) < 58)
                col[front & (eyes | smile)] = c["ink"]
            out[sel] = col
        return out

    def normal(self, p, eps=2e-3):
        k = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], np.float32)
        n = sum(k[i] * self.sdf(p + k[i] * eps)[..., None] for i in range(4))
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)

    def march(self, o, d, tmax, steps=96):
        t = np.zeros(len(o), np.float32)
        hit = np.zeros(len(o), bool)
        active = np.arange(len(o))
        for _ in range(steps):
            if len(active) == 0:
                break
            dist = self.sdf(o[active] + d[active] * t[active, None])
            t[active] += dist
            done = dist < 1e-3
            hit[active[done]] = True
            active = active[~done & (t[active] < tmax)]
        return t, hit

    def soft_shadow(self, p, L, k=10.0, steps=28):
        res = np.ones(len(p), np.float32)
        t = np.full(len(p), 0.03, np.float32)
        for _ in range(steps):
            h = self.sdf(p + L * t[:, None])
            res = np.minimum(res, np.clip(k * h / t, 0, 1))
            t += np.clip(h, 0.02, 0.4)
        return np.clip(res, 0, 1)

    def ao(self, p, n):
        occ, w = np.zeros(len(p), np.float32), 1.0
        for i in range(1, 6):
            h = 0.06 * i
            occ += (h - self.sdf(p + n * h)) * w
            w *= 0.7
        return np.clip(1 - 2.2 * occ, 0, 1)

    def render(self, height_px=520, yaw=22, pitch=-8, ss=2):
        # orthographic camera framing the whole pose
        C = rot_y(yaw) @ rot_x(pitch)
        right, up, fwd = C @ [1, 0, 0], C @ [0, 1, 0], C @ [0, 0, -1]
        half_w, y0, y1 = 3.6, -0.4, 5.9
        H = height_px * ss
        W = int(H * (2 * half_w) / (y1 - y0)) // ss * ss
        xs = np.linspace(-half_w, half_w, W, dtype=np.float32)
        ys = np.linspace(y1, y0, H, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        target = np.array([0, 2.6, 0], np.float32)
        o = (target + gx[..., None] * right + (gy[..., None] - 2.6) * up - fwd * 12).reshape(-1, 3).astype(np.float32)
        d = np.broadcast_to(fwd.astype(np.float32), o.shape).copy()

        t, hit = self.march(o, d, 24.0)
        rgb = np.zeros((len(o), 3), np.float32)
        idx = np.where(hit)[0]
        p = o[idx] + d[idx] * t[idx, None]
        _, ids = self.sdf(p, with_id=True)
        n = self.normal(p)
        alb = self.albedo(p, ids)

        key = -0.55 * right + 0.95 * up - 0.75 * fwd
        key /= np.linalg.norm(key)
        fill = 0.9 * right + 0.15 * up - 0.5 * fwd
        fill /= np.linalg.norm(fill)
        rim = 0.2 * up + 1.0 * fwd
        rim /= np.linalg.norm(rim)
        view = -fwd

        sh = self.soft_shadow(p + n * 0.01, key.astype(np.float32))
        occ = self.ao(p, n)
        dk = np.clip(n @ key, 0, 1) * sh
        df = np.clip(n @ fill, 0, 1)
        hlf = key + view
        hlf /= np.linalg.norm(hlf)
        spec = np.clip(n @ hlf, 0, 1) ** 40 * sh
        fres = (1 - np.clip(n @ view, 0, 1)) ** 3
        rimv = np.clip(n @ rim, 0, 1) * fres
        sky = 0.5 + 0.5 * n[:, 1]  # hemisphere ambient
        lit = alb * (0.34 * occ * (0.6 + 0.4 * sky) + 0.82 * dk + 0.26 * df)[:, None]
        lit += (0.18 * spec + 0.3 * rimv)[:, None] * np.array([1.0, 0.98, 0.95], np.float32)
        rgb[idx] = lit

        img = np.zeros((len(o), 4), np.float32)
        img[:, :3] = np.clip(rgb, 0, 1) ** (1 / 2.2)
        img[:, 3] = hit
        img = img.reshape(H, W, 4)
        # downsample with premultiplied alpha for clean edges
        img[..., :3] *= img[..., 3:]
        img = img.reshape(H // ss, ss, W // ss, ss, 4).mean((1, 3))
        a = img[..., 3:]
        img[..., :3] = np.where(a > 0, img[..., :3] / np.maximum(a, 1e-6), 0)
        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGBA")
        return out.crop(out.getbbox())


def render_avatar_png(path, shirt, pants, stripes=None, socks=None, pose="idle", height_px=520):
    img = Avatar(shirt, pants, stripes=stripes, socks=socks, pose=pose).render(height_px=height_px)
    img.save(path)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--shirt", default="#0d69ac")
    ap.add_argument("--pants", default="#a4bd47")
    ap.add_argument("--stripes")
    ap.add_argument("--socks")
    ap.add_argument("--pose", default="idle", choices=sorted(POSES))
    ap.add_argument("--height", type=int, default=520)
    a = ap.parse_args()
    render_avatar_png(a.out, a.shirt, a.pants, a.stripes, a.socks, a.pose, a.height)
    print(f"wrote {a.out}")
