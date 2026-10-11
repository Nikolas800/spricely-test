"""Roblox-style characters posed and shaped to fit a real player.

Built from 2D joints marked on the reference frame (head, neck, shoulders, elbows,
wrists, hips, knees, ankles, toes). Every body part is a rounded plastic piece laid
along the player's own bone and made as wide as that limb is in the footage, then
the render is trimmed to the player's silhouette. Rendered with the same raymarcher
and lighting as avatar3d.py.

Joint pairs are [screen-left, screen-right]. Coordinates are frame pixels.
"""

import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import avatar3d as av

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

OUTFITS = {
    "swat": {
        "vest": "#2a2f37", "shirt": "#1a1e25", "pants": "#1f232a", "glove": "#0f1013", "boot": "#0b0c0e",
        "helmet": "#15181d", "pad": "#3b414b", "belt": "#0d0e10", "pouch": "#363c46", "text": "#f2f2f2",
        "skin": av.SKIN,
    },
}


def _text_mask(text="SWAT"):
    font = ImageFont.truetype(FONT, 80)
    left, top, right, bottom = font.getbbox(text)
    im = Image.new("L", (right - left + 10, bottom - top + 10), 0)
    ImageDraw.Draw(im).text((5 - left, 5 - top), text, fill=255, font=font)
    return np.asarray(im) > 127


TEXT = _text_mask()


def W(p):
    """Frame pixel (x, y down) -> world (x, y up)."""
    return np.array([p[0], -p[1]], np.float32)


class Part:
    """Rounded box (or head / helmet) laid along a bone.

    Local axes: x across the bone, y along it (pointing from the distal end towards
    the body), z towards the camera.
    """

    def __init__(self, name, near, far, hw, hl_pad, hd, z=0.0, kind="box", round_=0.6, hl=None):
        a, b = W(near), W(far)
        d = a - b
        length = float(np.hypot(*d))
        along = d / length if length > 1e-6 else np.array([0, 1], np.float32)
        across = np.array([along[1], -along[0]], np.float32)
        self.name, self.kind = name, kind
        self.c = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z], np.float32)
        self.R = np.array([[across[0], along[0], 0], [across[1], along[1], 0], [0, 0, 1]], np.float32)
        self.hw, self.hd = float(hw), float(hd)
        self.hl = float(hl if hl is not None else length / 2 + hl_pad)
        self.r = round_ * min(self.hw, self.hd, self.hl)

    def local(self, p):
        return (p - self.c) @ self.R

    def sdf(self, p):
        q = self.local(p)
        if self.kind == "head":
            return av.sd_round_cyl(q, self.hw, 0.5 * self.hw, self.hl)
        if self.kind == "helmet":  # dome over the top of the head
            q = q - np.array([0, 0.12 * self.hw, 0], np.float32)
            dome = av.sd_round_cyl(q, 1.13 * self.hw, 0.6 * self.hw, 1.02 * self.hl)
            return np.maximum(dome, -(q[..., 1] - 0.18 * self.hw))
        return av.sd_round_box(q, (self.hw, self.hl, self.hd), self.r)

    def extent(self):
        return self.hl + self.hw + 4


def _measure(mask, p, n, maxd):
    """Width of the silhouette through frame point p along unit direction n (frame coords)."""
    H, Wd = mask.shape
    x, y = int(round(p[0])), int(round(p[1]))
    if not (0 <= x < Wd and 0 <= y < H) or not mask[y, x]:
        return None
    total = 0
    for sgn in (1, -1):
        k = 0
        while k < maxd:
            xx, yy = int(round(p[0] + sgn * n[0] * (k + 1))), int(round(p[1] + sgn * n[1] * (k + 1)))
            if not (0 <= xx < Wd and 0 <= yy < H) or not mask[yy, xx]:
                break
            k += 1
        total += k
    return total + 1


def fit_halfwidth(mask, a, b, default, lo, hi, ts=(0.35, 0.5, 0.65)):
    """Half-width of the limb from a to b as seen in the silhouette, clamped to [lo, hi]."""
    if mask is None:
        return default
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = np.hypot(*d)
    if L < 1:
        return default
    n = np.array([-d[1], d[0]]) / L
    ws = [w for w in (_measure(mask, a + t * d, n, int(hi * 2.5)) for t in ts) if w]
    if not ws:
        return default
    # the thinner readings: a limb touching the body or another player reads too wide
    return float(np.clip(np.percentile(ws, 30) / 2, lo, hi))


class Rig(av.Avatar):
    def __init__(self, kp, outfit, mask=None):
        self.kp = kp
        self.swat = outfit == "swat"
        col = OUTFITS["swat"] if self.swat else outfit
        self.col = {k: av._lin(v) for k, v in col.items() if isinstance(v, str)}
        self.col.setdefault("skin", av._lin(av.SKIN))
        self.col.setdefault("boot", av._lin("#141414"))
        self.col["ink"] = av._lin("#141414")
        if not self.swat:
            self.col.setdefault("socks", self.col["pants"] * 0.85)
        self.face = kp.get("face", True)
        self.parts = self._build(kp, mask)
        self.unit = float(np.mean([p.hw for p in self.parts if p.name.endswith("arm_u")]) or 10)

    def _build(self, kp, mask):
        P = {k: np.array(v, float) for k, v in kp.items() if k in ("head", "neck")}
        pairs = {k: [np.array(v, float) for v in kp[k]] for k in
                 ("shoulders", "elbows", "wrists", "hips", "knees", "ankles", "toes") if k in kp}
        hipc = (pairs["hips"][0] + pairs["hips"][1]) / 2
        neck = P["neck"]
        T = float(np.hypot(*(neck - hipc)))  # torso length drives all the default sizes
        front, back = set(kp.get("front", [])), set(kp.get("back", []))
        parts = []

        # torso: upper (chest / vest) and lower (belly / belt)
        mid = neck + 0.55 * (hipc - neck)
        sh_half = float(np.hypot(*(pairs["shoulders"][0] - pairs["shoulders"][1]))) / 2
        tw_u = fit_halfwidth(mask, neck, mid, max(sh_half, 0.33 * T), 0.26 * T, max(1.05 * sh_half, 0.38 * T),
                             ts=(0.4, 0.6, 0.8))
        tw_u = max(tw_u, 0.8 * sh_half)  # keep the shoulders attached
        tw_l = fit_halfwidth(mask, mid, hipc, 0.9 * tw_u, 0.24 * T, max(0.95 * tw_u, 0.34 * T))
        td = 0.42 * tw_u
        parts.append(Part("torso_u", neck, mid, tw_u, 0.06 * T, td))
        parts.append(Part("torso_l", mid, hipc, tw_l, 0.08 * T, td * 0.95))

        # head (+ helmet)
        r = float(kp.get("r", 0.3 * T))
        head, hl = P["head"], 0.92 * r
        up = head - neck
        up = up / max(np.hypot(*up), 1e-6)
        parts.append(Part("head", head + up * r, head - up * r, r, 0, r, kind="head", hl=hl))
        if self.swat:
            parts.append(Part("helmet", head + up * r, head - up * r, r, 0, r, kind="helmet", hl=hl))

        for i, side in enumerate(("l", "r")):
            arm, leg = f"{side}_arm", f"{side}_leg"
            za = td * (1.3 if arm in front else -0.6 if arm in back else 0.45)
            zl = td * (0.9 if leg in front else -0.6 if leg in back else 0.0)
            sh, el, wr = pairs["shoulders"][i], pairs["elbows"][i], pairs["wrists"][i]
            aw = fit_halfwidth(mask, sh, el, 0.1 * T, 0.065 * T, 0.15 * T)
            fw = fit_halfwidth(mask, el, wr, 0.9 * aw, 0.06 * T, 0.14 * T)
            parts.append(Part(f"{side}_arm_u", sh, el, aw, 0.55 * aw, aw * 0.95, z=za))
            parts.append(Part(f"{side}_arm_l", el, wr, fw, 0.5 * fw, fw * 0.95, z=za))
            d = (wr - el) / max(np.hypot(*(wr - el)), 1e-6)
            parts.append(Part(f"{side}_hand", wr - d * 0.2 * fw, wr + d * 1.5 * fw, 1.05 * fw, 0.2 * fw, 0.8 * fw,
                              z=za))

            hp, kn, an = pairs["hips"][i], pairs["knees"][i], pairs["ankles"][i]
            lw = fit_halfwidth(mask, hp, kn, 0.14 * T, 0.09 * T, 0.21 * T)
            sw = fit_halfwidth(mask, kn, an, 0.9 * lw, 0.08 * T, 0.19 * T)
            parts.append(Part(f"{side}_leg_u", hp, kn, lw, 0.5 * lw, lw * 0.95, z=zl))
            parts.append(Part(f"{side}_leg_l", kn, an, sw, 0.5 * sw, sw * 0.95, z=zl))
            toe = pairs["toes"][i] if "toes" in pairs else an + np.array([0.0, 0.4 * sw])
            parts.append(Part(f"{side}_foot", an, toe, 0.85 * sw, 0.7 * sw, 1.0 * sw, z=zl))
            if self.swat:
                # knee pad, centred on the knee and lying along the shin
                dk = (an - kn) / max(np.hypot(*(an - kn)), 1e-6)
                parts.append(Part(f"{side}_pad", kn - dk * 0.75 * sw, kn + dk * 0.75 * sw, 1.15 * sw, 0, 1.1 * sw,
                                  z=zl + 0.2 * sw))
        return parts

    def sdf(self, p, with_id=False):
        ds = np.stack([part.sdf(p) for part in self.parts], -1)
        d = ds.min(-1)
        return (d, ds.argmin(-1)) if with_id else d

    def albedo(self, p, ids):
        out = np.zeros(p.shape, np.float32)
        c, swat = self.col, self.swat
        for k, part in enumerate(self.parts):
            sel = ids == k
            if not sel.any():
                continue
            q = part.local(p[sel])
            u, v = q[:, 0] / part.hw, q[:, 1] / part.hl
            name = part.name
            col = np.empty((len(q), 3), np.float32)
            if name == "head":
                col[:] = c["skin"]
                if self.face:  # classic smile on the front of the head
                    s, fr = part.hw, q[:, 2] > 0.35 * part.hw
                    eu, ev = q[:, 0] / s, q[:, 1] / s
                    eyes = (((np.abs(eu) - 0.3) / 0.12) ** 2 + ((ev - 0.05) / 0.22) ** 2) < 1
                    rr = np.hypot(eu, ev - 0.02)
                    ang = np.degrees(np.arctan2(eu, -(ev - 0.02)))
                    smile = (rr > 0.42) & (rr < 0.53) & (np.abs(ang) < 58)
                    col[fr & (eyes | smile)] = c["ink"]
            elif name == "helmet":
                col[:] = c["helmet"]
                col[q[:, 1] / part.hw < 0.42] = c["helmet"] * 1.6  # rim
            elif name == "torso_u":
                if swat:
                    col[:] = c["vest"]
                    col[v > 0.82] = c["shirt"]
                    # SWAT lettering across the chest (or back)
                    tu, tv = (u + 0.62) / 1.24, (0.3 - v) / 0.75
                    inside = (tu >= 0) & (tu < 1) & (tv >= 0) & (tv < 1)
                    th, tw = TEXT.shape
                    hit = np.zeros(len(q), bool)
                    hit[inside] = TEXT[(tv[inside] * th).astype(int), (tu[inside] * tw).astype(int)]
                    col[hit] = c["text"]
                else:
                    col[:] = c["shirt"]
                    if "stripes" in c:
                        band = np.floor((u + 1.0) / 0.4).astype(int)
                        col[(band == 1) | (band == 3)] = c["stripes"]
            elif name == "torso_l":
                if swat:
                    col[:] = c["vest"]
                    pouch = (np.abs(u) > 0.25) & (np.abs(u) < 0.88) & (v > -0.4) & (v < 0.45)
                    col[pouch] = c["pouch"]
                    edge = pouch & ((np.abs(np.abs(u) - 0.565) > 0.27) | (np.abs(v - 0.025) > 0.38))
                    col[edge] = c["belt"]
                    col[v < -0.55] = c["belt"]
                    col[(v < -0.55) & (np.abs(u) < 0.15)] = c["pad"] * 1.4  # buckle
                else:
                    col[:] = c["shirt"]
                    if "stripes" in c:
                        band = np.floor((u + 1.0) / 0.4).astype(int)
                        col[(band == 1) | (band == 3)] = c["stripes"]
            elif name.endswith("arm_u"):
                col[:] = c["shirt"] if swat else np.where((v > -0.1)[:, None], c["shirt"], c["skin"])
            elif name.endswith("arm_l"):
                col[:] = c["shirt"] if swat else c["skin"]
            elif name.endswith("hand"):
                col[:] = c["glove"] if swat else c["skin"]
            elif name.endswith("leg_u"):
                col[:] = c["pants"]
            elif name.endswith("leg_l"):
                col[:] = c["pants"] if swat else c["socks"]
            elif name.endswith("foot"):
                col[:] = c["boot"]
            elif name.endswith("pad"):
                col[:] = c["pad"]
            out[sel] = col
        return out

    def render(self, ss=2):
        """Returns (RGBA image, (x, y) of its top-left corner in frame pixels)."""
        ext = [(p.c[0], -p.c[1], p.extent()) for p in self.parts]
        x0 = int(min(x - e for x, y, e in ext)) - 2
        x1 = int(max(x + e for x, y, e in ext)) + 2
        y0 = int(min(y - e for x, y, e in ext)) - 2
        y1 = int(max(y + e for x, y, e in ext)) + 2
        Wd, H = (x1 - x0) * ss, (y1 - y0) * ss
        xs = x0 + (np.arange(Wd, dtype=np.float32) + 0.5) / ss
        ys = y0 + (np.arange(H, dtype=np.float32) + 0.5) / ss
        gx, gy = np.meshgrid(xs, ys)
        zmax = max(abs(p.c[2]) + p.extent() for p in self.parts)
        o = np.stack([gx, -gy, np.full_like(gx, zmax)], -1).reshape(-1, 3)
        d = np.broadcast_to(np.float32([0, 0, -1]), o.shape).copy()
        t, hit = self.march(o, d, 2 * zmax, steps=80)
        rgb = np.zeros((len(o), 3), np.float32)
        idx = np.where(hit)[0]
        if len(idx):
            pts = o[idx] + d[idx] * t[idx, None]
            _, ids = self.sdf(pts, with_id=True)
            rgb[idx] = self.shade(pts, ids, np.float32([1, 0, 0]), np.float32([0, 1, 0]), np.float32([0, 0, -1]))
        img, (cx, cy) = av.finish(rgb, hit, H, Wd, ss)
        return img, (x0 + cx, y0 + cy)


def render_fitted(kp, outfit, mask=None, cache_dir=None):
    """Fitted character for one player, cached on disk by its inputs."""
    key = None
    if cache_dir:
        h = hashlib.sha1(json.dumps([kp, outfit], sort_keys=True).encode())
        for mod in (__file__, av.__file__):  # re-render when the renderer changes
            with open(mod, "rb") as fh:
                h.update(fh.read())
        if mask is not None:
            h.update(np.packbits(mask).tobytes())
        key = os.path.join(cache_dir, h.hexdigest()[:16])
        if os.path.exists(key + ".png"):
            with open(key + ".json") as fh:
                return Image.open(key + ".png").convert("RGBA"), tuple(json.load(fh))
    img, off = Rig(kp, outfit, mask).render()
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        img.save(key + ".png")
        with open(key + ".json", "w") as fh:
            json.dump([int(off[0]), int(off[1])], fh)
    return img, off
