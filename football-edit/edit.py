#!/usr/bin/env python3
"""Football -> Roblox edit.

Plays the clip as-is (same footage, same audio). At the transform moment (the Nth
boom after the skull, or `skull_hold` seconds after it) every annotated player gets
a Roblox character over them, warped to their shape, and the ball turns into a
holographic vault.

Two ways to do it:
  overlay_only: true   the video is left exactly as it is; the characters/vault are
                       drawn on top and tracked to the shot (follows the edit's
                       shake, zoom and brightness flicker/fades)
  overlay_only: false  freeze the frame, flash + shake, then the Roblox scene

  python edit.py contact input/clip.mp4 --from 2 --to 6      # find the skull moment
  python edit.py grab config.json                            # transform frame + pixel grid
  python edit.py preview config.json                         # one still of the effect
  python edit.py render config.json                          # final mp4
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import avatar3d
import puppet
import stage
import vault

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DEFAULTS = {
    "output": "output/football_roblox_edit.mp4",
    "upscale": 1,
    "overlay_only": False,
    "skull_hold": 1.0,
    "on_boom": None,
    "transform_time": None,
    "ref_time": None,
    "appear": "instant",
    "remove_players": True,
    "freeze_on_skull": True,
    "add_skull": False,
    "skull": {"x": 0.5, "y": 0.42, "size": 0.42, "after": "pop"},
    "keep_on_top": [],
    "erase": [],
    "kits": {},
    "fit": [1.1, 1.08],
    "players": [],
    "ball": None,
    "swap": None,
    "hero": None,
    "end_time": None,
    "flash": True,
    "shake": True,
    "slow_zoom": 0.05,
    "sfx": False,
    "crf": 18,
}


# ---------------------------------------------------------------- video io

def probe(path, upscale=1):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate:format=duration",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int((v.get("tags") or {}).get("rotate", 0))
    w, h = v["width"], v["height"]
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = Fraction(v["r_frame_rate"])
    if fps <= 0 or fps > 240:
        fps = Fraction(30)
    return {
        "w": w * upscale, "h": h * upscale, "upscale": upscale, "fps": fps,
        "duration": float(info["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def iter_frames(path, meta, max_frames=None):
    """Yield RGB uint8 frames at a constant frame rate (meta['fps']), upscaled if asked."""
    w, h = meta["w"] & ~1, meta["h"] & ~1
    vf = f"fps={meta['fps']}"
    if meta["upscale"] != 1:
        vf += f",scale={meta['w']}:{meta['h']}:flags=lanczos"
    vf += f",crop={w}:{h}:0:0"
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    if max_frames:
        cmd[-1:-1] = ["-frames:v", str(max_frames)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def frames_at(path, meta, indices):
    """Return {index: frame} for the requested frame indices."""
    want, got = set(indices), {}
    for i, f in enumerate(iter_frames(path, meta, max_frames=max(want) + 1)):
        if i in want:
            got[i] = f.copy()
    missing = want - set(got)
    if missing:
        sys.exit(f"could not read frame(s) {sorted(missing)} from {path}")
    return got


# ---------------------------------------------------------------- config

def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    for k in ("input", "output"):
        if not os.path.isabs(out[k]):
            out[k] = os.path.join(base, out[k])
    if "skull_time" not in out:
        sys.exit("config needs skull_time (seconds) - use `edit.py contact` to find it")
    out["_base"] = base
    if out["transform_time"] is None:
        if out["on_boom"]:
            booms = find_booms(out["input"], out["skull_time"])
            print("booms after the skull: " + ", ".join(f"#{k + 1} {b:.2f}s" for k, b in enumerate(booms)))
            if len(booms) < out["on_boom"]:
                sys.exit(f"only found {len(booms)} booms after the skull, can't use boom #{out['on_boom']}")
            out["transform_time"] = booms[out["on_boom"] - 1]
        else:
            out["transform_time"] = out["skull_time"] + out["skull_hold"]
    if out["overlay_only"]:
        # the video stays exactly as it is: no freeze, no added skull, no camera fx
        out.update(freeze_on_skull=False, add_skull=False, flash=False, shake=False, slow_zoom=0.0)
    return out


def timeline(cfg, meta):
    """Frame indices: skull hit, transformation, reference frame (the one the boxes are measured on), end."""
    fps = float(meta["fps"])
    skull_i = int(round(cfg["skull_time"] * fps))
    trans_i = int(round(cfg["transform_time"] * fps))
    if cfg["freeze_on_skull"]:
        ref_i = skull_i
    else:
        ref_i = int(round(cfg["ref_time"] * fps)) if cfg["ref_time"] is not None else trans_i
    end_t = cfg["end_time"] if cfg["end_time"] is not None else meta["duration"]
    end_i = int(math.floor(end_t * fps + 1e-6))
    return fps, skull_i, trans_i, ref_i, end_i


# ---------------------------------------------------------------- audio

def find_booms(path, after, sr=22050):
    """Times (s) of the bass hits after `after`: where the bass comes back in after a dip."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    hop, win = 128, 1024
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    e = np.convolve(spec[:, (freqs > 30) & (freqs < 160)].sum(1), np.ones(3) / 3, mode="same")
    t = (np.arange(len(e)) * hop + win / 2) / sr
    rad = int(0.25 * sr / hop)
    booms = []
    for i in range(len(e)):
        lo, hi = max(0, i - rad), min(len(e), i + rad)
        if e[i] == e[lo:hi].min() and e[i] < 0.6 * e[lo:hi].max():
            # the hit: where the bass has climbed a third of the way back up after the dip
            peak = e[i:hi].max()
            j = i + int(np.argmax(e[i:hi] >= e[i] + 0.3 * (peak - e[i])))
            if not booms or t[j] - booms[-1] > 0.15:
                booms.append(float(t[j]))
    return [b for b in booms if b > after + 0.15]


# ---------------------------------------------------------------- masks / inpainting

def grass(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 35)


def player_mask(frame, box, pad=0.12):
    """Mask of the player inside `box`: non-grass pixels, closed and dilated."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad * 0.5)))
    m = (~grass(frame[y0:y1, x0:x1])).astype(np.uint8) * 255
    k = max(3, int(min(w, h) * 0.13)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    # if the background isn't pitch the whole box goes; fine, the avatar covers it
    return full


def silhouette(frame, box, pad=0.3):
    """The player's own outline: non-grass pixels around `box` (bool), lightly closed."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = (~grass(frame[y0:y1, x0:x1])).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def clip_sprite(spr, off, sil, grow=4):
    """Trim a sprite placed at `off` to the (slightly grown, soft-edged) silhouette."""
    k = 2 * grow + 1
    soft = cv2.GaussianBlur(cv2.dilate(sil.astype(np.uint8) * 255, np.ones((k, k), np.uint8)).astype(np.float32),
                            (0, 0), 1.5) / 255
    x, y = int(off[0]), int(off[1])
    H, W = sil.shape
    m = np.zeros((spr.height, spr.width), np.float32)
    sx0, sy0, sx1, sy1 = max(0, -x), max(0, -y), min(spr.width, W - x), min(spr.height, H - y)
    if sx1 > sx0 and sy1 > sy0:
        m[sy0:sy1, sx0:sx1] = soft[y + sy0:y + sy1, x + sx0:x + sx1]
    arr = np.asarray(spr).copy()
    arr[..., 3] = (arr[..., 3] * m).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def solid_mask(frame, box):
    """Filled mask of the non-grass blob in `box` (e.g. an emoji sitting on the pitch)."""
    H, W = frame.shape[:2]
    x, y, w, h = [int(v) for v in box]
    m = (~grass(frame[y:y + h, x:x + w])).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill holes (eyes etc.): flood the outside, whatever isn't outside is inside
    flood = np.pad(m, 1)
    cv2.floodFill(flood, None, (0, 0), 128)
    m = np.where(flood[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)
    full = np.zeros((H, W), np.float32)
    full[y:y + h, x:x + w] = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0)
    return full


def inpaint(frame, mask, down=2):
    """Paint out `mask` (uint8) from `frame`. Works at 1/down resolution for smooth fills."""
    if not mask.any():
        return frame
    H, W = mask.shape
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (W // down, H // down), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(mask, (W // down, H // down), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(small, sm, 6, cv2.INPAINT_TELEA)
    fill = cv2.GaussianBlur(cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), down)
    fill = cv2.cvtColor(fill, cv2.COLOR_BGR2RGB)
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.5 * down)[..., None]
    return (frame * (1 - a) + fill * a).astype(np.uint8)


def shift(img, dx, dy):
    """img sampled at (x + dx, y + dy)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def clone_fill(img, mask, dx, dy, soft=3):
    """Fill `mask` with the texture found (dx, dy) away, e.g. more goal net."""
    if not mask.any():
        return img
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), soft)[..., None]
    return (img * (1 - a) + shift(img, dx, dy) * a).astype(np.uint8)


def _no_room(x0, x1, img):
    """True when there's no hole-sized strip of the frame left or right of [x0, x1]."""
    w = x1 - x0 + 4
    return x0 < w and img.shape[1] - 1 - x1 < w


def best_clone_offset(frame, mask, down=3):
    """Horizontal/vertical offset whose texture best continues the surroundings of `mask`."""
    small = cv2.resize(frame, (frame.shape[1] // down, frame.shape[0] // down), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
    xs = np.where(m.any(0))[0]
    x0, x1 = xs.min(), xs.max()
    src = small.astype(np.float32)
    best, best_cost = (0, 0), None
    for dx in range(-small.shape[1], small.shape[1]):
        # copy from beside the hole (never from inside it), and from inside the frame if possible
        if abs(dx) < x1 - x0 + 4 or (x0 + dx < 0 or x1 + dx >= small.shape[1]) and not _no_room(x0, x1, src):
            continue
        for dy in range(-4, 5, 2):
            s = shift(src, dx, dy)
            cost = np.abs(s[ring] - src[ring]).mean()
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
    return best[0] * down, best[1] * down


def avatar_sprite(p, cfg, frame):
    """The PNG pasted over a player: their own `png`, or a rendered Roblox avatar in their kit."""
    if p.get("png"):
        path = p["png"] if os.path.isabs(p["png"]) else os.path.join(cfg["_base"], p["png"])
        img = Image.open(path).convert("RGBA")
        return img.crop(img.getbbox())
    kit = dict(cfg["kits"].get(p.get("kit"), {}))
    for k in ("shirt", "stripes", "pants", "socks"):
        if p.get(k):
            kit[k] = p[k]
    if "shirt" not in kit or "pants" not in kit:
        shirt, pants = sample_colors(frame, p["box"])
        kit.setdefault("shirt", "#%02x%02x%02x" % shirt)
        kit.setdefault("pants", "#%02x%02x%02x" % pants)
    pose = p.get("pose", "idle")
    if pose not in avatar3d.POSES:
        sys.exit(f"unknown pose {pose!r}; pick one of {', '.join(sorted(avatar3d.POSES))}")
    name = p.get("kit") or "_".join(kit[k].lstrip("#") for k in ("shirt", "stripes", "pants") if kit.get(k))
    path = os.path.join(HERE, "assets", "avatars", f"{name}_{pose}.png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  rendering avatar {os.path.relpath(path, HERE)} ...")
        avatar3d.render_avatar_png(path, kit["shirt"], kit["pants"], kit.get("stripes"), kit.get("socks"), pose)
    return Image.open(path).convert("RGBA")


def sample_colors(frame, box):
    """Guess shirt / shorts colour from the player box (ignoring grass)."""
    x, y, w, h = [int(v) for v in box]

    def band(fy0, fy1):
        reg = frame[y + int(h * fy0):y + int(h * fy1), x + int(w * 0.3):x + int(w * 0.7)].reshape(-1, 3)
        if len(reg) == 0:
            return (200, 200, 200)
        keep = ~grass(reg[None])[0]
        reg = reg[keep] if keep.sum() > 10 else reg
        c = np.median(reg, axis=0)
        # punch up saturation a bit so it reads as a Roblox colour
        hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
        hsv[1] = min(255, int(hsv[1] * 1.35))
        hsv[2] = min(255, int(hsv[2] * 1.1 + 10))
        return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0])

    return band(0.18, 0.42), band(0.47, 0.6)


# ---------------------------------------------------------------- tracking

class Tracker:
    """Follows the shot from the reference frame: per-frame affine warp + brightness gain.

    Built for edits that freeze the picture and then shake / zoom / flicker it, so the
    overlays stay glued to the players and fade with the video.
    """

    def __init__(self, ref, k):
        self.k = max(1, int(k))
        self.ref = self._small(ref)
        self.orb = cv2.ORB_create(3000)
        self.kp, self.des = self.orb.detectAndCompute(self.ref, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last = np.float32([[1, 0, 0], [0, 1, 0]])

    def _small(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.k > 1:
            g = cv2.resize(g, (g.shape[1] // self.k, g.shape[0] // self.k), interpolation=cv2.INTER_AREA)
        return g

    def track(self, frame):
        """Returns (M, a, b): M maps reference coords -> this frame; colour ~ a*ref + b."""
        g = self._small(frame)
        h, w = g.shape
        M = self.last.copy()
        kp, des = self.orb.detectAndCompute(g, None)
        if des is not None and self.des is not None and len(kp) >= 30:
            m = self.bf.match(self.des, des)
            if len(m) >= 30:
                src = np.float32([self.kp[x.queryIdx].pt for x in m])
                dst = np.float32([kp[x.trainIdx].pt for x in m])
                est, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=2.0)
                if est is not None and inl.sum() >= 40:
                    M = est.astype(np.float32)
        try:  # sub-pixel refine; ECC is insensitive to brightness/contrast changes
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            _, M2 = cv2.findTransformECC(self.ref.astype(np.float32), g.astype(np.float32), M.copy(),
                                         cv2.MOTION_AFFINE, crit, None, 5)
            if np.abs(M2[:, :2] - M[:, :2]).max() < 0.02 and np.abs(M2[:, 2] - M[:, 2]).max() < 4:
                M = M2.astype(np.float32)
        except cv2.error:
            pass
        self.last = M

        rw = cv2.warpAffine(self.ref, M, (w, h)).astype(np.float32)
        valid = cv2.erode(cv2.warpAffine(np.full_like(self.ref, 255), M, (w, h)), np.ones((9, 9), np.uint8)) > 0
        x, y = rw[valid][::5], g[valid][::5].astype(np.float32)
        if len(x) > 100 and x.std() > 1:
            a, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
        else:
            a, b = 1.0, 0.0
        a = float(np.clip(a, 0.0, 2.0))

        # small-image transform -> full-res transform (pixel-centre aware)
        A, t = M[:, :2], M[:, 2]
        c = np.float32([(self.k - 1) / 2] * 2)
        Mf = np.hstack([A, (self.k * t + c - A @ c)[:, None]]).astype(np.float32)
        return Mf, a, float(b)


# ---------------------------------------------------------------- easing

def ease_out_back(p, s=1.9):
    p = min(max(p, 0.0), 1.0) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def ease_out_elastic(p):
    p = min(max(p, 0.0), 1.0)
    if p in (0.0, 1.0):
        return p
    return 2 ** (-9 * p) * math.sin((p * 10 - 0.75) * (2 * math.pi) / 3.2) + 1


# ---------------------------------------------------------------- scene

def _over(dst, spr, x, y):
    """Alpha-composite `spr` onto RGBA `dst` with its top-left at (x, y), clipping at the edges."""
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.width, x + spr.width), min(dst.height, y + spr.height)
    if x1 <= x0 or y1 <= y0:
        return
    dst.alpha_composite(spr.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def composite(base, ov, a=1.0, b=0.0):
    """base RGB uint8 + premultiplied RGBA float overlay (colours scaled by a, offset by b)."""
    alpha = ov[..., 3:] / 255.0
    rgb = ov[..., :3] * a + b * alpha
    return np.clip(base * (1 - alpha) + rgb, 0, 255).astype(np.uint8)


def premultiply(img):
    arr = np.asarray(img, np.float32)
    arr[..., :3] *= arr[..., 3:] / 255.0
    return arr


class Scene:
    """Everything that happens from the transform moment on, in reference-frame coords."""

    def __init__(self, frame, cfg):
        self.cfg = cfg
        self.frame = frame
        self.H, self.W = frame.shape[:2]
        self.down = max(2, int(cfg["upscale"]))
        self.masks = [player_mask(frame, p["box"]) for p in cfg["players"]]
        self.erase = np.zeros((self.H, self.W), np.uint8)
        for x, y, w, h in cfg["erase"]:
            self.erase[int(y):int(y + h), int(x):int(x + w)] = 255
        self.on_top = np.zeros((self.H, self.W), np.float32)
        for box in cfg["keep_on_top"]:
            self.on_top = np.maximum(self.on_top, solid_mask(frame, box))
        self._plate = None

        self.actors = []
        poof = cfg["appear"] == "poof"
        order = sorted(range(len(cfg["players"])), key=lambda i: cfg["players"][i]["box"][0])
        for n, i in enumerate(order):
            p = cfg["players"][i]
            x, y, w, h = p["box"]
            if p.get("rig"):
                # one Roblox character image, puppet-warped onto the player's skeleton and outline
                sil = silhouette(frame, p["box"])
                outfit = p.get("outfit", "swat")
                outfit = outfit if outfit == "swat" else cfg["kits"][outfit]
                print(f"  warping onto player {i} ({p.get('outfit', 'swat')}) ...")
                tpl = puppet.load_template(outfit, back=not p["rig"].get("face", True))
                spr, (ox, oy) = puppet.warp(tpl, p["rig"], sil, order=puppet.paint_order(p["rig"]))
                if p.get("clip", True):
                    spr = clip_sprite(spr, (ox, oy), sil)
                foot = (ox + spr.width / 2, oy + spr.height)
            else:
                # the avatar PNG is stretched to the player's box, whatever shape that is
                fw, fh = p.get("fit", cfg["fit"])
                spr = avatar_sprite(p, cfg, frame)
                spr = spr.resize((max(1, int(w * fw)), max(1, int(h * fh))), Image.LANCZOS)
                if p.get("flip"):
                    spr = spr.transpose(Image.FLIP_LEFT_RIGHT)
                foot = tuple(p.get("foot", (x + w / 2, y + h)))
            clone = None
            if p.get("fill") == "clone":
                clone = best_clone_offset(frame, self.masks[i], self.down)
                print(f"  player {i}: clone fill from offset {clone}")
            self.actors.append({
                "kind": "player", "idx": i, "sprite": spr, "anchor": (spr.width / 2, spr.height),
                "foot": foot, "delay": 0.035 * n if poof else 0.0,
                "mask": self.masks[i], "clone": clone,
            })
        nothing = np.zeros((self.H, self.W), np.uint8)
        if cfg["ball"]:
            (bx, by), r = cfg["ball"]["center"], cfg["ball"]["radius"]
            cw = cfg["ball"].get("width", max(r * 7.5, min(self.W, self.H) * 0.15))
            ball = nothing.copy()
            cv2.circle(ball, (int(bx), int(by)), int(r * 1.7 + 3), 255, -1)
            self.actors.append(self._vault((bx, by + r), cw, ball, 0.08 if poof else 0.0))
        self.swap = self.look = None
        if cfg["swap"]:  # the whole picture becomes this Roblox scene
            self.swap = stage.canvas(frame, cfg["swap"], cfg["_base"])
            self.rows = cfg["swap"].get("rows") or stage.content_rows(frame)
        if cfg["hero"]:  # the Roblox player in front, carrying the vault on their head
            h = cfg["hero"]
            spr = stage.hero_sprite(h, cfg["kits"], h["height"])
            fx_, fy_ = h["foot"]
            self.actors.append({"kind": "player", "idx": -2, "sprite": spr, "anchor": (spr.width / 2, spr.height),
                                "foot": (fx_, fy_), "delay": 0.0, "mask": nothing, "clone": None, "front": 1})
            if h.get("prop", "vault") == "vault":
                top = fy_ - spr.height * (1 - h.get("head_sink", 0.03))
                v = self._vault((fx_, top), h.get("prop_width", spr.width * 0.8), nothing, 0.0)
                v["front"] = 2
                self.actors.append(v)
        # far to near; the hero and what they carry always go on top
        self.actors.sort(key=lambda a: (a.get("front", 0), a["foot"][1]))
        if self.swap is not None:  # grade the new scene (with everyone in it) to match the clip
            scene = composite(self.swap, premultiply(self.layer(1.0)[0]))
            self.look = stage.Look(frame, scene, self.rows, cfg["swap"].get("look"), cfg["upscale"])

    def _vault(self, foot, width, mask, delay):
        spr, anchor = vault.sprite(width)
        return {
            "kind": "vault", "idx": -1, "sprite": spr, "anchor": anchor, "foot": foot, "delay": delay,
            "glow": vault.Glow(spr), "cw": width, "mask": mask,
            "sparkles": [(random.Random(k).uniform(-0.8, 0.8), random.Random(k + 50).uniform(0.1, 1.4),
                          random.Random(k + 99).uniform(0, 6.28)) for k in range(7)],
        }

    # -- pieces

    def _poof(self, draw, cx, cy, size, tau, seed):
        if not 0 <= tau < 0.5:
            return
        rnd = random.Random(seed)
        p = tau / 0.5
        for _ in range(14):
            ang = rnd.uniform(0, 6.28)
            dist = size * (0.15 + 0.85 * rnd.random()) * (0.3 + 0.9 * (1 - (1 - p) ** 3))
            r = size * rnd.uniform(0.18, 0.34) * (1.0 - 0.6 * p)
            x, y = cx + math.cos(ang) * dist, cy + math.sin(ang) * dist * 0.8
            g = rnd.randint(225, 255)
            a = int(255 * (1 - p) ** 1.4)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(g, g, g, a))

    def _put(self, ov, spr, anchor, foot, sx, sy):
        w, h = max(1, int(spr.width * sx)), max(1, int(spr.height * sy))
        _over(ov, spr.resize((w, h), Image.BILINEAR), foot[0] - anchor[0] * sx, foot[1] - anchor[1] * sy)

    def layer(self, t):
        """Overlay at t seconds after the transform.

        Returns (rgba overlay image, holes, mask of players still waiting for their poof).
        holes = (mask to inpaint, [(mask, dx, dy) to clone-fill]) for what's transformed so far.
        """
        size = (self.W, self.H)
        hole, clones = self.erase.copy(), []
        pending = np.zeros((self.H, self.W), np.uint8)
        for a in self.actors:
            if not self.cfg["remove_players"]:
                continue
            if t < a["delay"]:
                pending |= a["mask"]
            elif a.get("clone"):
                clones.append((a["mask"], *a["clone"]))
            else:
                hole |= a["mask"]

        ov = Image.new("RGBA", size, (0, 0, 0, 0))
        shadows = Image.new("RGBA", size, (0, 0, 0, 0))
        shd = ImageDraw.Draw(shadows)
        poof = self.cfg["appear"] == "poof"
        for a in self.actors:
            tau = t - a["delay"]
            if a["kind"] == "player" and tau >= 0 and self.cfg["remove_players"]:
                wd = a["sprite"].width * 0.95 * (min(1.0, tau / 0.12) if poof else 1.0)
                x, y = a["foot"]
                shd.ellipse([x - wd / 2, y - wd * 0.14, x + wd / 2, y + wd * 0.14], fill=(0, 0, 0, 80))
        ov.alpha_composite(shadows)

        fx = Image.new("RGBA", size, (0, 0, 0, 0))
        fxd = ImageDraw.Draw(fx)
        for a in self.actors:
            tau = t - a["delay"]
            if tau < 0:
                continue
            spr = a["sprite"]
            if a["kind"] == "player":
                if not poof:  # just pasted on, straight away
                    _over(ov, spr, a["foot"][0] - a["anchor"][0], a["foot"][1] - a["anchor"][1])
                    continue
                k = ease_out_elastic(tau / 0.6)
                if k > 0.02:
                    self._put(ov, spr, a["anchor"], a["foot"], k, k)
                self._poof(fxd, a["foot"][0], a["foot"][1] - spr.height * 0.5, spr.height * 0.45, tau, a["idx"])
            else:
                cw = a["cw"]
                fx_, fy_ = a["foot"]
                glow_a = min(1.0, max(0.0, (tau - 0.1) / 0.4)) if poof else 1.0
                if glow_a > 0:  # rainbow neon glow around the vault
                    g = a["glow"]
                    _over(ov, g.frame(tau, glow_a), fx_ - a["anchor"][0] - g.pad, fy_ - a["anchor"][1] - g.pad)
                k = ease_out_elastic(tau / 0.7) if poof and tau < 0.7 else 1 + 0.02 * math.sin(tau * 4)
                if k > 0.02:
                    # squash & stretch: wide while it grows, narrow while it overshoots
                    sx = math.sqrt(k) if k < 1 else 1 - 0.6 * (k - 1)
                    self._put(ov, vault.sheen(spr, tau), a["anchor"], (fx_, fy_), sx, k)
                if tau > 0.25 or not poof:
                    for ox, oy, ph in a["sparkles"]:
                        tw = 0.5 + 0.5 * math.sin(tau * 6 + ph)
                        if tw < 0.25:
                            continue
                        s = vault.sparkle(cw * 0.12 * tw)
                        _over(ov, s, fx_ + ox * cw - s.width / 2, fy_ - oy * cw * 0.75 - s.height / 2)
                if poof:
                    self._poof(fxd, fx_, fy_ - cw * 0.3, cw * 0.55, tau, 999)
        ov.alpha_composite(fx)
        return ov, (hole, clones), pending

    def clean(self, img, holes, M=None):
        """Remove transformed players/ball from `img` (warping the masks by M if given)."""
        hole, clones = holes
        H, W = img.shape[:2]

        def w(m):
            return m if M is None else cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
        for m, dx, dy in clones:
            img = clone_fill(img, w(m), dx, dy, soft=self.down)
        return inpaint(img, w(hole), self.down)

    def render_frozen(self, t):
        """Freeze-frame version: overlay on the cleaned-up still, plus flash / shake / zoom."""
        if self._plate is None:
            self._plate = self.clean(self.frame, self.layer(1e9)[1])
        ov, _, pending = self.layer(t)
        # players whose poof hasn't started yet are still the real footage
        base = np.where(pending[..., None] > 0, self.frame, self._plate)
        out = composite(base, premultiply(ov)).astype(np.float32)
        if self.cfg["flash"] and t < 0.18:
            out = out + (255 - out) * (0.85 * (1 - t / 0.18))
        out = np.clip(out, 0, 255).astype(np.uint8)
        dx = dy = 0.0
        if self.cfg["shake"] and t < 0.3:
            amp = 0.018 * self.W * (1 - t / 0.3)
            r = random.Random(int(t * 1000))
            dx, dy = r.uniform(-amp, amp), r.uniform(-amp, amp)
        z = 1 + self.cfg["slow_zoom"] * min(1.0, t / 4.0)
        return warp(out, z, dx, dy)

    def render_tracked(self, live, t, M, a, b):
        """Overlay-only version: the live frame stays, overlay follows the shot via M."""
        ov, holes, _ = self.layer(t)
        H, W = live.shape[:2]
        if self.swap is not None:
            # straight clip swap: the Roblox scene + characters replace the whole picture, but keep the
            # edit's own shake, flicker and fade-out
            scene = composite(self.swap, premultiply(ov))
            scene = self.look.apply(scene, seed=int(t * 1000))
            fx = self.cfg["swap"].get("punch", {})
            scene = stage.punch(scene, t, self.rows, fx.get("zoom", 0.08), fx.get("split", 10.0), fx.get("length", 0.25))
            scene = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return np.clip(scene.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        base = self.clean(live, holes, M)
        ovw = cv2.warpAffine(premultiply(ov), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        out = composite(base, ovw, a, b)
        if self.on_top.any():  # keep e.g. the clip's own 💀 in front of the avatars
            top = cv2.warpAffine(self.on_top, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
            out = (out * (1 - top) + live * top).astype(np.uint8)
        return out


def warp(img, zoom, dx, dy):
    if zoom == 1 and dx == 0 and dy == 0:
        return img
    H, W = img.shape[:2]
    M = np.float32([[zoom, 0, (1 - zoom) * W / 2 + dx], [0, zoom, (1 - zoom) * H / 2 + dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Skull:
    def __init__(self, cfg, W, H):
        s = cfg["skull"]
        self.size = int(min(W, H) * s["size"])
        self.c = (W * s["x"], H * s["y"])
        self.after = s.get("after", "pop")
        self.img = Image.open(os.path.join(HERE, "assets", "skull.png")).convert("RGBA")

    def draw(self, frame, t_hit, t_trans):
        """t_hit: seconds since the skull hit; t_trans: seconds since transform (<0 before)."""
        if t_hit < 0:
            return frame
        k = ease_out_back(t_hit / 0.18, 2.6)
        if t_trans >= 0 and self.after == "pop":
            k *= max(0.0, 1 - t_trans / 0.12)
        if k <= 0.01:
            return frame
        size = max(2, int(self.size * k))
        rot = 8 * math.sin(t_hit * 18) * max(0.0, 1 - t_hit / 0.4)
        s = self.img.resize((size, size), Image.LANCZOS).rotate(rot, resample=Image.BICUBIC, expand=True)
        im = Image.fromarray(frame)
        im.paste(s, (int(self.c[0] - s.width / 2), int(self.c[1] - s.height / 2)), s)
        out = np.asarray(im)
        if t_hit < 0.25:
            amp = 0.02 * frame.shape[1] * (1 - t_hit / 0.25)
            r = random.Random(int(t_hit * 1000) + 7)
            out = warp(out, 1.0, r.uniform(-amp, amp), r.uniform(-amp, amp))
        return out


# ---------------------------------------------------------------- sfx

def write_sfx(path, cfg, sr=44100):
    """Synthesised pop + boing + chime, positioned on the transform moment."""
    t0 = cfg["transform_time"]
    n = int(sr * (t0 + 2.5))
    out = np.zeros(n, np.float32)

    def add(sig, at):
        i = int(at * sr)
        out[i:i + len(sig)] += sig[:max(0, n - i)]

    t = np.arange(int(sr * 0.15)) / sr
    pop = (np.random.default_rng(1).uniform(-1, 1, len(t)) * 0.5 + np.sin(2 * np.pi * 90 * t)) * np.exp(-t * 35)
    add(pop * 0.6, t0)
    t = np.arange(int(sr * 0.6)) / sr
    f = 160 + 420 * (1 - np.exp(-t * 6)) + 25 * np.sin(2 * np.pi * 14 * t)
    boing = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 4.5)
    add(boing * 0.45, t0 + 0.30)
    for j, fr in enumerate((1318.5, 1568, 2093)):
        t = np.arange(int(sr * 0.35)) / sr
        add(np.sin(2 * np.pi * fr * t) * np.exp(-t * 9) * 0.18, t0 + 0.25 + j * 0.07)
    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(sr)
        wv.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- commands

class Composer:
    """Decides what each output frame is."""

    def __init__(self, cfg, meta):
        self.cfg = cfg
        self.fps, self.skull_i, self.trans_i, self.ref_i, self.end_i = timeline(cfg, meta)
        self.skull = Skull(cfg, meta["w"] & ~1, meta["h"] & ~1) if cfg["add_skull"] else None
        self.freeze = self.scene = self.tracker = None

    def reference(self, frame):
        self.freeze = frame.copy()
        self.scene = Scene(self.freeze, self.cfg)
        if self.cfg["overlay_only"]:
            self.tracker = Tracker(self.freeze, self.cfg["upscale"])

    def frame(self, frame, i):
        cfg = self.cfg
        t = (i - self.trans_i) / self.fps
        if i >= self.trans_i and cfg["overlay_only"]:
            return self.scene.render_tracked(frame, t, *self.tracker.track(frame))
        if i >= self.trans_i:
            out = self.scene.render_frozen(t)
        elif i >= self.skull_i and cfg["freeze_on_skull"]:
            out = self.freeze
        else:
            out = frame
        if self.skull is not None:
            out = self.skull.draw(out, (i - self.skull_i) / self.fps, t)
        return out


def cmd_render(args):
    cfg = load_config(args.config)
    meta = probe(cfg["input"], cfg["upscale"])
    comp = Composer(cfg, meta)
    fps, end_i = comp.fps, comp.end_i
    W, H = meta["w"] & ~1, meta["h"] & ~1
    os.makedirs(os.path.dirname(cfg["output"]) or ".", exist_ok=True)
    print(f"{W}x{H} @ {fps:.3f}fps  skull@{comp.skull_i}  transform@{comp.trans_i}  "
          f"reference@{comp.ref_i}  end@{end_i}  {'overlay-only' if cfg['overlay_only'] else 'freeze'}")
    comp.reference(frames_at(cfg["input"], meta, [comp.ref_i])[comp.ref_i])

    tmpdir = tempfile.mkdtemp()
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(meta["fps"]), "-i", "-"]
    audio_inputs = []
    if meta["audio"]:
        cmd += ["-i", cfg["input"]]
        audio_inputs.append("1:a")
    if cfg["sfx"]:
        sfx = os.path.join(tmpdir, "sfx.wav")
        write_sfx(sfx, cfg)
        cmd += ["-i", sfx]
        audio_inputs.append(f"{len(audio_inputs) + 1}:a")
    cmd += ["-map", "0:v"]
    if len(audio_inputs) == 2:
        cmd += ["-filter_complex", f"[{audio_inputs[0]}][{audio_inputs[1]}]amix=inputs=2:normalize=0:duration=first[a]",
                "-map", "[a]"]
    elif audio_inputs:
        cmd += ["-map", audio_inputs[0]]
    # no sfx: copy the original audio bit-for-bit; with sfx it has to be re-encoded after mixing
    acodec = ["-c:a", "aac", "-b:a", "192k"] if cfg["sfx"] else ["-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p",
            *acodec, "-t", f"{end_i / fps:.4f}", "-movflags", "+faststart", cfg["output"]]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    i = 0
    for i, frame in enumerate(iter_frames(cfg["input"], meta)):
        if i >= end_i:
            break
        enc.stdin.write(np.ascontiguousarray(comp.frame(frame, i)).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{end_i}", end="\r", flush=True)
    else:
        i += 1
    # clip ran out before end_time: keep the frozen scene going
    while i < end_i and comp.scene is not None and not cfg["overlay_only"]:
        enc.stdin.write(np.ascontiguousarray(comp.frame(None, i)).tobytes())
        i += 1
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg encode failed")
    print(f"\nwrote {cfg['output']}")


def cmd_preview(args):
    cfg = load_config(args.config)
    meta = probe(cfg["input"], cfg["upscale"])
    comp = Composer(cfg, meta)
    i = comp.trans_i + int(round(args.t * comp.fps))
    got = frames_at(cfg["input"], meta, [comp.ref_i, i])
    comp.reference(got[comp.ref_i])
    out = comp.frame(got[i], i)
    path = args.out or os.path.splitext(cfg["output"])[0] + f"_preview_{args.t:.2f}s.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(out).save(path)
    print(f"wrote {path}")


def grid(img, step=50):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, max(12, im.width // 60))
    for x in range(0, im.width, step):
        d.line([(x, 0), (x, im.height)], fill=(255, 0, 255) if x % 100 == 0 else (255, 160, 255), width=1)
        if x % 100 == 0:
            d.text((x + 2, 2), str(x), fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    for y in range(0, im.height, step):
        d.line([(0, y), (im.width, y)], fill=(255, 0, 255) if y % 100 == 0 else (255, 160, 255), width=1)
        if y % 100 == 0:
            d.text((2, y + 2), str(y), fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def cmd_grab(args):
    if args.target.endswith(".json"):
        cfg = load_config(args.target)
        meta = probe(cfg["input"], cfg["upscale"])
        fps, _, _, ref_i, _ = timeline(cfg, meta)
        src, idx = cfg["input"], ref_i
        if args.time is not None:
            idx = int(round(args.time * fps))
    else:
        src = args.target
        meta = probe(src)
        idx = int(round((args.time or 0) * float(meta["fps"])))
    f = frames_at(src, meta, [idx])[idx]
    im = grid(f, args.step) if not args.plain else Image.fromarray(f)
    path = args.out or f"frame_{idx}.png"
    im.save(path)
    print(f"wrote {path}  (frame {idx}, t={idx / float(meta['fps']):.3f}s, {f.shape[1]}x{f.shape[0]})")


def cmd_contact(args):
    meta = probe(args.video)
    fps = float(meta["fps"])
    t1 = args.to if args.to is not None else meta["duration"]
    wanted = {}
    t = args.frm
    while t <= t1 + 1e-6:
        wanted[int(round(t * fps))] = t
        t += args.step
    thumbs = []
    for i, f in enumerate(iter_frames(args.video, meta, max_frames=max(wanted) + 1)):
        if i in wanted:
            im = Image.fromarray(f)
            im.thumbnail((args.thumb, args.thumb))
            d = ImageDraw.Draw(im)
            d.text((6, 4), f"{i / fps:.2f}s  #{i}", fill=(255, 255, 0), font=ImageFont.truetype(FONT, 18),
                   stroke_width=3, stroke_fill=(0, 0, 0))
            thumbs.append(im)
    cols = args.cols
    tw, th = thumbs[0].size
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
    path = args.out or "contact.png"
    sheet.save(path)
    print(f"wrote {path}  ({len(thumbs)} frames, {meta['duration']:.2f}s clip @ {fps:.2f}fps, "
          f"{meta['w']}x{meta['h']}, audio={'yes' if meta['audio'] else 'no'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="render the final video")
    p.add_argument("config")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("preview", help="render one still of the effect")
    p.add_argument("config")
    p.add_argument("--t", type=float, default=1.5, help="seconds after the transform (default 1.5)")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("grab", help="save a frame with a pixel grid (for placing players/ball)")
    p.add_argument("target", help="config.json (grabs the reference frame) or a video file")
    p.add_argument("--time", type=float)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--plain", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_grab)

    p = sub.add_parser("contact", help="contact sheet of frames, to find the skull moment")
    p.add_argument("video")
    p.add_argument("--from", dest="frm", type=float, default=0.0)
    p.add_argument("--to", type=float)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--thumb", type=int, default=320)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_contact)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
