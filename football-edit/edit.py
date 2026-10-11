#!/usr/bin/env python3
"""Football -> Roblox edit.

Plays the clip as-is (same footage, same audio). When the skull hits it holds for
`skull_hold` seconds, then the frame freezes, every annotated player poofs into a
stretched-out Roblox avatar and the ball poofs into a treasure chest.

  python edit.py contact input/clip.mp4 --from 2 --to 6      # find the skull moment
  python edit.py grab config.json                            # freeze frame + pixel grid
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

import roblox as rb

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DEFAULTS = {
    "output": "output/football_roblox_edit.mp4",
    "skull_hold": 1.0,
    "freeze_on_skull": True,
    "add_skull": False,
    "skull": {"x": 0.5, "y": 0.42, "size": 0.42, "after": "pop"},
    "erase": [],
    "stretch": 2.4,
    "height_scale": 1.25,
    "players": [],
    "ball": None,
    "end_time": None,
    "flash": True,
    "shake": True,
    "slow_zoom": 0.05,
    "sfx": False,
    "crf": 18,
}


# ---------------------------------------------------------------- video io

def probe(path):
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
        "w": w, "h": h, "fps": fps, "duration": float(info["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def iter_frames(path, meta, max_frames=None):
    """Yield RGB uint8 frames at a constant frame rate (meta['fps'])."""
    w, h = meta["w"] & ~1, meta["h"] & ~1
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={meta['fps']},crop={w}:{h}:0:0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
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


def frame_at(path, meta, index):
    last = None
    for i, f in enumerate(iter_frames(path, meta, max_frames=index + 1)):
        last = f
    if last is None:
        sys.exit(f"could not read frame {index} from {path}")
    return last.copy()


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
    return out


def timeline(cfg, meta):
    fps = float(meta["fps"])
    skull_i = int(round(cfg["skull_time"] * fps))
    trans_i = int(round((cfg["skull_time"] + cfg["skull_hold"]) * fps))
    freeze_i = skull_i if cfg["freeze_on_skull"] else trans_i
    end_t = cfg["end_time"] if cfg["end_time"] is not None else meta["duration"]
    end_i = int(math.floor(end_t * fps + 1e-6))
    return fps, skull_i, trans_i, freeze_i, end_i


# ---------------------------------------------------------------- clean plate

def player_mask(frame, box, pad=0.12):
    """Mask of the player inside `box`: non-grass pixels, closed and dilated."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad * 0.5)))
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 35)
    m = (~grass).astype(np.uint8) * 255
    k = max(3, int(min(w, h) * 0.08)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    # if the background isn't pitch the whole box goes; fine, the avatar covers it
    return full


def clean_plate(frame, cfg):
    H, W = frame.shape[:2]
    masks = [player_mask(frame, p["box"]) for p in cfg["players"]]
    total = np.zeros((H, W), np.uint8)
    for m in masks:
        total |= m
    if cfg["ball"]:
        (bx, by), r = cfg["ball"]["center"], cfg["ball"]["radius"]
        cv2.circle(total, (int(bx), int(by)), int(r * 1.7 + 3), 255, -1)
    for x, y, w, h in cfg["erase"]:
        total[int(y):int(y + h), int(x):int(x + w)] = 255
    plate = frame.copy()
    if total.any():
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # inpaint at half res (smoother on big holes), then upsample into the hole
        small = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(total, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        fill = cv2.inpaint(small, sm, 6, cv2.INPAINT_TELEA)
        fill = cv2.GaussianBlur(cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
        fill = cv2.cvtColor(fill, cv2.COLOR_BGR2RGB)
        a = cv2.GaussianBlur(total.astype(np.float32) / 255, (0, 0), 3)[..., None]
        plate = (frame * (1 - a) + fill * a).astype(np.uint8)
    return plate, masks


def sample_colors(frame, box):
    """Guess shirt / shorts colour from the player box (ignoring grass)."""
    x, y, w, h = [int(v) for v in box]

    def band(fy0, fy1):
        reg = frame[y + int(h * fy0):y + int(h * fy1), x + int(w * 0.3):x + int(w * 0.7)].reshape(-1, 3)
        if len(reg) == 0:
            return (200, 200, 200)
        hsv = cv2.cvtColor(reg[None], cv2.COLOR_RGB2HSV)[0]
        keep = ~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 45))
        reg = reg[keep] if keep.sum() > 10 else reg
        c = np.median(reg, axis=0)
        # punch up saturation a bit so it reads as a Roblox colour
        hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
        hsv[1] = min(255, int(hsv[1] * 1.35))
        hsv[2] = min(255, int(hsv[2] * 1.1 + 10))
        return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0])

    return band(0.18, 0.42), band(0.47, 0.6)


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

class Scene:
    """Everything that happens from the transform moment on."""

    def __init__(self, frame, cfg):
        self.cfg = cfg
        self.frame = frame
        self.H, self.W = frame.shape[:2]
        self.plate, self.masks = clean_plate(frame, cfg)
        self.actors = []
        order = sorted(range(len(cfg["players"])), key=lambda i: cfg["players"][i]["box"][0])
        for n, i in enumerate(order):
            p = cfg["players"][i]
            x, y, w, h = p["box"]
            shirt, pants = sample_colors(frame, p["box"])
            shirt, pants = p.get("shirt", shirt), p.get("pants", pants)
            stretch = p.get("stretch", cfg["stretch"])
            pose = p.get("pose", rb.POSE_CYCLE[i % len(rb.POSE_CYCLE)])
            stud = h * cfg["height_scale"] * p.get("scale", 1.0) / (5.3 * stretch)
            spr, anchor = rb.render_avatar(stud, shirt, pants, pose=pose)
            self.actors.append({
                "kind": "player", "idx": i, "sprite": spr, "anchor": anchor, "stretch": stretch,
                "foot": (x + w / 2, y + h), "delay": 0.035 * n, "phase": random.Random(i).random() * 6.28,
                "stud": stud, "mask": self.masks[i],
            })
        if cfg["ball"]:
            (bx, by), r = cfg["ball"]["center"], cfg["ball"]["radius"]
            cw = cfg["ball"].get("chest_width", max(r * 7, min(self.W, self.H) * 0.13))
            spr, anchor = rb.render_chest(cw)
            self.actors.append({
                "kind": "chest", "sprite": spr, "anchor": anchor, "foot": (bx, by + r), "delay": 0.08,
                "glow": rb.glow_sprite(cw * 0.95), "cw": cw,
                "sparkles": [(random.Random(k).uniform(-0.8, 0.8), random.Random(k + 50).uniform(0.1, 1.4),
                              random.Random(k + 99).uniform(0, 6.28)) for k in range(7)],
            })
        self.actors.sort(key=lambda a: a["foot"][1])

    # -- per-actor pieces

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

    def _paste(self, img, spr, anchor, foot, sx, sy):
        w, h = max(1, int(spr.width * sx)), max(1, int(spr.height * sy))
        s = spr.resize((w, h), Image.BILINEAR)
        img.paste(s, (int(foot[0] - anchor[0] * sx), int(foot[1] - anchor[1] * sy)), s)

    def _shadow(self, draw, foot, width, alpha):
        x, y = foot
        draw.ellipse([x - width / 2, y - width * 0.12, x + width / 2, y + width * 0.12], fill=(0, 0, 0, alpha))

    def render(self, t):
        """t = seconds since the transform moment."""
        base = self.plate.copy()
        # players whose poof hasn't started yet are still the real footage
        for a in self.actors:
            if a["kind"] == "player" and t < a["delay"]:
                m = a["mask"][..., None] > 0
                base = np.where(m, self.frame, base)
        img = Image.fromarray(base)
        fx = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fxd = ImageDraw.Draw(fx)
        shadows = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shd = ImageDraw.Draw(shadows)
        for a in self.actors:
            tau = t - a["delay"]
            if a["kind"] == "player" and tau >= 0:
                self._shadow(shd, a["foot"], a["stud"] * 4.6 * min(1.0, tau / 0.12), 90)
        img.paste(shadows, (0, 0), shadows)

        for a in self.actors:
            tau = t - a["delay"]
            if tau < 0:
                continue
            spr = a["sprite"]
            if a["kind"] == "player":
                S = a["stretch"]
                if tau < 0.12:
                    k = ease_out_back(tau / 0.12)
                    sx, sy = k, k
                elif tau < 0.30:
                    sx, sy = 1.0, 1.0
                elif tau < 0.95:
                    e = ease_out_elastic((tau - 0.30) / 0.65)
                    sy = 1 + (S - 1) * e
                    sx = 1 - 0.15 * (sy - 1) / max(S - 1, 1e-6)
                else:
                    wob = math.sin(2 * math.pi * 1.6 * tau + a["phase"])
                    sy, sx = S * (1 + 0.03 * wob), 0.85 * (1 - 0.02 * wob)
                if sx > 0.02:
                    self._paste(img, spr, a["anchor"], a["foot"], sx, sy)
                cy = a["foot"][1] - a["stud"] * 2.6
                self._poof(fxd, a["foot"][0], cy, a["stud"] * 3.2, tau, a["idx"])
            else:
                cw = a["cw"]
                fx_, fy_ = a["foot"]
                glow_a = min(1.0, max(0.0, (tau - 0.1) / 0.4))
                if glow_a > 0:
                    g = a["glow"].rotate(tau * 25, resample=Image.BILINEAR)
                    pulse = 0.85 + 0.15 * math.sin(tau * 5)
                    al = g.getchannel("A").point(lambda v: int(v * glow_a * pulse))
                    g.putalpha(al)
                    img.paste(g, (int(fx_ - g.width / 2), int(fy_ - cw * 0.45 - g.height / 2)), g)
                k = ease_out_elastic(tau / 0.7) if tau < 0.7 else 1 + 0.02 * math.sin(tau * 4)
                if k > 0.02:
                    # squash & stretch: wide while it grows, narrow while it overshoots
                    sx = math.sqrt(k) if k < 1 else 1 - 0.6 * (k - 1)
                    self._paste(img, spr, a["anchor"], (fx_, fy_), sx, k)
                if tau > 0.25:
                    for i, (ox, oy, ph) in enumerate(a["sparkles"]):
                        tw = 0.5 + 0.5 * math.sin(tau * 6 + ph)
                        if tw < 0.25:
                            continue
                        s = rb.sparkle(cw * 0.12 * tw)
                        img.paste(s, (int(fx_ + ox * cw - s.width / 2), int(fy_ - oy * cw * 0.75 - s.height / 2)), s)
                self._poof(fxd, fx_, fy_ - cw * 0.3, cw * 0.55, tau, 999)

        img.paste(fx, (0, 0), fx)
        out = np.asarray(img).astype(np.float32)

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
    t0 = cfg["skull_time"] + cfg["skull_hold"]
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

def compose(frame, i, cfg, fps, skull_i, trans_i, freeze, scene, skull):
    if i >= trans_i:
        out = scene.render((i - trans_i) / fps)
    elif i >= skull_i and cfg["freeze_on_skull"]:
        out = freeze
    else:
        out = frame
    if skull is not None:
        out = skull.draw(out, (i - skull_i) / fps, (i - trans_i) / fps)
    return out


def cmd_render(args):
    cfg = load_config(args.config)
    meta = probe(cfg["input"])
    fps, skull_i, trans_i, freeze_i, end_i = timeline(cfg, meta)
    W, H = meta["w"] & ~1, meta["h"] & ~1
    os.makedirs(os.path.dirname(cfg["output"]) or ".", exist_ok=True)
    print(f"{W}x{H} @ {fps:.3f}fps  skull@{skull_i}  transform@{trans_i}  freeze@{freeze_i}  end@{end_i}")

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
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{end_i / fps:.4f}", "-movflags", "+faststart", cfg["output"]]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    skull = Skull(cfg, W, H) if cfg["add_skull"] else None
    freeze = scene = None
    last = None
    for i, frame in enumerate(iter_frames(cfg["input"], meta)):
        if i >= end_i:
            break
        if i == freeze_i:
            freeze = frame.copy()
            scene = Scene(freeze, cfg)
        last = compose(frame, i, cfg, fps, skull_i, trans_i, freeze, scene, skull)
        enc.stdin.write(np.ascontiguousarray(last).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{end_i}", end="\r", flush=True)
    else:
        i += 1
    # clip ran out before end_time: keep the frozen scene going
    while i < end_i and scene is not None:
        last = compose(None, i, cfg, fps, skull_i, trans_i, freeze, scene, skull)
        enc.stdin.write(np.ascontiguousarray(last).tobytes())
        i += 1
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg encode failed")
    print(f"\nwrote {cfg['output']}")


def cmd_preview(args):
    cfg = load_config(args.config)
    meta = probe(cfg["input"])
    fps, skull_i, trans_i, freeze_i, _ = timeline(cfg, meta)
    freeze = frame_at(cfg["input"], meta, freeze_i)
    scene = Scene(freeze, cfg)
    skull = Skull(cfg, freeze.shape[1], freeze.shape[0]) if cfg["add_skull"] else None
    i = trans_i + int(round(args.t * fps))
    out = compose(freeze, i, cfg, fps, skull_i, trans_i, freeze, scene, skull)
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
        meta = probe(cfg["input"])
        fps, _, _, freeze_i, _ = timeline(cfg, meta)
        src, idx = cfg["input"], freeze_i
        if args.time is not None:
            idx = int(round(args.time * fps))
    else:
        src = args.target
        meta = probe(src)
        idx = int(round((args.time or 0) * float(meta["fps"])))
    f = frame_at(src, meta, idx)
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
    p.add_argument("target", help="config.json (grabs the freeze frame) or a video file")
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
