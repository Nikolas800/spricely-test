"""The Roblox scene the shot cuts to, and the Roblox player standing in it.

`background()` takes a game screenshot, paints out the bits that shouldn't be there
(the screenshotter's own character, other players), crops it to the video's shape
and sharpens / grades it. `canvas()` puts it in the clip's picture area, between the
black bars, so the swap is a straight clip change. `hero_sprite()` renders the
Roblox player (Messi) that stands in front.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import avatar3d as av

HERE = os.path.dirname(os.path.abspath(__file__))


def content_rows(frame, thresh=20):
    """First and last picture rows of the frame (the rest is the black bars)."""
    rows = frame.mean(axis=(1, 2))
    lit = np.nonzero(rows > thresh)[0]
    return (int(lit[0]), int(lit[-1]) + 1) if len(lit) else (0, frame.shape[0])


def background(path, size, crop=None, erase=()):
    """Screenshot -> cleaned, cropped, sharpened RGB array of `size` (w, h)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).copy()
    if erase:
        mask = np.zeros(arr.shape[:2], np.uint8)
        for x, y, w, h in erase:
            mask[int(y):int(y + h), int(x):int(x + w)] = 255
        arr = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), mask, 7, cv2.INPAINT_TELEA),
                           cv2.COLOR_BGR2RGB)
    img = Image.fromarray(arr)
    w, h = size
    if crop is None:  # centred crop with the right shape
        sw, sh = img.size
        cw = min(sw, sh * w / h)
        ch = cw * h / w
        crop = ((sw - cw) / 2, (sh - ch) / 2, (sw + cw) / 2, (sh + ch) / 2)
    img = img.crop(tuple(int(round(v)) for v in crop)).resize((w, h), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=2.2, percent=90, threshold=2))
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Color(img).enhance(1.12)
    # soft vignette so the edges sit back a little, like the original edit
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vig = np.clip(1.0 - 0.22 * np.clip(r - 0.55, 0, None) ** 1.5, 0, 1)
    return (np.asarray(img, np.float32) * vig[..., None]).astype(np.uint8)


def canvas(frame, cfg_swap, base_dir):
    """Full-size frame: black bars as in the clip, the Roblox scene in the picture area."""
    H, W = frame.shape[:2]
    top, bottom = cfg_swap.get("rows") or content_rows(frame)
    path = cfg_swap["image"] if os.path.isabs(cfg_swap["image"]) else os.path.join(base_dir, cfg_swap["image"])
    out = np.zeros_like(frame)
    out[top:bottom] = background(path, (W, bottom - top), cfg_swap.get("crop"), cfg_swap.get("erase", ()))
    return out


def hero_sprite(hero, kits, height):
    """The Roblox player (cached render), scaled to `height` px tall."""
    kit = dict(kits.get(hero.get("kit"), {}))
    look = {k: hero[k] for k in ("skin", "hair", "beard") if hero.get(k)}
    for k in ("shirt", "stripes", "pants", "socks"):
        if hero.get(k):
            kit[k] = hero[k]
    name = "hero_" + "_".join(str(v).lstrip("#") for v in [*kit.values(), *look.values()])
    path = os.path.join(HERE, "assets", name + ".png")
    if not os.path.exists(path):
        print(f"  rendering {os.path.relpath(path, HERE)} ...")
        avatar = av.Avatar(kit["shirt"], kit["pants"], stripes=kit.get("stripes"), socks=kit.get("socks"),
                           pose=hero.get("pose", "idle"), skin=look.get("skin", av.SKIN), hair=look.get("hair"),
                           beard=look.get("beard"))
        img, _ = avatar.render(height_px=760, yaw=hero.get("yaw", -14), pitch=-6)
        img.save(path)
    img = Image.open(path).convert("RGBA")
    w = max(1, int(img.width * height / img.height))
    return img.resize((w, int(height)), Image.LANCZOS)
