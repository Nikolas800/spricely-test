#!/usr/bin/env python3
"""Puppet-warp ONE Roblox character image onto a real player.

1. A single template picture of a Roblox character (SWAT gear, or a football kit),
   standing with arms and legs slightly apart, plus a map of which body part each
   pixel belongs to and where its joints are.
2. A fine triangle mesh over that picture is bent with the player's skeleton
   (linear blend skinning on the bones torso, head, upper/lower arms, upper/lower
   legs, feet). Each bone is also stretched across to the player's limb thickness
   as measured in the footage, so the one image takes on the player's shape.
3. The result is trimmed to the player's silhouette by the caller.

  python puppet.py            # (re)render the template images into assets/templates/
"""

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import avatar3d as av

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "assets", "templates")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GROUPS = ["torso", "head", "l_arm", "r_arm", "l_leg", "r_leg"]
G = {name: i for i, name in enumerate(GROUPS)}
NONE = 255

SWAT = {
    "shirt": "#262d3b", "pants": "#262d3b", "vest": "#1b1e24", "pouch": "#121418", "flap": "#2d323b",
    "belt": "#0d0e10", "buckle": "#8d939c", "glove": "#0f1012", "boot": "#0b0c0e", "pad": "#3c424c",
    "helmet": "#15171b", "rim": "#2e3238", "lens": "#35546e", "patch": "#59606d", "text": "#f2f2f2",
}


def _text_mask(text="SWAT"):
    font = ImageFont.truetype(FONT, 80)
    left, top, right, bottom = font.getbbox(text)
    im = Image.new("L", (right - left + 10, bottom - top + 10), 0)
    ImageDraw.Draw(im).text((5 - left, 5 - top), text, fill=255, font=font)
    return np.asarray(im) > 127


TEXT = _text_mask()


def _stamp(u, v, box):
    """True where (u, v) falls on the SWAT lettering placed in box = (u0, u1, v_top, v_bottom)."""
    u0, u1, vt, vb = box
    tu, tv = (u - u0) / (u1 - u0), (vt - v) / (vt - vb)
    inside = (tu >= 0) & (tu < 1) & (tv >= 0) & (tv < 1)
    th, tw = TEXT.shape
    hit = np.zeros(len(u), bool)
    hit[inside] = TEXT[(tv[inside] * th).astype(int), (tu[inside] * tw).astype(int)]
    return hit


# ---------------------------------------------------------------- the template character

class TemplateAvatar(av.Avatar):
    """An R6 avatar in an A-pose; outfit is "swat" or a kit dict (shirt / stripes / pants / socks)."""

    def __init__(self, outfit):
        self.swat = outfit == "swat"
        kit = {"shirt": SWAT["shirt"], "pants": SWAT["pants"]} if self.swat else outfit
        super().__init__(kit["shirt"], kit["pants"], stripes=kit.get("stripes"), socks=kit.get("socks"),
                         pose="apose")
        if self.swat:
            self.col.update({k: av._lin(v) for k, v in SWAT.items()})
            self.parts.append(("helmet", "helmet", (0, 4.03, 0), np.eye(3, dtype=np.float32), (0, 0.6, 0), None))

    def sdf(self, p, with_id=False):
        ds = []
        for part in self.parts:
            q = self.local(p, part)
            if part[1] == "box":
                ds.append(av.sd_round_box(q, part[5], 0.13))
            elif part[1] == "head":
                ds.append(av.sd_round_cyl(q, 0.62, 0.24, 0.6))
            else:  # helmet: dome over the top of the head
                qq = q - np.float32([0, 0.1, 0])
                ds.append(np.maximum(av.sd_round_cyl(qq, 0.7, 0.32, 0.62), -(q[..., 1] - 0.2)))
        ds = np.stack(ds, -1)
        d = ds.min(-1)
        return (d, ds.argmin(-1)) if with_id else d

    def albedo(self, p, ids):
        out = super().albedo(p, ids)
        if not self.swat:
            return out
        c = self.col
        for k, part in enumerate(self.parts):
            sel = ids == k
            if not sel.any():
                continue
            q = self.local(p[sel], part)
            u, v, z = q[:, 0], q[:, 1], q[:, 2]
            name = part[0]
            col = out[sel]
            if name == "torso":
                front, back = z > 0.4, z < -0.4
                col[:] = c["vest"]
                strap = (np.abs(np.abs(u) - 0.45) < 0.15)
                col[(v > 0.8) & ~strap] = c["shirt"]
                col[front & _stamp(u, v, (-0.72, 0.72, 0.62, 0.2))] = c["text"]
                col[back & _stamp(-u, v, (-0.8, 0.8, 0.7, 0.15))] = c["text"]
                for cu in (-0.56, 0.0, 0.56):
                    pouch = front & (np.abs(u - cu) < 0.19) & (v > -0.64) & (v < -0.12)
                    col[pouch] = c["pouch"]
                    col[pouch & (v > -0.3) & (v < -0.22)] = c["flap"]
                col[v < -0.8] = c["belt"]
                col[front & (v < -0.8) & (np.abs(u) < 0.13)] = c["buckle"]
            elif name in ("larm", "rarm"):
                col[:] = c["shirt"]
                col[(z > 0.3) & (v > 0.35) & (v < 0.72) & (np.abs(u) < 0.32)] = c["patch"]
                col[v < -0.5] = c["glove"]
            elif name in ("lleg", "rleg"):
                col[:] = c["pants"]
                col[(z > 0.25) & (v > -0.3) & (v < 0.08) & (np.abs(u) < 0.4)] = c["pad"]
                if name == "rleg":  # thigh holster
                    col[(z > 0.25) & (v > 0.25) & (v < 0.72) & (u > -0.15) & (u < 0.45)] = c["pouch"]
                col[v < -0.72] = c["boot"]
            elif name == "helmet":
                col[:] = c["helmet"]
                col[v < 0.3] = c["rim"]
                lens = (z > 0.3) & (v > 0.3) & (v < 0.47) & (np.abs(u) > 0.06) & (np.abs(u) < 0.4)
                col[lens] = c["lens"]
            out[sel] = col
        return out


def _project(P, target, right, up, half_w, y0, y1, H, W):
    gx = float((np.asarray(P) - target) @ right)
    gy = float((np.asarray(P) - target) @ up) + 2.6
    return np.array([(gx + half_w) / (2 * half_w) * (W - 1), (y1 - gy) / (y1 - y0) * (H - 1)])


def render_template(outfit, back=False, height_px=760, yaw=10, pitch=-4, ss=2):
    """Returns (RGBA image, part-group map, joints dict, pixels per stud)."""
    avt = TemplateAvatar(outfit)
    C = av.rot_y(yaw + (180 if back else 0)) @ av.rot_x(pitch)
    right, up, fwd = C @ [1, 0, 0], C @ [0, 1, 0], C @ [0, 0, -1]
    half_w, y0, y1 = 3.0, -0.3, 5.9
    H = height_px * ss
    W = int(H * (2 * half_w) / (y1 - y0)) // ss * ss
    xs = np.linspace(-half_w, half_w, W, dtype=np.float32)
    ys = np.linspace(y1, y0, H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    target = np.array([0, 2.6, 0], np.float32)
    o = (target + gx[..., None] * right + (gy[..., None] - 2.6) * up - fwd * 12).reshape(-1, 3).astype(np.float32)
    d = np.broadcast_to(fwd.astype(np.float32), o.shape).copy()
    t, hit = avt.march(o, d, 24.0)
    rgb = np.zeros((len(o), 3), np.float32)
    ids = np.full(len(o), -1, np.int32)
    idx = np.where(hit)[0]
    pts = o[idx] + d[idx] * t[idx, None]
    _, pid = avt.sdf(pts, with_id=True)
    rgb[idx] = avt.shade(pts, pid, right, up, fwd)
    ids[idx] = pid
    img, (cx, cy) = av.finish(rgb, hit, H, W, ss)

    def P(p):  # 3D model point -> pixel in the cropped, downsampled image
        xy = _project(p, target, right, up, half_w, y0, y1, H, W) / ss
        return [float(xy[0] - cx), float(xy[1] - cy)]

    # which screen side each limb ends up on (it flips for the back view)
    pivots = {part[0]: part for part in avt.parts}
    side = {}
    for a, b in (("larm", "rarm"), ("lleg", "rleg")):
        xa, xb = P(pivots[a][2])[0], P(pivots[b][2])[0]
        side[a], side[b] = ("l", "r") if xa < xb else ("r", "l")
    group_of = {"torso": G["torso"], "head": G["head"], "helmet": G["head"]}
    for name in ("larm", "rarm"):
        group_of[name] = G[f"{side[name]}_arm"]
    for name in ("lleg", "rleg"):
        group_of[name] = G[f"{side[name]}_leg"]
    lut = np.array([group_of[part[0]] for part in avt.parts] + [NONE], np.uint8)
    gmap = lut[ids].reshape(H, W)[ss // 2::ss, ss // 2::ss][cy:cy + img.height, cx:cx + img.width]

    def along(part, y):
        _, _, pivot, R, _, _ = part
        return np.asarray(pivot, np.float32) + R @ np.float32([0, y, 0])

    joints = {"neck": P((0, 4.0, 0)), "head": P((0, 4.63, 0)), "head_top": P((0, 5.25, 0)), "hipc": P((0, 2.0, 0))}
    for name in ("larm", "rarm"):
        s = side[name]
        part = pivots[name]
        joints[f"{s}_shoulder"] = P(along(part, 0.38))  # top of the arm, so nothing pokes up past it
        joints[f"{s}_elbow"] = P(along(part, -0.5))
        joints[f"{s}_wrist"] = P(along(part, -1.15))
    for name in ("lleg", "rleg"):
        s = side[name]
        part = pivots[name]
        joints[f"{s}_hip"] = P(along(part, 0.0))
        joints[f"{s}_knee"] = P(along(part, -1.0))
        joints[f"{s}_ankle"] = P(along(part, -1.72))
        joints[f"{s}_toe"] = P(along(part, -2.0))
    stud_px = (H - 1) / (y1 - y0) / ss
    joints["r"] = 0.62 * stud_px
    return img, gmap, joints, stud_px


def template_name(outfit, back):
    if outfit == "swat":
        base = "swat"
    else:
        base = "kit_" + "_".join(outfit[k].lstrip("#") for k in ("shirt", "stripes", "pants") if outfit.get(k))
    return base + ("_back" if back else "")


def load_template(outfit, back=False):
    """The template image (rendered once, then read from assets/templates/)."""
    name = template_name(outfit, back)
    png = os.path.join(TEMPLATES, name + ".png")
    meta = os.path.join(TEMPLATES, name + ".json")
    parts = os.path.join(TEMPLATES, name + "_parts.png")
    if not (os.path.exists(png) and os.path.exists(meta) and os.path.exists(parts)):
        print(f"  rendering template {name} ...")
        img, gmap, joints, stud_px = render_template(outfit, back)
        os.makedirs(TEMPLATES, exist_ok=True)
        img.save(png)
        Image.fromarray(gmap).save(parts)
        with open(meta, "w") as fh:
            json.dump({"joints": joints, "stud_px": stud_px}, fh, indent=1)
    with open(meta) as fh:
        m = json.load(fh)
    return Image.open(png).convert("RGBA"), np.asarray(Image.open(parts)), m["joints"], m["stud_px"]


# ---------------------------------------------------------------- measuring the player

def _measure(mask, p, n, maxd):
    """Width of the silhouette through frame point p along unit direction n."""
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
    """Half-width of the limb from a to b in the silhouette (thinner readings win), clamped to [lo, hi]."""
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
    return float(np.clip(np.percentile(ws, 30) / 2, lo, hi))


# ---------------------------------------------------------------- the warp

# bone -> (group, start joint, end joint, template half-width in studs)
BONES = {
    "torso": ("torso", "hipc", "neck", 1.0),
    "head": ("head", "neck", "head_top", None),
    "l_arm_u": ("l_arm", "l_shoulder", "l_elbow", 0.5), "l_arm_l": ("l_arm", "l_elbow", "l_wrist", 0.5),
    "r_arm_u": ("r_arm", "r_shoulder", "r_elbow", 0.5), "r_arm_l": ("r_arm", "r_elbow", "r_wrist", 0.5),
    "l_leg_u": ("l_leg", "l_hip", "l_knee", 0.49), "l_leg_l": ("l_leg", "l_knee", "l_ankle", 0.49),
    "l_foot": ("l_leg", "l_ankle", "l_toe", 0.49),
    "r_leg_u": ("r_leg", "r_hip", "r_knee", 0.49), "r_leg_l": ("r_leg", "r_knee", "r_ankle", 0.49),
    "r_foot": ("r_leg", "r_ankle", "r_toe", 0.49),
}
NEIGHBOURS = {"torso": ("head", "l_arm", "r_arm", "l_leg", "r_leg"), "head": ("torso",),
              "l_arm": ("torso",), "r_arm": ("torso",), "l_leg": ("torso",), "r_leg": ("torso",)}


def player_joints(kp):
    """Config rig (screen-left/right pairs) -> flat joint dict like the template's."""
    j = {"neck": kp["neck"], "head": kp["head"], "r": kp.get("r", 15)}
    for key, name in (("shoulders", "shoulder"), ("elbows", "elbow"), ("wrists", "wrist"), ("hips", "hip"),
                      ("knees", "knee"), ("ankles", "ankle"), ("toes", "toe")):
        if key in kp:
            j[f"l_{name}"], j[f"r_{name}"] = kp[key]
    hc = (np.array(kp["hips"][0], float) + np.array(kp["hips"][1], float)) / 2
    j["hipc"] = hc.tolist()
    up = np.array(kp["head"], float) - np.array(kp["neck"], float)
    up /= max(np.hypot(*up), 1e-6)
    j["head_top"] = (np.array(kp["head"], float) + up * j["r"]).tolist()
    for s in "lr":
        if f"{s}_toe" not in j:
            j[f"{s}_toe"] = (np.array(j[f"{s}_ankle"], float) + [0, 0.3 * j["r"]]).tolist()
    return j


def bone_transforms(tj, pj, stud_px, mask):
    """2x3 affine per bone mapping template pixels onto the player (along: bone length, across: limb width)."""
    T = float(np.hypot(*(np.array(pj["neck"]) - np.array(pj["hipc"]))))
    sh_half = float(np.hypot(*(np.array(pj["l_shoulder"]) - np.array(pj["r_shoulder"])))) / 2
    widths = {
        "torso": fit_halfwidth(mask, pj["neck"], pj["hipc"], max(sh_half, 0.33 * T), 0.26 * T,
                               max(1.05 * sh_half, 0.38 * T), ts=(0.25, 0.45, 0.65)),
    }
    for s in "lr":
        aw = fit_halfwidth(mask, pj[f"{s}_shoulder"], pj[f"{s}_elbow"], 0.1 * T, 0.065 * T, 0.15 * T)
        widths[f"{s}_arm_u"] = aw
        widths[f"{s}_arm_l"] = fit_halfwidth(mask, pj[f"{s}_elbow"], pj[f"{s}_wrist"], 0.9 * aw, 0.06 * T, 0.14 * T)
        lw = fit_halfwidth(mask, pj[f"{s}_hip"], pj[f"{s}_knee"], 0.14 * T, 0.09 * T, 0.21 * T)
        widths[f"{s}_leg_u"] = lw
        sw = fit_halfwidth(mask, pj[f"{s}_knee"], pj[f"{s}_ankle"], 0.9 * lw, 0.08 * T, 0.19 * T)
        widths[f"{s}_leg_l"] = widths[f"{s}_foot"] = sw

    out = {}
    for name, (_, j0, j1, hw_studs) in BONES.items():
        a_t, b_t = np.array(tj[j0], float), np.array(tj[j1], float)
        a_p, b_p = np.array(pj[j0], float), np.array(pj[j1], float)
        Lt, Lp = max(np.hypot(*(b_t - a_t)), 1e-6), max(np.hypot(*(b_p - a_p)), 1e-6)
        ut, up = (b_t - a_t) / Lt, (b_p - a_p) / Lp
        nt, npp = np.array([-ut[1], ut[0]]), np.array([-up[1], up[0]])
        if name == "head":  # keep the head round: scale by head size, turn with the neck
            s_l = s_w = pj["r"] / tj["r"]
        else:
            s_l = Lp / Lt
            s_w = widths[name] / (hw_studs * stud_px)
            s_w = float(np.clip(s_w, 0.3 * s_l, 2.0 * s_l)) if name != "torso" else float(
                np.clip(s_w, 0.45 * s_l, 2.0 * s_l))
        A = np.column_stack([up, npp]) @ np.diag([s_l, s_w]) @ np.vstack([ut, nt])
        out[name] = (np.hstack([A, (a_p - A @ a_t)[:, None]]), a_t, b_t)
    return out


def _seg_dist(V, a, b):
    ab = b - a
    t = np.clip(((V - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.hypot(*(V - (a + t[:, None] * ab)).T)


def _fill_groups(gmap, alpha):
    """Give anti-aliased edge pixels (alpha > 0 but no part hit at their centre) the nearest part."""
    g = gmap.copy()
    need = (alpha > 0) & (g == NONE)
    for _ in range(4):
        if not need.any():
            break
        grown = g.copy()
        for gid in range(len(GROUPS)):
            d = cv2.dilate((g == gid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            grown[(grown == NONE) & d] = gid
        g = np.where(need, grown, g)
        need = (alpha > 0) & (g == NONE)
    return g


def warp(template, kp, mask=None, step=5, order=None):
    """Warp the template onto the player. Returns (RGBA image, (x, y) top-left in frame pixels)."""
    img, gmap, tj, stud_px = template
    pj = player_joints(kp)
    bones = bone_transforms(tj, pj, stud_px, mask)
    rgba = np.asarray(img, np.float32) / 255
    alpha = rgba[..., 3]
    prem = rgba.copy()
    prem[..., :3] *= alpha[..., None]
    gmap = _fill_groups(gmap, alpha)
    Ht, Wt = alpha.shape

    # mesh over the template
    xs = np.arange(0, Wt + step, step, dtype=np.float32)
    ys = np.arange(0, Ht + step, step, dtype=np.float32)
    VX, VY = np.meshgrid(xs, ys)
    V = np.stack([VX.ravel(), VY.ravel()], 1)
    nx = len(xs)
    eps2 = (0.25 * stud_px) ** 2

    def deform(group):
        """All mesh vertices moved by the bones of `group` (plus a little of its neighbours)."""
        num, den = np.zeros_like(V), np.zeros(len(V), np.float32)
        for name, (A, a, b) in bones.items():
            g = BONES[name][0]
            f = 1.0 if g == group else 0.2 if g in NEIGHBOURS[group] else 0.0
            if f == 0:
                continue
            if g != group and name not in ("torso", "head") and not name.endswith("_u"):
                continue
            w = f / (_seg_dist(V, a, b) ** 2 + eps2) ** 2
            num += w[:, None] * (V @ A[:, :2].T + A[:, 2])
            den += w
        return num / den[:, None]

    # cells: which part each covers (by its centre pixel), skipping empty ones
    cy, cx = np.meshgrid(np.arange(len(ys) - 1), np.arange(nx - 1), indexing="ij")
    ccx = np.clip((cx * step + step / 2).astype(int), 0, Wt - 1)
    ccy = np.clip((cy * step + step / 2).astype(int), 0, Ht - 1)
    pooled = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((step + 2, step + 2), np.uint8))
    occupied = pooled[ccy, ccx] > 0
    cell_group = gmap[ccy, ccx]

    order = order or ["l_leg", "r_leg", "torso", "l_arm", "r_arm", "head"]
    meshes = {}
    for gname in order:
        gid = G[gname]
        # a cell joins this part's mesh if its centre is this part, or it borders this part's pixels
        near = cv2.dilate((gmap == gid).astype(np.uint8), np.ones((step * 2 + 1, step * 2 + 1), np.uint8))
        sel = occupied & ((cell_group == gid) | (near[ccy, ccx] > 0))
        if not sel.any():
            continue
        moved = deform(gname)
        i0 = (cy[sel] * nx + cx[sel]).ravel()
        tris_src, tris_dst = [], []
        for quad in ((i0, i0 + 1, i0 + nx), (i0 + 1, i0 + nx + 1, i0 + nx)):
            tris_src.append(np.stack([V[q] for q in quad], 1))
            tris_dst.append(np.stack([moved[q] for q in quad], 1))
        meshes[gname] = (np.concatenate(tris_src), np.concatenate(tris_dst))  # (N, 3, 2) each

    allv = np.concatenate([d.reshape(-1, 2) for _, d in meshes.values()])
    x0, y0 = np.floor(allv.min(0)).astype(int) - 2
    x1, y1 = np.ceil(allv.max(0)).astype(int) + 2
    Wo, Ho = x1 - x0, y1 - y0
    out = np.zeros((Ho, Wo, 4), np.float32)

    for gname in order:
        if gname not in meshes:
            continue
        gid = G[gname]
        src, dst = meshes[gname]
        dst = dst - [x0, y0]

        # rasterise triangle ids, then map every covered pixel back into the template
        buf = np.zeros((Ho, Wo), np.int32)
        for k, tri in enumerate(np.round(dst * 16).astype(np.int32)):
            cv2.fillConvexPoly(buf, tri, k + 1, lineType=cv2.LINE_8, shift=4)
        yy, xx = np.nonzero(buf)
        if len(yy) == 0:
            continue
        k = buf[yy, xx] - 1
        D = np.concatenate([dst, np.ones((len(dst), 3, 1))], 2)  # (N, 3, 3)
        D_ok = np.abs(np.linalg.det(D)) > 1e-6
        M = np.zeros((len(dst), 3, 2))
        M[D_ok] = np.linalg.solve(D[D_ok], src[D_ok])  # [x y 1] @ M = source point
        sp = np.stack([xx, yy, np.ones_like(xx)], 1)[:, None, :] @ M[k]
        mapx = np.full((Ho, Wo), -1, np.float32)
        mapy = np.full((Ho, Wo), -1, np.float32)
        mapx[yy, xx] = sp[:, 0, 0]
        mapy[yy, xx] = sp[:, 0, 1]
        part = prem * (gmap == gid)[..., None]
        layer = cv2.remap(part, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = layer + out * (1 - layer[..., 3:])

    a = out[..., 3:]
    rgb = np.where(a > 1e-4, out[..., :3] / np.maximum(a, 1e-4), 0)
    res = Image.fromarray((np.clip(np.concatenate([rgb, a], 2), 0, 1) * 255).astype(np.uint8), "RGBA")
    bbox = res.getbbox() or (0, 0, 1, 1)
    return res.crop(bbox), (int(x0 + bbox[0]), int(y0 + bbox[1]))


def paint_order(kp):
    order = ["l_leg", "r_leg", "torso", "l_arm", "r_arm", "head"]
    back, front = kp.get("back", []), kp.get("front", [])
    order = [g for g in back if g in order] + [g for g in order if g not in back and g not in front]
    return order + [g for g in front if g in GROUPS]


if __name__ == "__main__":
    for outfit, back in (("swat", False), ("swat", True)):
        name = template_name(outfit, back)
        for ext in (".png", ".json", "_parts.png"):
            path = os.path.join(TEMPLATES, name + ext)
            if os.path.exists(path):
                os.remove(path)
        load_template(outfit, back)
        print("rendered", name)
