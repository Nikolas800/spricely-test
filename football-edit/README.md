# Football → Roblox edit

Takes a football clip and keeps it exactly as is, with the same footage and the same sound. On a boom after the 💀,
every player is instantly **overlaid with a Roblox character warped onto their own skeleton**: posed like them,
as wide as their limbs, and trimmed to their outline. The real players stay underneath. The **ball becomes a
holographic "secret prototype vault"** with a colour-cycling neon glow.

`config.json` is the finished setup for `input/videoplayback123.mp4` (the Messi edit). Its output is
`output/messi_roblox_edit.mp4`. The skull lands at 14.73s, and the booms after it are at 15.24, 15.70 and 16.16s.
On boom 1 (15.24s), the shot hard-cuts to the Roblox game (`assets/depot.webp`, the "Level 4: Armored Truck Depot"
screenshot, with the screenshotter's character and a passer-by painted out). The SWAT Roblox guards stand exactly
where the players were, and a Roblox Messi stands in front of them carrying the vault on his head. The clip's own
shake, flicker and fade-out carry on over the new scene.

There are two modes:

- **`"overlay_only": true`**: the video is left untouched, and the audio is copied bit-for-bit. The stickers are
  tracked to the shot, so they follow the edit's own shake and zoom and fade with its brightness flicker and
  fade-out. Use it when the clip already freezes on the skull, like this one.
- **`"overlay_only": false`**: the script freezes the frame itself, then adds a flash and screen shake before the
  Roblox scene. It can also stamp a 💀 on for you (`add_skull`).

## Quick start

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH
cp config.example.json config.json       # then edit it (see below)

python edit.py contact input/clip.mp4 --from 3 --to 6   # find the exact skull moment
python edit.py grab config.json                          # reference frame with a pixel grid
python edit.py preview config.json --t 0.5               # still of the finished effect
python edit.py render config.json                        # -> output/football_roblox_edit.mp4
```

1. Put the clip in `input/` (e.g. `input/clip.mp4`).
2. Use `contact` to find when the skull hits. It writes a sheet of timestamped thumbnails. Set `skull_time`.
   With `"on_boom": 3`, the booms the script found are printed when you run it.
3. Use `grab` to get the reference frame, with a pixel grid on it. Draw a box around each player
   (`[x, y, width, height]`, top-left corner plus size) and note the ball's centre and radius.
4. Use `preview` to check the result, then `render`.

## Clip swap (`swap` + `hero`)

`swap` replaces the whole picture at the transform moment with a still image (a game screenshot) instead of
overlaying the live footage:

```json
"swap": {"image": "assets/depot.webp", "crop": [x0, y0, x1, y1], "erase": [[x, y, w, h], ...]},
"hero": {"foot": [x, y], "height": 320, "kit": "barca", "skin": "#e2b38c", "hair": "#4a2f1d",
         "beard": "#4a2f1d", "prop": "vault"}
```

- `crop` is in screenshot pixels and should have the shape of the clip's picture area. `erase` boxes are painted
  out first. The result is sharpened, graded and placed between the clip's black bars.
- The fitted characters (`players`) are drawn at their usual spots. The `hero` (a Roblox player rendered by
  `stage.py`) is drawn in front of everyone, with the vault floating on their head (`head_sink` moves it up or
  down).
- Every frame of the new scene still gets the clip's tracked shake and brightness, so the cut feels like part of
  the same edit.

## The Roblox characters

### Fitted to the player (`rig`): one image, puppet-warped

Every fitted player starts from the **same single picture** of a Roblox character: `assets/templates/swat.png`
(and `swat_back.png` for players seen from behind), or a kit version such as Messi's Barça one. That one image is
bent onto the player like a puppet warp. A fine triangle mesh over the picture follows the player's skeleton, each
limb is stretched across to the player's limb thickness as measured in the footage, and the result is trimmed to
the player's silhouette (`"clip": false` turns that off).

Give the player a `rig` with their joints, measured on the reference frame. Pairs are `[screen-left, screen-right]`:

```json
"rig": {"head": [x, y], "r": 15, "neck": [x, y],
        "shoulders": [[x, y], [x, y]], "elbows": [...], "wrists": [...],
        "hips": [...], "knees": [...], "ankles": [...], "toes": [...],
        "face": true, "front": ["r_arm"], "back": []}
```

`outfit` is `"swat"` or the name of one of your `kits`. `face: false` uses the back view of the template.
`front` / `back` move a limb (`l_arm`, `r_arm`, `l_leg`, `r_leg`) in front of or behind the body.
`python puppet.py` re-renders the SWAT template images.

### Stickers

Players without a `rig` get a transparent PNG sticker, stretched to fill their box (times `fit`) and stood on the
bottom of it.

- **Your own PNG:** give a player `"png": "path/to/avatar.png"` (for example a screenshot of your Roblox avatar
  on a transparent background). It gets cropped to its edges and stretched the same way.
- **Rendered kit avatar:** otherwise `avatar3d.py` renders a smooth, plastic-looking R6 noob in the player's kit
  (`kits` + `kit`, or `shirt` / `stripes` / `pants` / `socks` on the player, or colours sampled from the
  footage). Renders are cached in `assets/avatars/<kit>_<pose>.png`. Delete a file there to re-render it.

You can also render one by hand:

```bash
python avatar3d.py out.png --shirt "#1b3f94" --stripes "#a50044" --pants "#1b3f94" --pose run
```

## Config

| key | default | what it does |
|---|---|---|
| `input` / `output` | – / `output/football_roblox_edit.mp4` | paths, relative to the config file |
| `upscale` | `1` | render at N× the clip's resolution so the stickers come out crisp (e.g. `3` turns 360×640 into 1080×1920) |
| `overlay_only` | `false` | `true` = leave the video as it is and only add the tracked overlays (see above) |
| `skull_time` | **required** | seconds into the clip where the 💀 hits |
| `on_boom` | `null` | transform on the Nth bass boom after the skull (found in the audio) |
| `transform_time` | `null` | or give the transform moment in seconds directly |
| `skull_hold` | `1.0` | used when neither of the above is set: transform this long after the skull |
| `ref_time` | transform time | the frame your boxes are measured on (`grab` shows it) |
| `appear` | `"instant"` | `"instant"` = pasted on in one frame; `"poof"` = smoke puff + elastic pop, one player at a time |
| `swap` / `hero` | `null` | cut to a still Roblox scene with a hero character in front (see "Clip swap") |
| `remove_players` | `true` | `false` = overlay only: the real players and ball stay underneath the characters and vault |
| `keep_on_top` | `[]` | `[x,y,w,h]` boxes around things in the clip (like its 💀) that should stay in front of the stickers |
| `kits` | `{}` | named colour sets: `{"barca": {"shirt": "#...", "stripes": "#...", "pants": "#...", "socks": "#..."}}` |
| `fit` | `[1.1, 1.08]` | sticker size relative to the player's box (width, height) |
| `players[]` | `[]` | `box` is required. Fitted: `rig`, `outfit`, `clip`. Sticker: `kit`, `pose` (`idle` `run` `star` `tpose` `cheer`), `flip`, `png`, `fit`, `foot` (`[x,y]` override), the colour keys above, and `fill: "clone"` (fill the hole with texture cloned from beside the player instead of smoothing it, good for players in front of a goal net) |
| `ball` | `null` | `{"center":[x,y], "radius":r}`, plus an optional vault `width` in px |
| `erase` | `[]` | extra `[x,y,w,h]` boxes to paint out at the transform |
| `end_time` | clip length | stop here; in freeze mode, a time past the end of the clip keeps the Roblox scene going |
| `freeze_on_skull` | `true` | freeze mode: freeze as soon as the skull hits (`false` = keep playing until the transform) |
| `add_skull` / `skull` | `false` / centred | freeze mode: stamp a 💀 on; position/size as fractions of the frame |
| `flash` / `shake` / `slow_zoom` | `true` / `true` / `0.05` | freeze mode camera effects on the transform |
| `sfx` | `false` | mixes a synthesized pop + boing + chime over the original audio (the original sound is always kept) |

All coordinates are pixels in the (upscaled) frame that `grab` gives you.

## How it works

- `edit.py` streams frames through ffmpeg. Everything before the transform is untouched, and the original audio
  track is copied into the output.
- Booms are found from the bass (30–160 Hz) energy: a boom is where the bass comes back after a dip.
- On the reference frame, the players and ball are removed (non-grass pixels inside each box get inpainted with
  OpenCV, or clone-filled), and the stickers are placed.
- In overlay mode, every later frame is matched to the reference (ORB features + ECC refinement), which gives an
  affine warp plus a brightness gain. The cleanup masks and the sticker layer are warped by it, so everything
  stays glued to the shot.
- `puppet.py` renders the template character once (image, part map and joint positions), then warps that one
  image onto each player with linear blend skinning over a triangle mesh.
- `avatar3d.py` is a small numpy SDF raymarcher: rounded R6 parts, soft shadows, ambient occlusion, rim light, the
  classic smile and striped kits.
- `stage.py` builds the swap scene from a screenshot and renders the hero player.
- `vault.py` models the vault (rainbow-framed holographic cube, magenta screen, rainbow keypad, dial, LEDs, magenta
  handle, glowing green pad), renders it once to `assets/vault.png`, and animates its neon glow, sheen and
  sparkles. `python vault.py` re-renders it.
