# Football → Roblox edit

Takes a football clip and keeps it exactly as is, with the same footage and the same sound. One second after the 💀
hits, every player poofs into a **stretched-out Roblox avatar** (they pop in normal-sized, then *boing* up to
goofy height) and the **ball poofs into a glowing treasure chest**.

`config.json` is the finished setup for `input/videoplayback123.mp4` (the Messi edit). Its output is
`output/messi_roblox_edit.mp4`.

There are two modes:

- **`"overlay_only": true`**: the video is left untouched, and the audio is copied bit-for-bit. The avatars and chest
  are drawn on top and tracked to the shot, so they follow the edit's own shake and zoom and fade with its
  brightness flicker and fade-out. Use it when the clip already freezes on the skull, like this one.
- **`"overlay_only": false`**: the script freezes the frame itself, then adds a flash and screen shake before the
  Roblox scene. It can also stamp a 💀 on for you (`add_skull`).

## Quick start

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH
cp config.example.json config.json       # then edit it (see below)

python edit.py contact input/clip.mp4 --from 3 --to 6   # find the exact skull moment
python edit.py grab config.json                          # reference frame with a pixel grid
python edit.py preview config.json --t 1.5               # still of the finished effect
python edit.py render config.json                        # -> output/football_roblox_edit.mp4
```

1. Put the clip in `input/` (e.g. `input/clip.mp4`).
2. Use `contact` to find when the skull hits. It writes a sheet of timestamped thumbnails. Set `skull_time`.
3. Use `grab` to get the reference frame (the one the transformation starts on), with a pixel grid on it. Draw a box
   around each player (`[x, y, width, height]`, top-left corner plus size) and note the ball's centre and
   radius.
4. Use `preview` to check the result, then `render`.

## Config

| key | default | what it does |
|---|---|---|
| `input` / `output` | – / `output/football_roblox_edit.mp4` | paths, relative to the config file |
| `upscale` | `1` | render at N× the clip's resolution so the avatars come out crisp (e.g. `3` turns 360×640 into 1080×1920) |
| `overlay_only` | `false` | `true` = leave the video as it is and only add the tracked overlays (see above) |
| `skull_time` | **required** | seconds into the clip where the 💀 hits |
| `skull_hold` | `1.0` | how long the skull stays before the transformation |
| `keep_on_top` | `[]` | `[x,y,w,h]` boxes around things in the clip (like its 💀) that should stay in front of the avatars |
| `freeze_on_skull` | `true` | freeze the picture as soon as the skull hits (`false` = keep playing until the transform) |
| `add_skull` | `false` | `true` if your clip doesn't already have the skull. The script stamps one on with a punch-in shake |
| `skull` | `{"x":0.5,"y":0.42,"size":0.42,"after":"pop"}` | position/size (fractions of the frame) of the added skull; `"after":"keep"` leaves it on screen |
| `erase` | `[]` | extra `[x,y,w,h]` boxes to paint out at the transform, e.g. a skull that's baked into the clip |
| `stretch` | `2.4` | how tall/goofy the avatars get (1 = normal Roblox proportions) |
| `height_scale` | `1.25` | final avatar height relative to the player's box |
| `players[]` | `[]` | `box` is required. Optional: `shirt`, `pants` (hex; auto-picked from the jersey otherwise), `stripes` (second shirt colour, drawn as vertical stripes), `pose` (`tpose` `cheer` `spread` `wave` `dab` `idle`), `stretch`, `scale`, `foot` (`[x,y]` override), `fill: "clone"` (fill the hole with texture cloned from beside the player instead of smoothing it, good for players in front of a goal net) |
| `ball` | `null` | `{"center":[x,y], "radius":r}`, plus an optional `chest_width` in px |
| `end_time` | clip length | stop here; in freeze mode, a time past the end of the clip keeps the Roblox scene going |
| `flash` / `shake` | `true` | white flash and screen shake on the transform |
| `slow_zoom` | `0.05` | slow push-in on the frozen scene |
| `sfx` | `false` | mixes a synthesized pop + boing + chime over the original audio (the original sound is always kept) |

All coordinates are pixels in the (upscaled) frame that `grab` gives you.

## How it works

- `edit.py` streams frames through ffmpeg. Everything before the skull is untouched, and the original audio track
  is copied into the output.
- On the reference frame, the players and ball are removed (non-grass pixels inside each box get inpainted with
  OpenCV, or clone-filled). Shirt and shorts colours are sampled from each player so the avatars wear the right kit.
- In overlay mode, every later frame is matched to the reference (ORB features + ECC refinement), which gives an
  affine warp plus a brightness gain. The cleanup masks and the overlay layer are warped by it, so everything
  stays glued to the shot.
- `roblox.py` draws the R6 avatars (yellow head with the classic smile, jersey-coloured torso and sleeves) and
  the chest as flat-shaded extruded blocks. Each sprite is rendered once, then squashed and stretched per frame.
