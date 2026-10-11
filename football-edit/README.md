# Football → Roblox edit

Takes a football clip and keeps it exactly as is, with the same footage and the same sound. When the 💀 hits, it
holds for a second. Then the frame freezes and, with a flash and a poof, every player turns into a
**stretched-out Roblox avatar** (they pop in normal-sized, then *boing* up to goofy height) and the **ball turns
into a glowing treasure chest**.

## Quick start

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH
cp config.example.json config.json       # then edit it (see below)

python edit.py contact input/clip.mp4 --from 3 --to 6   # find the exact skull moment
python edit.py grab config.json                          # freeze frame with a pixel grid
python edit.py preview config.json --t 1.5               # still of the finished effect
python edit.py render config.json                        # -> output/football_roblox_edit.mp4
```

1. Put the clip in `input/` (e.g. `input/clip.mp4`).
2. Use `contact` to find when the skull hits. It writes a sheet of timestamped thumbnails. Set `skull_time`.
3. Use `grab` to get the frame that gets frozen, with a pixel grid on it. Draw a box around each player
   (`[x, y, width, height]`, top-left corner plus size) and note the ball's centre and radius.
4. Use `preview` to check the result, then `render`.

## Config

| key | default | what it does |
|---|---|---|
| `input` / `output` | – / `output/football_roblox_edit.mp4` | paths, relative to the config file |
| `skull_time` | **required** | seconds into the clip where the 💀 hits |
| `skull_hold` | `1.0` | how long the skull stays before the transformation |
| `freeze_on_skull` | `true` | freeze the picture as soon as the skull hits (`false` = keep playing until the transform) |
| `add_skull` | `false` | `true` if your clip doesn't already have the skull. The script stamps one on with a punch-in shake |
| `skull` | `{"x":0.5,"y":0.42,"size":0.42,"after":"pop"}` | position/size (fractions of the frame) of the added skull; `"after":"keep"` leaves it on screen |
| `erase` | `[]` | extra `[x,y,w,h]` boxes to paint out at the transform, e.g. a skull that's baked into the clip |
| `stretch` | `2.4` | how tall/goofy the avatars get (1 = normal Roblox proportions) |
| `height_scale` | `1.25` | final avatar height relative to the player's box |
| `players[]` | `[]` | `box` is required. Optional: `shirt`, `pants` (hex; auto-picked from the jersey otherwise), `pose` (`tpose` `cheer` `spread` `wave` `dab` `idle`), `stretch`, `scale` |
| `ball` | `null` | `{"center":[x,y], "radius":r}`, plus an optional `chest_width` in px |
| `end_time` | clip length | stop here; if it's past the end of the clip, the Roblox scene keeps going |
| `flash` / `shake` | `true` | white flash and screen shake on the transform |
| `slow_zoom` | `0.05` | slow push-in on the frozen scene |
| `sfx` | `false` | mixes a synthesized pop + boing + chime over the original audio (the original sound is always kept) |

All coordinates are pixels in the source video, measured on the frame `grab` gives you.

## How it works

- `edit.py` streams frames through ffmpeg. Everything before the skull is untouched, and the original audio track
  is copied into the output.
- At the freeze frame, the players and ball are removed (non-grass pixels inside each box get inpainted with
  OpenCV). Shirt and shorts colours are sampled from each player so the avatars wear the right kit.
- `roblox.py` draws the R6 avatars (yellow head with the classic smile, jersey-coloured torso and sleeves) and
  the chest as flat-shaded extruded blocks. Each sprite is rendered once, then squashed and stretched per frame.
