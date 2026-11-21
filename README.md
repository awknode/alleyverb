# ALLEYVERB

**Convolution and algorithmic reverb with an impulse browser, FREEZE and a live "where it hits" view.**

ALLEYVERB loads your impulse responses (WAV) or runs 12 built-in algorithms. THE TAIL shows the decay with the RT60 marked, and WHERE IT HITS shows the reverb's spectrum over the dry, so you can see where the space is building up.

## Features

- **IMPULSE engine:** a browser with search, folders, favourites and arrow-key stepping.
- **ALGO engine:** 12 modes (HALL, BRIGHT HALL, PLATE, ROOM, CHAMBER, CATHEDRAL, AMBIENCE, CHORUS SPACE, WARP SPACE, LO-FI HALL, LO-FI PLATE, GATED) in three COLOR eras: VINTAGE, 80s and NOW.
- **FREEZE**, **REVERSE**, **SWELL**, shimmer and a one-knob **RISER**.
- **DUCK** from the track itself, the sidechain or both.
- **PARALLEL:** the original passes through bit-identical, like a send return.
- Pre-delay (ms or synced), low/high cut, width, MIX and SAFE.
- 44 factory presets.

## Install

Windows 64-bit, VST3. Built and tested in FL Studio.

1. Download `AlleyVerb-v0.6.0-win64.zip` from [Releases](../../releases/latest).
2. Copy `AlleyVerb.vst3` into `C:\Program Files\Common Files\VST3\`.
3. Copy the files in `presets` into `Documents\Awknode\Presets\AlleyVerb\` (create the folder if it doesn't exist).
4. Rescan plugins in your DAW.

## Using it

Every awk studios plugin has the same top strip: preset browser, A/B, undo/redo, power (bypass) and a global MIX. Knobs drag up and down (hold Ctrl or Shift for fine control), scroll to nudge, and double-click to reset. **SAFE** in the footer is an output limiter that keeps the plugin from going over 0 dBFS; it's on by default.

The full control reference is in the **[manual](docs/MANUAL.md)**.

## Build from source

You'll need Docker. Clone this repo next to the [nih-plug-awk](https://github.com/awknode/nih-plug-awk) framework fork, then run the build script:

```sh
git clone https://github.com/awknode/alleyverb
git clone -b awk-studios https://github.com/awknode/nih-plug-awk
cd alleyverb && ./build.sh
```

The plugin lands in `target/bundled/AlleyVerb.vst3`.

## License

GPL-3.0. See [LICENSE](LICENSE). Built with [nih-plug](https://github.com/robbert-vdh/nih-plug).

---

More awk studios plugins: [VISE](https://github.com/awknode/vise) · [SALVE](https://github.com/awknode/salve) · [HOIST](https://github.com/awknode/hoist) · [KILN](https://github.com/awknode/kiln) · [PATINA](https://github.com/awknode/patina) · [CHOPSHOP](https://github.com/awknode/chopshop) · [THROWBACK](https://github.com/awknode/throwback) · [THROW FX](https://github.com/awknode/throw-fx) · [KICKNULL](https://github.com/awknode/kicknull) · [WAVETAG](https://github.com/awknode/wavetag)
