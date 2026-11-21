# ALLEYVERB manual

## Signal flow

```
wet: convolution or ALGO ─► pre-delay ─► [FREEZE] ─► low cut ─► high cut ─► width ─► DUCK ─► WET level
dry: delayed by the convolution latency so both line up
```

With **PARALLEL** on, the original passes through untouched and every control shapes only the reverb copy.

## Impulse files

ALLEYVERB looks for WAV impulses in:

```
Documents\Awknode\Impulses\
```

Make sub-folders to group them. No impulses ship with the plugin; use your own, or switch to the ALGO engine.

## SEC.01

- **IMPULSE tab:** a search box, folders and impulse names, and stars for favourites. The up/down arrow keys step through the list.
- **ALGO tab:** the 12 algorithm modes and the COLOR era (VINTAGE, 80s or NOW). ALGO adds SIZE, DECAY, ATTACK, early/late diffusion, mod rate/depth, bass multiplier + crossover, and high shelf + frequency.

## Main controls

| Control | What it does |
|---|---|
| **DRY / WET** | Levels of the dry and reverb paths. |
| **PRE-DELAY** | In ms, or synced to the host (1/64 to 1/4D). |
| **LENGTH / DECAY** | Trims the impulse or sets the tail length. DECAY ORIGINAL keeps the impulse's own decay. |
| **LOW CUT / HIGH CUT** | Filters on the reverb only. |
| **WIDTH** | Stereo width of the reverb. |
| **DUCK** | Pushes the reverb down while the source plays. The source can be SELF, SIDECHAIN or BOTH (whichever is louder), with a frequency band. |
| **SHIMMER** | Pitch, feedback and tone of a shimmer layer. |
| **SWELL** | The reverb swells in after each phrase: length, amount, sensitivity and curve. |
| **RISER** | One-knob build: more reverb, wider and thinner, with shimmer creeping in. |
| **REVERSE / FREEZE** | Reverse the impulse, or hold the current tail forever. |
| **OUTPUT** | Output level. |

## Displays

- **THE TAIL:** the loaded impulse's envelope (or the ALGO decay shape) in dB over time, with the RT60 marked. It also has PARALLEL, PREV/NEXT, REVERSE and FREEZE.
- **WHERE IT HITS:** the live spectrum of the reverb (in colour) over the dry (grey), with the output cuts and the duck band.
- **IN / OUT** meters with held peaks.

## Latency

The convolution engine reports its latency and the dry is delayed to match, so FL lines everything up.
