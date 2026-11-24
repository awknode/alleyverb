//! ALGO engine: an algorithmic reverb, built to be cheap (~1% of a core).
//!
//! signal: input -> ATTACK swell taps + early reflections -> bandwidth (COLOR)
//!         -> EARLY DIFF allpasses -> early output + injection into the tank
//! tank:   8 modulated delay lines, orthogonal (Hadamard) mixing, and per line:
//!         LATE DIFF allpass -> BASS MULT split at BASS XOVER -> HIGH SHELF
//!         above HIGH FREQ. Decay gains come from DECAY (RT60) and the line
//!         length, so SIZE never changes the decay time.
//! Every gain in the loop is <= 1 and the mixing is orthogonal, so it can't
//! blow up at any setting.

use std::f32::consts::TAU;

pub const MODE_NAMES: [&str; 12] = [
    "HALL", "BRIGHT HALL", "PLATE", "ROOM", "CHAMBER", "CATHEDRAL", "AMBIENCE", "CHORUS SPACE", "WARP SPACE", "LO-FI HALL",
    "LO-FI PLATE", "GATED",
];
pub const MODE_INFO: [&str; 12] = [
    "Big, smooth concert space with a gentle build",
    "Hall with an open, airy top end",
    "Dense, instant, bright metal plate",
    "Small room: tight reflections, quick tail",
    "Warm, woody chamber with strong early bounce",
    "Huge stone space, slow and deep",
    "Mostly early reflections: space without a tail",
    "Lush and detuned, heavy movement",
    "Restless space: random drifting reflections",
    "Grainy, saturated old-hardware hall",
    "Gritty, darker vintage plate",
    "Dense burst that cuts off hard (DECAY = gate time)",
];
pub const COLOR_NAMES: [&str; 3] = ["VINTAGE", "80s", "NOW"];

#[derive(Clone, Copy)]
struct ModeDef {
    /// Line length scale (x the HALL set).
    scale: f32,
    /// Early reflection level, spacing scale.
    er: f32,
    er_scale: f32,
    /// Multipliers on the EARLY / LATE DIFF knobs.
    in_diff: f32,
    late_diff: f32,
    /// Multiplier on MOD DEPTH.
    modm: f32,
    /// Multiplier on HIGH SHELF loss.
    hf: f32,
    /// Bandwidth multiplier.
    bw: f32,
    random: bool,
    gated: bool,
    lofi: bool,
    tank: f32,
}

const fn m(scale: f32, er: f32, er_scale: f32, in_diff: f32, late_diff: f32, modm: f32, hf: f32, bw: f32, tank: f32) -> ModeDef {
    ModeDef { scale, er, er_scale, in_diff, late_diff, modm, hf, bw, random: false, gated: false, lofi: false, tank }
}

const MODES: [ModeDef; 12] = [
    m(1.0, 0.35, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),  // HALL
    m(1.0, 0.30, 1.0, 1.0, 1.0, 1.0, 0.45, 1.6, 1.0), // BRIGHT HALL
    m(0.5, 0.0, 0.5, 1.25, 1.2, 0.7, 0.8, 1.3, 1.1),  // PLATE
    m(0.35, 0.75, 0.45, 0.9, 0.9, 0.5, 1.2, 1.0, 0.8), // ROOM
    m(0.55, 0.6, 0.7, 1.0, 1.0, 0.6, 1.4, 0.9, 0.9),  // CHAMBER
    m(1.8, 0.3, 1.8, 1.1, 1.1, 1.2, 1.0, 1.0, 1.0),   // CATHEDRAL
    m(0.25, 1.0, 0.4, 0.8, 0.8, 0.4, 1.0, 1.0, 0.35), // AMBIENCE
    m(0.9, 0.25, 1.0, 1.0, 1.0, 3.2, 1.0, 1.0, 1.0),  // CHORUS SPACE
    ModeDef { random: true, ..m(1.0, 0.3, 1.2, 1.0, 1.0, 2.2, 1.0, 1.0, 1.0) }, // WARP SPACE
    ModeDef { lofi: true, ..m(1.0, 0.3, 1.0, 1.0, 1.0, 1.4, 1.6, 0.55, 1.0) }, // LO-FI HALL
    ModeDef { lofi: true, ..m(0.5, 0.0, 0.5, 1.25, 1.2, 1.0, 1.7, 0.5, 1.1) }, // LO-FI PLATE
    ModeDef { gated: true, ..m(0.45, 0.3, 0.6, 1.3, 1.2, 0.3, 0.8, 1.2, 1.2) }, // GATED
];

const LINE_MS: [f32; 8] = [43.7, 51.3, 59.9, 67.1, 73.9, 83.3, 91.7, 101.9];
const LATE_AP_MS: [f32; 8] = [3.1, 4.3, 5.3, 6.7, 3.7, 4.9, 5.9, 7.3];
const IN_AP_MS: [f32; 4] = [4.77, 3.59, 12.73, 9.31];
const ER_MS: [f32; 8] = [7.1, 11.3, 17.9, 23.3, 31.7, 41.9, 53.1, 67.7];
const MAX_SCALE: f32 = 1.8 * 2.0;
pub const MAX_ATTACK_MS: f32 = 400.0;
const MAX_MOD_MS: f32 = 6.0;
/// Overall level, matched to the impulse engine's loudness.
const OUT_GAIN: f32 = 1.45;
/// Per-mode level trims (dB) so flipping modes doesn't jump in loudness.
/// 0.7.0: every measured mode brought to ValhallaVintageVerb's Concert Hall level (awk-bench 2026-11-22,
/// average of noise + 1 kHz bursts), still level with each other; WARP / LO-FI / GATED moved by the average.
const TRIM_DB: [f32; 12] = [3.0, 2.2, -1.8, 0.6, 1.8, 5.1, 5.3, 2.8, 3.1, 4.4, 2.2, -1.1];
/// 0.6.1 trims, so the shimmer tank (a feedback loop) can take the 0.7.0 level change back out.
const TRIM_DB_061: [f32; 12] = [0.0, 0.2, -1.6, -2.0, -0.7, 2.2, -0.4, 0.9, 0.5, 1.8, -0.4, -3.7];
/// VINTAGE / 80s read quieter once darker; level them with NOW.
const COLOR_TRIM_DB: [f32; 3] = [2.0, 0.8, 0.0];

pub struct AlgoParams {
    pub mode: usize,
    pub color: usize,
    pub decay: f32,
    pub size: f32,
    pub attack: f32,
    pub early_diff: f32,
    pub late_diff: f32,
    pub mod_rate: f32,
    pub mod_depth: f32,
    pub bass_mult: f32,
    pub bass_xover: f32,
    pub high_shelf_db: f32,
    pub high_freq: f32,
}

struct Delay {
    buf: Vec<f32>,
    w: usize,
}

impl Delay {
    fn new(n: usize) -> Self {
        Self { buf: vec![0.0; n.max(4)], w: 0 }
    }
    #[inline]
    fn push(&mut self, x: f32) {
        self.buf[self.w] = x;
        self.w += 1;
        if self.w == self.buf.len() {
            self.w = 0;
        }
    }
    /// `d` samples ago (d = 1: the newest), linear interpolation.
    #[inline]
    fn read(&self, d: f32) -> f32 {
        let n = self.buf.len();
        let d = d.clamp(1.0, (n - 2) as f32);
        let mut p = self.w as f32 - d;
        if p < 0.0 {
            p += n as f32;
        }
        let i = p as usize;
        let f = p - i as f32;
        let a = self.buf[i % n];
        let b = self.buf[(i + 1) % n];
        a + (b - a) * f
    }
    #[inline]
    fn tap(&self, d: usize) -> f32 {
        let n = self.buf.len();
        self.buf[(self.w + n - d.clamp(1, n - 1)) % n]
    }
    fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
    }
}

struct Ap {
    d: Delay,
}

impl Ap {
    fn new(n: usize) -> Self {
        Self { d: Delay::new(n) }
    }
    #[inline]
    fn tick(&mut self, x: f32, len: f32, g: f32) -> f32 {
        // whole samples: a fractional (linear) read is a low-pass inside the loop on every pass,
        // which darkened the small modes most (ROOM / CHAMBER / AMBIENCE go round most often)
        let dl = self.d.tap(len.round().max(1.0) as usize);
        let v = x + g * dl;
        self.d.push(v);
        dl - g * v
    }
}

#[derive(Clone, Copy, Default)]
struct OnePole {
    z: f32,
}

impl OnePole {
    #[inline]
    fn lp(&mut self, x: f32, a: f32) -> f32 {
        self.z += a * (x - self.z);
        self.z
    }
}

fn coef(f: f32, sr: f32) -> f32 {
    1.0 - (-TAU * (f / sr).min(0.49)).exp()
}

/// Cheap sine for the LFOs (phase in cycles).
#[inline]
fn fsin(ph: f32) -> f32 {
    let x = ph - ph.floor();
    let t = if x < 0.5 { 4.0 * x - 1.0 } else { 3.0 - 4.0 * x }; // triangle -1..1
    t * (1.5 - 0.5 * t * t) // soften the corners
}

pub struct Algo {
    sr: f32,
    inbuf: [Delay; 2],
    in_ap: [Vec<Ap>; 2],
    lines: Vec<Delay>,
    late_ap: Vec<Ap>,
    lo: [OnePole; 8],
    hs: [OnePole; 8],
    bw_in: [OnePole; 2],
    bw_out: [OnePole; 2],
    len_s: [f32; 8],
    /// allpass-interpolator state per tank line (flat magnitude; a linear read at a moving
    /// fraction low-passes every pass and made every mode dark: ~6.6 kHz vs VintageVerb's ~9.3 kHz)
    ap_y: [f32; 8],
    lfo: [f32; 8],
    rnd_cur: [f32; 8],
    rnd_tgt: [f32; 8],
    rnd_count: u32,
    rng: u32,
    gate_hold: f32,
    gate: f32,
    // computed by set()
    def: ModeDef,
    target_len: [f32; 8],
    ap_len: [f32; 8],
    in_ap_len: [f32; 4],
    g_lo: [f32; 8],
    g_mid: [f32; 8],
    hs_keep: [f32; 8],
    a_lo: f32,
    a_hs: f32,
    a_bw: f32,
    bw_on: bool,
    in_g: f32,
    late_g: f32,
    mod_s: f32,
    mod_inc: f32,
    atk_taps: [usize; 8],
    atk_w: [f32; 8],
    er_taps: [usize; 8],
    er_g: [f32; 8],
    er_level: f32,
    gate_len: f32,
    out_gain: f32,
    /// multiply the output by this for 0.6.1's level (used inside feedback loops)
    pub lvl_061: f32,
}

impl Algo {
    /// Allocates for the largest settings (call off the audio thread).
    pub fn new(sr: f32) -> Self {
        let ms = |v: f32| (v * 0.001 * sr) as usize;
        let max_line = ms(LINE_MS[7] * MAX_SCALE + MAX_MOD_MS * 2.0) + 8;
        let in_len = ms(MAX_ATTACK_MS + ER_MS[7] * MAX_SCALE * 1.1) + 8;
        let mk_in = || IN_AP_MS.iter().map(|&v| Ap::new(ms(v * 2.0) + 4)).collect::<Vec<_>>();
        let mut a = Self {
            sr,
            inbuf: [Delay::new(in_len), Delay::new(in_len)],
            in_ap: [mk_in(), mk_in()],
            lines: (0..8).map(|_| Delay::new(max_line)).collect(),
            late_ap: LATE_AP_MS.iter().map(|&v| Ap::new(ms(v * 2.0) + 4)).collect(),
            lo: [OnePole::default(); 8],
            hs: [OnePole::default(); 8],
            bw_in: [OnePole::default(); 2],
            bw_out: [OnePole::default(); 2],
            len_s: [0.0; 8],
            ap_y: [0.0; 8],
            lfo: [0.0, 0.13, 0.29, 0.41, 0.57, 0.66, 0.78, 0.91],
            rnd_cur: [0.0; 8],
            rnd_tgt: [0.0; 8],
            rnd_count: 0,
            rng: 0xa1b2_c3d4,
            gate_hold: 0.0,
            gate: 0.0,
            def: MODES[0],
            target_len: [1000.0; 8],
            ap_len: [100.0; 8],
            in_ap_len: [100.0; 4],
            g_lo: [0.0; 8],
            g_mid: [0.0; 8],
            hs_keep: [1.0; 8],
            a_lo: 0.1,
            a_hs: 0.5,
            a_bw: 1.0,
            bw_on: false,
            in_g: 0.5,
            late_g: 0.5,
            mod_s: 0.0,
            mod_inc: 0.0,
            atk_taps: [1; 8],
            atk_w: [0.125; 8],
            er_taps: [1; 8],
            er_g: [0.0; 8],
            er_level: 0.0,
            gate_len: 0.0,
            out_gain: OUT_GAIN,
            lvl_061: 1.0,
        };
        a.set(&AlgoParams {
            mode: 0,
            color: 2,
            decay: 2.0,
            size: 1.0,
            attack: 0.3,
            early_diff: 0.6,
            late_diff: 0.6,
            mod_rate: 0.5,
            mod_depth: 0.3,
            bass_mult: 1.0,
            bass_xover: 400.0,
            high_shelf_db: -6.0,
            high_freq: 6000.0,
        });
        a.len_s = a.target_len;
        a
    }

    /// Recompute coefficients (once per block; cheap).
    pub fn set(&mut self, p: &AlgoParams) {
        let sr = self.sr;
        let def = MODES[p.mode.min(11)];
        self.def = def;
        let scale = def.scale * p.size.clamp(0.5, 2.0);
        let rt = if def.gated { 3.0 } else { p.decay.clamp(0.05, 60.0) };
        let color_hf = [1.5, 1.2, 1.0][p.color.min(2)];
        let color_mod = [1.4, 1.15, 1.0][p.color.min(2)];
        // 0.7.0: VINTAGE / 80s / NOW set against VintageVerb 70s / 80s / now (tails ~5.1 / ~8.3 / ~9.3 kHz)
        let color_bw = [3200.0, 7000.0, 10000.0][p.color.min(2)];
        for k in 0..8 {
            let len = LINE_MS[k] * scale * 0.001 * sr;
            self.target_len[k] = len;
            self.ap_len[k] = LATE_AP_MS[k] * scale.sqrt() * 0.001 * sr;
            self.g_mid[k] = 10f32.powf(-3.0 * len / (rt * sr));
            self.g_lo[k] = 10f32.powf(-3.0 * len / (rt * p.bass_mult.clamp(0.1, 8.0) * sr)).min(0.99995);
            // Extra loss for the highs: HIGH SHELF dB per second of travel.
            let shelf = p.high_shelf_db.min(0.0) * def.hf * color_hf;
            self.hs_keep[k] = 10f32.powf(shelf / 20.0 * len / sr);
        }
        for (k, l) in self.in_ap_len.iter_mut().enumerate() {
            *l = IN_AP_MS[k] * scale.sqrt().min(1.9) * 0.001 * sr;
        }
        self.a_lo = coef(p.bass_xover, sr);
        self.a_hs = coef(p.high_freq, sr);
        let bw = (color_bw * def.bw).min(0.45 * sr);
        self.bw_on = bw < 19000.0;
        self.a_bw = coef(bw, sr);
        self.in_g = (p.early_diff * 0.72 * def.in_diff).clamp(0.0, 0.85);
        self.late_g = (p.late_diff * 0.65 * def.late_diff).clamp(0.0, 0.8);
        self.mod_s = (p.mod_depth * 0.0018 * sr * def.modm * color_mod).min(MAX_MOD_MS * 0.001 * sr);
        self.mod_inc = p.mod_rate / sr;
        // ATTACK: spread the injection over up to 400 ms, weighted toward the end.
        let t = p.attack.clamp(0.0, 1.0) * MAX_ATTACK_MS * 0.001 * sr;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for k in 0..8 {
            self.atk_taps[k] = 1 + (t * k as f32 / 7.0) as usize;
            let w = ((k + 1) as f32).powf(p.attack * 2.5);
            self.atk_w[k] = w;
            sum += w;
            sum2 += w * w;
        }
        let spread = (t / (0.005 * sr)).clamp(0.0, 1.0);
        let norm = sum * (1.0 - spread) + sum2.sqrt() * spread;
        self.atk_w.iter_mut().for_each(|w| *w /= norm);
        for k in 0..8 {
            self.er_taps[k] = 1 + (ER_MS[k] * def.er_scale * p.size * 0.001 * sr) as usize;
            self.er_g[k] = 0.82f32.powi(k as i32) * if k % 3 == 1 { -1.0 } else { 1.0 } * 0.45;
        }
        self.er_level = def.er * (1.0 - 0.6 * p.attack);
        self.gate_len = p.decay.clamp(0.05, 5.0) * sr;
        self.out_gain = OUT_GAIN * 10f32.powf((TRIM_DB[p.mode.min(11)] + COLOR_TRIM_DB[p.color.min(2)]) / 20.0);
        self.lvl_061 = 10f32.powf((TRIM_DB_061[p.mode.min(11)] - TRIM_DB[p.mode.min(11)] - COLOR_TRIM_DB[p.color.min(2)]) / 20.0);
    }

    pub fn clear(&mut self) {
        for d in self.inbuf.iter_mut().chain(self.lines.iter_mut()) {
            d.clear();
        }
        for a in self.in_ap.iter_mut().flatten().chain(self.late_ap.iter_mut()) {
            a.d.clear();
        }
        self.lo = [OnePole::default(); 8];
        self.hs = [OnePole::default(); 8];
        self.gate = 0.0;
        self.gate_hold = 0.0;
        self.ap_y = [0.0; 8];
    }

    #[inline]
    pub fn tick(&mut self, l: f32, r: f32) -> (f32, f32) {
        let def = self.def;
        self.inbuf[0].push(l);
        self.inbuf[1].push(r);
        let mut d = [0.0f32; 2];
        let mut early = [0.0f32; 2];
        for c in 0..2 {
            let buf = &self.inbuf[c];
            let mut s = 0.0;
            for k in 0..8 {
                s += self.atk_w[k] * buf.tap(self.atk_taps[k]);
            }
            let mut e = 0.0;
            if self.er_level > 0.0 {
                for k in 0..8 {
                    let tap = if c == 0 { self.er_taps[k] } else { self.er_taps[k] + self.er_taps[k] / 14 };
                    e += self.er_g[k] * buf.tap(tap);
                }
            }
            let mut x = s + e * self.er_level;
            if self.bw_on {
                x = self.bw_in[c].lp(x, self.a_bw);
            }
            for (k, ap) in self.in_ap[c].iter_mut().enumerate() {
                x = ap.tick(x, self.in_ap_len[k], self.in_g);
            }
            d[c] = x;
            early[c] = e * self.er_level;
        }
        // Modulation.
        if def.random {
            self.rnd_count += 1;
            if self.rnd_count >= (self.sr * 0.25) as u32 {
                self.rnd_count = 0;
                for t in self.rnd_tgt.iter_mut() {
                    self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    *t = (self.rng >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                }
            }
        }
        let rnd_c = (self.mod_inc * 40.0).clamp(0.00002, 0.01);
        let mut o = [0.0f32; 8];
        for k in 0..8 {
            self.len_s[k] += (self.target_len[k] - self.len_s[k]) * 0.0005;
            self.lfo[k] += self.mod_inc * (1.0 + 0.071 * k as f32);
            if self.lfo[k] >= 1.0 {
                self.lfo[k] -= 1.0;
            }
            let mv = if def.random {
                self.rnd_cur[k] += (self.rnd_tgt[k] - self.rnd_cur[k]) * rnd_c;
                self.rnd_cur[k]
            } else {
                fsin(self.lfo[k])
            };
            let d = self.len_s[k] + self.mod_s * (1.0 + mv);
            let n = (d - 0.5).floor().max(1.0);
            let fr = d - n;
            let a = (1.0 - fr) / (1.0 + fr);
            let v = a * self.lines[k].tap(n as usize) + self.lines[k].tap(n as usize + 1) - a * self.ap_y[k];
            self.ap_y[k] = v;
            o[k] = v;
        }
        // Hadamard 8 (orthogonal after /sqrt(8)).
        let mut mx = o;
        let mut h = 1;
        while h < 8 {
            let mut i = 0;
            while i < 8 {
                for j in i..i + h {
                    let (a, b) = (mx[j], mx[j + h]);
                    mx[j] = a + b;
                    mx[j + h] = a - b;
                }
                i += h * 2;
            }
            h *= 2;
        }
        let norm = 0.353_553_4;
        for k in 0..8 {
            let mut v = mx[k] * norm;
            v = self.late_ap[k].tick(v, self.ap_len[k], self.late_g);
            let lo = self.lo[k].lp(v, self.a_lo);
            v = lo * self.g_lo[k] + (v - lo) * self.g_mid[k];
            let hl = self.hs[k].lp(v, self.a_hs);
            v = hl + (v - hl) * self.hs_keep[k];
            if def.lofi {
                v = (v * 1.6).tanh() / 1.6;
            }
            let inj = if k % 2 == 0 { d[0] } else { d[1] };
            let sign = if k & 2 == 0 { 1.0 } else { -1.0 };
            self.lines[k].push(v + inj * sign * 0.35);
        }
        let tl = (o[0] - o[2] + o[4] - o[6]) * 0.5;
        let tr = (o[1] - o[3] + o[5] - o[7]) * 0.5;
        let mut out = [early[0] * 0.5 + tl * def.tank, early[1] * 0.5 + tr * def.tank];
        if self.bw_on {
            for c in 0..2 {
                out[c] = self.bw_out[c].lp(out[c], self.a_bw);
            }
        }
        if def.gated {
            let lvl = l.abs().max(r.abs());
            if lvl > 0.003 {
                self.gate_hold = self.gate_len;
            }
            let target = if self.gate_hold > 0.0 { 1.0 } else { 0.0 };
            self.gate_hold -= 1.0;
            let c = if target > self.gate { 0.01 } else { 0.004 };
            self.gate += (target - self.gate) * c;
            out[0] *= self.gate;
            out[1] *= self.gate;
        }
        (out[0] * self.out_gain, out[1] * self.out_gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn params(mode: usize, decay: f32) -> AlgoParams {
        AlgoParams {
            mode,
            color: 2,
            decay,
            size: 1.0,
            attack: 0.3,
            early_diff: 0.6,
            late_diff: 0.6,
            mod_rate: 0.5,
            mod_depth: 0.3,
            bass_mult: 1.0,
            bass_xover: 400.0,
            high_shelf_db: -6.0,
            high_freq: 6000.0,
        }
    }

    fn impulse_response(p: &AlgoParams, secs: f32, sr: f32) -> crate::wav::Ir {
        let mut a = Algo::new(sr);
        a.set(p);
        a.len_s = a.target_len;
        let n = (secs * sr) as usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let (a0, a1) = a.tick(x, x);
            l.push(a0);
            r.push(a1);
        }
        crate::wav::Ir { sr, ch: [l, r] }
    }

    #[test]
    fn every_mode_is_stable_and_decays() {
        let sr = 48000.0;
        for mode in 0..12 {
            for &(decay, bass) in &[(2.0f32, 1.0f32), (60.0, 4.0)] {
                let mut p = params(mode, decay);
                p.bass_mult = bass;
                p.size = 2.0;
                p.late_diff = 1.0;
                p.early_diff = 1.0;
                p.mod_depth = 1.0;
                let ir = impulse_response(&p, 6.0, sr);
                assert!(ir.ch.iter().all(|c| c.iter().all(|v| v.is_finite() && v.abs() < 4.0)), "mode {mode} blew up");
                let e = |a: usize, b: usize| ir.ch[0][a..b].iter().map(|v| v * v).sum::<f32>();
                let s = sr as usize;
                if decay < 10.0 && mode != 11 {
                    assert!(e(5 * s, 6 * s) < e(s / 2, 3 * s / 2) * 0.01, "mode {mode} doesn't decay");
                }
            }
        }
    }

    #[test]
    fn decay_knob_sets_rt60() {
        let sr = 48000.0;
        for &(mode, want) in &[(0usize, 2.0f32), (0, 5.0), (2, 1.5), (5, 8.0)] {
            let ir = impulse_response(&params(mode, want), want * 1.6 + 1.0, sr);
            let got = crate::wav::analyse(&ir).rt60;
            assert!((got - want).abs() / want < 0.25, "mode {mode}: want {want} got {got}");
        }
    }

    #[test]
    fn loudness_is_in_line_with_the_impulse_engine() {
        let sr = 48000.0;
        let mut bad = 0;
        for mode in 0..12 {
            let ir = impulse_response(&params(mode, 2.0), 4.0, sr);
            let e: f32 = ir.ch.iter().map(|c| c.iter().map(|v| v * v).sum::<f32>()).sum::<f32>() / 2.0;
            println!("{:<13} energy {:.3} ({:+.1} dB vs impulses)", MODE_NAMES[mode], e, 10.0 * (e / crate::wav::TARGET_ENERGY).log10());
            bad += (!(e > crate::wav::TARGET_ENERGY * 0.25 && e < crate::wav::TARGET_ENERGY * 4.0)) as usize;
        }
        assert_eq!(bad, 0);
    }
}

/// ALGO_REPORT=1: per mode / colour, tail level and centroid of a noise burst with the tone controls flat
/// (same settings as awk-bench gen_alley.py), for matching against ValhallaVintageVerb.
#[cfg(test)]
mod algo_report {
    use super::*;
    fn run(mode: usize, color: usize) -> (f32, f32) {
        let sr = 48000.0f32;
        let mut a = Algo::new(sr);
        let p = AlgoParams { mode, color, decay: 2.0, size: 1.0, attack: 0.5, early_diff: 1.0, late_diff: 1.0, mod_rate: 0.5, mod_depth: 0.3,
            bass_mult: 1.0, bass_xover: 400.0, high_shelf_db: 0.0, high_freq: 6000.0 };
        a.set(&p);
        a.len_s = a.target_len;
        let mut s = 99u32;
        let (b0, b1) = (24000usize, 24480usize);
        let n = 96000;
        let mut l = vec![0.0f32; n];
        let mut e_in = 0.0f32;
        for i in 0..n {
            let x = if i >= b0 && i < b1 { s = s.wrapping_mul(1664525).wrapping_add(1013904223); 0.25 * ((s >> 8) as f32 / 8388608.0 - 1.0) } else { 0.0 };
            e_in += x * x;
            if i % 64 == 0 { a.set(&p); }
            let (y, z) = a.tick(x, x);
            l[i] = 0.5 * (y * y + z * z);
            l[i] = l[i].sqrt() * y.signum();
            if i >= b1 { l[i] = y; } else { l[i] = 0.0; }
            let _ = z;
        }
        let seg = &l[b1 + 4800..b1 + 48000];
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for f in seg.chunks(2048).filter(|c| c.len() == 2048) {
            for k in 1..1024 {
                let w = std::f64::consts::TAU * k as f64 / 2048.0;
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, v) in f.iter().enumerate() { let h = 0.5 - 0.5 * (std::f64::consts::TAU * j as f64 / 2047.0).cos(); re += *v as f64 * h * (w * j as f64).cos(); im -= *v as f64 * h * (w * j as f64).sin(); }
                let m = (re * re + im * im).sqrt();
                num += m * k as f64 * sr as f64 / 2048.0; den += m;
            }
        }
        let e_out: f32 = l[b1..b1 + 48000].iter().map(|v| v * v).sum();
        (10.0 * (e_out / e_in).log10(), (num / den.max(1e-12)) as f32)
    }
    #[test]
    fn algo_report() {
        if std::env::var("ALGO_REPORT").is_err() { return; }
        for m in 0..8 { let (lv, c) = run(m, 2); println!("{:13} NOW: tail {lv:+5.1} dB, centroid {c:5.0} Hz", MODE_NAMES[m]); }
        for c in 0..3 { let (lv, ce) = run(0, c); println!("HALL {:8}: tail {lv:+5.1} dB, centroid {ce:5.0} Hz", COLOR_NAMES[c]); }
    }
}
