//! FREEZE + wet filters.
//!
//! FREEZE is a spectral freeze. While live, it keeps an 8192-point snapshot of
//! the reverb's spectrum (averaged over the last ~1/4 s). It also re-synthesises
//! that snapshot all the time, so the frozen sound is ready the instant you
//! press FREEZE. Freezing just stops the snapshot from updating.
//! Each bin's phase advances at its own frequency, with a small random drift, so
//! tones stay clean, nothing loops and there's no grain. Its level is matched to
//! the live reverb at the moment you freeze.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const FREEZE_N: usize = 8192;
const HOP: usize = FREEZE_N / 4;
const BINS: usize = FREEZE_N / 2 + 1;
/// Random phase drift per hop (radians): keeps the hold from sounding like a loop.
const DRIFT: f32 = 0.35;

struct Chan {
    input: Vec<f32>,
    out: Vec<f32>,
    cap: Vec<f32>,
    phase: Vec<f32>,
}

pub struct FreezeTank {
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    win: Vec<f32>,
    ch: [Chan; 2],
    /// Write position in `input` / read position in `out` (both length N rings).
    pos: usize,
    since_hop: usize,
    tbuf: Vec<f32>,
    spec: Vec<Complex<f32>>,
    sf: Vec<Complex<f32>>,
    si: Vec<Complex<f32>>,
    rng: u32,
    live_rms: f32,
    synth_rms: f32,
    rms_c: f32,
    pub makeup: f32,
    frozen: bool,
}

impl FreezeTank {
    pub fn new(_sr: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FREEZE_N);
        let ifft = planner.plan_fft_inverse(FREEZE_N);
        let win = (0..FREEZE_N).map(|k| (std::f32::consts::PI * (k as f32 + 0.5) / FREEZE_N as f32).sin()).collect();
        let mk = || Chan { input: vec![0.0; FREEZE_N], out: vec![0.0; FREEZE_N], cap: vec![0.0; BINS], phase: vec![0.0; BINS] };
        let mut t = Self {
            sf: fft.make_scratch_vec(),
            si: ifft.make_scratch_vec(),
            spec: fft.make_output_vec(),
            fft,
            ifft,
            win,
            ch: [mk(), mk()],
            pos: 0,
            since_hop: 0,
            tbuf: vec![0.0; FREEZE_N],
            rng: 0x1234_5678,
            live_rms: 0.0,
            synth_rms: 0.0,
            rms_c: (-1.0 / (0.3 * _sr)).exp(),
            makeup: 1.0,
            frozen: false,
        };
        t.scatter_phases();
        t
    }

    /// Random starting phases: if every bin started in step they would add up
    /// into a click train (static).
    fn scatter_phases(&mut self) {
        for c in self.ch.iter_mut() {
            for ph in c.phase.iter_mut() {
                self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *ph = (self.rng >> 8) as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
            }
        }
    }

    /// `l`,`r`: the live reverb. Returns the frozen sound (level-matched).
    #[inline]
    pub fn tick(&mut self, l: f32, r: f32, freeze: bool) -> (f32, f32) {
        if freeze && !self.frozen {
            self.makeup = if self.synth_rms > 1e-12 { (self.live_rms / self.synth_rms).sqrt().clamp(0.0, 30.0) } else { 0.0 };
        }
        self.frozen = freeze;
        let p = self.pos;
        let raw = (self.ch[0].out[p], self.ch[1].out[p]);
        self.ch[0].out[p] = 0.0;
        self.ch[1].out[p] = 0.0;
        self.ch[0].input[p] = l;
        self.ch[1].input[p] = r;
        self.pos = (p + 1) % FREEZE_N;
        if !freeze {
            let lv = 0.5 * (l * l + r * r);
            let sv = 0.5 * (raw.0 * raw.0 + raw.1 * raw.1);
            self.live_rms = lv + (self.live_rms - lv) * self.rms_c;
            self.synth_rms = sv + (self.synth_rms - sv) * self.rms_c;
        }
        self.since_hop += 1;
        if self.since_hop == HOP {
            self.since_hop = 0;
            self.hop(freeze);
        }
        (raw.0 * self.makeup, raw.1 * self.makeup)
    }

    fn hop(&mut self, frozen: bool) {
        let two_pi = std::f32::consts::TAU;
        for c in 0..2 {
            if !frozen {
                // Analyse the last N input samples (oldest first).
                for k in 0..FREEZE_N {
                    self.tbuf[k] = self.ch[c].input[(self.pos + k) % FREEZE_N] * self.win[k];
                }
                let _ = self.fft.process_with_scratch(&mut self.tbuf, &mut self.spec, &mut self.sf);
                for (m, x) in self.ch[c].cap.iter_mut().zip(self.spec.iter()) {
                    // ~4-frame average: smooths the snapshot.
                    *m = 0.75 * *m + 0.25 * x.norm();
                }
            }
            // Synthesise one frame from the snapshot.
            for k in 0..BINS {
                self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let j = ((self.rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * DRIFT;
                let ph = (self.ch[c].phase[k] + two_pi * (k * HOP) as f32 / FREEZE_N as f32 + j).rem_euclid(two_pi);
                self.ch[c].phase[k] = ph;
                self.spec[k] = Complex::from_polar(self.ch[c].cap[k], ph);
            }
            self.spec[0].im = 0.0;
            self.spec[BINS - 1].im = 0.0;
            let _ = self.ifft.process_with_scratch(&mut self.spec, &mut self.tbuf, &mut self.si);
            for k in 0..FREEZE_N {
                let i = (self.pos + k) % FREEZE_N;
                self.ch[c].out[i] += self.tbuf[k] * self.win[k];
            }
        }
    }

    pub fn clear(&mut self) {
        for c in self.ch.iter_mut() {
            c.input.iter_mut().for_each(|v| *v = 0.0);
            c.out.iter_mut().for_each(|v| *v = 0.0);
            c.cap.iter_mut().for_each(|v| *v = 0.0);
        }
        self.live_rms = 0.0;
        self.synth_rms = 0.0;
        self.scatter_phases();
    }
}

/// DUCK band: picks out `lo..hi` of the reverb (Linkwitz-Riley 24 dB/oct
/// edges). Ducking subtracts part of that band, so with no ducking the reverb
/// comes through bit-exact.
#[derive(Clone, Copy, Default)]
pub struct DuckBand {
    hp: [Biquad; 2],
    lp: [Biquad; 2],
    pub full: bool,
}

impl DuckBand {
    pub fn set(&mut self, lo: f32, hi: f32, sr: f32, lo_off: bool, hi_off: bool) {
        self.full = lo_off && hi_off;
        for f in self.hp.iter_mut() {
            f.bypass = lo_off;
            f.set(true, lo, sr);
        }
        for f in self.lp.iter_mut() {
            f.bypass = hi_off;
            f.set(false, hi, sr);
        }
    }
    /// Returns `x` with its band scaled by `g`.
    #[inline]
    pub fn tick(&mut self, x: f32, g: f32) -> f32 {
        if self.full {
            return x * g;
        }
        let mut b = x;
        for f in self.hp.iter_mut().chain(self.lp.iter_mut()) {
            b = f.tick(b);
        }
        x - b * (1.0 - g)
    }
}

/// 12 dB/oct RBJ high/low-pass for LOW CUT / HIGH CUT.
#[derive(Clone, Copy, Default)]
pub struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
    pub bypass: bool,
}

impl Biquad {
    pub fn set(&mut self, high_pass: bool, freq: f32, sr: f32) {
        let (b, a) = coefs(high_pass, freq, sr);
        self.b = b;
        self.a = a;
    }
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        if self.bypass {
            return x;
        }
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

pub fn coefs(high_pass: bool, freq: f32, sr: f32) -> ([f32; 3], [f32; 2]) {
    let w0 = 2.0 * std::f32::consts::PI * (freq / sr).min(0.49);
    let (sin, cos) = w0.sin_cos();
    let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1) = if high_pass { ((1.0 + cos) / 2.0, -(1.0 + cos)) } else { ((1.0 - cos) / 2.0, 1.0 - cos) };
    ([b0 / a0, b1 / a0, b0 / a0], [-2.0 * cos / a0, (1.0 - alpha) / a0])
}

/// Magnitude (dB) of a biquad at `f` — for drawing the curve.
pub fn response_db(high_pass: bool, cut: f32, f: f32, sr: f32) -> f32 {
    let (b, a) = coefs(high_pass, cut, sr);
    let w = 2.0 * std::f32::consts::PI * f / sr;
    let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
    let nr = b[0] + b[1] * c1 + b[2] * c2;
    let ni = -(b[1] * s1 + b[2] * s2);
    let dr = 1.0 + a[0] * c1 + a[1] * c2;
    let di = -(a[0] * s1 + a[1] * s2);
    10.0 * ((nr * nr + ni * ni) / (dr * dr + di * di)).max(1e-12).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goertzel(x: &[f32], f: f32, sr: f32) -> f32 {
        let w = std::f32::consts::TAU * f / sr;
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &v in x {
            let s0 = v + 2.0 * w.cos() * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2).sqrt() / x.len() as f32
    }

    /// Freeze a reverb-ish tone: the hold keeps its pitch and level, never
    /// drifts, and has no static (no sample-to-sample spikes).
    #[test]
    fn freeze_holds_the_sound_cleanly() {
        let sr = 48000.0;
        let mut t = FreezeTank::new(sr);
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut live = Vec::new();
        for n in 0..(sr as usize * 2) {
            let v = 0.3 * (std::f32::consts::TAU * 440.0 * n as f32 / sr).sin() + 0.05 * rnd();
            live.push(v);
            t.tick(v, v, false);
        }
        let live_rms = (live[live.len() - 24000..].iter().map(|v| v * v).sum::<f32>() / 24000.0).sqrt();
        let mut grab = |t: &mut FreezeTank, secs: f32| -> Vec<f32> { (0..(secs * sr) as usize).map(|_| t.tick(0.0, 0.0, true).0).collect() };
        let a = grab(&mut t, 1.0);
        let _ = grab(&mut t, 10.0);
        let b = grab(&mut t, 1.0);
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let (ra, rb) = (rms(&a[4800..]), rms(&b));
        assert!((ra / live_rms - 1.0).abs() < 0.25, "level matched: live {live_rms} frozen {ra}");
        assert!((rb / ra - 1.0).abs() < 0.1, "no drift: {ra} -> {rb}");
        // The tone survives: 440 Hz dominates over a nearby non-harmonic frequency.
        let (g440, g610) = (goertzel(&b, 440.0, sr), goertzel(&b, 610.0, sr));
        assert!(g440 > 20.0 * g610, "440 {g440} vs 610 {g610}");
        // No static: sample-to-sample jumps no worse than the live sound's own
        // (relative to level).
        let step = |x: &[f32]| x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max) / rms(x);
        let live_tail = &live[live.len() - 48000..];
        assert!(step(&b) < 2.0 * step(live_tail), "spiky: frozen {} vs live {}", step(&b), step(live_tail));
    }

    /// A pure tone frozen must come out as a pure tone: any static would show
    /// up as energy away from the note.
    #[test]
    fn frozen_pure_tone_has_no_static() {
        let sr = 48000.0;
        let mut t = FreezeTank::new(sr);
        for n in 0..(sr as usize * 2) {
            let v = 0.3 * (std::f32::consts::TAU * 440.0 * n as f32 / sr).sin();
            t.tick(v, v, false);
        }
        for _ in 0..(sr as usize * 3) {
            t.tick(0.0, 0.0, true);
        }
        let b: Vec<f32> = (0..sr as usize).map(|_| t.tick(0.0, 0.0, true).0).collect();
        let total = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
        // Energy far from 440 Hz (above 2 kHz): crude highpass = first difference twice.
        let hp: Vec<f32> = b.windows(3).map(|w| w[2] - 2.0 * w[1] + w[0]).collect();
        let hp_rms = (hp.iter().map(|v| v * v).sum::<f32>() / hp.len() as f32).sqrt();
        let expect = total * (std::f32::consts::TAU * 440.0 / sr).powi(2); // what a clean 440 Hz gives
        assert!(hp_rms < 3.0 * expect, "static: hp {hp_rms} vs clean {expect}");
    }

    /// No ducking = bit-exact; ducking a low band leaves the highs alone.
    #[test]
    fn duck_band_only_touches_its_band() {
        let sr = 48000.0;
        let mut b = DuckBand::default();
        b.set(40.0, 250.0, sr, false, false);
        let sig = |f: f32, n: usize| (std::f32::consts::TAU * f * n as f32 / sr).sin();
        for n in 0..1000 {
            let x = sig(100.0, n) + sig(5000.0, n);
            assert_eq!(b.tick(x, 1.0), x);
        }
        let mut lo_in = 0.0;
        let mut hi_left = 0.0;
        let mut b = DuckBand::default();
        b.set(40.0, 250.0, sr, false, false);
        let n_total = 48000;
        let mut out = Vec::new();
        for n in 0..n_total {
            out.push(b.tick(sig(100.0, n) + sig(5000.0, n), 0.0));
        }
        let tail = &out[24000..];
        // Correlate with each tone.
        for (k, v) in tail.iter().enumerate() {
            let n = k + 24000;
            lo_in += v * sig(100.0, n);
            hi_left += v * sig(5000.0, n);
        }
        let half = tail.len() as f32 / 2.0;
        assert!((hi_left / half - 1.0).abs() < 0.1, "5 kHz kept: {}", hi_left / half);
        assert!((lo_in / half).abs() < 0.35, "100 Hz ducked: {}", lo_in / half);
    }

    #[test]
    fn filter_curve_is_right() {
        let sr = 48000.0;
        assert!((response_db(true, 200.0, 200.0, sr) + 3.0).abs() < 0.2);
        assert!(response_db(true, 200.0, 20.0, sr) < -35.0);
        assert!(response_db(false, 5000.0, 100.0, sr).abs() < 0.1);
    }
}

// ---------------------------------------------------------------------------
// SHIMMER pitch shifter: two crossfading read heads (granular), cheap and
// smooth. `ratio` 2.0 = +12 semitones.
pub struct Pitch {
    buf: Vec<f32>,
    w: usize,
    phase: f32,
    win: f32,
    /// Hann (sin^2) grain windows: the two grains always add up to exactly 1, so the
    /// shifter has unity gain. The old sine windows peaked at +3 dB, which made a
    /// feedback loop around it build up instead of dying away.
    pub hann: bool,
}

impl Pitch {
    pub fn new(sr: f32) -> Self {
        let win = 0.06 * sr;
        Self { buf: vec![0.0; (win as usize + 8).next_power_of_two() * 2], w: 0, phase: 0.0, win, hann: true }
    }
    #[inline]
    fn read(&self, d: f32) -> f32 {
        let n = self.buf.len();
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
    pub fn tick(&mut self, x: f32, ratio: f32) -> f32 {
        let n = self.buf.len();
        self.buf[self.w] = x;
        self.w = (self.w + 1) % n;
        self.phase = (self.phase + (1.0 - ratio) / self.win).rem_euclid(1.0);
        let p2 = (self.phase + 0.5) % 1.0;
        let mut g1 = (std::f32::consts::PI * self.phase).sin();
        let mut g2 = (std::f32::consts::PI * p2).sin();
        if self.hann {
            g1 *= g1;
            g2 *= g2;
        }
        self.read(self.phase * self.win + 2.0) * g1 + self.read(p2 * self.win + 2.0) * g2
    }
    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ---------------------------------------------------------------------------
// SWELL: captures each hit and builds a reverse-reverb swell from its sound
// that rises into the hit. Works on the *input* timeline; the plugin delays
// everything else by SWELL_LOOKAHEAD so the swell can start before the hit.
//
// hit detected at t0 -> capture SWELL_W samples -> spectrum snapshot ->
// a voice of random-phase noise shaped like the hit, rising (exponential
// curve) over LEN and ending exactly where the delayed hit plays.

pub const SWELL_W: usize = 4096;
pub const SWELL_MAX_S: f32 = 2.0;
const SW_N: usize = 4096;
const SW_HOP: usize = SW_N / 2;
const SW_BINS: usize = SW_N / 2 + 1;
const VOICES: usize = 6;

pub fn swell_lookahead(sr: f32) -> usize {
    (SWELL_MAX_S * sr) as usize + SWELL_W + (0.02 * sr) as usize
}

struct Voice {
    active: bool,
    /// Samples until the swell starts (after capture).
    wait: i64,
    len: usize,
    pos: usize,
    mag: Vec<f32>,
    level: f32,
    phase: [Vec<f32>; 2],
    ola: [Vec<f32>; 2],
    ola_pos: usize,
    hop_count: usize,
}

pub struct Swell {
    sr: f32,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    win: Vec<f32>,
    cap: Vec<f32>,
    capturing: Option<usize>,
    fast: f32,
    slow: f32,
    since: usize,
    voices: Vec<Voice>,
    tbuf: Vec<f32>,
    spec: Vec<Complex<f32>>,
    sf: Vec<Complex<f32>>,
    si: Vec<Complex<f32>>,
    rng: u32,
    pub hits: u32,
    /// Voices cut off while still loud (would click). Should stay 0.
    pub loud_steals: u32,
}

impl Swell {
    pub fn new(sr: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(SW_N);
        let ifft = planner.plan_fft_inverse(SW_N);
        let voice = || Voice {
            active: false,
            wait: 0,
            len: 1,
            pos: 0,
            mag: vec![0.0; SW_BINS],
            level: 0.0,
            phase: [vec![0.0; SW_BINS], vec![0.0; SW_BINS]],
            ola: [vec![0.0; SW_N], vec![0.0; SW_N]],
            ola_pos: 0,
            hop_count: 0,
        };
        let mut s = Self {
            sr,
            win: (0..SW_N).map(|k| (std::f32::consts::PI * (k as f32 + 0.5) / SW_N as f32).sin()).collect(),
            cap: vec![0.0; SWELL_W],
            capturing: None,
            fast: 0.0,
            slow: 0.0,
            since: usize::MAX / 2,
            voices: (0..VOICES).map(|_| voice()).collect(),
            tbuf: vec![0.0; SW_N],
            spec: fft.make_output_vec(),
            sf: fft.make_scratch_vec(),
            si: ifft.make_scratch_vec(),
            fft,
            ifft,
            rng: 0x2545_f491,
            hits: 0,
            loud_steals: 0,
        };
        for v in s.voices.iter_mut() {
            for ph in v.phase.iter_mut().flatten() {
                s.rng = s.rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *ph = (s.rng >> 8) as f32 / (1u32 << 24) as f32 * std::f32::consts::TAU;
            }
        }
        s
    }

    pub fn clear(&mut self) {
        for v in self.voices.iter_mut() {
            v.active = false;
        }
        self.capturing = None;
        self.fast = 0.0;
        self.slow = 0.0;
    }

    /// `x`: mono input. `len_s`: swell length (s). `sens` 0..1 (higher = more hits).
    /// `curve` 0..1 (higher = holds back longer, then rushes in). Returns stereo swell.
    #[inline]
    pub fn tick(&mut self, x: f32, len_s: f32, sens: f32, curve: f32) -> (f32, f32) {
        let sr = self.sr;
        // Hit detection: fast envelope jumps well above the slow one.
        let a = x.abs();
        let cf = if a > self.fast { 0.3 } else { 1.0 - (-1.0 / (0.01 * sr)).exp() };
        self.fast += (a - self.fast) * cf;
        let cs = if a > self.slow { 1.0 - (-1.0 / (0.06 * sr)).exp() } else { 1.0 - (-1.0 / (0.4 * sr)).exp() };
        self.slow += (a - self.slow) * cs;
        self.since = self.since.saturating_add(1);
        let ratio = 6.0 - 4.5 * sens.clamp(0.0, 1.0);
        let floor = 10f32.powf((-50.0 + 20.0 * (1.0 - sens)) / 20.0);
        if self.capturing.is_none() && self.since > (0.12 * sr) as usize && self.fast > floor && self.fast > self.slow * ratio {
            self.capturing = Some(0);
            self.since = 0;
        }
        if let Some(i) = self.capturing {
            self.cap[i] = x;
            if i + 1 == SWELL_W {
                self.capturing = None;
                self.start_voice(len_s);
            } else {
                self.capturing = Some(i + 1);
            }
        }
        // Voices.
        let mut out = (0.0, 0.0);
        let k = 2.0 + curve.clamp(0.0, 1.0) * 8.0;
        for vi in 0..VOICES {
            if !self.voices[vi].active {
                continue;
            }
            if self.voices[vi].wait > 0 {
                self.voices[vi].wait -= 1;
                continue;
            }
            if self.voices[vi].hop_count == 0 {
                self.synth_frame(vi);
            }
            let v = &mut self.voices[vi];
            v.hop_count = (v.hop_count + 1) % SW_HOP;
            let t = v.pos as f32 / v.len as f32;
            let left = (v.len - v.pos) as f32 / (0.01 * sr);
            let env = ((t - 1.0) * k).exp() * (1.0 - (-t * 40.0).exp()) * left.min(1.0);
            let i = v.ola_pos;
            let (l, r) = (v.ola[0][i], v.ola[1][i]);
            v.ola[0][i] = 0.0;
            v.ola[1][i] = 0.0;
            v.ola_pos = (i + 1) % SW_N;
            out.0 += l * env * v.level;
            out.1 += r * env * v.level;
            v.pos += 1;
            if v.pos >= v.len {
                v.active = false;
            }
        }
        out
    }

    fn start_voice(&mut self, len_s: f32) {
        let sr = self.sr;
        // Spectrum of the captured hit.
        for k in 0..SW_N {
            self.tbuf[k] = if k < SWELL_W { self.cap[k] * self.win[k * SW_N / SW_N] } else { 0.0 };
        }
        let _ = self.fft.process_with_scratch(&mut self.tbuf, &mut self.spec, &mut self.sf);
        let rms = (self.cap.iter().map(|v| v * v).sum::<f32>() / SWELL_W as f32).sqrt();
        if rms < 1e-6 {
            return;
        }
        // Pick a free voice, else steal the quietest one: the one that has risen
        // least (cutting off a swell near its peak was an audible click).
        let vi = (0..VOICES).find(|&i| !self.voices[i].active).unwrap_or_else(|| {
            (0..VOICES).min_by(|&a, &b| {
                let fa = self.voices[a].pos as f32 / self.voices[a].len.max(1) as f32;
                let fb = self.voices[b].pos as f32 / self.voices[b].len.max(1) as f32;
                fa.total_cmp(&fb)
            }).unwrap_or(0)
        });
        if self.voices[vi].active && self.voices[vi].wait <= 0 {
            let t = self.voices[vi].pos as f32 / self.voices[vi].len.max(1) as f32;
            if t > 0.7 {
                self.loud_steals += 1;
            }
        }
        let len = ((len_s.clamp(0.1, SWELL_MAX_S)) * sr) as usize;
        let total = swell_lookahead(sr);
        let v = &mut self.voices[vi];
        // Smoothed magnitudes, a little darker on top (reverb-like).
        for b in 0..SW_BINS {
            let (lo, hi) = (b.saturating_sub(3), (b + 4).min(SW_BINS));
            let m = (lo..hi).map(|j| self.spec[j].norm()).sum::<f32>() / (hi - lo) as f32;
            let f = b as f32 * sr / SW_N as f32;
            v.mag[b] = m / (1.0 + (f / 7000.0).powi(2)).sqrt();
        }
        // Level: the unnormalised inverse FFT of random-phase bins has an RMS of
        // sqrt(2 * sum|mag|^2), and the sine-windowed 50% overlap-add keeps it.
        // (This was divided by sqrt(2N) too many times: the swell came out ~40 dB hot.)
        // Aim the swell's RMS 6 dB under the hit's, so noise peaks land near the hit's level.
        let e: f32 = v.mag.iter().map(|m| m * m).sum::<f32>();
        let synth_rms = (2.0 * e).sqrt();
        v.level = if synth_rms > 1e-12 { 0.5 * rms / synth_rms } else { 0.0 };
        v.active = true;
        v.wait = total as i64 - SWELL_W as i64 - len as i64;
        v.len = len.max(1);
        v.pos = 0;
        v.hop_count = 0;
        v.ola_pos = 0;
        for o in v.ola.iter_mut() {
            o.iter_mut().for_each(|s| *s = 0.0);
        }
        self.hits = self.hits.wrapping_add(1);
    }

    fn synth_frame(&mut self, vi: usize) {
        let two_pi = std::f32::consts::TAU;
        for c in 0..2 {
            for b in 0..SW_BINS {
                self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let j = (self.rng >> 8) as f32 / (1u32 << 24) as f32 * two_pi;
                let v = &mut self.voices[vi];
                v.phase[c][b] = (v.phase[c][b] + j * 0.6).rem_euclid(two_pi);
                self.spec[b] = Complex::from_polar(v.mag[b], v.phase[c][b]);
            }
            self.spec[0].im = 0.0;
            self.spec[SW_BINS - 1].im = 0.0;
            let _ = self.ifft.process_with_scratch(&mut self.spec, &mut self.tbuf, &mut self.si);
            let v = &mut self.voices[vi];
            for k in 0..SW_N {
                let i = (v.ola_pos + k) % SW_N;
                v.ola[c][i] += self.tbuf[k] * self.win[k];
            }
        }
    }
}

#[cfg(test)]
mod swell_tests {
    use super::*;

    /// A hit at t0 -> the swell rises and peaks right where the (delayed) hit
    /// lands, i.e. at input-time t0 + lookahead, and is silent long before.
    #[test]
    fn swell_lands_on_the_hit() {
        let sr = 48000.0;
        let mut s = Swell::new(sr);
        let la = swell_lookahead(sr);
        let t0 = 10000usize;
        let n = t0 + la + 4000;
        let mut out = Vec::with_capacity(n);
        let mut seed = 3u32;
        for i in 0..n {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let x = if i >= t0 && i < t0 + 6000 { noise * (-((i - t0) as f32) / 2000.0).exp() } else { 0.0 };
            out.push(s.tick(x, 1.0, 0.5, 0.5).0);
        }
        assert_eq!(s.hits, 1, "one hit detected");
        let rms = |a: usize, b: usize| (out[a..b].iter().map(|v| v * v).sum::<f32>() / (b - a) as f32).sqrt();
        let land = t0 + la;
        let end_of_swell = rms(land - 4800, land - 100);
        let early = rms(land - (0.9 * sr) as usize, land - (0.7 * sr) as usize);
        let after = rms(land + 1000, land + 3000);
        assert!(end_of_swell > 0.02, "swell is audible near the hit: {end_of_swell}");
        assert!(early < end_of_swell * 0.5, "it rises: early {early} vs end {end_of_swell}");
        assert!(after < end_of_swell * 0.05, "stops at the hit: after {after}");
    }

    #[test]
    fn pitch_shifts_up_an_octave() {
        let sr = 48000.0;
        let mut p = Pitch::new(sr);
        let n = 48000;
        let y: Vec<f32> = (0..n).map(|i| p.tick((std::f32::consts::TAU * 300.0 * i as f32 / sr).sin(), 2.0)).collect();
        let g = |f: f32| {
            let w = std::f32::consts::TAU * f / sr;
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for &v in &y[8000..] {
                let s0 = v + 2.0 * w.cos() * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2).sqrt()
        };
        assert!(g(600.0) > 4.0 * g(300.0), "600 {} vs 300 {}", g(600.0), g(300.0));
    }
}
