//! Impulse loading: WAV parse -> stereo f32 -> resample to the host rate ->
//! SIZE stretch -> DECAY reshape (grows new tail) -> LENGTH -> REVERSE -> loudness match.

use std::path::Path;

pub struct Ir {
    pub sr: f32,
    pub ch: [Vec<f32>; 2],
}

/// Reads PCM 16/24/32-bit, float 32/64, mono or stereo (extra channels ignored).
pub fn read_wav(path: &Path) -> Result<Ir, String> {
    let b = std::fs::read(path).map_err(|e| format!("can't read file: {e}"))?;
    parse_wav(&b)
}

pub fn parse_wav(b: &[u8]) -> Result<Ir, String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let (mut fmt, mut chans, mut sr, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32_at(i + 4) as usize;
        let body = i + 8;
        let end = (body + len).min(b.len());
        if id == b"fmt " && len >= 16 {
            fmt = u16_at(body);
            chans = u16_at(body + 2) as usize;
            sr = u32_at(body + 4);
            bits = u16_at(body + 14);
            if fmt == 0xFFFE && len >= 26 {
                fmt = u16_at(body + 24); // sub-format GUID starts with the real code
            }
        } else if id == b"data" {
            data = Some(&b[body..end]);
        }
        i = body + len + (len & 1);
    }
    let data = data.ok_or("no audio data")?;
    if chans == 0 || sr == 0 {
        return Err("bad WAV header".into());
    }
    let bps = (bits / 8) as usize;
    let frame = bps * chans;
    if frame == 0 {
        return Err("bad sample size".into());
    }
    let n = data.len() / frame;
    let sample = |k: usize| -> Result<f32, String> {
        let s = &data[k..k + bps];
        Ok(match (fmt, bits) {
            (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            (3, 64) => f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32,
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (1, 24) => (((s[2] as i32) << 24 | (s[1] as i32) << 16 | (s[0] as i32) << 8) >> 8) as f32 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
            _ => return Err(format!("unsupported WAV format {fmt}/{bits}-bit")),
        })
    };
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for f in 0..n {
        let base = f * frame;
        let a = sample(base)?;
        let c = if chans > 1 { sample(base + bps)? } else { a };
        l.push(a);
        r.push(c);
    }
    Ok(Ir { sr: sr as f32, ch: [l, r] })
}

/// Cubic (Catmull-Rom) resampling. Fine for impulses, which are mostly noise-like.
pub fn resample(x: &[f32], from: f32, to: f32) -> Vec<f32> {
    if (from - to).abs() < 0.5 || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((x.len() as f64) / ratio).ceil() as usize;
    let at = |i: isize| -> f32 { if i < 0 || i as usize >= x.len() { 0.0 } else { x[i as usize] } };
    (0..out_len)
        .map(|n| {
            let pos = n as f64 * ratio;
            let i = pos.floor() as isize;
            let t = (pos - i as f64) as f32;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        })
        .collect()
}

/// Loudness target: every impulse comes out at about the same level, so
/// flicking through 326 of them doesn't jump between whisper and blast.
pub const TARGET_ENERGY: f32 = 0.25;

/// Longest tail DECAY may grow to.
pub const MAX_TAIL_S: f32 = 60.0;

/// SIZE: stretch (>1) or squeeze (<1) the impulse in time. A bigger space
/// also comes out a little darker, like a real one.
pub fn stretch(ir: &mut Ir, size: f32) {
    if (size - 1.0).abs() < 0.005 {
        return;
    }
    let sr = ir.sr;
    for ch in ir.ch.iter_mut() {
        if size < 1.0 {
            // Anti-alias before squeezing.
            lowpass(ch, 0.45 * sr * size, sr);
        }
        *ch = resample(ch, sr, sr * size);
        if size > 1.0 {
            lowpass(ch, (20000.0 / size.powf(1.5)).max(5000.0), sr);
        }
    }
}

fn lowpass(x: &mut [f32], freq: f32, sr: f32) {
    let mut f = crate::fx::Biquad::default();
    f.set(false, freq, sr);
    x.iter_mut().for_each(|v| *v = f.tick(*v));
}

/// What the decay analysis found (sample indices).
pub struct Decay {
    pub peak: usize,
    /// Last point well above the noise floor; a grown tail joins here.
    pub join: usize,
    pub rt60: f32,
}

/// Measures RT60 from the 10 ms energy envelope: a straight-line fit of the
/// dB decay between -5 dB and -35 dB under the peak (or down to the noise floor).
/// Measured on the mids (lowpassed at 1.5 kHz), the way reverbs quote decay
/// time, so the faster-dying highs don't skew it.
pub fn analyse(ir: &Ir) -> Decay {
    let sr = ir.sr;
    let mut mids = [ir.ch[0].clone(), ir.ch[1].clone()];
    for ch in mids.iter_mut() {
        lowpass(ch, 1500.0, sr);
    }
    let ir = &Ir { sr, ch: mids };
    let win = ((0.01 * sr) as usize).max(1);
    let n = ir.ch[0].len();
    let nw = n.div_ceil(win).max(1);
    let raw: Vec<f32> = (0..nw)
        .map(|k| {
            let (a, b) = (k * win, ((k + 1) * win).min(n));
            let e: f32 = (a..b).map(|i| 0.5 * (ir.ch[0][i].powi(2) + ir.ch[1][i].powi(2))).sum::<f32>() / (b - a).max(1) as f32;
            10.0 * (e + 1e-20).log10()
        })
        .collect();
    // 3-window smoothing so one quiet window doesn't count as the end.
    let e: Vec<f32> = (0..nw).map(|k| (raw[k.saturating_sub(1)] + raw[k] + raw[(k + 1).min(nw - 1)]) / 3.0).collect();
    let kp = (0..nw).fold(0, |m, k| if e[k] > e[m] { k } else { m });
    let top = e[kp];
    let mut tail: Vec<f32> = e[nw - (nw / 10).max(1)..].to_vec();
    tail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = tail[tail.len() / 2];
    let usable = (floor + 8.0).max(top - 80.0);
    let kend = (kp + 1..nw).find(|&k| e[k] < usable).unwrap_or(nw);
    // Join early (25 dB down), where the original is still clean: lifting a
    // long way amplifies any bend in its decay into an audible swell.
    let kjoin = (kp + 1..kend).find(|&k| e[k] < (usable + 12.0).max(top - 25.0)).unwrap_or(kend).max(kp + 1).min(nw);
    // Least-squares slope in dB per window.
    let lo = (top - 35.0).max(usable + 3.0);
    let pts: Vec<(f32, f32)> = (kp..kend).filter(|&k| e[k] <= top - 5.0 && e[k] >= lo).map(|k| (k as f32, e[k])).collect();
    let win_s = win as f32 / sr;
    let mut rt = if pts.len() >= 3 {
        let m = pts.len() as f32;
        let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        let (mx, my) = (sx / m, sy / m);
        let (num, den) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + (p.0 - mx) * (p.1 - my), a.1 + (p.0 - mx).powi(2)));
        let slope = num / den.max(1e-9);
        if slope < -1e-4 { -60.0 / (slope / win_s) } else { 0.0 }
    } else {
        0.0
    };
    if rt <= 0.0 {
        let drop = (top - usable).max(10.0);
        rt = (kend - kp).max(1) as f32 * win_s * 60.0 / drop;
    }
    Decay { peak: kp * win, join: (kjoin * win).min(n), rt60: rt.clamp(0.02, 60.0) }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// DECAY: reshape the impulse to a new RT60 of `target` seconds.
/// Shorter: the envelope is steepened. Longer: the envelope is lifted until
/// it reaches the noise floor, and from there a new tail is grown from the
/// impulse's own late spectrum, decaying at the new rate (highs faster).
/// Returns (measured RT60, new RT60) in seconds.
pub fn reshape_decay(ir: &mut Ir, target: f32) -> (f32, f32) {
    let d = analyse(ir);
    let rt_new = target.clamp(0.02, 60.0);
    let ratio = rt_new / d.rt60;
    if (ratio - 1.0).abs() < 0.01 {
        return (d.rt60, d.rt60);
    }
    let sr = ir.sr;
    // log10 gain per sample (after the peak) that turns rt60 into rt_new.
    let a = -3.0 * (1.0 / rt_new - 1.0 / d.rt60) / sr;
    let gain = |i: usize| -> f32 { if i <= d.peak { 1.0 } else { 10f32.powf(a * (i - d.peak) as f32) } };
    let end = (d.peak + (rt_new * 80.0 / 60.0 * sr) as usize).min((MAX_TAIL_S * sr) as usize).max(d.peak + 1);
    if ratio < 1.0 {
        for ch in ir.ch.iter_mut() {
            for (i, v) in ch.iter_mut().enumerate() {
                *v *= gain(i);
            }
            let keep = end.min(ch.len());
            ch.truncate(keep);
            let fade = ((0.01 * sr) as usize).min(keep);
            for k in 0..fade {
                ch[keep - fade + k] *= 1.0 - (k as f32 + 1.0) / fade as f32;
            }
        }
        return (d.rt60, rt_new);
    }
    // Longer: lift up to the join point, then grow a new tail.
    let join = d.join.max(d.peak + 1);
    let xf = ((0.05 * sr) as usize).min((join - d.peak) / 2).max(1);
    let ts = join - xf; // the grown tail starts here, crossfading over xf samples
    if end <= join {
        for ch in ir.ch.iter_mut() {
            for (i, v) in ch.iter_mut().enumerate() {
                *v *= gain(i);
            }
        }
        return (d.rt60, rt_new);
    }
    let mut rng = Lcg(0x5eed_1234);
    for (ci, ch) in ir.ch.iter_mut().enumerate() {
        ch.truncate(join);
        for (i, v) in ch.iter_mut().enumerate() {
            *v *= gain(i);
        }
        // Level to continue at: RMS just before the tail starts (lifted).
        let w0 = ts.saturating_sub((0.03 * sr) as usize).max(d.peak);
        let level = if ts > w0 { (ch[w0..ts].iter().map(|v| v * v).sum::<f32>() / (ts - w0) as f32).sqrt() } else { 0.0 };
        let tail = grow_tail(&ch[..ts], ts - w0.min(ts), end - ts, rt_new, sr, level, &mut rng, ci as u32);
        for k in 0..xf {
            let w = (k as f32 + 0.5) / xf as f32;
            let (go, gt) = ((1.0 - w).sqrt(), w.sqrt()); // equal power: both are noise-like
            ch[ts + k] = ch[ts + k] * go + tail[k] * gt;
        }
        ch.extend_from_slice(&tail[xf..]);
    }
    (d.rt60, rt_new)
}

/// Random-phase noise shaped like the end of `x`, overlap-added with a sine
/// window (power complementary), each bin decaying at its own rate.
fn grow_tail(x: &[f32], seg_hint: usize, len: usize, rt: f32, sr: f32, level: f32, rng: &mut Lcg, salt: u32) -> Vec<f32> {
    use realfft::num_complex::Complex;
    const F: usize = 2048;
    const H: usize = F / 2;
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(F);
    let inv = planner.plan_fft_inverse(F);
    // Spectral shape of the late tail.
    let seg = seg_hint.max(F).min(x.len());
    let mut buf = vec![0.0f32; F];
    let mut mag = vec![0.0f32; F / 2 + 1];
    let mut spec = fwd.make_output_vec();
    let mut frames = 0;
    let mut s = x.len() - seg;
    while s + F <= x.len() || frames == 0 {
        for k in 0..F {
            let w = (std::f32::consts::PI * (k as f32 + 0.5) / F as f32).sin();
            buf[k] = x.get(s + k).copied().unwrap_or(0.0) * w;
        }
        let _ = fwd.process(&mut buf, &mut spec);
        for (m, c) in mag.iter_mut().zip(spec.iter()) {
            *m += c.norm_sqr();
        }
        frames += 1;
        if s + F + H > x.len() {
            break;
        }
        s += H;
    }
    // Smooth over +-3 bins, to amplitude.
    let smooth: Vec<f32> = (0..mag.len()).map(|k| {
        let (a, b) = (k.saturating_sub(3), (k + 4).min(mag.len()));
        (mag[a..b].iter().sum::<f32>() / (b - a) as f32).sqrt()
    }).collect();
    let rt_bin: Vec<f32> = (0..mag.len()).map(|k| {
        let f = k as f32 * sr / F as f32;
        rt / (1.0 + 0.8 * ((f - 1500.0).max(0.0) / 12000.0))
    }).collect();
    let mut rng2 = Lcg(rng.0 ^ salt.wrapping_mul(0x9e37_79b9));
    rng.next();
    let total = len + F;
    let mut out = vec![0.0f32; total];
    let mut frame = inv.make_input_vec();
    let mut tbuf = vec![0.0f32; F];
    let mut start = 0usize; // frame k covers out[start..start+F], tail sample 0 = out[H]
    while start < total {
        let t = (start as f32 + H as f32 - H as f32) / sr; // frame centre relative to tail start
        for (k, c) in frame.iter_mut().enumerate() {
            let g = 10f32.powf(-3.0 * t.max(0.0) / rt_bin[k]);
            let ph = rng2.next() * std::f32::consts::TAU;
            *c = Complex::from_polar(smooth[k] * g, ph);
        }
        frame[0].im = 0.0;
        frame[F / 2].im = 0.0;
        let _ = inv.process(&mut frame, &mut tbuf);
        for k in 0..F {
            if start + k >= total {
                break;
            }
            let w = (std::f32::consts::PI * (k as f32 + 0.5) / F as f32).sin();
            out[start + k] += tbuf[k] * w;
        }
        start += H;
    }
    let mut tail = out[H..H + len].to_vec();
    // Match the level: RMS of the first ~40 ms with the broadband decay undone.
    let m = ((0.04 * sr) as usize).clamp(64, len.max(64)).min(len);
    let rms = if m > 0 {
        (tail[..m].iter().enumerate().map(|(i, v)| (v * 10f32.powf(3.0 * i as f32 / (rt * sr))).powi(2)).sum::<f32>() / m as f32).sqrt()
    } else {
        0.0
    };
    let g = if rms > 1e-12 { level / rms } else { 0.0 };
    tail.iter_mut().for_each(|v| *v *= g);
    // Gentle 20 ms fade at the very end.
    let fade = ((0.02 * sr) as usize).min(len);
    for k in 0..fade {
        tail[len - fade + k] *= 1.0 - (k as f32 + 1.0) / fade as f32;
    }
    tail
}

/// SIZE -> DECAY -> LENGTH -> REVERSE, then the loudness match. The loudness
/// gain is taken before DECAY, so the early sound keeps its level and a longer
/// decay just means more tail. Returns (measured RT60, new RT60).
/// `decay`: target RT60 in seconds, or None to keep the impulse's own.
pub fn prepare(ir: &mut Ir, size: f32, decay: Option<f32>, length: f32, reverse: bool) -> (f32, f32) {
    stretch(ir, size);
    let energy: f32 = ir.ch.iter().map(|c| c.iter().map(|v| v * v).sum::<f32>()).sum::<f32>() / 2.0;
    let norm = if energy > 1e-12 { (TARGET_ENERGY / energy).sqrt() } else { 1.0 };
    // Each side gets its own level, so an impulse that's much louder on one side
    // comes out balanced (within 12 dB of the shared gain, so a near-empty side
    // isn't pumped up into noise).
    let norms: Vec<f32> = ir
        .ch
        .iter()
        .map(|c| {
            let e: f32 = c.iter().map(|v| v * v).sum();
            if e > 1e-12 { (TARGET_ENERGY / e).sqrt().clamp(norm / 4.0, norm * 4.0) } else { norm }
        })
        .collect();
    let rts = match decay {
        Some(t) => reshape_decay(ir, t),
        None => {
            let r = analyse(ir).rt60;
            (r, r)
        }
    };
    for (ci, ch) in ir.ch.iter_mut().enumerate() {
        let norm = norms[ci];
        let keep = ((ch.len() as f32 * length.clamp(0.02, 1.0)).ceil() as usize).max(1).min(ch.len());
        ch.truncate(keep);
        if length < 0.999 {
            let fade = (keep / 5).max(1);
            for k in 0..fade {
                let g = 1.0 - (k as f32 + 1.0) / fade as f32;
                ch[keep - fade + k] *= g * g;
            }
        }
        if reverse {
            ch.reverse();
        }
        ch.iter_mut().for_each(|v| *v *= norm);
    }
    rts
}

/// Peak envelope for the display.
pub fn envelope(ir: &Ir, points: usize) -> Vec<f32> {
    let n = ir.ch[0].len().max(1);
    let mut env = vec![0.0f32; points];
    for (k, e) in env.iter_mut().enumerate() {
        let a = k * n / points;
        let b = ((k + 1) * n / points).max(a + 1).min(n);
        for i in a..b {
            *e = e.max(ir.ch[0][i].abs()).max(ir.ch[1][i].abs());
        }
    }
    let m = env.iter().cloned().fold(1e-9, f32::max);
    env.iter_mut().for_each(|e| *e /= m);
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(fmt: u16, bits: u16, chans: u16, sr: u32, samples: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + samples.len() as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&fmt.to_le_bytes());
        v.extend_from_slice(&chans.to_le_bytes());
        v.extend_from_slice(&sr.to_le_bytes());
        v.extend_from_slice(&(sr * chans as u32 * bits as u32 / 8).to_le_bytes());
        v.extend_from_slice(&(chans * bits / 8).to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        v.extend_from_slice(samples);
        v
    }

    #[test]
    fn parses_float_stereo_and_int16_mono() {
        let mut s = Vec::new();
        for x in [0.5f32, -0.25, 1.0, 0.0] {
            s.extend_from_slice(&x.to_le_bytes());
        }
        let ir = parse_wav(&wav_bytes(3, 32, 2, 44100, &s)).unwrap();
        assert_eq!(ir.ch[0], vec![0.5, 1.0]);
        assert_eq!(ir.ch[1], vec![-0.25, 0.0]);
        let s16: Vec<u8> = [16384i16, -32768].iter().flat_map(|v| v.to_le_bytes()).collect();
        let ir = parse_wav(&wav_bytes(1, 16, 1, 48000, &s16)).unwrap();
        assert_eq!(ir.ch[0], vec![0.5, -1.0]);
        assert_eq!(ir.ch[1], ir.ch[0]);
    }

    #[test]
    fn resample_keeps_duration() {
        let x = vec![0.1f32; 44100];
        assert_eq!(resample(&x, 44100.0, 48000.0).len(), 48000);
    }

    /// An impulse 12 dB hotter on the left comes out level on both sides.
    #[test]
    fn prepare_balances_left_and_right() {
        let sr = 48000.0;
        let mk = |a: f32| (0..24000).map(|i| a * (-(i as f32) / 6000.0).exp() * if i % 7 == 0 { 1.0 } else { -0.5 }).collect::<Vec<f32>>();
        let mut ir = Ir { sr, ch: [mk(1.0), mk(0.25)] };
        prepare(&mut ir, 1.0, None, 1.0, false);
        let e: Vec<f32> = ir.ch.iter().map(|c| c.iter().map(|v| v * v).sum::<f32>()).collect();
        let diff = 10.0 * (e[0] / e[1]).log10();
        assert!(diff.abs() < 0.1, "L/R differ by {diff} dB");
    }

    #[test]
    fn prepare_normalises_trims_and_reverses() {
        let mut ir = Ir { sr: 1000.0, ch: [(0..1000).map(|i| 1.0 - i as f32 / 1000.0).collect(), vec![0.3; 1000]] };
        let full: f32 = ir.ch.iter().map(|c| c.iter().map(|v| v * v).sum::<f32>()).sum::<f32>() / 2.0;
        prepare(&mut ir, 1.0, None, 0.5, true);
        assert_eq!(ir.ch[0].len(), 500);
        assert!(ir.ch[0][0].abs() < 1e-6, "reversed: faded end is now the start");
        // Loudness gain comes from the untrimmed impulse, per side (within 12 dB of the shared gain).
        let shared = (TARGET_ENERGY / full).sqrt();
        let side = (TARGET_ENERGY / (0.09 * 1000.0)).sqrt().clamp(shared / 4.0, shared * 4.0);
        assert!((ir.ch[1][250] - 0.3 * side).abs() < 1e-4);
    }

    /// Synthetic room: noise with a known RT60 and a noise floor.
    fn room(rt: f32, secs: f32, sr: f32) -> Ir {
        let mut r = Lcg(99);
        let n = (secs * sr) as usize;
        let mk = |r: &mut Lcg| -> Vec<f32> {
            (0..n).map(|i| (r.next() - 0.5) * 10f32.powf(-3.0 * i as f32 / (rt * sr)) + (r.next() - 0.5) * 1e-4).collect()
        };
        Ir { sr, ch: [mk(&mut r), mk(&mut r)] }
    }

    #[test]
    fn measures_rt60() {
        let d = analyse(&room(1.5, 4.0, 44100.0));
        assert!((d.rt60 - 1.5).abs() < 0.15, "rt60 {}", d.rt60);
    }

    #[test]
    fn decay_shorter_and_longer_hit_the_target() {
        let sr = 44100.0;
        let mut ir = room(1.0, 3.0, sr);
        let (_, want) = reshape_decay(&mut ir, 0.5);
        let got = analyse(&ir).rt60;
        assert!((got - want).abs() / want < 0.15, "shorter: want {want} got {got}");

        let mut ir = room(1.0, 3.0, sr);
        let early: f32 = ir.ch[0][..441].iter().map(|v| v * v).sum();
        let (_, want) = reshape_decay(&mut ir, 4.0);
        assert!(ir.ch[0].len() as f32 / sr > 4.5, "tail grew: {} s", ir.ch[0].len() as f32 / sr);
        let got = analyse(&ir).rt60;
        assert!((got - want).abs() / want < 0.2, "longer: want {want} got {got}");
        let early2: f32 = ir.ch[0][..441].iter().map(|v| v * v).sum();
        assert!((early2 / early - 1.0).abs() < 0.15, "early part kept its level: {}", early2 / early);
        // No jump where the grown tail joins: 10 ms windows never step up by >3 dB.
        let w = 441;
        let e: Vec<f32> = ir.ch[0].chunks(w).map(|c| 10.0 * (c.iter().map(|v| v * v).sum::<f32>() / w as f32 + 1e-20).log10()).collect();
        for k in 2..e.len() - 2 {
            let a = (e[k - 2] + e[k - 1]) / 2.0;
            let b = (e[k + 1] + e[k + 2]) / 2.0;
            assert!(b - a < 3.0, "jump of {} dB at {} s", b - a, k as f32 * 0.01);
        }
    }

    #[test]
    fn size_stretches() {
        let mut ir = room(1.0, 2.0, 44100.0);
        stretch(&mut ir, 2.0);
        assert!((ir.ch[0].len() as f32 / 44100.0 - 4.0).abs() < 0.01);
        assert!((analyse(&ir).rt60 - 2.0).abs() < 0.25);
    }
}
