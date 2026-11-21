//! Uniform-partitioned FFT convolution (overlap-save), stereo (L*irL, R*irR).
//! Latency = BLOCK samples, reported to the host so FL lines it up.
//! A 20 s impulse is ~860 partitions; each block does one FFT, one inverse FFT
//! and a multiply-add per partition, so even the huge halls stay cheap.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const BLOCK: usize = 1024;
const N: usize = BLOCK * 2;
const BINS: usize = BLOCK + 1;

pub struct Engine {
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Impulse spectrum per partition, per channel (already scaled by 1/N).
    h: [Vec<Vec<Complex<f32>>>; 2],
    /// Frequency-domain delay line of past input blocks.
    fdl: [Vec<Vec<Complex<f32>>>; 2],
    pos: usize,
    prev: [Vec<f32>; 2],
    cur: [Vec<f32>; 2],
    out: [Vec<f32>; 2],
    idx: usize,
    tbuf: Vec<f32>,
    acc: Vec<Complex<f32>>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
}

impl Engine {
    /// Allocates everything (call off the audio thread).
    pub fn new(ir: &[Vec<f32>; 2]) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N);
        let ifft = planner.plan_fft_inverse(N);
        let len = ir[0].len().max(ir[1].len()).max(1);
        let parts = len.div_ceil(BLOCK);
        let mut tbuf = vec![0.0f32; N];
        let mut spec = vec![Complex::new(0.0, 0.0); BINS];
        let mut scratch_f = fft.make_scratch_vec();
        let scale = 1.0 / N as f32;
        let mut h: [Vec<Vec<Complex<f32>>>; 2] = [Vec::with_capacity(parts), Vec::with_capacity(parts)];
        for ch in 0..2 {
            for p in 0..parts {
                tbuf.iter_mut().for_each(|v| *v = 0.0);
                for k in 0..BLOCK {
                    tbuf[k] = ir[ch].get(p * BLOCK + k).copied().unwrap_or(0.0) * scale;
                }
                let _ = fft.process_with_scratch(&mut tbuf, &mut spec, &mut scratch_f);
                h[ch].push(spec.clone());
            }
        }
        let zeros = || vec![vec![Complex::new(0.0f32, 0.0); BINS]; parts];
        Self {
            scratch_i: ifft.make_scratch_vec(),
            fft,
            ifft,
            h,
            fdl: [zeros(), zeros()],
            pos: 0,
            prev: [vec![0.0; BLOCK], vec![0.0; BLOCK]],
            cur: [vec![0.0; BLOCK], vec![0.0; BLOCK]],
            out: [vec![0.0; BLOCK], vec![0.0; BLOCK]],
            idx: 0,
            tbuf,
            acc: vec![Complex::new(0.0, 0.0); BINS],
            scratch_f,
        }
    }

    /// Silence all history (no allocation): used when switching back from ALGO
    /// so stale audio doesn't ring out.
    pub fn clear(&mut self) {
        for ch in 0..2 {
            for b in self.fdl[ch].iter_mut() {
                b.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
            }
            self.prev[ch].iter_mut().for_each(|v| *v = 0.0);
            self.cur[ch].iter_mut().for_each(|v| *v = 0.0);
            self.out[ch].iter_mut().for_each(|v| *v = 0.0);
        }
        self.idx = 0;
    }

    #[inline]
    pub fn tick(&mut self, l: f32, r: f32) -> (f32, f32) {
        let o = (self.out[0][self.idx], self.out[1][self.idx]);
        self.cur[0][self.idx] = l;
        self.cur[1][self.idx] = r;
        self.idx += 1;
        if self.idx == BLOCK {
            self.idx = 0;
            self.block();
        }
        o
    }

    fn block(&mut self) {
        let parts = self.h[0].len();
        for ch in 0..2 {
            self.tbuf[..BLOCK].copy_from_slice(&self.prev[ch]);
            self.tbuf[BLOCK..].copy_from_slice(&self.cur[ch]);
            let _ = self.fft.process_with_scratch(&mut self.tbuf, &mut self.fdl[ch][self.pos], &mut self.scratch_f);
            self.acc.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
            for p in 0..parts {
                let x = &self.fdl[ch][(self.pos + parts - p) % parts];
                let hh = &self.h[ch][p];
                for ((a, xv), hv) in self.acc.iter_mut().zip(x.iter()).zip(hh.iter()) {
                    *a += xv * hv;
                }
            }
            self.acc[0].im = 0.0;
            self.acc[BLOCK].im = 0.0;
            let _ = self.ifft.process_with_scratch(&mut self.acc, &mut self.tbuf, &mut self.scratch_i);
            self.out[ch].copy_from_slice(&self.tbuf[BLOCK..]);
            std::mem::swap(&mut self.prev[ch], &mut self.cur[ch]);
        }
        self.pos = (self.pos + 1) % parts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must equal plain (direct) convolution, delayed by BLOCK samples.
    #[test]
    fn matches_direct_convolution() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let ir_len = BLOCK * 3 + 77;
        let ir = [(0..ir_len).map(|_| rnd()).collect::<Vec<f32>>(), (0..ir_len).map(|_| rnd()).collect()];
        let x: Vec<f32> = (0..BLOCK * 8).map(|_| rnd()).collect();
        let mut e = Engine::new(&ir);
        let out: Vec<(f32, f32)> = x.iter().map(|&v| e.tick(v, -v)).collect();
        let mut worst = 0.0f32;
        for n in BLOCK..x.len() {
            let m = n - BLOCK;
            let (mut yl, mut yr) = (0.0f32, 0.0f32);
            for k in 0..ir_len.min(m + 1) {
                yl += ir[0][k] * x[m - k];
                yr += ir[1][k] * -x[m - k];
            }
            worst = worst.max((out[n].0 - yl).abs()).max((out[n].1 - yr).abs());
        }
        assert!(worst < 1e-3, "max error {worst}");
    }
}
