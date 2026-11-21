//! Measures how loud the SWELL gets compared to the input that triggers it.
use alley_verb::fx::{swell_lookahead, Swell};

fn run(name: &str, sig: &dyn Fn(usize) -> f32, secs: f32) -> (f32, f32) {
    // (returns input peak, swell peak)
    let sr = 48000.0;
    let mut s = Swell::new(sr);
    let n = (secs * sr) as usize + swell_lookahead(sr);
    let (mut pin, mut pout) = (0.0f32, 0.0f32);
    for i in 0..n {
        let x = sig(i);
        pin = pin.max(x.abs());
        let (l, r) = s.tick(x, 1.0, 0.5, 0.5);
        pout = pout.max(l.abs()).max(r.abs());
    }
    println!("{name:<28} input peak {:+6.1} dB   swell peak {:+6.1} dB  (hits {})", 20.0 * pin.log10(), 20.0 * pout.max(1e-9).log10(), s.hits);
    (pin, pout)
}

#[test]
fn swell_level_report() {
    let mut seed = 7u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let nz: Vec<f32> = (0..48000 * 8).map(|_| noise()).collect();
    let beat = 24000usize; // 120 BPM
    let mut worst = -99.0f32;
    let mut chk = |r: (f32, f32)| worst = worst.max(20.0 * (r.1 / r.0).log10());
    chk(run("snare, 0 dBFS, every beat", &|i| { let t = i % beat; if i < 48000 * 6 { nz[i] * (-(t as f32) / 3000.0).exp() } else { 0.0 } }, 6.0));
    chk(run("kick sine 60Hz 0 dBFS", &|i| { let t = i % beat; if i < 48000 * 6 { (6.283 * 60.0 * t as f32 / 48000.0).sin() * (-(t as f32) / 8000.0).exp() } else { 0.0 } }, 6.0));
    chk(run("vocal-ish bursts 0 dBFS", &|i| { let t = i % (beat / 2); if i < 48000 * 6 && t < 9000 { (6.283 * 220.0 * i as f32 / 48000.0).sin() * 0.9 + 0.1 * nz[i] } else { 0.0 } }, 6.0));
    // The swell (before its own level knob) never peaks above the hit that made it.
    assert!(worst <= 0.0, "swell peaks {worst:+.1} dB over its hit");
}

/// Hits closer together than the swell length never cut a swell off near its
/// peak (that was a click). Each swell also fades out over 10 ms as its hit lands.
fn loud_steals(spacing_s: f32, len_s: f32) -> (u32, u32) {
    let sr = 48000.0;
    let mut s = Swell::new(sr);
    let mut seed = 11u32;
    let n = (10.0 * sr) as usize + swell_lookahead(sr);
    let step = (spacing_s * sr) as usize;
    for i in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let nz = (seed >> 8) as f32 / 16777216.0 * 2.0 - 1.0;
        let t = i % step;
        let x = if i < (8.0 * sr) as usize { nz * (-(t as f32) / 1500.0).exp() } else { 0.0 };
        s.tick(x, len_s, 0.6, 0.5);
    }
    (s.hits, s.loud_steals)
}

#[test]
fn swell_has_no_clicks() {
    for len in [1.0f32, 2.0] {
        for sp in [0.15f32, 0.25, 0.4, 0.5, 0.7] {
            let (h, st) = loud_steals(sp, len);
            println!("swell {len} s, hit every {sp} s: hits {h}, loud steals {st}");
            assert!(h > 5);
            assert_eq!(st, 0, "a loud swell was cut off (swell {len} s, hits every {sp} s)");
        }
    }
}
