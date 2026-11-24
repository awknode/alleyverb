//! The SHIMMER loop (also switched on by RISER) must die away after the input stops.
use alley_verb::algo::{Algo, AlgoParams};
use alley_verb::fx::{Biquad, Pitch};

fn tail_db(fb: f32, ratio: f32, hann: bool, hp: bool) -> Vec<f32> {
    let sr = 48000.0;
    let mut t = Algo::new(sr);
    t.set(&AlgoParams { mode: 0, color: 2, decay: 4.5, size: 1.4, attack: 0.25, early_diff: 0.7, late_diff: 0.7, mod_rate: 0.3, mod_depth: 0.35, bass_mult: 0.6, bass_xover: 500.0, high_shelf_db: -3.0, high_freq: 7000.0 });
    let mut ps = [Pitch::new(sr), Pitch::new(sr)];
    ps.iter_mut().for_each(|p| p.hann = hann);
    let mut lp = [Biquad::default(); 2];
    let mut hpf = [Biquad::default(); 2];
    for i in 0..2 { lp[i].set(false, 8000.0, sr); hpf[i].set(true, 150.0, sr); }
    let mut prev = (0.0f32, 0.0f32);
    let mut seed = 1u32;
    let mut out = vec![];
    let mut pk = 0.0f32;
    for n in 0..(sr as usize * 20) {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = if n < 24000 { ((seed >> 8) as f32 / 16777216.0 - 0.5) * 0.5 } else { 0.0 };
        let il = ((x + prev.0 * fb) * 0.8).tanh();
        let ir = ((x + prev.1 * fb) * 0.8).tanh();
        let mut pl = lp[0].tick(ps[0].tick(il, ratio));
        let mut pr = lp[1].tick(ps[1].tick(ir, ratio));
        if hp { pl = hpf[0].tick(pl); pr = hpf[1].tick(pr); }
        prev = t.tick(pl, pr);
        prev = (prev.0 * t.lvl_061, prev.1 * t.lvl_061);
        pk = pk.max(prev.0.abs());
        if n % 48000 == 47999 { out.push(20.0 * pk.max(1e-9).log10()); pk = 0.0; }
    }
    out
}

/// Fixed loop: dead silent (under -90 dB) within 9 s after the input stops, even at max feedback.
/// (0.7.0: was 8 s; the tank no longer loses highs on every pass, so max-feedback octave-down
/// shimmer reaches -88 dB at 8 s and -102 dB at 9 s, still falling ~12 dB/s. The loop uses the
/// tank at 0.6.1's level, like the plugin does.)
#[test]
fn shimmer_dies_away() {
    for fb in [0.425f32, 0.85] {
        for ratio in [0.5f32, 2.0, 4.0] {
            let v = tail_db(fb, ratio, true, true);
            assert!(v[9] < -90.0, "fb {fb} ratio {ratio}: {:?}", v);
        }
    }
}

#[test]
fn shimmer_tail_report() {
    for (fb, label) in [(0.425f32, "default fb"), (0.85, "max fb")] {
        for (hann, hp) in [(false, false), (true, true)] {
            let v = tail_db(fb, 2.0, hann, hp);
            println!("{label:<10} hann={hann:<5} hp={hp:<5} peak per second: {}", v.iter().map(|d| format!("{d:.0}")).collect::<Vec<_>>().join(" "));
        }
    }
}
