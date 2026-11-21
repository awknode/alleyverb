use alley_verb::wav;
#[test]
fn where_is_the_jump() {
    let Ok(f) = std::env::var("F") else { return };
    let d: f32 = std::env::var("D").unwrap().parse().unwrap();
    let sr = 48000.0;
    let mut ir = wav::read_wav(std::path::Path::new(&f)).unwrap();
    for ch in ir.ch.iter_mut() { *ch = wav::resample(ch, ir.sr, sr); }
    ir.sr = sr;
    let a = wav::analyse(&ir);
    println!("peak {:.3}s join {:.3}s rt {:.2} len {:.2}s", a.peak as f32 / sr, a.join as f32 / sr, a.rt60, ir.ch[0].len() as f32 / sr);
    wav::reshape_decay(&mut ir, d);
    let w = 960;
    let e: Vec<f32> = ir.ch[0].chunks(w).map(|c| 10.0 * (c.iter().map(|v| v * v).sum::<f32>() / w as f32 + 1e-20).log10()).collect();
    for k in 2..e.len() - 2 {
        let j = (e[k + 1] + e[k + 2]) / 2.0 - (e[k - 2] + e[k - 1]) / 2.0;
        if j > 3.0 { println!("  +{j:.1} dB at {:.2}s (level {:.1})", k as f32 * 0.02, e[k]); }
    }
}
