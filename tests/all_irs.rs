// IRDIR=<folder> cargo test --release --test all_irs -- --nocapture
use alley_verb::wav;
use std::time::Instant;

fn wavs(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(d).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() { wavs(&p, out) } else if p.extension().map(|x| x == "wav").unwrap_or(false) { out.push(p) }
    }
}

/// Biggest step up (dB) between neighbouring 20 ms windows after the peak.
fn worst_jump(ir: &wav::Ir) -> f32 {
    let w = (0.02 * ir.sr) as usize;
    let e: Vec<f32> = ir.ch[0].chunks(w).zip(ir.ch[1].chunks(w))
        .map(|(a, b)| 10.0 * ((a.iter().chain(b).map(|v| v * v).sum::<f32>()) / (2 * w) as f32 + 1e-20).log10()).collect();
    let kp = (0..e.len()).fold(0, |m, k| if e[k] > e[m] { k } else { m });
    let top = e[kp];
    let mut worst = 0.0f32;
    for k in kp + 3..e.len().saturating_sub(2) {
        if e[k + 1] < top - 70.0 { break; }
        let a = (e[k - 2] + e[k - 1]) / 2.0;
        let b = (e[k + 1] + e[k + 2]) / 2.0;
        worst = worst.max(b - a);
    }
    worst
}

#[test]
fn every_impulse_reshapes_cleanly() {
    let Ok(dir) = std::env::var("IRDIR") else { return };
    let mut files = Vec::new();
    wavs(std::path::Path::new(&dir), &mut files);
    files.sort();
    let sr = 48000.0;
    let mut slowest = (0.0f64, String::new());
    let mut bad = Vec::new();
    let mut errs = Vec::new();
    for f in &files {
        let mut base = wav::read_wav(f).unwrap();
        for ch in base.ch.iter_mut() { *ch = wav::resample(ch, base.sr, sr); }
        base.sr = sr;
        let orig_jump = worst_jump(&base);
        for (size, decay) in [(1.0f32, Some(0.5f32)), (1.0, Some(2.0)), (1.0, Some(8.0)), (1.0, Some(30.0)), (2.0, None)] {
            let mut ir = wav::Ir { sr, ch: base.ch.clone() };
            let t = Instant::now();
            let (_, want) = wav::prepare(&mut ir, size, decay, 1.0, false);
            let dt = t.elapsed().as_secs_f64();
            let _ = alley_verb::conv::Engine::new(&ir.ch);
            let total = t.elapsed().as_secs_f64();
            if total > slowest.0 { slowest = (total, format!("{} s{size} d{decay:?} prep {dt:.3}", f.file_name().unwrap().to_string_lossy())); }
            let j = worst_jump(&ir);
            if j > orig_jump.max(0.0) + 4.0 { bad.push(format!("{:>5.1} dB jump (orig {:.1}) {} size {size} decay {decay:?}", j, orig_jump, f.display())); }
            if ir.ch.iter().any(|c| c.iter().any(|v| !v.is_finite())) { bad.push(format!("NaN {}", f.display())); }
            if decay.is_some() && size == 1.0 && want < 25.0 {
                let got = wav::analyse(&ir).rt60;
                errs.push(((got - want).abs() / want, format!("{} d{decay:?}: want {want:.2} got {got:.2}", f.file_name().unwrap().to_string_lossy())));
            }
        }
    }
    errs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = errs.len();
    println!("files {}  slowest rebuild {:.3}s ({})", files.len(), slowest.0, slowest.1);
    println!("decay accuracy: median err {:.1}%  90th {:.1}%  worst {:.1}%", errs[n / 2].0 * 100.0, errs[n * 9 / 10].0 * 100.0, errs[n - 1].0 * 100.0);
    for e in errs.iter().rev().take(6) { println!("  {}", e.1); }
    println!("jumps/NaN flagged: {}", bad.len());
    for b in bad.iter().take(15) { println!("  {b}"); }
}
