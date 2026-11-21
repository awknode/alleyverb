// Real-time check on a real impulse: run with IR=<wav> cargo test --release --test bench -- --nocapture
use std::time::Instant;
#[test]
fn realtime_load_on_longest_ir() {
    let Ok(path) = std::env::var("IR") else { return };
    let mut ir = alley_verb::wav::read_wav(std::path::Path::new(&path)).unwrap();
    let sr = 48000.0;
    for ch in ir.ch.iter_mut() { *ch = alley_verb::wav::resample(ch, ir.sr, sr); }
    ir.sr = sr;
    alley_verb::wav::prepare(&mut ir, 1.0, std::env::var("DECAY").ok().and_then(|v| v.parse().ok()), 1.0, false);
    let t0 = Instant::now();
    let mut e = alley_verb::conv::Engine::new(&ir.ch);
    let build = t0.elapsed();
    let secs = 10.0;
    let n = (sr * secs) as usize;
    let t1 = Instant::now();
    let mut acc = 0.0f32;
    for i in 0..n { let x = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5; let (l, r) = e.tick(x, x); acc += l + r; }
    let run = t1.elapsed().as_secs_f64();
    println!("IR {:.1}s: build {:?}, {secs}s of audio in {:.3}s = {:.1}% of one core (acc {acc})", ir.ch[0].len() as f32 / sr, build, run, run / secs as f64 * 100.0);
}

#[test]
fn algo_cpu() {
    use alley_verb::algo::{Algo, AlgoParams};
    let sr = 48000.0;
    for mode in [0usize, 5, 8, 9] {
        let mut a = Algo::new(sr);
        let p = AlgoParams { mode, color: 0, decay: 30.0, size: 2.0, attack: 0.5, early_diff: 1.0, late_diff: 1.0, mod_rate: 2.0, mod_depth: 1.0, bass_mult: 2.0, bass_xover: 300.0, high_shelf_db: -12.0, high_freq: 5000.0 };
        let n = (sr * 10.0) as usize;
        let t = std::time::Instant::now();
        let mut acc = 0.0f32;
        for i in 0..n {
            if i % 256 == 0 { a.set(&p); }
            let x = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5;
            let (l, r) = a.tick(x, -x);
            acc += l + r;
        }
        println!("ALGO mode {mode}: {:.2}% of one core (acc {acc:.1})", t.elapsed().as_secs_f64() / 10.0 * 100.0);
    }
}
