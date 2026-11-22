//! AlleyVerb — awk studios. Convolution reverb with an IR browser,
//! FREEZE, and a live "where the reverb hits" frequency view.
//!
//! wet chain: convolution (old/new crossfade on rebuild) -> pre-delay ->
//!            [FREEZE crossfade] -> low cut -> high cut -> width -> DUCK -> wet level
//! The dry signal is delayed by the convolution latency so both line up.
//! PARALLEL: the original passes through bit-identical (DRY and OUTPUT don't
//! touch it); WET/OUTPUT/cuts/DUCK only shape the reverb copy, like a send return.
//! Header MIX + power (global): blend the untouched input (delayed to match the
//! reported latency) with everything above; bypass = the aligned untouched input.

use crossbeam::queue::ArrayQueue;
use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub mod algo;
mod awk_ui;
pub mod conv;
mod editor;
#[cfg(test)]
mod shot;
mod presets;
pub mod fx;
#[cfg(test)]
#[allow(dead_code)]
mod style;
pub mod wav;

use conv::{Engine, BLOCK};
use algo::{Algo, AlgoParams, COLOR_NAMES, MODE_NAMES};
use awk_ui::{BypassRamp, DryDelay, Meters, PeakFollower};
use fx::{swell_lookahead, Biquad, DuckBand, FreezeTank, Pitch, Swell};

pub const MAX_PREDELAY_MS: f32 = 1000.0;
pub const PRE_DIVS: [&str; 9] = ["OFF", "1/64", "1/32", "1/16", "1/16D", "1/8", "1/8D", "1/4", "1/4D"];
const PRE_BEATS: [f64; 9] = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5];
pub const LOW_CUT_OFF: f32 = 20.5;
pub const HIGH_CUT_OFF: f32 = 19900.0;
pub const SPECTRUM_QUEUE: usize = 1 << 15;
pub const DUCK_SOURCES: [&str; 3] = ["SELF", "SIDECHAIN", "BOTH"];

fn documents() -> PathBuf {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    PathBuf::from(home).join("Documents")
}

/// Default impulse folder: %USERPROFILE%\Documents\Awknode\Impulses
pub fn ir_root() -> PathBuf {
    documents().join("Awknode").join("Impulses")
}

/// Every impulse library the browser shows: (label prefix, folder).
pub fn ir_roots() -> Vec<(String, PathBuf)> {
    let mut v = vec![(String::new(), ir_root())];
    let triaz = documents().join("Triaz").join("IR Files");
    if triaz.is_dir() {
        v.push(("TRIAZ".to_string(), triaz));
    }
    v
}

/// What a fresh AlleyVerb loads: Triaz "Black Hole" if it's there, else a hall.
pub fn default_ir() -> Option<PathBuf> {
    let candidates = [
        documents().join("Triaz").join("IR Files").join("Effects").join("Black Hole.wav"),
        ir_root().join("CLA Epic").join("CLA Hall 001.wav"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Favourites live in a file so they're shared by every AlleyVerb in every project.
pub fn favorites_file() -> PathBuf {
    ir_root().join("favorites.txt")
}

pub struct IrInfo {
    pub name: String,
    pub secs: f32,
    /// Measured RT60 of the impulse (after SIZE) and the RT60 after DECAY.
    pub rt_orig: f32,
    pub rt_new: f32,
    pub env: Vec<f32>,
    pub error: Option<String>,
}

/// What an engine was built from; a rebuild runs when the params drift from it.
#[derive(Clone, Default, PartialEq)]
pub struct BuildKey {
    pub path: String,
    pub reverse: bool,
    pub length: f32,
    pub size: f32,
    pub decay: Option<f32>,
}

impl BuildKey {
    fn from_params(p: &AlleyParams, path: String) -> Self {
        let decay = if p.decay_orig.value() { None } else { Some(p.decay.value()) };
        Self { path, reverse: p.reverse.value(), length: p.length.value(), size: p.size.value(), decay }
    }
    fn close_to(&self, o: &BuildKey) -> bool {
        let near = |a: f32, b: f32| (a - b).abs() <= 0.004 * a.abs().max(b.abs()).max(1.0);
        self.path == o.path && self.reverse == o.reverse && near(self.length, o.length) && near(self.size, o.size)
            && match (self.decay, o.decay) {
                (Some(a), Some(b)) => (a - b).abs() <= 0.004 * a.max(b),
                (None, None) => true,
                _ => false,
            }
    }
}

pub struct Shared {
    /// A freshly built engine waiting to be swapped in.
    pub slot: Mutex<Option<Box<Engine>>>,
    /// Retired engines, freed by the background thread (never on the audio thread).
    pub garbage: ArrayQueue<Box<Engine>>,
    pub info: Mutex<IrInfo>,
    pub loading: AtomicBool,
    pub sr: AtomicU32,
    /// What the current/pending engine was built with.
    pub built: Mutex<BuildKey>,
    /// Current duck gain reduction in dB (<= 0), for the display.
    pub duck_db: AtomicU32,
    pub spec_wet: ArrayQueue<f32>,
    pub spec_dry: ArrayQueue<f32>,
    /// IN / OUT peak meters for the editor.
    pub meters: Meters,
}

#[derive(Params)]
pub struct AlleyParams {
    #[persist = "editor-state-v3"]
    pub editor_state: Arc<EguiState>,
    #[persist = "ir-path"]
    pub ir_path: Arc<RwLock<String>>,

    #[id = "dry"]
    pub dry: FloatParam,
    #[id = "wet"]
    pub wet: FloatParam,
    #[id = "pre"]
    pub predelay: FloatParam,
    #[id = "lcut"]
    pub low_cut: FloatParam,
    #[id = "hcut"]
    pub high_cut: FloatParam,
    #[id = "len"]
    pub length: FloatParam,
    #[id = "width"]
    pub width: FloatParam,
    #[id = "out"]
    pub output: FloatParam,
    #[id = "size"]
    pub size: FloatParam,
    #[id = "decay"]
    pub decay: FloatParam,
    #[id = "decorig"]
    pub decay_orig: BoolParam,
    #[id = "duck"]
    pub duck: FloatParam,
    /// 0 SELF (the track itself), 1 SIDECHAIN, 2 BOTH (whichever is louder).
    #[id = "ducksrc"]
    pub duck_src: IntParam,
    #[id = "ducklo"]
    pub duck_lo: FloatParam,
    #[id = "duckhi"]
    pub duck_hi: FloatParam,
    /// 0 IMPULSE (convolution), 1 ALGO.
    #[id = "engine"]
    pub engine: IntParam,
    #[id = "amode"]
    pub algo_mode: IntParam,
    #[id = "acolor"]
    pub algo_color: IntParam,
    #[id = "attack"]
    pub attack: FloatParam,
    #[id = "ediff"]
    pub early_diff: FloatParam,
    #[id = "ldiff"]
    pub late_diff: FloatParam,
    #[id = "mrate"]
    pub mod_rate: FloatParam,
    #[id = "mdepth"]
    pub mod_depth: FloatParam,
    #[id = "bmult"]
    pub bass_mult: FloatParam,
    #[id = "bxover"]
    pub bass_xover: FloatParam,
    #[id = "hshelf"]
    pub high_shelf: FloatParam,
    #[id = "hfreq"]
    pub high_freq: FloatParam,
    #[id = "shim"]
    pub shimmer: FloatParam,
    #[id = "shpitch"]
    pub shim_pitch: FloatParam,
    #[id = "shfb"]
    pub shim_fb: FloatParam,
    #[id = "shtone"]
    pub shim_tone: FloatParam,
    /// One-knob build: more reverb, wider, thinner, shimmer creeping in.
    #[id = "riser"]
    pub riser: FloatParam,
    #[id = "swell"]
    pub swell: BoolParam,
    #[id = "swlen"]
    pub swell_len: FloatParam,
    #[id = "swamt"]
    pub swell_amt: FloatParam,
    #[id = "swsens"]
    pub swell_sens: FloatParam,
    #[id = "swcurve"]
    pub swell_curve: FloatParam,
    #[id = "inlo"]
    pub in_lo: FloatParam,
    #[id = "inhi"]
    pub in_hi: FloatParam,
    #[id = "prediv"]
    pub pre_div: IntParam,
    #[id = "rev"]
    pub reverse: BoolParam,
    #[id = "freeze"]
    pub freeze: BoolParam,
    #[id = "send"]
    pub send: BoolParam,
    /// Header MIX: the whole plugin's dry/wet (100% = as before).
    #[id = "gmix"]
    pub mix: FloatParam,
    /// Header power button (also the host's bypass).
    #[id = "bypass"]
    pub bypass: BoolParam,
    /// SAFE: output limiter at -0.1 dBFS (footer switch, on by default).
    #[id = "safe"]
    pub safe: BoolParam,
}

fn db(name: &str, default: f32, min: f32, max: f32) -> FloatParam {
    FloatParam::new(
        name,
        util::db_to_gain(default),
        FloatRange::Skewed { min: util::db_to_gain(min), max: util::db_to_gain(max), factor: FloatRange::gain_skew_factor(min, max) },
    )
    .with_smoother(SmoothingStyle::Logarithmic(30.0))
    .with_unit(" dB")
    .with_value_to_string(Arc::new(move |v| {
        if v <= util::db_to_gain(min) * 1.001 { "OFF".to_string() } else { format!("{:.1}", util::gain_to_db(v)) }
    }))
    .with_string_to_value(formatters::s2v_f32_gain_to_db())
}

fn pct(name: &str, default: f32) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min: 0.0, max: 1.0 })
        .with_unit("%")
        .with_value_to_string(formatters::v2s_f32_percentage(0))
        .with_string_to_value(formatters::s2v_f32_percentage())
}

fn hz(name: &str, default: f32, min: f32, max: f32, off_low: bool) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Skewed { min, max, factor: FloatRange::skew_factor(-2.0) })
        .with_smoother(SmoothingStyle::Logarithmic(30.0))
        .with_value_to_string(Arc::new(move |v| {
            let off = if off_low { v <= LOW_CUT_OFF } else { v >= HIGH_CUT_OFF };
            if off {
                "OFF".to_string()
            } else if v >= 1000.0 {
                format!("{:.1}k", v / 1000.0)
            } else {
                format!("{v:.0} Hz")
            }
        }))
}

impl Default for AlleyParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(editor::WIDTH, editor::HEIGHT),
            ir_path: Arc::new(RwLock::new(String::new())),
            dry: db("Dry", 0.0, -60.0, 6.0),
            wet: db("Wet", -6.0, -60.0, 6.0),
            predelay: FloatParam::new("Pre-delay", 0.0, FloatRange::Skewed { min: 0.0, max: MAX_PREDELAY_MS, factor: FloatRange::skew_factor(-1.0) })
                .with_unit(" ms")
                .with_value_to_string(Arc::new(|v| format!("{v:.0}"))),
            low_cut: hz("Low Cut", 300.0, 20.0, 2000.0, true),
            high_cut: hz("High Cut", 15000.0, 1000.0, 20000.0, false),
            length: FloatParam::new("Length", 1.0, FloatRange::Linear { min: 0.05, max: 1.0 })
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            width: FloatParam::new("Width", 1.0, FloatRange::Linear { min: 0.0, max: 2.0 })
                .with_smoother(SmoothingStyle::Linear(30.0))
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            output: db("Output", 0.0, -24.0, 12.0),
            size: FloatParam::new("Size", 1.0, FloatRange::Skewed { min: 0.5, max: 2.0, factor: FloatRange::skew_factor(-1.0) })
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            // Reverb time (RT60) in seconds. Past the impulse's own tail a new one is grown.
            decay: FloatParam::new("Decay", 4.0, FloatRange::Skewed { min: 0.1, max: 60.0, factor: FloatRange::skew_factor(-2.2) })
                .with_unit(" s")
                .with_value_to_string(Arc::new(|v| if v < 10.0 { format!("{v:.2}") } else { format!("{v:.1}") }))
                .with_string_to_value(Arc::new(|s| s.trim().trim_end_matches(['s', 'S', ' ']).parse().ok())),
            decay_orig: BoolParam::new("Decay Original", false),
            duck: FloatParam::new("Duck", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            duck_src: IntParam::new("Duck Source", 0, IntRange::Linear { min: 0, max: 2 })
                .with_value_to_string(Arc::new(|v| DUCK_SOURCES[v.clamp(0, 2) as usize].to_string())),
            duck_lo: hz("Duck Low", 20.0, 20.0, 20000.0, true),
            duck_hi: hz("Duck High", 20000.0, 20.0, 20000.0, false),
            engine: IntParam::new("Engine", 0, IntRange::Linear { min: 0, max: 1 })
                .with_value_to_string(Arc::new(|v| if v == 0 { "IMPULSE".into() } else { "ALGO".into() })),
            algo_mode: IntParam::new("Algo Mode", 0, IntRange::Linear { min: 0, max: 11 })
                .with_value_to_string(Arc::new(|v| MODE_NAMES[v.clamp(0, 11) as usize].to_string())),
            algo_color: IntParam::new("Color", 2, IntRange::Linear { min: 0, max: 2 })
                .with_value_to_string(Arc::new(|v| COLOR_NAMES[v.clamp(0, 2) as usize].to_string())),
            attack: pct("Attack", 0.3),
            early_diff: pct("Early Diffusion", 0.6),
            late_diff: pct("Late Diffusion", 0.6),
            mod_rate: FloatParam::new("Mod Rate", 0.5, FloatRange::Skewed { min: 0.05, max: 10.0, factor: FloatRange::skew_factor(-2.0) })
                .with_unit(" Hz")
                .with_value_to_string(Arc::new(|v| format!("{v:.2}"))),
            mod_depth: pct("Mod Depth", 0.3),
            bass_mult: FloatParam::new("Bass Mult", 1.0, FloatRange::Skewed { min: 0.25, max: 4.0, factor: FloatRange::skew_factor(-1.0) })
                .with_value_to_string(Arc::new(|v| format!("x{v:.2}")))
                .with_string_to_value(Arc::new(|s| s.trim().trim_start_matches(['x', 'X']).parse().ok())),
            bass_xover: hz("Bass Xover", 400.0, 50.0, 4000.0, false),
            high_shelf: FloatParam::new("High Shelf", -6.0, FloatRange::Linear { min: -24.0, max: 0.0 })
                .with_unit(" dB")
                .with_value_to_string(Arc::new(|v| format!("{v:.1}"))),
            high_freq: hz("High Freq", 6000.0, 1000.0, 20000.0, false),
            shimmer: pct("Shimmer", 0.0),
            shim_pitch: FloatParam::new("Shimmer Pitch", 12.0, FloatRange::Linear { min: -12.0, max: 24.0 })
                .with_step_size(1.0)
                .with_value_to_string(Arc::new(|v| format!("{:+.0} st", v)))
                .with_string_to_value(Arc::new(|s| s.trim().trim_end_matches("st").trim().parse().ok())),
            shim_fb: pct("Shimmer Feedback", 0.5),
            shim_tone: hz("Shimmer Tone", 8000.0, 1000.0, 20000.0, false),
            riser: pct("Riser", 0.0),
            swell: BoolParam::new("Swell", false),
            swell_len: FloatParam::new("Swell Length", 1.0, FloatRange::Linear { min: 0.1, max: fx::SWELL_MAX_S })
                .with_unit(" s")
                .with_value_to_string(Arc::new(|v| format!("{v:.2}"))),
            swell_amt: db("Swell Level", -6.0, -40.0, 6.0),
            swell_sens: pct("Swell Sensitivity", 0.5),
            swell_curve: pct("Swell Curve", 0.5),
            in_lo: hz("Input Low Cut", 20.0, 20.0, 2000.0, true),
            in_hi: hz("Input High Cut", 20000.0, 1000.0, 20000.0, false),
            pre_div: IntParam::new("Pre-delay Sync", 0, IntRange::Linear { min: 0, max: 8 })
                .with_value_to_string(Arc::new(|v| PRE_DIVS[v.clamp(0, 8) as usize].to_string())),
            reverse: BoolParam::new("Reverse", false),
            freeze: BoolParam::new("Freeze", false),
            send: BoolParam::new("Parallel", false),
            mix: awk_ui::mix_param(),
            safe: awk_ui::safe_param(),
            bypass: awk_ui::bypass_param(),
        }
    }
}

pub enum Task {
    /// Load/rebuild the impulse at this path with the current LENGTH/REVERSE.
    Load(String),
}

/// Background job: read, resample, prepare, build the engine, hand it over.
pub(crate) fn load_task(params: &AlleyParams, shared: &Shared, path: String) {
    // Free retired engines here, off the audio thread.
    while shared.garbage.pop().is_some() {}
    let sr = f32::from_bits(shared.sr.load(Ordering::Relaxed));
    let key = BuildKey::from_params(params, path.clone());
    let result = wav::read_wav(std::path::Path::new(&path)).map(|mut ir| {
        for ch in ir.ch.iter_mut() {
            *ch = wav::resample(ch, ir.sr, sr);
        }
        ir.sr = sr;
        let rts = wav::prepare(&mut ir, key.size, key.decay, key.length, key.reverse);
        (ir, rts)
    });
    match result {
        Ok((ir, (rt_orig, rt_new))) => {
            let secs = ir.ch[0].len() as f32 / sr;
            let env = wav::envelope(&ir, 360);
            let engine = Box::new(Engine::new(&ir.ch));
            let name = std::path::Path::new(&path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            if let Ok(mut slot) = shared.slot.lock() {
                // An engine that never got swapped in is simply replaced (freed here).
                *slot = Some(engine);
            }
            if let Ok(mut i) = shared.info.lock() {
                *i = IrInfo { name, secs, rt_orig, rt_new, env, error: None };
            }
        }
        Err(e) => {
            if let Ok(mut i) = shared.info.lock() {
                i.error = Some(e);
            }
        }
    }
    if let Ok(mut b) = shared.built.lock() {
        *b = key;
    }
    shared.loading.store(false, Ordering::Release);
}

struct AlleyVerb {
    params: Arc<AlleyParams>,
    shared: Arc<Shared>,
    sr: f32,
    engine: Option<Box<Engine>>,
    /// The previous engine keeps ringing while it fades out after a swap, so
    /// turning SIZE/DECAY or picking a new impulse never cuts the tail off.
    old: Option<Box<Engine>>,
    /// 0 -> 1 over the crossfade.
    swap_fade: f32,
    duck_env: f32,
    duck_band: [DuckBand; 2],
    duck_cuts: (f32, f32),
    duck_gain: f32,
    dry_level: f32,
    pre: [Vec<f32>; 2],
    pre_pos: usize,
    dry_delay: [Vec<f32>; 2],
    dry_pos: usize,
    tank: Option<FreezeTank>,
    algo: Option<Box<Algo>>,
    shim_tank: Option<Box<Algo>>,
    pitch: Option<Box<[Pitch; 2]>>,
    shim_lp: [Biquad; 2],
    shim_hp: [Biquad; 2],
    shim_prev: (f32, f32),
    shim_stale: bool,
    swell: Option<Box<Swell>>,
    la: [Vec<f32>; 2],
    la_pos: usize,
    swell_latched: bool,
    riser_hp: [Biquad; 2],
    in_f: [Biquad; 4],
    in_keys: (f32, f32, f32, f32),
    fx_delay: [Vec<f32>; 2],
    bpm: f64,
    /// 0 = IMPULSE, 1 = ALGO (glides for a click-free switch).
    engine_mix: f32,
    conv_stale: bool,
    algo_stale: bool,
    freeze_mix: f32,
    lcut: [Biquad; 2],
    hcut: [Biquad; 2],
    cuts: (f32, f32),
    /// Untouched input, delayed to the reported latency, for MIX / bypass.
    gdry: [DryDelay; 2],
    byp: BypassRamp,
    peaks: PeakFollower,
    ceil: awk_ui::Ceiling,
}

impl Default for AlleyVerb {
    fn default() -> Self {
        Self {
            params: Arc::new(AlleyParams::default()),
            shared: Arc::new(Shared {
                slot: Mutex::new(None),
                garbage: ArrayQueue::new(16),
                info: Mutex::new(IrInfo { name: "NO IMPULSE".into(), secs: 0.0, rt_orig: 0.0, rt_new: 0.0, env: Vec::new(), error: None }),
                loading: AtomicBool::new(false),
                sr: AtomicU32::new(44100f32.to_bits()),
                built: Mutex::new(BuildKey::default()),
                duck_db: AtomicU32::new(0f32.to_bits()),
                spec_wet: ArrayQueue::new(SPECTRUM_QUEUE),
                spec_dry: ArrayQueue::new(SPECTRUM_QUEUE),
                meters: Meters::default(),
            }),
            sr: 44100.0,
            engine: None,
            old: None,
            swap_fade: 1.0,
            duck_env: 0.0,
            duck_band: [DuckBand::default(); 2],
            duck_cuts: (0.0, 0.0),
            shim_tank: None,
            pitch: None,
            shim_lp: [Biquad::default(); 2],
            shim_hp: [Biquad::default(); 2],
            shim_prev: (0.0, 0.0),
            shim_stale: true,
            swell: None,
            la: [Vec::new(), Vec::new()],
            la_pos: 0,
            swell_latched: false,
            riser_hp: [Biquad::default(); 2],
            in_f: [Biquad::default(); 4],
            in_keys: (0.0, 0.0, 0.0, 0.0),
            fx_delay: [Vec::new(), Vec::new()],
            bpm: 120.0,
            algo: None,
            engine_mix: 0.0,
            conv_stale: false,
            algo_stale: true,
            duck_gain: 1.0,
            dry_level: 0.0,
            pre: [Vec::new(), Vec::new()],
            pre_pos: 0,
            dry_delay: [Vec::new(), Vec::new()],
            dry_pos: 0,
            tank: None,
            freeze_mix: 0.0,
            lcut: [Biquad::default(); 2],
            hcut: [Biquad::default(); 2],
            cuts: (0.0, 0.0),
            gdry: [DryDelay::new(1), DryDelay::new(1)],
            byp: BypassRamp::default(),
            peaks: PeakFollower::default(),
            ceil: awk_ui::Ceiling::default(),
        }
    }
}

impl Plugin for AlleyVerb {
    const NAME: &'static str = "AlleyVerb";
    const VENDOR: &'static str = "awk studios";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[new_nonzero_u32(2)],
            names: PortNames {
                layout: Some("Stereo"),
                main_input: Some("Input"),
                main_output: Some("Output"),
                aux_inputs: &["Duck (sidechain)"],
                aux_outputs: &[],
            },
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            aux_input_ports: &[new_nonzero_u32(1)],
            names: PortNames {
                layout: Some("Mono"),
                main_input: Some("Input"),
                main_output: Some("Output"),
                aux_inputs: &["Duck (sidechain)"],
                aux_outputs: &[],
            },
            ..AudioIOLayout::const_default()
        },
    ];
    type SysExMessage = ();
    type BackgroundTask = Task;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let params = self.params.clone();
        let shared = self.shared.clone();
        Box::new(move |task| match task {
            Task::Load(path) => load_task(&params, &shared, path),
        })
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // The editor must NOT keep the AsyncExecutor: it holds a strong ref to
        // nih-plug's wrapper, which owns the editor -> a cycle that is never
        // freed, so nih-plug's hidden message window outlived the DLL and FL
        // crashed on unload ("AlleyVerb.vst3_unloaded"). The editor has its
        // own loader thread instead (editor::Loader).
        editor::create(self.params.clone(), self.shared.clone())
    }

    fn initialize(&mut self, _l: &AudioIOLayout, cfg: &BufferConfig, context: &mut impl InitContext<Self>) -> bool {
        self.sr = cfg.sample_rate;
        self.shared.sr.store(self.sr.to_bits(), Ordering::Relaxed);
        let pre_len = (MAX_PREDELAY_MS * 0.001 * self.sr) as usize + 2;
        self.pre = [vec![0.0; pre_len], vec![0.0; pre_len]];
        self.dry_delay = [vec![0.0; BLOCK], vec![0.0; BLOCK]];
        self.tank = Some(FreezeTank::new(self.sr));
        self.algo = Some(Box::new(Algo::new(self.sr)));
        self.shim_tank = Some(Box::new(Algo::new(self.sr)));
        self.pitch = Some(Box::new([Pitch::new(self.sr), Pitch::new(self.sr)]));
        self.swell = Some(Box::new(Swell::new(self.sr)));
        let la = swell_lookahead(self.sr);
        self.la = [vec![0.0; la], vec![0.0; la]];
        self.la_pos = 0;
        self.fx_delay = [vec![0.0; BLOCK], vec![0.0; BLOCK]];
        self.gdry = [DryDelay::new(BLOCK + la), DryDelay::new(BLOCK + la)];
        self.byp.set_sr(self.sr);
        self.peaks.set_sr(self.sr);
        self.ceil.set_sr(self.sr);
        self.in_keys = (0.0, 0.0, 0.0, 0.0);
        self.swell_latched = self.params.swell.value();
        self.engine_mix = self.params.engine.value() as f32;
        // A brand-new AlleyVerb starts on the default impulse.
        if self.params.ir_path.read().map(|p| p.is_empty()).unwrap_or(false) {
            if let (Some(d), Ok(mut w)) = (default_ir(), self.params.ir_path.write()) {
                *w = d.to_string_lossy().to_string();
            }
        }
        self.cuts = (0.0, 0.0);
        context.set_latency_samples((BLOCK + if self.params.swell.value() { swell_lookahead(self.sr) } else { 0 }) as u32);
        // Reload the saved impulse (runs right here, before audio starts).
        let path = self.params.ir_path.read().map(|p| p.clone()).unwrap_or_default();
        if !path.is_empty() {
            self.shared.loading.store(true, Ordering::Release);
            context.execute(Task::Load(path));
        }
        true
    }

    fn reset(&mut self) {
        for b in self.pre.iter_mut().chain(self.dry_delay.iter_mut()) {
            b.iter_mut().for_each(|v| *v = 0.0);
        }
        if let Some(t) = self.tank.as_mut() {
            t.clear();
        }
        if let Some(a) = self.algo.as_mut() {
            a.clear();
        }
        if let Some(a) = self.shim_tank.as_mut() {
            a.clear();
        }
        if let Some(s) = self.swell.as_mut() {
            s.clear();
        }
        for b in self.la.iter_mut().chain(self.fx_delay.iter_mut()) {
            b.iter_mut().for_each(|v| *v = 0.0);
        }
        self.shim_prev = (0.0, 0.0);
        self.gdry.iter_mut().for_each(DryDelay::clear);
    }

    fn process(&mut self, buffer: &mut Buffer, aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        let sr = self.sr;
        // Swap in a freshly built engine (never allocates or frees here).
        if let Ok(mut slot) = self.shared.slot.try_lock() {
            if let Some(new) = slot.take() {
                // If a crossfade is still running, retire the oldest engine.
                if let Some(g) = std::mem::replace(&mut self.old, self.engine.take()) {
                    let _ = self.shared.garbage.push(g);
                }
                self.engine = Some(new);
                self.swap_fade = 0.0;
            }
        }
        // SIZE / DECAY / LENGTH / REVERSE changed: rebuild in the background.
        if !self.shared.loading.load(Ordering::Acquire) {
            if let Ok(b) = self.shared.built.try_lock() {
                if !b.path.is_empty() {
                    let want = BuildKey::from_params(&self.params, b.path.clone());
                    if !want.close_to(&b) {
                        drop(b);
                        self.shared.loading.store(true, Ordering::Release);
                        context.execute_background(Task::Load(want.path));
                    }
                }
            }
        }
        let p = &self.params;
        let freeze = p.freeze.value();
        let use_algo = p.engine.value() == 1;
        if let Some(a) = self.algo.as_mut() {
            a.set(&AlgoParams {
                mode: p.algo_mode.value().clamp(0, 11) as usize,
                color: p.algo_color.value().clamp(0, 2) as usize,
                decay: p.decay.value(),
                size: p.size.value(),
                attack: p.attack.value(),
                early_diff: p.early_diff.value(),
                late_diff: p.late_diff.value(),
                mod_rate: p.mod_rate.value(),
                mod_depth: p.mod_depth.value(),
                bass_mult: p.bass_mult.value(),
                bass_xover: p.bass_xover.value(),
                high_shelf_db: p.high_shelf.value(),
                high_freq: p.high_freq.value(),
            });
        }
        let engine_step = 1.0 / (0.08 * sr);
        if let Some(b) = context.transport().tempo {
            self.bpm = b;
        }
        // SWELL needs lookahead: tell FL when it's switched (FL re-aligns playback).
        let swell_on = p.swell.value();
        if swell_on != self.swell_latched {
            self.swell_latched = swell_on;
            context.set_latency_samples((BLOCK + if swell_on { swell_lookahead(sr) } else { 0 }) as u32);
            if let Some(s) = self.swell.as_mut() {
                s.clear();
            }
            for b in self.la.iter_mut() {
                b.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        let pre_div = p.pre_div.value().clamp(0, 8) as usize;
        let synced_pre = if pre_div > 0 { Some(((PRE_BEATS[pre_div] * 60.0 / self.bpm.max(20.0)) as f32 * sr) as usize) } else { None };
        if let Some(t) = self.shim_tank.as_mut() {
            t.set(&AlgoParams {
                mode: 0,
                color: 2,
                decay: p.decay.value().max(3.0) * 1.5,
                size: 1.4,
                attack: 0.25,
                early_diff: 0.7,
                late_diff: 0.7,
                mod_rate: 0.3,
                mod_depth: 0.35,
                bass_mult: 0.6,
                bass_xover: 500.0,
                high_shelf_db: -3.0,
                high_freq: 7000.0,
            });
        }
        let shim_ratio = 2f32.powf(p.shim_pitch.value() / 12.0);
        let shim_fb = p.shim_fb.value() * 0.85;
        let shim_tone = p.shim_tone.value();
        for f in self.shim_lp.iter_mut() {
            f.set(false, shim_tone, sr);
        }
        // Keeps rumble and DC out of the shimmer loop so it can't build into a hum.
        for f in self.shim_hp.iter_mut() {
            f.set(true, 150.0, sr);
        }
        let la_len = self.la[0].len();
        let fade_step = 1.0 / (0.04 * sr);
        let swap_step = 1.0 / (0.25 * sr);
        let send = p.send.value();
        let duck_amt = p.duck.value();
        let duck_att = (-1.0 / (0.005 * sr)).exp();
        let duck_rel = (-1.0 / (0.25 * sr)).exp();
        let gain_c = (-1.0 / (0.01 * sr)).exp();
        let dry_c = (-1.0 / (0.02 * sr)).exp();
        let duck_src = p.duck_src.value();
        let sidechain: Option<&[&mut [f32]]> = aux.inputs.first().map(|b| b.as_slice_immutable());
        let nch = buffer.channels();
        let pre_len = self.pre[0].len();
        let bypassed = p.bypass.value();
        for (i, mut frame) in buffer.iter_samples().enumerate() {
            let dry_g = p.dry.smoothed.next();
            let wet_g = p.wet.smoothed.next();
            let out_g = p.output.smoothed.next();
            let width = p.width.smoothed.next();
            let pre_ms = p.predelay.smoothed.next();
            let pre = synced_pre.unwrap_or((pre_ms * 0.001 * sr) as usize).min(pre_len - 2);
            let lc = p.low_cut.smoothed.next();
            let hc = p.high_cut.smoothed.next();
            if (lc, hc) != self.cuts {
                for ch in 0..2 {
                    self.lcut[ch].bypass = lc <= LOW_CUT_OFF;
                    self.lcut[ch].set(true, lc, sr);
                    self.hcut[ch].bypass = hc >= HIGH_CUT_OFF;
                    self.hcut[ch].set(false, hc, sr);
                }
                self.cuts = (lc, hc);
            }
            let dlo = p.duck_lo.smoothed.next();
            let dhi = p.duck_hi.smoothed.next();
            if (dlo, dhi) != self.duck_cuts {
                for b in self.duck_band.iter_mut() {
                    b.set(dlo, dhi.max(dlo * 1.05), sr, dlo <= LOW_CUT_OFF, dhi >= HIGH_CUT_OFF);
                }
                self.duck_cuts = (dlo, dhi);
            }
            let x0 = *frame.get_mut(0).unwrap();
            let x1 = if nch > 1 { *frame.get_mut(1).unwrap() } else { x0 };
            // INPUT EQ: shapes only what goes into the reverb.
            let (ilo, ihi) = (p.in_lo.smoothed.next(), p.in_hi.smoothed.next());
            if (ilo, ihi) != (self.in_keys.0, self.in_keys.1) {
                for c in 0..2 {
                    self.in_f[c].bypass = ilo <= LOW_CUT_OFF;
                    self.in_f[c].set(true, ilo, sr);
                    self.in_f[2 + c].bypass = ihi >= HIGH_CUT_OFF;
                    self.in_f[2 + c].set(false, ihi, sr);
                }
                self.in_keys.0 = ilo;
                self.in_keys.1 = ihi;
            }
            let a0 = self.in_f[0].tick(x0);
            let xf0 = self.in_f[2].tick(a0);
            let a1 = self.in_f[1].tick(x1);
            let xf1 = self.in_f[3].tick(a1);
            // RISER: one knob for builds.
            let riser = p.riser.smoothed.next();

            // Dry, delayed to match the convolution (ALGO is fed this too, so both line up).
            let dp = self.dry_pos;
            let (dl, dr) = (self.dry_delay[0][dp], self.dry_delay[1][dp]);
            self.dry_delay[0][dp] = x0;
            self.dry_delay[1][dp] = x1;
            let (fl, fr) = (self.fx_delay[0][dp], self.fx_delay[1][dp]);
            self.fx_delay[0][dp] = xf0;
            self.fx_delay[1][dp] = xf1;
            self.dry_pos = (dp + 1) % BLOCK;

            // ENGINE switch: glide, and fully idle the engine that isn't heard.
            let em_t = if use_algo { 1.0 } else { 0.0 };
            if self.engine_mix != em_t {
                self.engine_mix = if em_t > self.engine_mix { (self.engine_mix + engine_step).min(1.0) } else { (self.engine_mix - engine_step).max(0.0) };
            }
            let em = self.engine_mix;
            let (mut al, mut ar) = (0.0, 0.0);
            if em > 0.0 {
                if let Some(a) = self.algo.as_mut() {
                    if self.algo_stale {
                        a.clear();
                        self.algo_stale = false;
                    }
                    (al, ar) = a.tick(fl, fr);
                }
            } else {
                self.algo_stale = true;
            }
            let run_conv = em < 1.0;
            if !run_conv {
                self.conv_stale = true;
            } else if self.conv_stale {
                if let Some(e) = self.engine.as_mut() {
                    e.clear();
                }
                self.conv_stale = false;
            }
            // Convolution (+ the outgoing engine during a crossfade).
            let (mut wl, mut wr) = match self.engine.as_mut() {
                Some(e) if run_conv => e.tick(xf0, xf1),
                _ => (0.0, 0.0),
            };
            if self.swap_fade < 1.0 && run_conv {
                self.swap_fade = (self.swap_fade + swap_step).min(1.0);
                let f = self.swap_fade;
                let (gi, go) = ((f * std::f32::consts::FRAC_PI_2).sin(), (f * std::f32::consts::FRAC_PI_2).cos());
                let (ol, or) = match self.old.as_mut() {
                    Some(e) => e.tick(x0, x1),
                    None => (0.0, 0.0),
                };
                wl = wl * gi + ol * go;
                wr = wr * gi + or * go;
                if f >= 1.0 {
                    if let Some(g) = self.old.take() {
                        let _ = self.shared.garbage.push(g);
                    }
                }
            }
            if em > 0.0 {
                let (gc, ga) = ((em * std::f32::consts::FRAC_PI_2).cos(), (em * std::f32::consts::FRAC_PI_2).sin());
                wl = wl * gc + al * ga;
                wr = wr * gc + ar * ga;
            }
            // Pre-delay.
            self.pre[0][self.pre_pos] = wl;
            self.pre[1][self.pre_pos] = wr;
            let rd = (self.pre_pos + pre_len - pre) % pre_len;
            let (mut wl, mut wr) = (self.pre[0][rd], self.pre[1][rd]);
            self.pre_pos = (self.pre_pos + 1) % pre_len;
            // FREEZE.
            if let Some(t) = self.tank.as_mut() {
                let (tl, tr) = t.tick(wl, wr, freeze);
                let target = if freeze { 1.0 } else { 0.0 };
                if self.freeze_mix != target {
                    self.freeze_mix = if target > self.freeze_mix {
                        (self.freeze_mix + fade_step).min(1.0)
                    } else {
                        (self.freeze_mix - fade_step).max(0.0)
                    };
                }
                let f = self.freeze_mix;
                wl = wl * (1.0 - f) + tl * f;
                wr = wr * (1.0 - f) + tr * f;
            }
            // SHIMMER: pitch-shifted, fed back through its own long hall.
            let shim_amt = (p.shimmer.smoothed.next() + riser * 0.5).min(1.0);
            if shim_amt > 0.0005 {
                if let (Some(t), Some(ps)) = (self.shim_tank.as_mut(), self.pitch.as_mut()) {
                    if self.shim_stale {
                        t.clear();
                        ps[0].clear();
                        ps[1].clear();
                        self.shim_prev = (0.0, 0.0);
                        self.shim_stale = false;
                    }
                    let il = ((wl + self.shim_prev.0 * shim_fb) * 0.8).tanh();
                    let ir = ((wr + self.shim_prev.1 * shim_fb) * 0.8).tanh();
                    let pl = self.shim_hp[0].tick(self.shim_lp[0].tick(ps[0].tick(il, shim_ratio)));
                    let pr = self.shim_hp[1].tick(self.shim_lp[1].tick(ps[1].tick(ir, shim_ratio)));
                    let (sl, sr2) = t.tick(pl, pr);
                    self.shim_prev = (sl, sr2);
                    wl += sl * shim_amt;
                    wr += sr2 * shim_amt;
                }
            } else {
                self.shim_stale = true;
            }
            // Cuts (RISER pushes the low cut up), width, level.
            let rl = 20.0 * 100f32.powf(riser);
            if riser > 0.001 && rl > self.cuts.0 {
                for c in 0..2 {
                    self.lcut[c].bypass = false;
                    self.lcut[c].set(true, rl, sr);
                }
                self.cuts.0 = -1.0; // force a refresh when the riser drops
            }
            wl = self.hcut[0].tick(self.lcut[0].tick(wl));
            wr = self.hcut[1].tick(self.lcut[1].tick(wr));
            let m = 0.5 * (wl + wr);
            let s = 0.5 * (wl - wr) * (width * (1.0 + riser)).min(2.0);
            // DUCK: the reverb dips while the source plays, swells back in the gaps.
            // The detector hears the input ~23 ms before the delayed output: free lookahead.
            let own = x0.abs().max(x1.abs());
            let sc = sidechain.map(|c| c.iter().map(|ch| ch.get(i).map(|v| v.abs()).unwrap_or(0.0)).fold(0.0f32, f32::max)).unwrap_or(0.0);
            let lvl = match duck_src {
                0 => own,
                1 => sc,
                _ => own.max(sc),
            };
            let c = if lvl > self.duck_env { duck_att } else { duck_rel };
            self.duck_env = lvl + (self.duck_env - lvl) * c;
            let target = if duck_amt > 0.0 {
                let over = (util::gain_to_db(self.duck_env.max(1e-6)) + 36.0).max(0.0);
                util::db_to_gain(-(over * 0.75).min(24.0) * duck_amt)
            } else {
                1.0
            };
            self.duck_gain = target + (self.duck_gain - target) * gain_c;
            let g = self.duck_gain;
            let rg = 1.0 + riser; // up to +6 dB of reverb
            let wl = self.duck_band[0].tick(m + s, g) * wet_g * rg;
            let wr = self.duck_band[1].tick(m - s, g) * wet_g * rg;
            // RISER also thins the original (only while it's up).
            let (dl, dr) = if riser > 0.005 {
                let f = 20.0 * 50f32.powf(riser);
                for h in self.riser_hp.iter_mut() {
                    h.set(true, f, sr);
                }
                (self.riser_hp[0].tick(dl), self.riser_hp[1].tick(dr))
            } else {
                (dl, dr)
            };

            let _ = self.shared.spec_wet.push(0.5 * (wl + wr));
            let _ = self.shared.spec_dry.push(0.5 * (dl + dr));
            // `par` fades 0..1 so flipping PARALLEL never clicks; at 1 the dry is untouched.
            let par_t = if send { 1.0 } else { 0.0 };
            self.dry_level = par_t + (self.dry_level - par_t) * dry_c;
            let par = if (self.dry_level - par_t).abs() < 1e-4 { par_t } else { self.dry_level };
            let dry_gain = par + (1.0 - par) * dry_g * out_g;
            let mut outs = [dl * dry_gain + wl * out_g, dr * dry_gain + wr * out_g];
            // SWELL: everything waits in the lookahead, the swell (built from
            // the hit just captured) rises into it.
            if swell_on {
                let i = self.la_pos;
                let (ol, or) = (self.la[0][i], self.la[1][i]);
                self.la[0][i] = outs[0];
                self.la[1][i] = outs[1];
                self.la_pos = (i + 1) % la_len;
                let sg = p.swell_amt.smoothed.next() * out_g;
                let (sl, sr2) = match self.swell.as_mut() {
                    Some(sw) => sw.tick(0.5 * (x0 + x1), p.swell_len.value(), p.swell_sens.value(), p.swell_curve.value()),
                    None => (0.0, 0.0),
                };
                outs = [ol + sl * sg, or + sr2 * sg];
            }
            // Header MIX + power: untouched input lined up with the reported latency.
            let lat = BLOCK + if swell_on { la_len } else { 0 };
            let gd = [self.gdry[0].tick(x0, lat), self.gdry[1].tick(x1, lat)];
            let gm = p.mix.smoothed.next() * self.byp.next(bypassed);
            if gm < 1.0 {
                outs = [gd[0] + (outs[0] - gd[0]) * gm, gd[1] + (outs[1] - gd[1]) * gm];
            }
            // SAFE: never above -0.1 dBFS (whatever the impulse, riser or swell does).
            let (cl, cr) = self.ceil.tick(outs[0], outs[1], p.safe.value());
            outs = [cl, cr];
            let (ol, or) = if nch == 1 { let m = 0.5 * (outs[0] + outs[1]); (m, m) } else { (outs[0], outs[1]) };
            self.peaks.feed(x0, x1, ol, or);
            for (ch, s) in frame.iter_mut().enumerate() {
                *s = if nch == 1 { 0.5 * (outs[0] + outs[1]) } else { outs[ch.min(1)] };
            }
        }
        self.peaks.store(&self.shared.meters);
        self.ceil.store(&self.shared.meters);
        self.shared.duck_db.store(util::gain_to_db(self.duck_gain.max(1e-4)).min(0.0).to_bits(), Ordering::Relaxed);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for AlleyVerb {
    const CLAP_ID: &'static str = "com.awknode.alleyverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Convolution reverb with freeze");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo, ClapFeature::Reverb];
}

impl Vst3Plugin for AlleyVerb {
    const VST3_CLASS_ID: [u8; 16] = *b"AwknodeAlleyVrb1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

nih_export_clap!(AlleyVerb);
nih_export_vst3!(AlleyVerb);

#[cfg(test)]
mod dump_defaults;

#[cfg(test)]
mod header_tests {
    use super::*;

    /// Header MIX + power exist with the shared IDs and safe defaults.
    #[test]
    fn mix_and_bypass_params() {
        let p = AlleyParams::default();
        let ids: Vec<String> = p.param_map().into_iter().map(|(id, _, _)| id).collect();
        assert!(ids.iter().any(|i| i == "gmix") && ids.iter().any(|i| i == "bypass"));
        assert_eq!(p.mix.value(), 1.0);
        assert!(!p.bypass.value());
    }

    /// The global dry path lines up with the reported latency (BLOCK, + lookahead with SWELL).
    #[test]
    fn global_dry_matches_latency() {
        let sr = 48000.0;
        let la = swell_lookahead(sr);
        let mut d = DryDelay::new(BLOCK + la);
        for lat in [BLOCK, BLOCK + la] {
            d.clear();
            let out: Vec<f32> = (0..lat + 4).map(|n| d.tick(if n == 0 { 1.0 } else { 0.0 }, lat)).collect();
            assert_eq!(out.iter().position(|&v| v == 1.0), Some(lat));
        }
    }
}
