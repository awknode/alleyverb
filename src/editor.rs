//! AlleyVerb editor — awk studios "Vector HUD" look (purple accent, cyan lines).
//! - Header (shared PresetBar): presets, A/B, undo/redo, power (bypass), global MIX.
//! - SEC.01, two tabs:
//!     IMPULSE: search box, folders + impulse names, star favourites,
//!              up/down arrow keys step through the list.
//!     ALGO:    the 12 algorithm modes, plus the COLOR era.
//! - THE TAIL: the loaded impulse's envelope (or the ALGO decay shape) on a
//!   dB-over-time screen with the RT60 marked, PARALLEL, PREV/NEXT, REVERSE, FREEZE.
//! - WHERE IT HITS: live spectrum of the reverb (accent) over the dry (grey) in
//!   dBFS, with the output cuts and the duck band.
//! - IN / OUT meters (dBFS, held peak readouts).
//! Knobs: drag (Ctrl or Shift = fine), scroll to nudge, double-click to reset.

use crate::algo::{COLOR_NAMES, MAX_ATTACK_MS, MODE_INFO, MODE_NAMES};
use crate::awk_ui::*;
use crate::conv::BLOCK;
use crate::fx::{response_db, swell_lookahead};
use crate::presets::PresetBar;
use crate::{favorites_file, ir_roots, load_task, AlleyParams, Shared, DUCK_SOURCES, HIGH_CUT_OFF, LOW_CUT_OFF, PRE_DIVS};
use nih_plug::prelude::*;
use nih_plug_egui::create_egui_editor;
use nih_plug_egui::egui::{self, pos2, vec2, Align2, Color32, Key, Painter, Pos2, Rect, Sense, Stroke, Vec2};
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub const WIDTH: u32 = 960;
pub const HEIGHT: u32 = 984;
const ACC: Color32 = rgb(0xb4, 0x8c, 0xff);
const FFT: usize = 4096;
const SPEC_POINTS: usize = 200;

struct Item {
    path: PathBuf,
    key: String,
    name: String,
    cat: usize,
}

struct Ui {
    cats: Vec<String>,
    items: Vec<Item>,
    favs: BTreeSet<String>,
    scanned: bool,
    /// Category index, or FAV_CAT for favourites.
    cat: usize,
    search: String,
    presets: PresetBar,
    /// EXTRAS row tab: 0 ALGO, 1 SHIMMER, 2 RISER + SWELL, 3 INPUT EQ.
    extra_tab: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    ring_wet: Vec<f32>,
    ring_dry: Vec<f32>,
    pos: usize,
    tbuf: Vec<f32>,
    spec: Vec<Complex<f32>>,
    wet_db: Vec<f32>,
    dry_db: Vec<f32>,
}

const FAV_CAT: usize = usize::MAX;

/// "TR Vocals Hall 001" -> "Vocals Hall"; "CLA Slap 15 002" -> "Slap 15 (2)".
fn clean_name(stem: &str) -> String {
    let mut s = stem.trim().to_string();
    for pre in ["TR ", "CLA ", "VVV ", "FP ", "FL "] {
        if s.starts_with(pre) && s.len() > pre.len() + 1 {
            s = s[pre.len()..].to_string();
            break;
        }
    }
    if let Some((head, tail)) = s.rsplit_once(' ') {
        if tail.len() == 3 && tail.chars().all(|c| c.is_ascii_digit()) {
            let n: u32 = tail.parse().unwrap_or(1);
            s = if n <= 1 { head.to_string() } else { format!("{head} ({n})") };
        }
    }
    s
}

fn clean_cat(prefix: &str, dir: &str) -> String {
    let base = match dir {
        "CLA Epic" => "CLA EPIC".to_string(),
        "FP FrankPole" => "FRANK POLE".to_string(),
        "TR Hall" => "HALLS".to_string(),
        "TR Inverse" => "REVERSE + GATED".to_string(),
        "TR Plate" => "PLATES".to_string(),
        "TR Room" => "ROOMS".to_string(),
        "ValhallaVintageVerb" => "VINTAGE VERB".to_string(),
        d => d.to_uppercase(),
    };
    if prefix.is_empty() { base } else { format!("{prefix} {base}") }
}

fn scan(st: &mut Ui) {
    st.cats.clear();
    st.items.clear();
    for (prefix, root) in ir_roots() {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
        dirs.sort();
        for d in dirs {
            let mut files = Vec::new();
            let mut stack = vec![d.clone()];
            while let Some(x) = stack.pop() {
                if let Ok(rd) = std::fs::read_dir(&x) {
                    for e in rd.flatten() {
                        let p = e.path();
                        if p.is_dir() {
                            stack.push(p);
                        } else if p.extension().map(|v| v.eq_ignore_ascii_case("wav")).unwrap_or(false) {
                            files.push(p);
                        }
                    }
                }
            }
            if files.is_empty() {
                continue;
            }
            let ci = st.cats.len();
            st.cats.push(clean_cat(&prefix, &d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
            let mut its: Vec<Item> = files
                .into_iter()
                .map(|f| {
                    let name = clean_name(&f.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
                    Item { key: f.to_string_lossy().to_string(), path: f, name, cat: ci }
                })
                .collect();
            its.sort_by_key(|i| i.name.to_lowercase());
            st.items.extend(its);
        }
    }
    st.favs = std::fs::read_to_string(favorites_file()).map(|s| s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()).unwrap_or_default();
    st.scanned = true;
}

fn save_favs(favs: &BTreeSet<String>) {
    let f = favorites_file();
    if let Some(d) = f.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(f, favs.iter().cloned().collect::<Vec<_>>().join("\n"));
}

/// Indices of the items currently listed (search > favourites > category).
fn visible(st: &Ui) -> Vec<usize> {
    let words: Vec<String> = st.search.to_lowercase().split_whitespace().map(|w| w.to_string()).collect();
    st.items
        .iter()
        .enumerate()
        .filter(|(_, it)| {
            if !words.is_empty() {
                let hay = format!("{} {}", it.name, st.cats[it.cat]).to_lowercase();
                words.iter().all(|w| hay.contains(w))
            } else if st.cat == FAV_CAT {
                st.favs.contains(&it.key)
            } else {
                it.cat == st.cat
            }
        })
        .map(|(i, _)| i)
        .collect()
}

fn set_int(setter: &ParamSetter, p: &IntParam, v: i32) {
    setter.begin_set_parameter(p);
    setter.set_parameter(p, v);
    setter.end_set_parameter(p);
}

/// Loads impulses off the GUI thread. One worker per editor; dropping the
/// editor closes the queue and WAITS for the worker, so no thread of ours is
/// left running when the host unloads the DLL.
pub struct Loader {
    tx: Option<std::sync::mpsc::Sender<String>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Loader {
    fn new(params: Arc<AlleyParams>, shared: Arc<Shared>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let worker = std::thread::Builder::new()
            .name("alleyverb-loader".into())
            .spawn(move || {
                while let Ok(mut path) = rx.recv() {
                    // Only the newest request matters (fast PREV/NEXT clicking).
                    while let Ok(newer) = rx.try_recv() {
                        path = newer;
                    }
                    load_task(&params, &shared, path);
                }
            })
            .ok();
        Self { tx: Some(tx), worker }
    }
    fn load(&self, path: String) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(path);
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

pub fn create(params: Arc<AlleyParams>, shared: Arc<Shared>) -> Option<Box<dyn Editor>> {
    let state = params.editor_state.clone();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
    let ui_state = Ui {
        cats: Vec::new(),
        items: Vec::new(),
        favs: BTreeSet::new(),
        scanned: false,
        cat: 0,
        search: String::new(),
        presets: PresetBar::new("AlleyVerb"),
        extra_tab: 0,
        spec: fft.make_output_vec(),
        fft,
        ring_wet: vec![0.0; FFT],
        ring_dry: vec![0.0; FFT],
        pos: 0,
        tbuf: vec![0.0; FFT],
        wet_db: vec![-120.0; SPEC_POINTS],
        dry_db: vec![-120.0; SPEC_POINTS],
    };
    let skin = Skin::new(Look::Hud, ACC, "ALLEYVERB", env!("CARGO_PKG_VERSION"), "IMPULSE + ALGO // FREEZE // SHIMMER // RISER + SWELL // DUCK");
    // Owned by the update closure below, so it lives exactly as long as the editor.
    let loader = Loader::new(params.clone(), shared.clone());
    create_egui_editor(
        state,
        (ui_state, skin),
        |ctx, _| install_fonts(ctx),
        move |ctx, setter, state| frame(ctx, setter, state, &params, &shared, &loader),
    )
}

/// Left-bottom text, squeezed to fit `max_w` by dropping characters.
fn fit_txt(p: &Painter, pos: Pos2, s: &str, font: egui::FontId, col: Color32, sp: f32, max_w: f32) {
    let mut out = s.to_string();
    if txt_w(p, &out, font.clone(), sp) > max_w {
        let mut chars: Vec<char> = s.chars().collect();
        while chars.len() > 4 {
            chars.pop();
            out = chars.iter().collect::<String>() + "…";
            if txt_w(p, &out, font.clone(), sp) <= max_w {
                break;
            }
        }
    }
    txt(p, pos, Align2::LEFT_BOTTOM, out, font, col, sp);
}

fn freeze_button(ui: &mut egui::Ui, setter: &ParamSetter, r: Rect, param: &BoolParam) {
    let on = param.value();
    let resp = ui.interact(r, ui.id().with("freeze"), Sense::click());
    if resp.clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, !on);
        setter.end_set_parameter(param);
    }
    let p = ui.painter();
    if on {
        cham_fill(p, r.expand(4.0), 10.0, 10.0, al(CY, 0.12), Stroke::NONE);
    }
    let fill = if on { al(CY, 0.35) } else if resp.hovered() { al(CY, 0.12) } else { al(CY, 0.06) };
    cham_fill(p, r, 8.0, 8.0, fill, Stroke::new(1.5_f32, if on { Color32::WHITE } else { CY }));
    txt(p, r.center(), Align2::CENTER_CENTER, if on { "FROZEN" } else { "FREEZE" }, disp(11.0), Color32::WHITE, 3.0);
}

/// Time-axis labels for a span in seconds.
fn tsec(s: f32, span: f32) -> String {
    if span < 1.0 {
        format!("{:.0}ms", s * 1000.0)
    } else if span < 10.0 {
        format!("{s:.1}s")
    } else {
        format!("{s:.0}s")
    }
}

/// Shared dB-over-time frame for THE TAIL. Returns (clipped painter, pal, x of seconds, y of dB, base y).
fn tail_frame(painter: &Painter, skin: &Skin, wave: Rect, span: f32, dlabel: &str) -> (Painter, Pal, impl Fn(f32) -> f32 + Copy, impl Fn(f32) -> f32 + Copy, f32, f32, f32) {
    let (sp, pal) = skin.screen(painter, wave);
    let (lft, rgt, top, base) = (wave.left() + 36.0, wave.right() - 10.0, wave.top() + 10.0, wave.bottom() - 18.0);
    let px = move |s: f32| lft + (s / span).clamp(0.0, 1.0) * (rgt - lft);
    let py = move |d: f32| top + (-d / 72.0).clamp(0.0, 1.0) * (base - top);
    skin.db_axis(&sp, &pal, lft, rgt, py, &[0.0, -24.0, -48.0, -72.0], lft - 5.0);
    for k in 0..=6 {
        let s = span * k as f32 / 6.0;
        let xx = px(s).round() + 0.5;
        line(&sp, pos2(xx, top), pos2(xx, base), pal.grid, 1.0);
        let a = if k == 0 { Align2::LEFT_TOP } else if k == 6 { Align2::RIGHT_TOP } else { Align2::CENTER_TOP };
        txt(&sp, pos2(xx, base + 4.0), a, tsec(s, span), mono(8.5), pal.axis, 0.0);
    }
    txt(&sp, pos2(wave.left() + 6.0, top - 2.0), Align2::LEFT_TOP, dlabel, mono(7.5), pal.alt, 0.0);
    (sp, pal, px, py, lft, rgt, base)
}

#[allow(clippy::too_many_arguments)]
fn rt60_marker(sp: &Painter, px: &dyn Fn(f32) -> f32, py: &dyn Fn(f32) -> f32, lft: f32, rgt: f32, at: f32, span: f32, top: f32, base: f32) {
    if at > 0.0 && at < span {
        let xx = px(at);
        dashed(sp, pos2(xx, top), pos2(xx, base), Color32::from_white_alpha(150), 1.0, 3.0, 3.0);
        dashed(sp, pos2(lft, py(-60.0)), pos2(rgt, py(-60.0)), Color32::from_white_alpha(64), 1.0, 2.0, 4.0);
        let right = xx > rgt - 90.0;
        txt(sp, pos2(if right { xx - 4.0 } else { xx + 4.0 }, top + 4.0), if right { Align2::RIGHT_TOP } else { Align2::LEFT_TOP }, format!("RT60 {at:.2} s"), mono(9.0), Color32::WHITE, 0.0);
    }
}

/// Loaded impulse: peak envelope in dB (re the impulse's peak) over its length.
fn draw_tail_ir(painter: &Painter, skin: &Skin, wave: Rect, env: &[f32], secs: f32, rt60: f32, frozen: bool) {
    let span = secs.max(0.05);
    let (sp, pal, px, py, lft, rgt, base) = tail_frame(painter, skin, wave, span, "dB");
    if !env.is_empty() {
        let n = env.len();
        let pts: Vec<Pos2> = env.iter().enumerate().map(|(k, e)| pos2(px(span * (k as f32 + 0.5) / n as f32), py((20.0 * e.max(1e-6).log10()).max(-72.0)))).collect();
        fill_under(&sp, &pts, base, pal.fill(0.18));
        polyline(&sp, pts, pal.trace, 1.4, pal.glow);
        rt60_marker(&sp, &px, &py, lft, rgt, rt60, span, wave.top() + 10.0, base);
    } else {
        txt(&sp, wave.center(), Align2::CENTER_CENTER, "NO IMPULSE LOADED", mono(10.0), pal.alt, 1.0);
    }
    if frozen {
        skin.readout(&sp, Rect::from_min_size(pos2(rgt - 112.0, wave.top() + 8.0), vec2(108.0, 18.0)), "FROZEN · HOLD", CY);
    }
    skin.screen_end(painter, wave);
}

/// ALGO: attack swell then the RT60 slope (shown over 1.5x decay), after the pre-delay.
fn draw_tail_algo(painter: &Painter, skin: &Skin, wave: Rect, p: &AlleyParams, gated: bool) {
    let decay = p.decay.value().max(0.05);
    let pre = p.predelay.value() * 0.001;
    let atk = p.attack.value() * MAX_ATTACK_MS * 0.001;
    let span = (decay * 1.5).max(0.5) + pre;
    let (sp, pal, px, py, lft, rgt, base) = tail_frame(painter, skin, wave, span, "dB");
    let n = 360;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let ts = span * k as f32 / n as f32;
            let t = ts - pre;
            let a = if t < 0.0 {
                0.0
            } else if gated {
                if t < decay.min(5.0) { 1.0 } else { 0.0 }
            } else {
                let rise = if atk > 0.001 { (t / atk).min(1.0) } else { 1.0 };
                rise * 10f32.powf(-3.0 * t / decay)
            };
            pos2(px(ts), py((20.0 * a.max(1e-6).log10()).max(-72.0)))
        })
        .collect();
    fill_under(&sp, &pts, base, pal.fill(0.18));
    polyline(&sp, pts, pal.trace, 1.4, pal.glow);
    if !gated {
        rt60_marker(&sp, &px, &py, lft, rgt, pre + decay, span, wave.top() + 10.0, base);
    }
    if p.freeze.value() {
        skin.readout(&sp, Rect::from_min_size(pos2(rgt - 112.0, wave.top() + 8.0), vec2(108.0, 18.0)), "FROZEN · HOLD", CY);
    }
    skin.screen_end(painter, wave);
}

fn impulse_browser(ui: &mut egui::Ui, painter: &Painter, skin: &Skin, stash: Rect, st: &mut Ui, current: &str, load: &dyn Fn(&PathBuf)) {
    if st.items.is_empty() {
        skin.small(painter, stash.center(), Align2::CENTER_CENTER, "NO IMPULSES IN Documents\\Awknode\\Impulses", skin.text_dim());
        return;
    }
    // Search box.
    let search_r = Rect::from_min_max(pos2(stash.left() + 204.0, stash.top() + 8.0), pos2(stash.right() - 12.0, stash.top() + 30.0));
    cham_fill(painter, search_r, 5.0, 5.0, Color32::from_rgba_unmultiplied(3, 10, 18, 230), Stroke::new(1.0_f32, al(CY, 0.35)));
    let total = st.items.len();
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(search_r.shrink2(vec2(8.0, 3.0))), |ui| {
        let v = ui.visuals_mut();
        v.extreme_bg_color = Color32::TRANSPARENT;
        v.widgets.inactive.bg_stroke = Stroke::NONE;
        v.widgets.hovered.bg_stroke = Stroke::NONE;
        v.selection.stroke = Stroke::NONE;
        v.selection.bg_fill = al(ACC, 0.35);
        v.text_cursor.stroke = Stroke::new(1.5_f32, ACC);
        ui.add(
            egui::TextEdit::singleline(&mut st.search)
                .frame(false)
                .hint_text(egui::RichText::new(format!("search {total} impulses: hall, dark, vocal...")).color(al(rgb(0x8f, 0xc9, 0xe3), 0.6)))
                .text_color(rgb(0xe6, 0xf8, 0xff))
                .desired_width(search_r.width() - 16.0)
                .font(mono(11.0)),
        );
    });
    let searching = !st.search.trim().is_empty();
    // Folders (left).
    let cat_rect = Rect::from_min_max(pos2(stash.left() + 10.0, stash.top() + 40.0), pos2(stash.left() + 172.0, stash.bottom() - 10.0));
    let list_rect = Rect::from_min_max(pos2(stash.left() + 186.0, stash.top() + 40.0), stash.right_bottom() - vec2(10.0, 10.0));
    line(painter, pos2(stash.left() + 179.0, cat_rect.top()), pos2(stash.left() + 179.0, cat_rect.bottom()), al(CY, 0.2), 1.0);
    let mut pick_cat = None;
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(cat_rect), |ui| {
        egui::ScrollArea::vertical().id_salt("cats").show(ui, |ui| {
            let favn = st.items.iter().filter(|i| st.favs.contains(&i.key)).count();
            if row(ui, &format!("★ FAVOURITES  {favn}"), !searching && st.cat == FAV_CAT, false, None).0 {
                pick_cat = Some(FAV_CAT);
            }
            for (i, c) in st.cats.iter().enumerate() {
                let n = st.items.iter().filter(|it| it.cat == i).count();
                if row(ui, &format!("{c}  {n}"), !searching && i == st.cat, false, None).0 {
                    pick_cat = Some(i);
                }
            }
        });
    });
    if let Some(i) = pick_cat {
        st.cat = i;
        st.search.clear();
    }
    // Impulses (right), with a star to favourite.
    let list = visible(st);
    let mut pick = None;
    let mut fav_toggle = None;
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(list_rect), |ui| {
        egui::ScrollArea::vertical().id_salt(("files", st.cat, searching)).show(ui, |ui| {
            if list.is_empty() {
                ui.label(egui::RichText::new(if st.cat == FAV_CAT && !searching { "Click the ☆ next to an impulse to favourite it" } else { "Nothing matches" }).font(mono(10.0)).color(al(rgb(0x8f, 0xc9, 0xe3), 0.7)));
            }
            for &i in &list {
                let it = &st.items[i];
                let label = if searching { format!("{}  ({})", it.name, st.cats[it.cat]) } else { it.name.clone() };
                let (clicked, star) = row(ui, &label, it.key == current, true, Some(st.favs.contains(&it.key)));
                if star {
                    fav_toggle = Some(it.key.clone());
                } else if clicked {
                    pick = Some(it.path.clone());
                }
            }
        });
    });
    if let Some(k) = fav_toggle {
        if !st.favs.remove(&k) {
            st.favs.insert(k);
        }
        save_favs(&st.favs);
    }
    if let Some(f) = pick {
        load(&f);
    }
}

fn algo_browser(ui: &mut egui::Ui, painter: &Painter, skin: &Skin, stash: Rect, setter: &ParamSetter, p: &AlleyParams) {
    let mode = p.algo_mode.value().clamp(0, 11) as usize;
    let grid = Rect::from_min_max(pos2(stash.left() + 12.0, stash.top() + 40.0), pos2(stash.right() - 12.0, stash.bottom() - 44.0));
    let (cols, rows) = (3, 4);
    let cw = grid.width() / cols as f32;
    let rh = grid.height() / rows as f32;
    for m in 0..12 {
        let r = Rect::from_min_size(pos2(grid.left() + cw * (m % cols) as f32, grid.top() + rh * (m / cols) as f32), vec2(cw, rh)).shrink(3.0);
        if skin.chip(ui, r, MODE_NAMES[m], m == mode, ("mode", m)).on_hover_text(MODE_INFO[m]).clicked() {
            set_int(setter, &p.algo_mode, m as i32);
        }
    }
    // COLOR era.
    let color = p.algo_color.value().clamp(0, 2) as usize;
    txt(painter, pos2(stash.left() + 16.0, stash.bottom() - 24.0), Align2::LEFT_CENTER, "COLOR", disp(8.5), rgb(0xcd, 0xef, 0xff), 2.0);
    for (i, name) in COLOR_NAMES.iter().enumerate() {
        let r = Rect::from_min_size(pos2(stash.left() + 86.0 + i as f32 * 104.0, stash.bottom() - 34.0), vec2(98.0, 20.0));
        if skin.chip(ui, r, name, i == color, ("color", i)).clicked() {
            set_int(setter, &p.algo_color, i as i32);
        }
    }
}

/// One browser row. `file`: impulse row (chamfered highlight) vs folder row (side bar).
/// `fav`: Some(is_favourite) draws a star at the right. Returns (row clicked, star clicked).
fn row(ui: &mut egui::Ui, text: &str, selected: bool, file: bool, fav: Option<bool>) -> (bool, bool) {
    let w = ui.available_width();
    let (r, resp) = ui.allocate_exact_size(Vec2::new(w, if file { 22.0 } else { 20.0 }), Sense::click());
    let p = ui.painter();
    if selected {
        if file {
            cham_fill(p, r.shrink2(vec2(0.0, 1.0)), 4.0, 4.0, al(ACC, 0.2), Stroke::new(1.0_f32, ACC));
        } else {
            p.rect_filled(r, 0.0, al(ACC, 0.15));
            p.rect_filled(Rect::from_min_size(r.min, vec2(2.0, r.height())), 0.0, ACC);
        }
    } else if resp.hovered() {
        p.rect_filled(r, 0.0, al(CY, 0.07));
    }
    let col = if selected { Color32::WHITE } else if file { rgb(0xcd, 0xef, 0xff) } else { rgb(0x8f, 0xc9, 0xe3) };
    let tp = p.with_clip_rect(Rect::from_min_max(r.min, pos2(r.right() - if fav.is_some() { 24.0 } else { 2.0 }, r.bottom())).intersect(p.clip_rect()));
    txt(&tp, r.left_center() + vec2(8.0, 0.0), Align2::LEFT_CENTER, text, mono(if file { 11.0 } else { 10.0 }), col, 0.5);
    let mut star = false;
    if let Some(on) = fav {
        let sr = Rect::from_center_size(r.right_center() - vec2(12.0, 0.0), Vec2::splat(18.0));
        let hovered = resp.hover_pos().map(|h| sr.contains(h)).unwrap_or(false);
        let c = if on { ACC } else if hovered { Color32::WHITE } else { al(CY, 0.3) };
        p.text(sr.center(), Align2::CENTER_CENTER, if on { "★" } else { "☆" }, mono(13.0), c);
        star = resp.clicked() && hovered;
    }
    (resp.clicked() && !star, star)
}

/// Pull new samples from the audio thread and update the smoothed spectra (dBFS, sine peak = 0 dB).
fn pump_spectrum(st: &mut Ui, shared: &Shared, sr: f32) {
    let mut got = false;
    while let (Some(w), Some(d)) = (shared.spec_wet.pop(), shared.spec_dry.pop()) {
        st.ring_wet[st.pos] = w;
        st.ring_dry[st.pos] = d;
        st.pos = (st.pos + 1) % FFT;
        got = true;
    }
    if !got {
        // Let the display fall when audio stops.
        st.wet_db.iter_mut().chain(st.dry_db.iter_mut()).for_each(|v| *v = (*v - 1.5).max(-120.0));
        return;
    }
    for which in 0..2 {
        let ring = if which == 0 { &st.ring_wet } else { &st.ring_dry };
        for k in 0..FFT {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / FFT as f32).cos();
            st.tbuf[k] = ring[(st.pos + k) % FFT] * w;
        }
        let _ = st.fft.process(&mut st.tbuf, &mut st.spec);
        let out = if which == 0 { &mut st.wet_db } else { &mut st.dry_db };
        for (i, o) in out.iter_mut().enumerate() {
            // Log-spaced bands 20 Hz .. 20 kHz.
            let f0 = 20.0 * 1000f32.powf(i as f32 / SPEC_POINTS as f32);
            let f1 = 20.0 * 1000f32.powf((i + 1) as f32 / SPEC_POINTS as f32);
            let b0 = ((f0 / sr * FFT as f32) as usize).max(1);
            let b1 = ((f1 / sr * FFT as f32) as usize).max(b0 + 1).min(FFT / 2);
            let mut m = 0.0f32;
            for b in b0..b1 {
                m = m.max(st.spec[b].norm_sqr());
            }
            let v = 10.0 * (m.max(1e-14)).log10() - 20.0 * (FFT as f32 / 4.0).log10();
            *o = if v > *o { v } else { *o * 0.85 + v * 0.15 };
        }
    }
}

fn draw_spectrum(painter: &Painter, skin: &Skin, r: Rect, st: &Ui, params: &AlleyParams, duck_db: f32, sr: f32) {
    let (sp, pal) = skin.screen(painter, r);
    let (sl, sr_, stp, sb) = (r.left() + 44.0, r.right() - 12.0, r.top() + 26.0, r.bottom() - 22.0);
    let x_of = |u: f32| sl + u.clamp(0.0, 1.0) * (sr_ - sl);
    let y_of = |db: f32| sb - ((db + 72.0) / 72.0).clamp(0.0, 1.0) * (sb - stp);
    skin.db_axis(&sp, &pal, sl, sr_, y_of, &[0.0, -12.0, -24.0, -36.0, -48.0, -60.0], sl - 6.0);
    skin.freq_axis(&sp, &pal, x_of, stp, sb, sb + 5.0);
    txt(&sp, pos2(sl - 6.0, sb), Align2::RIGHT_CENTER, "dB", mono(8.0), pal.alt, 0.0);
    let n = SPEC_POINTS;
    let u_at = |i: usize| (i as f32 + 0.5) / n as f32;
    // Output cuts: shade what the reverb won't keep.
    let (lc, hc) = (params.low_cut.value(), params.high_cut.value());
    if lc > LOW_CUT_OFF {
        sp.rect_filled(Rect::from_min_max(pos2(sl, stp), pos2(x_of(fu(lc)), sb)), 0.0, Color32::from_black_alpha(90));
    }
    if hc < HIGH_CUT_OFF {
        sp.rect_filled(Rect::from_min_max(pos2(x_of(fu(hc)), stp), pos2(sr_, sb)), 0.0, Color32::from_black_alpha(90));
    }
    // DUCK band: zone over the ducked frequencies, filling down with the dip.
    let duck_on = params.duck.value() > 0.0;
    if duck_on {
        let (dl, dh) = (params.duck_lo.value(), params.duck_hi.value());
        let x0 = if dl <= LOW_CUT_OFF { sl } else { x_of(fu(dl)) };
        let x1 = if dh >= HIGH_CUT_OFF { sr_ } else { x_of(fu(dh)) };
        let band = Rect::from_min_max(pos2(x0, stp), pos2(x1.max(x0 + 2.0), sb));
        sp.rect_filled(band, 0.0, Color32::from_white_alpha(8));
        let depth = y_of(duck_db.max(-72.0)) - y_of(0.0);
        if depth > 0.5 {
            sp.rect_filled(Rect::from_min_max(band.left_top(), pos2(band.right(), band.top() + depth)), 0.0, Color32::from_white_alpha(22));
        }
        dashed(&sp, band.left_top(), band.left_bottom(), Color32::from_white_alpha(110), 1.0, 2.0, 3.0);
        dashed(&sp, band.right_top(), band.right_bottom(), Color32::from_white_alpha(110), 1.0, 2.0, 3.0);
    }
    // Dry (grey line) then reverb (accent fill).
    let dry: Vec<Pos2> = (0..n).map(|i| pos2(x_of(u_at(i)), y_of(st.dry_db[i]))).collect();
    polyline(&sp, dry, Color32::from_rgba_unmultiplied(200, 220, 235, 115), 1.2, false);
    let wet: Vec<Pos2> = (0..n).map(|i| pos2(x_of(u_at(i)), y_of(st.wet_db[i]))).collect();
    fill_under(&sp, &wet, sb, pal.fill(0.22));
    polyline(&sp, wet, pal.trace, 1.8, pal.glow);
    // The cut response on the same dB scale (0 dB = untouched).
    if lc > LOW_CUT_OFF || hc < HIGH_CUT_OFF {
        let resp = |f: f32| -> f32 {
            let mut d = 0.0;
            if lc > LOW_CUT_OFF {
                d += response_db(true, lc, f, sr);
            }
            if hc < HIGH_CUT_OFF {
                d += response_db(false, hc, f, sr);
            }
            d
        };
        let curve: Vec<Pos2> = (0..=300)
            .map(|k| {
                let u = k as f32 / 300.0;
                pos2(x_of(u), y_of(resp(20.0 * 1000f32.powf(u)).max(-72.0)))
            })
            .collect();
        polyline(&sp, curve, al(CY, 0.85), 1.4, false);
        let fmt = |f: f32| if f >= 1000.0 { format!("{:.1} kHz", f / 1000.0) } else { format!("{f:.0} Hz") };
        if lc > LOW_CUT_OFF {
            let x = x_of(fu(lc));
            dashed(&sp, pos2(x, stp), pos2(x, sb), ACC, 1.0, 4.0, 3.0);
            txt(&sp, pos2(x + 4.0, stp + 4.0), Align2::LEFT_TOP, format!("LOW CUT {}", fmt(lc)), mono(9.0), ACC, 0.0);
        }
        if hc < HIGH_CUT_OFF {
            let x = x_of(fu(hc));
            dashed(&sp, pos2(x, stp), pos2(x, sb), ACC, 1.0, 4.0, 3.0);
            txt(&sp, pos2(x - 4.0, stp + 4.0), Align2::RIGHT_TOP, format!("HIGH CUT {}", fmt(hc)), mono(9.0), ACC, 0.0);
        }
    }
    // Title + legend.
    txt(&sp, pos2(r.left() + 22.0, r.top() + 17.0), Align2::LEFT_BOTTOM, "WHERE IT HITS // REVERB vs DRY", disp(8.0), rgb(0xcd, 0xef, 0xff), 2.0);
    let mut x = sr_;
    let ly = r.top() + 12.0;
    if duck_on {
        let rr = txt(&sp, pos2(x, ly), Align2::RIGHT_CENTER, format!("DUCK {duck_db:.1} dB"), mono(9.0), Color32::WHITE, 0.0);
        x = rr.left() - 14.0;
    }
    for (name, col) in [("DRY", Color32::from_rgba_unmultiplied(200, 220, 235, 160)), ("REVERB", ACC)] {
        let rr = txt(&sp, pos2(x, ly), Align2::RIGHT_CENTER, name, mono(9.0), pal.axis, 0.0);
        line(&sp, pos2(rr.left() - 18.0, ly), pos2(rr.left() - 4.0, ly), col, 2.0);
        x = rr.left() - 30.0;
    }
    skin.screen_end(painter, r);
}

/// One editor frame (pulled out of the update closure so tests can render it).
fn frame(ctx: &egui::Context, setter: &ParamSetter, state: &mut (Ui, Skin), params: &Arc<AlleyParams>, shared: &Arc<Shared>, loader: &Loader) {
            let (st, skin) = (&mut state.0, &mut state.1);
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
            skin.begin(ctx);
            let p = &params;
            let current = p.ir_path.read().map(|s| s.clone()).unwrap_or_default();
            if !st.scanned {
                scan(st);
                if let Some(it) = st.items.iter().find(|i| i.key == current) {
                    st.cat = it.cat;
                }
            }
            let load = |path: &PathBuf| {
                let s = path.to_string_lossy().to_string();
                if let Ok(mut w) = p.ir_path.write() {
                    *w = s.clone();
                }
                shared.loading.store(true, Ordering::Release);
                loader.load(s);
            };
            let sr = f32::from_bits(shared.sr.load(Ordering::Relaxed)).max(8000.0);
            pump_spectrum(st, &shared, sr);
            let algo = p.engine.value() == 1;

            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(rgb(0x04, 0x07, 0x0c))).show(ctx, |ui| {
                let whole = ui.max_rect();
                skin.background(ui.painter(), whole);
                let full = st.presets.strip(ui, setter, params.as_ref(), whole, skin, &p.mix, &p.bypass);
                if let Some(fields) = st.presets.pending_fields.take() {
                    if let Some(path) = fields.get("ir-path").and_then(|j| serde_json::from_str::<String>(j).ok()) {
                        if !path.is_empty() && path != current {
                            load(&PathBuf::from(path));
                        }
                    }
                }
                let painter = ui.painter().clone();
                let (l, r, t) = (full.left(), full.right(), full.top());
                let foot = Rect::from_min_max(pos2(l, full.bottom() - 30.0), full.right_bottom());
                // Main column ends here; the meters take the strip to the right.
                let mr = r - 100.0;

                // ---- SEC.01: IMPULSE / ALGO ----
                let stash = Rect::from_min_max(pos2(l + 16.0, t + 12.0), pos2(l + 466.0, t + 286.0));
                skin.panel(&painter, stash, "", "");
                for (i, label) in ["IMPULSE", "ALGO"].iter().enumerate() {
                    let cr = Rect::from_min_size(pos2(stash.left() + 12.0 + 92.0 * i as f32, stash.top() + 8.0), vec2(86.0, 22.0));
                    let on = (i == 1) == algo;
                    if skin.chip(ui, cr, label, on, ("tab", i)).clicked() && !on {
                        set_int(setter, &p.engine, i as i32);
                    }
                }
                if !algo {
                    impulse_browser(ui, &painter, skin, stash, st, &current, &load);
                } else {
                    algo_browser(ui, &painter, skin, stash, setter, p);
                }

                // ---- SEC.02: THE TAIL ----
                let tail = Rect::from_min_max(pos2(stash.right() + 12.0, stash.top()), pos2(r - 16.0, stash.bottom()));
                skin.panel(&painter, tail, "THE TAIL", "SEC.02");
                let send_on = p.send.value();
                // Centred in the header row, clear of the title and the SEC.02 tag.
                let par_r = Rect::from_center_size(pos2(tail.center().x, tail.top() + 18.0), vec2(150.0, 20.0));
                skin.chip_param(ui, setter, par_r, &p.send, ("PARALLEL: ON", "PARALLEL: OFF"))
                    .on_hover_text(if send_on { "Original passes untouched; the reverb runs on a copy (like a send)" } else { "DRY and OUTPUT shape the original" });
                let wave = Rect::from_min_max(pos2(tail.left() + 12.0, tail.top() + 80.0), pos2(tail.right() - 12.0, tail.top() + 220.0));
                let by = tail.top() + 248.0;
                let hi = rgb(0xf0, 0xfb, 0xff);
                let dim_txt = al(rgb(0x8f, 0xc9, 0xe3), 0.75);
                if !algo {
                    let (name, secs, rt0, rt1, env, err) = match shared.info.lock() {
                        Ok(i) => (i.name.clone(), i.secs, i.rt_orig, i.rt_new, i.env.clone(), i.error.clone()),
                        Err(_) => (String::new(), 0.0, 0.0, 0.0, Vec::new(), None),
                    };
                    let loading = shared.loading.load(Ordering::Acquire);
                    let cat = st.items.iter().find(|i| i.key == current).map(|i| st.cats[i.cat].clone()).unwrap_or_else(|| "IMPULSE".into());
                    fit_txt(&painter, pos2(tail.left() + 16.0, tail.top() + 56.0), &clean_name(&name).to_uppercase(), disp(15.0), hi, 2.0, tail.width() - 32.0);
                    txt(&painter, pos2(tail.left() + 16.0, tail.top() + 72.0), Align2::LEFT_BOTTOM, format!("{cat}  ·  {}", if p.reverse.value() { "REVERSED" } else { "FORWARD" }), mono(9.0), dim_txt, 1.0);
                    draw_tail_ir(&painter, skin, wave, &env, secs, rt1, p.freeze.value());
                    let sub = if loading {
                        "LOADING...".to_string()
                    } else if let Some(e) = err {
                        e
                    } else if (rt1 - rt0).abs() > 0.005 {
                        format!("DECAY {rt0:.2} s  ›  {rt1:.2} s     TAIL {secs:.1} s")
                    } else {
                        format!("DECAY {rt0:.2} s     TAIL {secs:.1} s")
                    };
                    txt(&painter, pos2(tail.left() + 16.0, tail.top() + 238.0), Align2::LEFT_BOTTOM, sub, mono(10.0), rgb(0xcd, 0xef, 0xff), 1.0);
                    let list = visible(st);
                    let idx = list.iter().position(|&i| st.items[i].key == current);
                    let step = |d: isize| -> Option<PathBuf> {
                        if list.is_empty() {
                            return None;
                        }
                        let n = list.len() as isize;
                        let i = idx.map(|i| (i as isize + d).rem_euclid(n)).unwrap_or(0) as usize;
                        Some(st.items[list[i]].path.clone())
                    };
                    let mut go = None;
                    if skin.chip(ui, Rect::from_min_size(pos2(tail.left() + 16.0, by), vec2(64.0, 18.0)), "‹ PREV", false, "prev").clicked() {
                        go = step(-1);
                    }
                    if skin.chip(ui, Rect::from_min_size(pos2(tail.left() + 86.0, by), vec2(64.0, 18.0)), "NEXT ›", false, "next").clicked() {
                        go = step(1);
                    }
                    // Arrow keys step too (when not typing in the search box).
                    if !ctx.wants_keyboard_input() {
                        if ui.input(|i| i.key_pressed(Key::ArrowDown) || i.key_pressed(Key::ArrowRight)) {
                            go = step(1);
                        } else if ui.input(|i| i.key_pressed(Key::ArrowUp) || i.key_pressed(Key::ArrowLeft)) {
                            go = step(-1);
                        }
                    }
                    if let Some(f) = go {
                        load(&f);
                    }
                    skin.chip_param(ui, setter, Rect::from_min_size(pos2(tail.left() + 156.0, by), vec2(110.0, 18.0)), &p.reverse, ("REVERSE: ON", "REVERSE: OFF"));
                } else {
                    let mode = p.algo_mode.value().clamp(0, 11) as usize;
                    let color = COLOR_NAMES[p.algo_color.value().clamp(0, 2) as usize];
                    txt(&painter, pos2(tail.left() + 16.0, tail.top() + 56.0), Align2::LEFT_BOTTOM, MODE_NAMES[mode], disp(15.0), hi, 2.0);
                    fit_txt(&painter, pos2(tail.left() + 16.0, tail.top() + 72.0), &format!("{color}  ·  ALGO ENGINE  ·  {}", MODE_INFO[mode].to_uppercase()), mono(9.0), dim_txt, 1.0, tail.width() - 32.0);
                    draw_tail_algo(&painter, skin, wave, p, mode == 11);
                    txt(&painter, pos2(tail.left() + 16.0, tail.top() + 238.0), Align2::LEFT_BOTTOM, format!("DECAY {:.2} s     PRE-DELAY {:.0} ms", p.decay.value(), p.predelay.value()), mono(10.0), rgb(0xcd, 0xef, 0xff), 1.0);
                    if skin.chip(ui, Rect::from_min_size(pos2(tail.left() + 16.0, by), vec2(64.0, 18.0)), "‹ PREV", false, "aprev").clicked() {
                        set_int(setter, &p.algo_mode, ((mode + 11) % 12) as i32);
                    }
                    if skin.chip(ui, Rect::from_min_size(pos2(tail.left() + 86.0, by), vec2(64.0, 18.0)), "NEXT ›", false, "anext").clicked() {
                        set_int(setter, &p.algo_mode, ((mode + 1) % 12) as i32);
                    }
                }
                freeze_button(ui, setter, Rect::from_min_size(pos2(tail.right() - 138.0, tail.top() + 226.0), vec2(126.0, 40.0)), &p.freeze);

                // ---- SEC.03: WHERE IT HITS ----
                let spec = Rect::from_min_max(pos2(l + 16.0, t + 298.0), pos2(mr, t + 448.0));
                let duck_db = f32::from_bits(shared.duck_db.load(Ordering::Relaxed));
                draw_spectrum(&painter, skin, spec, st, p, duck_db, sr);

                // ---- METERS ----
                let mp = Rect::from_min_max(pos2(mr + 12.0, spec.top()), pos2(r - 16.0, t + 756.0));
                skin.panel(&painter, mp, "", "");
                let m = &shared.meters;
                let (my, mh) = (mp.top() + 40.0, mp.height() - 66.0);
                skin.meter(&painter, mp.left() + 9.0, my, mh, "IN", m.db(0), m.db(1), Side::Right, "in");
                skin.meter(&painter, mp.left() + 49.0, my, mh, "OUT", m.db(2), m.db(3), Side::None, "out");

                // ---- SEC.04 SPACE | SEC.05 DUCK ----
                let space_r = Rect::from_min_max(pos2(l + 16.0, t + 460.0), pos2(l + 540.0, t + 580.0));
                skin.panel(&painter, space_r, "SPACE", "SEC.04");
                let ky = space_r.top() + 62.0;
                let sx = |r: Rect, n: usize, i: usize| r.left() + r.width() * (i as f32 + 0.5) / n as f32;
                let space: [(&str, &FloatParam, bool); 5] = [("LENGTH", &p.length, algo), ("SIZE", &p.size, false), ("DECAY", &p.decay, false), ("PRE-DELAY", &p.predelay, false), ("WIDTH", &p.width, false)];
                for (i, (lab_, prm, dim)) in space.iter().enumerate() {
                    skin.knob(ui, setter, pos2(sx(space_r, 5, i), ky), 20.0, lab_, *prm, false, false, *dim);
                }
                if !algo {
                    skin.chip_param(ui, setter, Rect::from_min_size(pos2(space_r.left() + 94.0, space_r.top() + 6.0), vec2(158.0, 18.0)), &p.decay_orig, ("DECAY: IMPULSE'S OWN", "DECAY: SET BY KNOB"));
                }
                // Pre-delay tempo sync (cycles OFF, 1/64 ... 1/4D).
                let pd = p.pre_div.value().clamp(0, 8);
                if skin.chip(ui, Rect::from_min_size(pos2(space_r.left() + 260.0, space_r.top() + 6.0), vec2(128.0, 18.0)), &format!("PRE SYNC: {} ›", PRE_DIVS[pd as usize]), pd != 0, "prediv").clicked() {
                    set_int(setter, &p.pre_div, (pd + 1) % 9);
                }
                let duck_r = Rect::from_min_max(pos2(space_r.right() + 12.0, space_r.top()), pos2(mr, space_r.bottom()));
                skin.panel(&painter, duck_r, "DUCK", "SEC.05");
                let duck: [(&str, &FloatParam); 3] = [("DUCK", &p.duck), ("DUCK LOW", &p.duck_lo), ("DUCK HIGH", &p.duck_hi)];
                for (i, (lab_, prm)) in duck.iter().enumerate() {
                    skin.knob(ui, setter, pos2(sx(duck_r, 3, i), ky), 20.0, lab_, *prm, false, false, false);
                }
                let src = p.duck_src.value().clamp(0, 2);
                if skin.chip(ui, Rect::from_min_size(pos2(duck_r.left() + 76.0, duck_r.top() + 6.0), vec2(124.0, 18.0)), &format!("HEARS: {} ›", DUCK_SOURCES[src as usize]), src != 0, "ducksrc").clicked() {
                    set_int(setter, &p.duck_src, (src + 1) % 3);
                }

                // ---- SEC.06 EXTRAS: tabbed row ----
                let ex = Rect::from_min_max(pos2(l + 16.0, t + 592.0), pos2(mr, t + 756.0));
                skin.panel(&painter, ex, "EXTRAS", "");
                let ey = ex.top() + 82.0;
                let tabs = ["ALGO", "SHIMMER", "RISER + SWELL", "INPUT EQ"];
                for (i, tname) in tabs.iter().enumerate() {
                    let cr = Rect::from_min_size(pos2(ex.left() + 94.0 + i as f32 * 124.0, ex.top() + 6.0), vec2(118.0, 20.0));
                    if skin.chip(ui, cr, tname, st.extra_tab == i, ("xtab", i)).clicked() {
                        st.extra_tab = i;
                    }
                    let lit = match i {
                        1 => p.shimmer.value() > 0.0,
                        2 => p.riser.value() > 0.0 || p.swell.value(),
                        3 => p.in_lo.value() > LOW_CUT_OFF || p.in_hi.value() < HIGH_CUT_OFF,
                        _ => algo,
                    };
                    if lit {
                        painter.circle_filled(cr.right_top() + vec2(-6.0, 6.0), 4.5, al(ACC, 0.25));
                        painter.circle_filled(cr.right_top() + vec2(-6.0, 6.0), 2.2, ACC);
                    }
                }
                let wide = |i: usize| ex.left() + 70.0 + i as f32 * 140.0;
                match st.extra_tab {
                    0 => {
                        let ak: [(&str, &FloatParam); 9] = [
                            ("ATTACK", &p.attack),
                            ("EARLY DIFF", &p.early_diff),
                            ("LATE DIFF", &p.late_diff),
                            ("MOD RATE", &p.mod_rate),
                            ("MOD DEPTH", &p.mod_depth),
                            ("BASS MULT", &p.bass_mult),
                            ("BASS XOVER", &p.bass_xover),
                            ("HIGH SHELF", &p.high_shelf),
                            ("HIGH FREQ", &p.high_freq),
                        ];
                        for (i, (lab_, prm)) in ak.iter().enumerate() {
                            skin.knob(ui, setter, pos2(sx(ex, 9, i), ey), 18.0, lab_, *prm, false, false, !algo);
                        }
                        if !algo {
                            skin.small(&painter, pos2(ex.right() - 14.0, ex.top() + 16.0), Align2::RIGHT_CENTER, "ALGO ENGINE ONLY · ALGO TAB, TOP LEFT", skin.text_faint());
                        }
                    }
                    1 => {
                        let k: [(&str, &FloatParam); 4] = [("SHIMMER", &p.shimmer), ("PITCH", &p.shim_pitch), ("FEEDBACK", &p.shim_fb), ("TONE", &p.shim_tone)];
                        for (i, (lab_, prm)) in k.iter().enumerate() {
                            skin.knob(ui, setter, pos2(wide(i), ey), 18.0, lab_, *prm, false, false, false);
                        }
                    }
                    2 => {
                        let k: [(&str, &FloatParam); 5] = [("RISER", &p.riser), ("SWELL LENGTH", &p.swell_len), ("SWELL LEVEL", &p.swell_amt), ("SENSITIVITY", &p.swell_sens), ("CURVE", &p.swell_curve)];
                        let swell = p.swell.value();
                        for (i, (lab_, prm)) in k.iter().enumerate() {
                            skin.knob(ui, setter, pos2(wide(i), ey), 18.0, lab_, *prm, false, false, i > 0 && !swell);
                        }
                        let pdc = swell_lookahead(sr) as f32 / sr;
                        skin.chip_param(ui, setter, Rect::from_min_size(pos2(ex.right() - 202.0, ex.top() + 6.0), vec2(190.0, 20.0)), &p.swell, (&format!("SWELL: ON (+{pdc:.1} s PDC)"), "SWELL: OFF"));
                    }
                    _ => {
                        let k: [(&str, &FloatParam); 2] = [("INPUT LOW CUT", &p.in_lo), ("INPUT HIGH CUT", &p.in_hi)];
                        for (i, (lab_, prm)) in k.iter().enumerate() {
                            skin.knob(ui, setter, pos2(wide(i), ey), 18.0, lab_, *prm, false, false, false);
                        }
                        let nx = ex.left() + 370.0;
                        let c = rgb(0xcd, 0xef, 0xff);
                        let d = al(rgb(0x8f, 0xc9, 0xe3), 0.75);
                        txt(&painter, pos2(nx, ey - 26.0), Align2::LEFT_CENTER, "SHAPES WHAT GOES INTO THE REVERB", disp(8.5), c, 2.0);
                        skin.small(&painter, pos2(nx, ey - 4.0), Align2::LEFT_CENTER, "Cut lows going in = no mud building up in the tail.", d);
                        skin.small(&painter, pos2(nx, ey + 12.0), Align2::LEFT_CENTER, "Cut highs going in = no harsh esses or hats in the tail.", d);
                        skin.small(&painter, pos2(nx, ey + 28.0), Align2::LEFT_CENTER, "LOW/HIGH CUT in MIX shape what comes out.", d);
                    }
                }

                // ---- SEC.07 MIX ----
                let mix_r = Rect::from_min_max(pos2(l + 16.0, t + 768.0), pos2(r - 16.0, foot.top() - 10.0));
                skin.panel(&painter, mix_r, "MIX", "SEC.07");
                let mx: [(&str, &FloatParam, bool); 5] = [
                    (if send_on { "DRY (LOCKED)" } else { "DRY" }, &p.dry, send_on),
                    ("WET", &p.wet, false),
                    ("LOW CUT", &p.low_cut, false),
                    ("HIGH CUT", &p.high_cut, false),
                    (if send_on { "OUTPUT (REVERB)" } else { "OUTPUT" }, &p.output, false),
                ];
                let mky = mix_r.top() + 56.0;
                for (i, (lab_, prm, dim)) in mx.iter().enumerate() {
                    skin.knob(ui, setter, pos2(sx(mix_r, 5, i), mky), 26.0, lab_, *prm, true, false, *dim);
                }

                // ---- footer ----
                let lat = BLOCK + if p.swell.value() { swell_lookahead(sr) } else { 0 };
                skin.footer(&painter, foot, foot.left() + 300.0);
                skin.safe_chip(ui, setter, Rect::from_center_size(pos2(foot.left() + 226.0, foot.center().y), vec2(112.0, 18.0)), &p.safe, shared.meters.db(4));
                skin.small(&painter, pos2(foot.left() + 16.0, foot.center().y), Align2::LEFT_CENTER, format!("PDC {lat} SMP · {:.0} ms", lat as f32 / sr * 1000.0), rgb(0x9f, 0xdc, 0xf5));
            });
        }

#[cfg(test)]
mod shot_tests {
    use super::*;
    use crate::shot::*;
    use std::collections::HashMap;
    /// SHOT_DIR=/src/target/shots cargo test --lib screenshots  ->  alleyverb0.rgb
    #[test]
    fn screenshots() {
        let Ok(dir) = std::env::var("SHOT_DIR") else { return };
        let plug = crate::AlleyVerb::default();
        let (params, shared) = (plug.params.clone(), plug.shared.clone());
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
    let ui_state = Ui {
        cats: Vec::new(),
        items: Vec::new(),
        favs: BTreeSet::new(),
        scanned: false,
        cat: 0,
        search: String::new(),
        presets: PresetBar::new("AlleyVerb"),
        extra_tab: 0,
        spec: fft.make_output_vec(),
        fft,
        ring_wet: vec![0.0; FFT],
        ring_dry: vec![0.0; FFT],
        pos: 0,
        tbuf: vec![0.0; FFT],
        wet_db: vec![-120.0; SPEC_POINTS],
        dry_db: vec![-120.0; SPEC_POINTS],
    };
    let skin = Skin::new(Look::Hud, ACC, "ALLEYVERB", env!("CARGO_PKG_VERSION"), "IMPULSE + ALGO // FREEZE // SHIMMER // RISER + SWELL // DUCK");
        let mut state = (ui_state, skin);
        let loader = Loader::new(params.clone(), shared.clone());
        let ctx = egui::Context::default();
        install_fonts(&ctx);
        let mock = Mock;
        let setter = ParamSetter::new(&mock);
        let mut texs = HashMap::new();
        let mut out = None;
        for i in 0..4 {
            out = Some(ctx.run(raw(i, vec![]), |c| frame(c, &setter, &mut state, &params, &shared, &loader)));
            if i < 3 { apply_delta(&mut texs, out.as_ref().unwrap()); }
        }
        render(&ctx, out.unwrap(), &mut texs, &format!("{dir}/alleyverb0.rgb"));
    }
}
