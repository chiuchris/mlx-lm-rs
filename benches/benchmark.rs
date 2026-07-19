//! In-process throughput benchmark for `mlx-lm-rs`.
//!
//! The model + tokenizer + chat template are loaded ONCE (lazy `OnceLock`)
//! and reused across every criterion sample. Each `iter_custom` iteration
//! runs the same generation pipeline as `main.rs` but skips:
//!   * subprocess spawn
//!   * weight reload from safetensors
//!   * disk page-cache warmup
//!   * stdout streaming
//!
//! Two metric families per prompt size:
//!   * `prefill/<size>` — prompt tokens/sec (until first output token)
//!   * `decode/<size>`  — generation tokens/sec (excluding first token)
//!
//! Run:
//!   cargo bench --bench benchmark
//!   cargo bench --bench benchmark -- decode/short
//!   BENCH_MODEL=mlx-community/Qwen3-4B-bf16 cargo bench --bench benchmark

use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tokenizers::Tokenizer;

use mlx_lm_rs::{
    chat_template::ChatTemplate,
    config::load_config,
    generate::Generator,
    loader::{list_weight_files, resolve_model_dir},
    models::qwen3::Model,
    tokenizer::load_tokenizer,
};

const DEFAULT_MODEL: &str = "mlx-community/Qwen3-0.6B-bf16";
const DEFAULT_MAX_TOKENS: usize = 256;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
fn model_id() -> String {
    env_or("BENCH_MODEL", DEFAULT_MODEL)
}
fn max_tokens() -> usize {
    env_or("BENCH_MAX_TOKENS", &DEFAULT_MAX_TOKENS.to_string())
        .parse()
        .expect("BENCH_MAX_TOKENS must be a usize")
}

struct Loaded {
    model: Model,
    tokenizer: Tokenizer,
    chat: Option<ChatTemplate>,
    eos_ids: Vec<u32>,
}

fn loaded() -> &'static Mutex<Loaded> {
    static LOADED: OnceLock<Mutex<Loaded>> = OnceLock::new();
    LOADED.get_or_init(|| Mutex::new(load_once()))
}

fn load_once() -> Loaded {
    let id = model_id();
    eprintln!("[bench] loading {id} (one-time) …");
    let dir = resolve_model_dir(&id).expect("resolve model dir");
    let cfg = load_config(&dir).expect("config");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");
    let chat = ChatTemplate::load(&dir).expect("chat template load");
    let eos_ids = cfg
        .eos_token_id
        .as_ref()
        .map(|x| x.ids())
        .unwrap_or_default();
    let mut model = Model::new(cfg).expect("construct model");
    let shards = list_weight_files(&dir).expect("list weight files");
    model.load_weights(&shards).expect("load weights");
    eprintln!("[bench] loaded.");
    Loaded {
        model,
        tokenizer,
        chat,
        eos_ids,
    }
}

#[derive(Debug, Clone)]
struct RunStats {
    prompt_tokens: u32,
    prompt_secs: f64,
    gen_tokens: u32,
    gen_secs: f64,
}

fn render_and_encode(g: &Loaded, prompt: &str) -> Vec<u32> {
    let (final_prompt, chat_applied) = match &g.chat {
        Some(t) => (t.render(prompt, true).expect("render"), true),
        None => (prompt.to_string(), false),
    };
    let enc = g
        .tokenizer
        .encode(final_prompt, !chat_applied)
        .expect("encode");
    enc.get_ids().to_vec()
}

fn run_once(prompt: &str) -> RunStats {
    let mtx = loaded();
    let mut g = mtx.lock().expect("loaded mutex");
    let ids = render_and_encode(&g, prompt);
    let eos_ids = g.eos_ids.clone();
    let max_tok = max_tokens();
    let step = NonZeroUsize::new(2048).unwrap();

    let prefill_start = Instant::now();
    let mut gen =
        Generator::new(&mut g.model, &ids, max_tok, 0.0, 1.0, eos_ids, step).expect("generator");
    let _ = gen.next().transpose().expect("first token");
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    // The first token was materialized inside the prefill timer (its wait
    // counts as time-to-first-token). Decode timing/count covers only
    // tokens produced after that boundary.
    let decode_start = Instant::now();
    let mut produced = 0usize;
    for tok in gen {
        tok.expect("token");
        produced += 1;
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();

    RunStats {
        prompt_tokens: ids.len() as u32,
        prompt_secs: prefill_secs,
        gen_tokens: produced as u32,
        gen_secs: decode_secs,
    }
}

fn long_prompt() -> String {
    let body = "The quick brown fox jumps over the lazy dog. ".repeat(60);
    format!("Summarize the following text in two sentences.\n\n{body}")
}

fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("short", "The capital of France is".to_string()),
        (
            "medium",
            "Explain in three short paragraphs how a transformer language model generates \
             text token by token, including how attention works and why a key/value cache \
             speeds up generation."
                .to_string(),
        ),
        ("long", long_prompt()),
    ]
}

fn timed(iters: u64, prompt: &str, pick_secs: fn(&RunStats) -> f64) -> Duration {
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let s = run_once(prompt);
        total += Duration::from_secs_f64(pick_secs(&s));
    }
    total
}

fn bench_prefill(c: &mut Criterion) {
    for (label, prompt) in prompts() {
        let probe = run_once(&prompt);
        let mut g = c.benchmark_group(format!("prefill/{label}"));
        g.sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(20))
            .throughput(Throughput::Elements(probe.prompt_tokens as u64));
        let p = prompt.clone();
        g.bench_function("rust", move |b| {
            b.iter_custom(|iters| timed(iters, &p, |s| s.prompt_secs));
        });
        g.finish();
    }
}

fn bench_decode(c: &mut Criterion) {
    for (label, prompt) in prompts() {
        let probe = run_once(&prompt);
        let mut g = c.benchmark_group(format!("decode/{label}"));
        g.sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(60))
            .throughput(Throughput::Elements(probe.gen_tokens as u64));
        let p = prompt.clone();
        g.bench_function("rust", move |b| {
            b.iter_custom(|iters| timed(iters, &p, |s| s.gen_secs));
        });
        g.finish();
    }
}

criterion_group!(benches, bench_prefill, bench_decode);
criterion_main!(benches);
