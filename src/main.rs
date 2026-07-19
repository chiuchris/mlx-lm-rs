use std::io::Write;
use std::num::NonZeroUsize;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

fn parse_temp(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("not a float: {e}"))?;
    if v == 0.0 || (v.is_finite() && v > 0.0) {
        Ok(v)
    } else {
        Err(format!(
            "temperature must be 0.0 or a finite positive value, got {v}"
        ))
    }
}

use mlx_lm_rs::{
    chat_template::ChatTemplate,
    config::load_config,
    generate::Generator,
    loader::{list_weight_files, resolve_model_dir},
    models::qwen3::Model,
    server::{serve, ServeConfig},
    stats::{fmt_peak_memory, fmt_throughput, peak_memory_bytes, reset_peak_memory},
    tokenizer::load_tokenizer,
};

#[derive(Parser)]
#[command(
    name = "mlx-lm-rs",
    version,
    about = "Greenfield Rust port of mlx-lm — Qwen3 inference"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate text from a prompt.
    Generate {
        /// Local model directory or HF repo (e.g. mlx-community/Qwen3-4B-bf16).
        #[arg(long, default_value = "mlx-community/Qwen3-4B-bf16")]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0, value_parser = parse_temp)]
        temp: f32,
        /// Repetition penalty (1.0 = no penalty, > 1.0 discourages repeats).
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value = "2048")]
        prefill_step_size: NonZeroUsize,
        /// Skip applying the chat template (use raw prompt).
        #[arg(long)]
        no_chat_template: bool,
        /// Suppress timing/peak-memory stats at the end.
        #[arg(long)]
        no_stats: bool,
        /// Don't flush stdout per-token. Useful for benchmarking — saves the
        /// per-token sync flush (~1ms/token on a TTY).
        #[arg(long)]
        no_stream: bool,
    },
    /// Serve an OpenAI-compatible HTTP API.
    Serve {
        /// Local model directory or HF repo (e.g. mlx-community/Qwen3-4B-bf16).
        #[arg(long, default_value = "mlx-community/Qwen3-4B-bf16")]
        model: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Default generation length and maximum accepted per-request cap.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0, value_parser = parse_temp)]
        temp: f32,
        /// Default repetition penalty (1.0 = no penalty, > 1.0 discourages repeats).
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value = "2048")]
        prefill_step_size: NonZeroUsize,
        /// Skip applying the chat template (render messages as plain text).
        #[arg(long)]
        no_chat_template: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate {
            model,
            prompt,
            max_tokens,
            temp,
            rep_penalty,
            prefill_step_size,
            no_chat_template,
            no_stats,
            no_stream,
        } => {
            let model_dir = resolve_model_dir(&model).context("resolving model dir")?;
            eprintln!("[mlx-lm-rs] using model dir: {}", model_dir.display());

            let cfg = load_config(&model_dir).context("loading config.json")?;
            let tokenizer = load_tokenizer(&model_dir).context("loading tokenizer.json")?;

            // `add_special_tokens` mirrors upstream mlx-lm: when the chat
            // template runs, its rendered output already contains BOS/system
            // specials, so the tokenizer must not add them again.
            let (final_prompt, add_special_tokens) = if no_chat_template {
                (prompt, true)
            } else {
                match ChatTemplate::load(&model_dir).context("loading chat template")? {
                    Some(t) => (
                        t.render(&prompt, true).context("rendering chat template")?,
                        false,
                    ),
                    None => (prompt, true),
                }
            };

            let encoding = tokenizer
                .encode(final_prompt.as_str(), add_special_tokens)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
            let prompt_len = prompt_ids.len();
            eprintln!("[mlx-lm-rs] prompt tokens: {prompt_len}");

            let mut model_obj = Model::new(cfg.clone()).context("constructing model")?;
            let shards = list_weight_files(&model_dir).context("listing weight files")?;
            eprintln!("[mlx-lm-rs] loading {} weight shard(s)…", shards.len());
            model_obj.load_weights(&shards).context("loading weights")?;
            eprintln!("[mlx-lm-rs] weights loaded; starting generation");

            // Reset MLX peak counter so it captures only the generation.
            reset_peak_memory();

            let eos_ids = cfg
                .eos_token_id
                .as_ref()
                .map(|x| x.ids())
                .unwrap_or_default();

            // Match python mlx-lm: `prompt_secs` runs until the first token
            // is materialized (yielded), not until the first decode graph is
            // built. The first iterator pull below blocks on `cur.item()`.
            let prefill_start = Instant::now();
            let mut gen = Generator::new(
                &mut model_obj,
                &prompt_ids,
                max_tokens,
                temp,
                rep_penalty,
                eos_ids,
                prefill_step_size,
            )?;

            let mut stream = tokenizer.decode_stream(false);
            let mut stdout = std::io::stdout().lock();
            let mut produced = 0usize;

            let first = gen.next().transpose()?;
            let prefill_secs = prefill_start.elapsed().as_secs_f64();
            let decode_start = Instant::now();

            let mut emit = |s: &str| -> Result<()> {
                write!(stdout, "{s}")?;
                if !no_stream {
                    stdout.flush()?;
                }
                Ok(())
            };
            for tok in first.into_iter().map(Ok).chain(gen) {
                let tok = tok?;
                produced += 1;
                if let Some(s) = stream
                    .step(tok)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?
                {
                    emit(&s)?;
                }
            }
            let decode_secs = decode_start.elapsed().as_secs_f64();
            writeln!(stdout)?;

            if !no_stats {
                eprintln!("==========");
                eprintln!("{}", fmt_throughput("Prompt", prompt_len, prefill_secs));
                eprintln!("{}", fmt_throughput("Generation", produced, decode_secs));
                eprintln!("{}", fmt_peak_memory(peak_memory_bytes()));
            }
            Ok(())
        }
        Cmd::Serve {
            model,
            host,
            port,
            max_tokens,
            temp,
            rep_penalty,
            prefill_step_size,
            no_chat_template,
        } => {
            serve(ServeConfig {
                model,
                host,
                port,
                default_max_tokens: max_tokens,
                default_temperature: temp,
                default_rep_penalty: rep_penalty,
                prefill_step_size,
                no_chat_template,
            })
            .await
        }
    }
}
