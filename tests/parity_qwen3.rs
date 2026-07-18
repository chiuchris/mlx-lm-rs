//! Greedy parity check vs python `mlx_lm.generate` on the same Qwen3 model.
//!
//! This test is `#[ignore]` by default. Its default case requires:
//!   * `pip install mlx-lm` available on PATH (`python3 -m mlx_lm generate`)
//!   * `mlx-community/Qwen3-0.6B-bf16` already downloaded into the local HF cache
//!
//! Override `MLX_LM_RS_PARITY_MODEL`, `MLX_LM_RS_PARITY_PROMPT`,
//! `MLX_LM_RS_PARITY_MAX_TOKENS`, and `MLX_LM_RS_PARITY_PYTHON` to probe another
//! cached checkpoint without weakening the exact output comparison.

use std::process::Command;

const DEFAULT_MODEL: &str = "mlx-community/Qwen3-0.6B-bf16";
const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_MAX_TOKENS: usize = 16;
const DEFAULT_PYTHON: &str = "python3";

struct ParityCase {
    model: String,
    prompt: String,
    max_tokens: usize,
    python: String,
}

impl ParityCase {
    fn from_env() -> Self {
        let max_tokens = std::env::var("MLX_LM_RS_PARITY_MAX_TOKENS")
            .map(|value| {
                value
                    .parse()
                    .expect("MLX_LM_RS_PARITY_MAX_TOKENS must be a positive integer")
            })
            .unwrap_or(DEFAULT_MAX_TOKENS);
        assert!(max_tokens > 0, "parity max_tokens must be positive");

        Self {
            model: std::env::var("MLX_LM_RS_PARITY_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            prompt: std::env::var("MLX_LM_RS_PARITY_PROMPT")
                .unwrap_or_else(|_| DEFAULT_PROMPT.to_string()),
            max_tokens,
            python: std::env::var("MLX_LM_RS_PARITY_PYTHON")
                .unwrap_or_else(|_| DEFAULT_PYTHON.to_string()),
        }
    }
}

fn run_python_greedy(case: &ParityCase) -> String {
    let max_tokens = case.max_tokens.to_string();
    let out = Command::new(&case.python)
        .args([
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            &case.model,
            "--prompt",
            &case.prompt,
            "--max-tokens",
            &max_tokens,
            "--temp",
            "0",
        ])
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "{} -m mlx_lm generate failed to launch: {error}",
                case.python
            )
        });
    assert!(
        out.status.success(),
        "python mlx_lm exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn run_rust_greedy(case: &ParityCase) -> String {
    let bin = env!("CARGO_BIN_EXE_mlx-lm-rs");
    let max_tokens = case.max_tokens.to_string();
    let out = Command::new(bin)
        .args([
            "generate",
            "--model",
            &case.model,
            "--prompt",
            &case.prompt,
            "--max-tokens",
            &max_tokens,
            "--temp",
            "0",
        ])
        .output()
        .expect("our binary failed to launch");
    assert!(
        out.status.success(),
        "rust mlx-lm-rs exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Extract just the generated text. Python wraps it in `==========` lines and
/// then prints stats; ours streams generation to stdout and logs to stderr.
fn extract_generated(stream: &str, is_python: bool) -> String {
    if is_python {
        let mut in_block = false;
        let mut out = String::new();
        for line in stream.lines() {
            if line.starts_with("==========") {
                if in_block {
                    break;
                }
                in_block = true;
                continue;
            }
            if in_block {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(line);
            }
        }
        out.trim().to_string()
    } else {
        stream.trim().to_string()
    }
}

#[test]
#[ignore = "requires python mlx_lm and a downloaded Qwen3 checkpoint"]
fn greedy_matches_python() {
    let case = ParityCase::from_env();
    let python = run_python_greedy(&case);
    let rust = run_rust_greedy(&case);
    let python_generated = extract_generated(&python, true);
    let rust_generated = extract_generated(&rust, false);
    eprintln!("model:  {:?}", case.model);
    eprintln!("prompt: {:?}", case.prompt);
    eprintln!("python: {python_generated:?}");
    eprintln!("rust:   {rust_generated:?}");
    assert_eq!(
        python_generated, rust_generated,
        "rust greedy output should match python mlx_lm.generate token-for-token"
    );
}
