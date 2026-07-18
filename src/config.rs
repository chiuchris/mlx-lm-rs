use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    Float(f32),
    String(String),
    Vec(Vec<f32>),
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

impl Qwen3Config {
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
    }
}

fn default_tie() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multi(Vec<u32>),
}

impl EosTokenId {
    pub fn ids(&self) -> Vec<u32> {
        match self {
            Self::Single(x) => vec![*x],
            Self::Multi(xs) => xs.clone(),
        }
    }
}

pub fn load_config(model_dir: impl AsRef<Path>) -> Result<Qwen3Config> {
    let path = model_dir.as_ref().join("config.json");
    let s = std::fs::read_to_string(&path)?;
    let cfg: Qwen3Config = serde_json::from_str(&s)?;
    if cfg.model_type != "qwen3" {
        return Err(Error::Config(format!(
            "expected model_type=qwen3, got {}",
            cfg.model_type
        )));
    }
    if let Some(quantization) = cfg.quantization() {
        if quantization.group_size <= 0 {
            return Err(Error::Config(format!(
                "quantization group_size must be positive, got {}",
                quantization.group_size
            )));
        }
        if !matches!(quantization.bits, 2 | 4 | 8) {
            return Err(Error::Config(format!(
                "quantization bits must be 2, 4, or 8, got {}",
                quantization.bits
            )));
        }
    }
    Ok(cfg)
}
