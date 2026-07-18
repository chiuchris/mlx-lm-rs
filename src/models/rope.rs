use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    nn::{Rope, RopeBuilder},
};

use crate::config::RopeScalingValue;
use crate::error::{Error, Result};

/// Build a RoPE module from Qwen3 config. Supports default, linear, and yarn
/// scaling types — ds-8b and other extended-context DeepSeek-R1 distills use YaRN.
pub fn build_rope(
    head_dim: i32,
    rope_theta: f32,
    rope_scaling: &Option<HashMap<String, RopeScalingValue>>,
) -> Result<Rope> {
    let scale = match rope_scaling {
        None => 1.0,
        Some(cfg) => {
            let ty = cfg
                .get("type")
                .or_else(|| cfg.get("rope_type"))
                .and_then(|v| match v {
                    RopeScalingValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("default");
            match ty {
                "default" => 1.0,
                "linear" | "yarn" => {
                    let factor = cfg
                        .get("factor")
                        .and_then(|v| match v {
                            RopeScalingValue::Float(f) => Some(*f),
                            _ => None,
                        })
                        .unwrap_or(1.0);
                    1.0 / factor
                }
                other => {
                    return Err(Error::Config(format!(
                        "unsupported rope_type {other:?} (only default+linear+yarn in this slice)"
                    )))
                }
            }
        }
    };
    RopeBuilder::new(head_dim)
        .traditional(false)
        .base(rope_theta)
        .scale(scale)
        .build()
        .map_err(|_: std::convert::Infallible| Error::Config("rope build".into()))
}
