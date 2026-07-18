use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    fast,
    module::Module,
    nn::{self, Rope as StandardRope, RopeBuilder},
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp},
    },
    Array,
};

use crate::config::RopeScalingValue;
use crate::error::{Error, Result};

const DEFAULT_BETA_FAST: f32 = 32.0;
const DEFAULT_BETA_SLOW: f32 = 1.0;
const DEFAULT_MSCALE: f32 = 1.0;
const DEFAULT_MSCALE_ALL_DIM: f32 = 0.0;

#[derive(Debug, Clone)]
struct YarnParameters {
    factor: f32,
    original_max_position_embeddings: f32,
    beta_fast: f32,
    beta_slow: f32,
    mscale: f32,
    mscale_all_dim: f32,
}

#[derive(Debug, Clone)]
enum Scaling {
    Standard { scale: f32 },
    Yarn(YarnParameters),
}

#[derive(Debug, Clone)]
struct YarnRope {
    dims: i32,
    mscale: f32,
    freqs: Array,
}

/// RoPE implementation used by Qwen3. Default and linear scaling delegate to
/// mlx-rs; YaRN supplies Python-compatible custom denominator frequencies.
#[derive(Debug, Clone)]
pub struct Rope(RopeKind);

#[derive(Debug, Clone)]
enum RopeKind {
    Standard(StandardRope),
    Yarn(YarnRope),
}

impl Rope {
    pub fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        match &mut self.0 {
            RopeKind::Standard(rope) => Ok(rope.forward(nn::RopeInput { x, offset })?),
            RopeKind::Yarn(rope) => rope.forward(x, offset),
        }
    }
}

impl YarnRope {
    fn forward(&self, x: &Array, offset: i32) -> Result<Array> {
        let last_dim =
            x.shape().last().copied().ok_or_else(|| {
                Error::Config("YaRN input must have at least one dimension".into())
            })?;
        if self.dims > last_dim {
            return Err(Error::Config(format!(
                "YaRN rotary dimensions {} exceed input dimension {last_dim}",
                self.dims
            )));
        }

        let scaled = if self.mscale == 1.0 {
            x.clone()
        } else if self.dims == last_dim {
            x.multiply(Array::from_f32(self.mscale))?
        } else {
            let rotary = x
                .index((Ellipsis, 0..self.dims))
                .multiply(Array::from_f32(self.mscale))?;
            let pass_through = x.index((Ellipsis, self.dims..last_dim));
            concatenate_axis(&[rotary, pass_through], -1)?
        };

        Ok(fast::rope(
            &scaled,
            self.dims,
            false,
            None,
            1.0,
            offset,
            Some(&self.freqs),
        )?)
    }
}

/// Build a RoPE module from Qwen3 config. Supports default, linear, and YaRN
/// scaling while matching mlx_lm's required YaRN fields and defaults.
pub fn build_rope(
    head_dim: i32,
    rope_theta: f32,
    rope_scaling: &Option<HashMap<String, RopeScalingValue>>,
) -> Result<Rope> {
    validate_positive("head_dim", head_dim as f32)?;
    if head_dim % 2 != 0 {
        return Err(Error::Config(format!(
            "head_dim must be even for RoPE, got {head_dim}"
        )));
    }
    validate_positive("rope_theta", rope_theta)?;

    match parse_scaling(rope_scaling)? {
        Scaling::Standard { scale } => {
            let rope = RopeBuilder::new(head_dim)
                .traditional(false)
                .base(rope_theta)
                .scale(scale)
                .build()
                .map_err(|_: std::convert::Infallible| Error::Config("rope build".into()))?;
            Ok(Rope(RopeKind::Standard(rope)))
        }
        Scaling::Yarn(parameters) => {
            let values = yarn_frequencies(head_dim, rope_theta, &parameters);
            let freqs = Array::from_slice(&values, &[values.len() as i32]);
            let mscale = yarn_get_mscale(parameters.factor, parameters.mscale)
                / yarn_get_mscale(parameters.factor, parameters.mscale_all_dim);
            Ok(Rope(RopeKind::Yarn(YarnRope {
                dims: head_dim,
                mscale,
                freqs,
            })))
        }
    }
}

fn parse_scaling(rope_scaling: &Option<HashMap<String, RopeScalingValue>>) -> Result<Scaling> {
    let Some(config) = rope_scaling else {
        return Ok(Scaling::Standard { scale: 1.0 });
    };
    let rope_type = config
        .get("type")
        .or_else(|| config.get("rope_type"))
        .and_then(|value| match value {
            RopeScalingValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .unwrap_or("default");

    match rope_type {
        "default" => Ok(Scaling::Standard { scale: 1.0 }),
        "linear" => {
            let factor = optional_float(config, "factor")?.unwrap_or(1.0);
            validate_positive("rope_scaling.factor", factor)?;
            Ok(Scaling::Standard {
                scale: 1.0 / factor,
            })
        }
        "yarn" => {
            let factor = required_float(config, "factor")?;
            let original_max_position_embeddings =
                required_float(config, "original_max_position_embeddings")?;
            let beta_fast = optional_float(config, "beta_fast")?.unwrap_or(DEFAULT_BETA_FAST);
            let beta_slow = optional_float(config, "beta_slow")?.unwrap_or(DEFAULT_BETA_SLOW);
            let mscale = optional_float(config, "mscale")?.unwrap_or(DEFAULT_MSCALE);
            let mscale_all_dim =
                optional_float(config, "mscale_all_dim")?.unwrap_or(DEFAULT_MSCALE_ALL_DIM);

            validate_positive("rope_scaling.factor", factor)?;
            validate_positive(
                "rope_scaling.original_max_position_embeddings",
                original_max_position_embeddings,
            )?;
            validate_positive("rope_scaling.beta_fast", beta_fast)?;
            validate_positive("rope_scaling.beta_slow", beta_slow)?;
            validate_finite("rope_scaling.mscale", mscale)?;
            validate_finite("rope_scaling.mscale_all_dim", mscale_all_dim)?;

            Ok(Scaling::Yarn(YarnParameters {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
            }))
        }
        other => Err(Error::Config(format!(
            "unsupported rope_type {other:?} (only default, linear, and yarn are supported)"
        ))),
    }
}

fn optional_float(config: &HashMap<String, RopeScalingValue>, key: &str) -> Result<Option<f32>> {
    match config.get(key) {
        None => Ok(None),
        Some(RopeScalingValue::Float(value)) => Ok(Some(*value)),
        Some(_) => Err(Error::Config(format!("rope_scaling.{key} must be numeric"))),
    }
}

fn required_float(config: &HashMap<String, RopeScalingValue>, key: &str) -> Result<f32> {
    optional_float(config, key)?.ok_or_else(|| {
        Error::Config(format!(
            "rope_scaling.{key} is required for rope_type=\"yarn\""
        ))
    })
}

fn validate_positive(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

fn validate_finite(name: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::Config(format!("{name} must be finite, got {value}")))
    }
}

fn yarn_correction_dim(
    rotations: f32,
    dims: i32,
    base: f32,
    original_max_position_embeddings: f32,
) -> f32 {
    dims as f32 * (original_max_position_embeddings / (rotations * std::f32::consts::TAU)).ln()
        / (2.0 * base.ln())
}

fn yarn_correction_range(dims: i32, base: f32, parameters: &YarnParameters) -> (i32, i32) {
    let low = yarn_correction_dim(
        parameters.beta_fast,
        dims,
        base,
        parameters.original_max_position_embeddings,
    )
    .floor() as i32;
    let high = yarn_correction_dim(
        parameters.beta_slow,
        dims,
        base,
        parameters.original_max_position_embeddings,
    )
    .ceil() as i32;
    (low.max(0), high.min(dims - 1))
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_frequencies(dims: i32, base: f32, parameters: &YarnParameters) -> Vec<f32> {
    let (low, high) = yarn_correction_range(dims, base, parameters);
    let max_ramp = if low == high {
        high as f32 + 0.001
    } else {
        high as f32
    };

    (0..dims / 2)
        .map(|index| {
            let ramp = ((index as f32 - low as f32) / (max_ramp - low as f32)).clamp(0.0, 1.0);
            let freq_extra = base.powf((2 * index) as f32 / dims as f32);
            let freq_inter = parameters.factor * freq_extra;
            let freq_mask = 1.0 - ramp;
            (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yarn_parameters() -> YarnParameters {
        YarnParameters {
            factor: 4.0,
            original_max_position_embeddings: 32_768.0,
            beta_fast: DEFAULT_BETA_FAST,
            beta_slow: DEFAULT_BETA_SLOW,
            mscale: DEFAULT_MSCALE,
            mscale_all_dim: DEFAULT_MSCALE_ALL_DIM,
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = expected.abs().max(1.0) * 1.0e-6;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    #[test]
    fn yarn_matches_python_correction_range() {
        assert_eq!(
            yarn_correction_range(128, 1_000_000.0, &yarn_parameters()),
            (23, 40)
        );
    }

    #[test]
    fn yarn_matches_python_selected_frequencies() {
        let frequencies = yarn_frequencies(128, 1_000_000.0, &yarn_parameters());
        assert_eq!(frequencies.len(), 64);
        for (index, expected) in [
            (0, 1.0),
            (22, 115.478_195),
            (30, 939.530_94),
            (40, 22_493.652),
            (63, 3_223_368.8),
        ] {
            assert_close(frequencies[index], expected);
        }
    }

    #[test]
    fn yarn_matches_python_mscale() {
        let parameters = yarn_parameters();
        let mscale = yarn_get_mscale(parameters.factor, parameters.mscale)
            / yarn_get_mscale(parameters.factor, parameters.mscale_all_dim);
        assert_close(mscale, 1.138_629_4);
    }

    #[test]
    fn default_and_linear_keep_standard_rope_scaling() {
        assert!(matches!(
            parse_scaling(&None).expect("default scaling"),
            Scaling::Standard { scale } if scale == 1.0
        ));

        let linear = Some(HashMap::from([
            (
                "rope_type".to_string(),
                RopeScalingValue::String("linear".to_string()),
            ),
            ("factor".to_string(), RopeScalingValue::Float(4.0)),
        ]));
        assert!(matches!(
            parse_scaling(&linear).expect("linear scaling"),
            Scaling::Standard { scale } if scale == 0.25
        ));
    }

    #[test]
    fn yarn_uses_python_defaults_and_requires_context_values() {
        let config = Some(HashMap::from([
            (
                "rope_type".to_string(),
                RopeScalingValue::String("yarn".to_string()),
            ),
            ("factor".to_string(), RopeScalingValue::Float(4.0)),
            (
                "original_max_position_embeddings".to_string(),
                RopeScalingValue::Float(32_768.0),
            ),
        ]));
        let Scaling::Yarn(parameters) = parse_scaling(&config).expect("YaRN scaling") else {
            panic!("expected YaRN parameters");
        };
        assert_eq!(parameters.beta_fast, DEFAULT_BETA_FAST);
        assert_eq!(parameters.beta_slow, DEFAULT_BETA_SLOW);
        assert_eq!(parameters.mscale, DEFAULT_MSCALE);
        assert_eq!(parameters.mscale_all_dim, DEFAULT_MSCALE_ALL_DIM);

        let missing_original = Some(HashMap::from([
            (
                "rope_type".to_string(),
                RopeScalingValue::String("yarn".to_string()),
            ),
            ("factor".to_string(), RopeScalingValue::Float(4.0)),
        ]));
        assert!(parse_scaling(&missing_original).is_err());
    }
}
