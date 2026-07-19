use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    fast,
    module::Module,
    nn::{self, Rope as StandardRope, RopeBuilder},
    ops::{
        clip, concatenate_axis,
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
            x.multiply(Array::from_f32(self.mscale).as_dtype(x.dtype())?)?
        } else {
            let rotary = x
                .index((Ellipsis, 0..self.dims))
                .multiply(Array::from_f32(self.mscale).as_dtype(x.dtype())?)?;
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
            let freqs = yarn_frequencies(head_dim, rope_theta, &parameters)?;
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

fn yarn_frequencies(dims: i32, base: f32, parameters: &YarnParameters) -> Result<Array> {
    let (low, high) = yarn_correction_range(dims, base, parameters);
    let max_ramp = if low == high {
        high as f32 + 0.001
    } else {
        high as f32
    };

    let exponents = Array::arange::<_, f32>(0, dims, 2)?.divide(Array::from_f32(dims as f32))?;
    let freq_extra = Array::from_f32(base).power(&exponents)?;
    let freq_inter = Array::from_f32(parameters.factor).multiply(&freq_extra)?;
    let ramp = Array::arange::<_, f32>(0, dims / 2, None)?
        .subtract(Array::from_f32(low as f32))?
        .divide(Array::from_f32(max_ramp - low as f32))?;
    let freq_mask = Array::from_f32(1.0).subtract(clip(&ramp, (0.0, 1.0))?)?;
    let numerator = freq_inter.multiply(&freq_extra)?;
    let denominator = freq_inter
        .multiply(&freq_mask)?
        .add(&freq_extra.multiply(Array::from_f32(1.0).subtract(&freq_mask)?)?)?;
    Ok(numerator.divide(&denominator)?)
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
    fn yarn_matches_python_frequency_bits() {
        let frequencies =
            yarn_frequencies(128, 1_000_000.0, &yarn_parameters()).expect("build YaRN frequencies");
        let actual = frequencies
            .as_slice::<f32>()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expected = [
            0x3f800000, 0x3f9ed70c, 0x3fc51c50, 0x3ff49a1b, 0x4017c496, 0x403c55a5, 0x4069b621,
            0x409102bc, 0x40b3f300, 0x40df4e48, 0x410a8de7, 0x412beff0, 0x41555d09, 0x418462a8,
            0x41a44832, 0x41cbdd1f, 0x41fcfb72, 0x421cf7b5, 0x4242c979, 0x4271b7f4, 0x4295fa95,
            0x42ba1d4b, 0x42e6f4d6, 0x430f4d1f, 0x433a090f, 0x43720768, 0x439dcea7, 0x43ce51dc,
            0x440742da, 0x4431ebf3, 0x446ae1fb, 0x449bac8f, 0x44cf512c, 0x450aca02, 0x453afdb9,
            0x457dcc68, 0x45adc3b3, 0x45f082e9, 0x4628b1d0, 0x4670bd8c, 0x46afbb4e, 0x46da1273,
            0x47074e93, 0x4727e851, 0x47505cdc, 0x47814858, 0x47a06e81, 0x47c715f0, 0x47f70d8e,
            0x481949e6, 0x483e38c1, 0x486c0da4, 0x489276b6, 0x48b5c09b, 0x48e18b1a, 0x490bf150,
            0x492da8fc, 0x4957805b, 0x4985b63f, 0x49a5ed9b, 0x49cde812, 0x49ff8464, 0x4a1e8a5b,
            0x4a44bd24,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn yarn_matches_python_mscale() {
        let parameters = yarn_parameters();
        let mscale = yarn_get_mscale(parameters.factor, parameters.mscale)
            / yarn_get_mscale(parameters.factor, parameters.mscale_all_dim);
        assert_close(mscale, 1.138_629_4);
    }

    #[test]
    fn yarn_matches_python_bfloat16_output_bits() {
        let parameters = yarn_parameters();
        let mscale = yarn_get_mscale(parameters.factor, parameters.mscale)
            / yarn_get_mscale(parameters.factor, parameters.mscale_all_dim);
        let rope = YarnRope {
            dims: 128,
            mscale,
            freqs: yarn_frequencies(128, 1_000_000.0, &parameters).expect("build YaRN frequencies"),
        };
        let input = Array::arange::<_, f32>(0, 256, None)
            .expect("build input")
            .subtract(Array::from_f32(128.0))
            .expect("center input")
            .divide(Array::from_f32(31.0))
            .expect("scale input")
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .expect("cast input")
            .reshape(&[1, 2, 1, 128])
            .expect("reshape input");
        let output = rope.forward(&input, 46).expect("apply YaRN");
        let bits = output
            .view_dtype(mlx_rs::Dtype::Uint16)
            .expect("view BF16 bits");
        let expected = [
            16517, 49316, 49178, 49287, 49200, 16534, 49294, 16378, 16471, 49252, 49283, 15612,
            16478, 16532, 16513, 16430, 16291, 15133, 49021, 49120, 49172, 49198, 49216, 49229,
            49239, 49245, 49249, 49251, 49251, 49251, 49250, 49249,
        ];
        assert_eq!(&bits.as_slice::<u16>()[..expected.len()], &expected);
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
