use mlx_rs::{ops::indexing::argmax_axis, random, Array};
use std::collections::HashSet;

use crate::error::{Error, Result};

/// Sample one token id per batch row from the final-step logits.
///
/// `logits` is shape `[B, V]` (already last-step). At temp == 0.0 we take
/// argmax (greedy). Otherwise we sample categorically from logits/T.
///
/// `temp` must be `0.0` or a finite positive value; negatives invert
/// preference and `NaN` is undefined.
///
/// `rep_penalty` (default 1.0 = no penalty) penalizes tokens that appear in
/// `generated_ids`. Values > 1.0 discourage repetition; 1.0 is a no-op.
pub fn sample(
    logits: &Array,
    temp: f32,
    rep_penalty: f32,
    generated_ids: &HashSet<u32>,
) -> Result<Array> {
    if rep_penalty != 1.0 && !generated_ids.is_empty() {
        // Apply repetition penalty to already-generated tokens.
        // Standard formula: logit < 0 => logit * penalty; logit >= 0 => logit / penalty.
        let logits = apply_rep_penalty(logits, rep_penalty, generated_ids)?;
        finish_sample(&logits, temp)
    } else if temp == 0.0 {
        argmax_axis(logits, -1, None).map_err(Into::into)
    } else if temp.is_finite() && temp > 0.0 {
        let scaled = logits.divide(Array::from_f32(temp))?;
        random::categorical(scaled, -1, None, None).map_err(Into::into)
    } else {
        Err(Error::Config(format!(
            "temperature must be 0.0 or a finite positive value, got {temp}"
        )))
    }
}

/// Apply repetition penalty to logits for tokens in `generated_ids`.
/// Formula: logit < 0 => logit * penalty; logit >= 0 => logit / penalty.
fn apply_rep_penalty(
    logits: &Array,
    penalty: f32,
    generated_ids: &HashSet<u32>,
) -> Result<Array> {
    let vocab_size = logits.shape()[1] as usize;
    let mut penalty_factors = vec![1.0f32; vocab_size];
    for &id in generated_ids {
        let idx = id as usize;
        if idx < vocab_size {
            penalty_factors[idx] = penalty;
        }
    }

    let penalty_arr = Array::from_slice(&penalty_factors, &[1, vocab_size as i32]);

    // where(logits < 0, logits * penalty, logits / penalty)
    let neg_mask = mlx_rs::ops::lt(logits, &Array::from_f32(0.0))?;
    let multiplied = mlx_rs::ops::multiply(logits, &penalty_arr)?;
    let divided = mlx_rs::ops::divide(logits, &penalty_arr)?;
    let result = mlx_rs::ops::r#where(&neg_mask, &multiplied, &divided)?;
    Ok(result)
}

fn finish_sample(scaled: &Array, temp: f32) -> Result<Array> {
    if temp == 0.0 {
        argmax_axis(scaled, -1, None).map_err(Into::into)
    } else {
        let divided = scaled.divide(Array::from_f32(temp))?;
        random::categorical(divided, -1, None, None).map_err(Into::into)
    }
}
