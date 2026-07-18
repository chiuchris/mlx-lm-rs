use std::path::Path;
use tokenizers::Tokenizer;

use crate::error::Result;

fn declares_llama_tokenizer(model_dir: &Path) -> Result<bool> {
    let path = model_dir.join("tokenizer_config.json");
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let config: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(matches!(
        config
            .get("tokenizer_class")
            .and_then(|value| value.as_str()),
        Some("LlamaTokenizer" | "LlamaTokenizerFast")
    ))
}

fn apply_llama_fast_compat(tokenizer_json: &mut serde_json::Value) -> bool {
    let Some(model) = tokenizer_json
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return false;
    };
    if model.get("type").and_then(serde_json::Value::as_str) != Some("BPE") {
        return false;
    }

    model.insert("byte_fallback".to_string(), serde_json::Value::Bool(true));
    model.insert("fuse_unk".to_string(), serde_json::Value::Bool(true));
    model.insert(
        "continuing_subword_prefix".to_string(),
        serde_json::Value::Null,
    );
    model.insert("end_of_word_suffix".to_string(), serde_json::Value::Null);

    tokenizer_json["normalizer"] = serde_json::Value::Null;
    tokenizer_json["pre_tokenizer"] = serde_json::json!({
        "type": "Metaspace",
        "replacement": "\u{2581}",
        "prepend_scheme": "always",
        "split": false
    });
    true
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer> {
    let model_dir = model_dir.as_ref();
    let path = model_dir.join("tokenizer.json");
    if !declares_llama_tokenizer(model_dir)? {
        return Ok(Tokenizer::from_file(&path)?);
    }

    let raw = std::fs::read_to_string(&path)?;
    let mut tokenizer_json: serde_json::Value = serde_json::from_str(&raw)?;
    apply_llama_fast_compat(&mut tokenizer_json);
    let bytes = serde_json::to_vec(&tokenizer_json)?;
    Ok(Tokenizer::from_bytes(bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_fast_compat_matches_mlx_lm_pipeline_shape() {
        let mut json = serde_json::json!({
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "byte_fallback": false,
                "fuse_unk": false,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": ""
            }
        });

        assert!(apply_llama_fast_compat(&mut json));
        assert!(json["normalizer"].is_null());
        assert_eq!(json["pre_tokenizer"]["type"], "Metaspace");
        assert_eq!(json["pre_tokenizer"]["replacement"], "\u{2581}");
        assert_eq!(json["pre_tokenizer"]["prepend_scheme"], "always");
        assert_eq!(json["pre_tokenizer"]["split"], false);
        assert_eq!(json["model"]["byte_fallback"], true);
        assert_eq!(json["model"]["fuse_unk"], true);
        assert!(json["model"]["continuing_subword_prefix"].is_null());
        assert!(json["model"]["end_of_word_suffix"].is_null());
        assert_eq!(json["decoder"]["type"], "ByteLevel");
    }

    #[test]
    #[ignore = "requires MLX_LM_RS_TEST_MODEL_DIR"]
    fn llama_fast_real_checkpoint_matches_mlx_lm_reference() {
        let model_dir = std::env::var_os("MLX_LM_RS_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set MLX_LM_RS_TEST_MODEL_DIR to the checkpoint snapshot");
        let tokenizer = load_tokenizer(model_dir).expect("load real checkpoint tokenizer");
        let text = "<｜begin▁of▁sentence｜>You are a local AI Agent.<｜User｜>\
                    convert 100 USD to CAD<｜Assistant｜>";
        let encoding = tokenizer
            .encode(text, false)
            .expect("encode parity fixture");
        assert_eq!(
            encoding.get_ids(),
            &[
                151643, 2610, 546, 278, 3683, 32, 5863, 15772, 13, 151669, 14166, 16, 15, 15, 2034,
                14797, 48570, 151670,
            ]
        );
        let decoded = tokenizer
            .decode(&[198, 198, 5338, 11, 279, 1196, 1053, 25], false)
            .expect("decode ByteLevel generation fixture");
        assert_eq!(decoded, "\n\nFirst, the user said:");

        let model_dir = std::env::var_os("MLX_LM_RS_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set MLX_LM_RS_TEST_MODEL_DIR to the checkpoint snapshot");
        let template = crate::chat_template::ChatTemplate::load(model_dir)
            .expect("load chat template")
            .expect("checkpoint has a chat template");
        let rendered = template
            .render("Say hello in exactly five words.", true)
            .expect("render Gate 2 fixture");
        let encoding = tokenizer
            .encode(rendered, false)
            .expect("encode rendered Gate 2 fixture");
        assert_eq!(
            encoding.get_ids(),
            &[151643, 151669, 45764, 14990, 258, 327, 32739, 69, 344, 365, 2260, 13, 151670,]
        );
    }
}
