use std::iter::once;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{self, ScaledDotProductAttentionMask},
    macros::ModuleParameters,
    module::{Module, ModuleParameters as _, ModuleParametersExt, Param},
    nn::{self, Embedding, Linear, LinearBuilder, RmsNorm, RmsNormBuilder},
    ops::{self, indexing::IndexOp},
    quantization::{MaybeQuantized, Quantizable},
    Array,
};

use crate::cache::KvCache;
use crate::config::Qwen3Config;
use crate::error::Result;
use crate::models::rope::{build_rope, Rope};

type LinearLayer = MaybeQuantized<Linear>;
type EmbeddingLayer = MaybeQuantized<EmbeddingModule>;

#[derive(Debug, Clone, ModuleParameters)]
pub struct EmbeddingModule {
    #[param]
    inner: Embedding,
}

impl EmbeddingModule {
    fn new(embedding_count: i32, dimensions: i32) -> Result<Self> {
        Ok(Self {
            inner: Embedding::new(embedding_count, dimensions)?,
        })
    }

    fn as_linear(&self, x: &Array) -> std::result::Result<Array, Exception> {
        self.inner.as_linear(x)
    }
}

impl Module<&Array> for EmbeddingModule {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        self.inner.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct PackedEmbedding {
    group_size: i32,
    bits: i32,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    #[param]
    inner: Embedding,
}

impl PackedEmbedding {
    fn as_linear(&self, x: &Array) -> std::result::Result<Array, Exception> {
        ops::quantized_matmul(
            x,
            &self.inner.weight,
            &self.scales,
            &*self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

impl Module<&Array> for PackedEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let shape = x.shape().to_vec();
        let indices = x.flatten(None, None)?;
        let weight = self.inner.weight.index(&indices);
        let scales = self.scales.index(&indices);
        let biases = self.biases.index(&indices);
        let output = ops::dequantize(&weight, &scales, &biases, self.group_size, self.bits)?;
        let output_shape = shape.into_iter().chain(once(-1)).collect::<Vec<_>>();
        output.reshape(&output_shape)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}

impl Quantizable for EmbeddingModule {
    type Quantized = PackedEmbedding;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Self::Quantized, Self::QuantizationError> {
        let quantized = nn::QuantizedEmbedding::try_from_embedding(self.inner, group_size, bits)?;
        Ok(PackedEmbedding {
            group_size: quantized.group_size,
            bits: quantized.bits,
            scales: quantized.scales,
            biases: quantized.biases,
            inner: quantized.inner,
        })
    }
}

fn linear(in_dim: i32, out_dim: i32, cfg: &Qwen3Config) -> Result<LinearLayer> {
    let layer = MaybeQuantized::new(LinearBuilder::new(in_dim, out_dim).bias(false).build()?);
    match cfg.quantization() {
        Some(quantization) => Ok(nn::quantize(
            layer,
            quantization.group_size,
            quantization.bits,
        )?),
        None => Ok(layer),
    }
}

fn embedding(cfg: &Qwen3Config) -> Result<EmbeddingLayer> {
    let layer = MaybeQuantized::new(EmbeddingModule::new(cfg.vocab_size, cfg.hidden_size)?);
    match cfg.quantization() {
        Some(quantization) => Ok(nn::quantize(
            layer,
            quantization.group_size,
            quantization.bits,
        )?),
        None => Ok(layer),
    }
}

fn embedding_as_linear(layer: &EmbeddingLayer, x: &Array) -> Result<Array> {
    match layer {
        MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(x)?),
        MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(x)?),
    }
}

fn rms(dim: i32, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNormBuilder::new(dim).eps(eps).build()?)
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,

    #[param]
    q_proj: LinearLayer,
    #[param]
    k_proj: LinearLayer,
    #[param]
    v_proj: LinearLayer,
    #[param]
    o_proj: LinearLayer,
    #[param]
    q_norm: RmsNorm,
    #[param]
    k_norm: RmsNorm,

    rope: Rope,
}

impl Attention {
    pub fn new(cfg: &Qwen3Config) -> Result<Self> {
        let dim = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj: linear(dim, n_heads * head_dim, cfg)?,
            k_proj: linear(dim, n_kv_heads * head_dim, cfg)?,
            v_proj: linear(dim, n_kv_heads * head_dim, cfg)?,
            o_proj: linear(n_heads * head_dim, dim, cfg)?,
            q_norm: rms(head_dim, cfg.rms_norm_eps)?,
            k_norm: rms(head_dim, cfg.rms_norm_eps)?,
            rope: build_rope(head_dim, cfg.rope_theta, &cfg.rope_scaling)?,
        })
    }

    fn forward(&mut self, x: &Array, cache: Option<&mut KvCache>) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[b, l, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
        let v = v
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);
        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        let (k, v) = match cache {
            Some(c) => c.update_and_fetch(k, v)?,
            None => (k, v),
        };

        // Causal mask only needed when q_len > 1 (prefill); decode's single
        // query attends only to past keys by construction.
        let mask = (l > 1).then_some(ScaledDotProductAttentionMask::Causal);
        let out = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        Ok(self.o_proj.forward(&out)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Mlp {
    #[param]
    gate_proj: LinearLayer,
    #[param]
    down_proj: LinearLayer,
    #[param]
    up_proj: LinearLayer,
}

impl Mlp {
    pub fn new(cfg: &Qwen3Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, cfg)?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, cfg)?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, cfg)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let h = nn::silu(&g)?.multiply(&u)?;
        Ok(self.down_proj.forward(&h)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    #[param]
    self_attn: Attention,
    #[param]
    mlp: Mlp,
    #[param]
    input_layernorm: RmsNorm,
    #[param]
    post_attention_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn new(cfg: &Qwen3Config) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg)?,
            mlp: Mlp::new(cfg)?,
            input_layernorm: rms(cfg.hidden_size, cfg.rms_norm_eps)?,
            post_attention_layernorm: rms(cfg.hidden_size, cfg.rms_norm_eps)?,
        })
    }

    fn forward(&mut self, x: &Array, cache: Option<&mut KvCache>) -> Result<Array> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x)?, cache)?;
        let h = x.add(&attn)?;
        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        Ok(h.add(&r)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3Backbone {
    #[param]
    pub embed_tokens: EmbeddingLayer,
    #[param]
    layers: Vec<TransformerBlock>,
    #[param]
    norm: RmsNorm,
}

impl Qwen3Backbone {
    pub fn new(cfg: &Qwen3Config) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| TransformerBlock::new(cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: embedding(cfg)?,
            layers,
            norm: rms(cfg.hidden_size, cfg.rms_norm_eps)?,
        })
    }

    fn forward(&mut self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array> {
        if cache.len() != self.layers.len() {
            return Err(crate::error::Error::Config(format!(
                "cache length {} does not match layer count {}",
                cache.len(),
                self.layers.len()
            )));
        }
        let mut h = self.embed_tokens.forward(tokens)?;
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, Some(c))?;
        }
        Ok(self.norm.forward(&h)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub config: Qwen3Config,
    #[param]
    pub model: Qwen3Backbone,
    #[param]
    lm_head: Option<LinearLayer>,
}

impl Model {
    pub fn new(cfg: Qwen3Config) -> Result<Self> {
        let model = Qwen3Backbone::new(&cfg)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(linear(cfg.hidden_size, cfg.vocab_size, &cfg)?)
        };
        Ok(Self {
            config: cfg,
            model,
            lm_head,
        })
    }

    pub fn forward(&mut self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array> {
        let shape = tokens.shape();
        if shape.len() != 2 || shape[0] != 1 || shape[1] < 1 {
            return Err(crate::error::Error::Config(format!(
                "model input must have shape [1, L] with L >= 1, got {shape:?}"
            )));
        }
        let h = self.model.forward(tokens, cache)?;
        match &mut self.lm_head {
            Some(head) => Ok(head.forward(&h)?),
            None => embedding_as_linear(&self.model.embed_tokens, &h),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.model.layers.len()
    }

    pub fn make_cache(&self) -> Vec<KvCache> {
        (0..self.n_layers()).map(|_| KvCache::new()).collect()
    }

    /// Load all weight shards in one pass, with a single eval at the end.
    /// Quantized checkpoints stay packed and are consumed by mlx-rs fused
    /// quantized embedding and matrix-multiplication operations.
    pub fn load_weights(&mut self, shards: &[std::path::PathBuf]) -> Result<()> {
        use std::collections::{HashMap, HashSet};

        let mut tensors: HashMap<String, Array> = HashMap::new();
        for shard in shards {
            tensors.extend(Array::load_safetensors(shard)?);
        }

        let mut params = self.parameters_mut().flatten();
        let mut loaded_keys: HashSet<String> = HashSet::new();
        let param_names: Vec<String> = params.keys().map(|key| key.to_string()).collect();

        for param_name in param_names {
            let checkpoint_name = match param_name.strip_suffix(".inner.weight") {
                Some(prefix) => format!("{prefix}.weight"),
                None => param_name.clone(),
            };
            let Some(value) = tensors.remove(&checkpoint_name) else {
                continue;
            };
            if let Some(param) = params.get_mut(param_name.as_str()) {
                **param = value;
                loaded_keys.insert(param_name);
            }
        }

        let mut missing: Vec<String> = params
            .keys()
            .map(|key| key.to_string())
            .filter(|key| !loaded_keys.contains(key))
            .collect();
        if !missing.is_empty() {
            missing.sort();
            let head = missing
                .iter()
                .take(5)
                .map(|key| key.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let tail = if missing.len() > 5 {
                format!(" (+{} more)", missing.len() - 5)
            } else {
                String::new()
            };
            return Err(crate::error::Error::MissingWeight(format!("{head}{tail}")));
        }
        self.eval()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::load_config, loader::list_weight_files};

    fn bf16_hash(array: &Array) -> u64 {
        array
            .flatten(None, None)
            .expect("flatten BF16 values")
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("copy BF16 values")
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .expect("restore BF16 values")
            .view_dtype(mlx_rs::Dtype::Uint16)
            .expect("view BF16 bits")
            .as_slice::<u16>()
            .iter()
            .fold(0xcbf29ce484222325_u64, |state, value| {
                (state ^ u64::from(*value)).wrapping_mul(0x100000001b3)
            })
    }

    #[test]
    #[ignore = "requires MLX_LM_RS_TEST_MODEL_DIR"]
    fn ds8_layer0_intermediates_match_python_bits() {
        let model_dir = std::env::var_os("MLX_LM_RS_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set MLX_LM_RS_TEST_MODEL_DIR to the checkpoint snapshot");
        let config = load_config(&model_dir).expect("load config");
        let mut model = Model::new(config).expect("construct model");
        let shards = list_weight_files(&model_dir).expect("list weights");
        model.load_weights(&shards).expect("load weights");

        let token = Array::from_slice(&[387_u32], &[1, 1]);
        let embed = model
            .model
            .embed_tokens
            .forward(&token)
            .expect("embed token");
        let layer = &mut model.model.layers[0];
        let norm = layer.input_layernorm.forward(&embed).expect("input norm");
        let q = layer.self_attn.q_proj.forward(&norm).expect("q projection");
        let k = layer.self_attn.k_proj.forward(&norm).expect("k projection");
        let v = layer.self_attn.v_proj.forward(&norm).expect("v projection");
        let qnorm = layer
            .self_attn
            .q_norm
            .forward(&q.reshape(&[1, 1, 32, 128]).expect("reshape q"))
            .expect("q norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose q");
        let knorm = layer
            .self_attn
            .k_norm
            .forward(&k.reshape(&[1, 1, 8, 128]).expect("reshape k"))
            .expect("k norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose k");
        let krope_zero = layer
            .self_attn
            .rope
            .forward(&knorm, 0)
            .expect("apply rope at offset zero");
        let krope = layer
            .self_attn
            .rope
            .forward(&knorm, 46)
            .expect("apply rope");
        let single_value = v
            .reshape(&[1, 1, 8, 128])
            .expect("reshape v")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose v");

        let actual = [
            ("embed", bf16_hash(&embed)),
            ("norm", bf16_hash(&norm)),
            ("q", bf16_hash(&q)),
            ("k", bf16_hash(&k)),
            ("v", bf16_hash(&v)),
            ("qnorm", bf16_hash(&qnorm)),
            ("knorm", bf16_hash(&knorm)),
            ("krope_zero", bf16_hash(&krope_zero)),
            ("krope", bf16_hash(&krope)),
            ("value", bf16_hash(&single_value)),
        ];
        let expected = [
            ("embed", 0x1dc119f48e844974),
            ("norm", 0x491388f13a1a9877),
            ("q", 0xc4ba062602b40fa8),
            ("k", 0xe0644aebe940e540),
            ("v", 0xe5073d4a5312339f),
            ("qnorm", 0xa75015074680f1e1),
            ("knorm", 0xac19672b30f07960),
            ("krope_zero", 0x08253d7535037885),
            ("krope", 0xfda4637578980328),
            ("value", 0xe5073d4a5312339f),
        ];
        assert_eq!(actual, expected);

        let prefill = Array::from_slice(
            &[
                151643_u32, 151669, 45764, 14990, 258, 327, 32739, 69, 344, 365, 2260, 13,
            ],
            &[1, 12],
        );
        let embed = model
            .model
            .embed_tokens
            .forward(&prefill)
            .expect("embed prefill");
        let layer = &mut model.model.layers[0];
        let norm = layer
            .input_layernorm
            .forward(&embed)
            .expect("prefill input norm");
        let q = layer.self_attn.q_proj.forward(&norm).expect("prefill q");
        let k = layer.self_attn.k_proj.forward(&norm).expect("prefill k");
        let v = layer.self_attn.v_proj.forward(&norm).expect("prefill v");
        let qnorm = layer
            .self_attn
            .q_norm
            .forward(&q.reshape(&[1, 12, 32, 128]).expect("reshape prefill q"))
            .expect("prefill q norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose prefill q");
        let knorm = layer
            .self_attn
            .k_norm
            .forward(&k.reshape(&[1, 12, 8, 128]).expect("reshape prefill k"))
            .expect("prefill k norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose prefill k");
        let krope = layer
            .self_attn
            .rope
            .forward(&knorm, 0)
            .expect("prefill rope");
        let value = v
            .reshape(&[1, 12, 8, 128])
            .expect("reshape prefill v")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose prefill v");
        assert_eq!(
            [
                ("embed", bf16_hash(&embed)),
                ("norm", bf16_hash(&norm)),
                ("q", bf16_hash(&q)),
                ("k", bf16_hash(&k)),
                ("v", bf16_hash(&v)),
                ("qnorm", bf16_hash(&qnorm)),
                ("knorm", bf16_hash(&knorm)),
                ("krope", bf16_hash(&krope)),
                ("value", bf16_hash(&value)),
            ],
            [
                ("embed", 0xf942f12184d85c0c),
                ("norm", 0x94b076813d563f7f),
                ("q", 0x15a395dc6e6fc2c2),
                ("k", 0x2fa943f1cbfc5c50),
                ("v", 0xe515362796f37613),
                ("qnorm", 0x513065e73313900f),
                ("knorm", 0x94798588402a2139),
                ("krope", 0x0618b2e09a652eb3),
                ("value", 0xd82b8f1b2eac18c3),
            ]
        );

        let full_prefix = Array::from_slice(
            &[
                151643_u32, 151669, 45764, 14990, 258, 327, 32739, 69, 344, 365, 2260, 13, 151670,
                151667,
            ],
            &[1, 14],
        );
        let embed = model
            .model
            .embed_tokens
            .forward(&full_prefix)
            .expect("embed full prefix");
        let norm = layer
            .input_layernorm
            .forward(&embed)
            .expect("full-prefix input norm");
        let q = layer
            .self_attn
            .q_proj
            .forward(&norm)
            .expect("full-prefix q");
        let k = layer
            .self_attn
            .k_proj
            .forward(&norm)
            .expect("full-prefix k");
        let v = layer
            .self_attn
            .v_proj
            .forward(&norm)
            .expect("full-prefix v");
        let qnorm = layer
            .self_attn
            .q_norm
            .forward(&q.reshape(&[1, 14, 32, 128]).expect("reshape full-prefix q"))
            .expect("full-prefix q norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose full-prefix q");
        let knorm = layer
            .self_attn
            .k_norm
            .forward(&k.reshape(&[1, 14, 8, 128]).expect("reshape full-prefix k"))
            .expect("full-prefix k norm")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose full-prefix k");
        let qrope = layer
            .self_attn
            .rope
            .forward(&qnorm, 0)
            .expect("full-prefix q rope");
        let krope = layer
            .self_attn
            .rope
            .forward(&knorm, 0)
            .expect("full-prefix k rope");
        let value = v
            .reshape(&[1, 14, 8, 128])
            .expect("reshape full-prefix v")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose full-prefix v");
        assert_eq!(
            (qrope.dtype(), krope.dtype(), value.dtype()),
            (
                mlx_rs::Dtype::Bfloat16,
                mlx_rs::Dtype::Bfloat16,
                mlx_rs::Dtype::Bfloat16,
            )
        );
        assert_eq!(layer.self_attn.scale.to_bits(), 0x3db5_04f3);
        let sdpa = fast::scaled_dot_product_attention(
            &qrope,
            &krope,
            &value,
            layer.self_attn.scale,
            ScaledDotProductAttentionMask::Causal,
            None,
        )
        .expect("full-prefix SDPA");
        let flat = sdpa
            .transpose_axes(&[0, 2, 1, 3])
            .expect("transpose full-prefix SDPA")
            .reshape(&[1, 14, 4096])
            .expect("flatten full-prefix SDPA");
        let out = layer
            .self_attn
            .o_proj
            .forward(&flat)
            .expect("full-prefix attention output");
        assert_eq!(
            [
                ("embed", bf16_hash(&embed)),
                ("norm", bf16_hash(&norm)),
                ("q", bf16_hash(&q)),
                ("k", bf16_hash(&k)),
                ("v", bf16_hash(&v)),
                ("qnorm", bf16_hash(&qnorm)),
                ("knorm", bf16_hash(&knorm)),
                ("qrope", bf16_hash(&qrope)),
                ("krope", bf16_hash(&krope)),
                ("value", bf16_hash(&value)),
                ("sdpa", bf16_hash(&sdpa)),
                ("flat", bf16_hash(&flat)),
                ("out", bf16_hash(&out)),
            ],
            [
                ("embed", 0xef0006be2976f276),
                ("norm", 0xe7a6e7e6eeeccfc2),
                ("q", 0x0d38f13e10b147fe),
                ("k", 0x5951c1d02b1fb8b0),
                ("v", 0xa945a0a940af6fe2),
                ("qnorm", 0xb110964321ad4b34),
                ("knorm", 0xe134a31a55653915),
                ("qrope", 0x3370c087e3e809f1),
                ("krope", 0xc757b57623b36400),
                ("value", 0x0baf9f219a29b28a),
                ("sdpa", 0xc591114599abbd67),
                ("flat", 0x7740e35b51ad9d33),
                ("out", 0x37c68f356e4378a4),
            ]
        );

        let mut direct_cache = KvCache::new();
        direct_cache
            .update_and_fetch(krope_zero.clone(), single_value.clone())
            .expect("direct cache insertion");
        let (direct_key, direct_value) = direct_cache.active().expect("direct cache");
        let key_equal: bool = direct_key
            .eq(&krope_zero)
            .expect("compare direct key")
            .all(None)
            .expect("reduce direct key equality")
            .item();
        let value_equal: bool = direct_value
            .eq(&single_value)
            .expect("compare direct value")
            .all(None)
            .expect("reduce direct value equality")
            .item();
        assert!(
            key_equal && value_equal,
            "cache insertion changed K/V values"
        );

        let mut cache = model.make_cache();
        model.forward(&token, &mut cache).expect("cached forward");
        let (cached_key, cached_value) = cache[0].active().expect("layer-0 cache");
        assert_eq!(
            (bf16_hash(&cached_key), bf16_hash(&cached_value)),
            (0x08253d7535037885, 0xe5073d4a5312339f)
        );
    }
}
