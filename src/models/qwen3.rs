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
            &self.biases,
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

        let q = q
            .reshape(&[b, l, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

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
        let out = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
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
