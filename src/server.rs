use std::{
    convert::Infallible,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::{
    chat_template::{ChatTemplate, ChatTemplateMessage},
    config::load_config,
    generate::Generator,
    loader::{list_weight_files, resolve_model_dir},
    models::qwen3::Model,
    tokenizer::load_tokenizer,
};

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_rep_penalty: f32,
    pub prefill_step_size: NonZeroUsize,
    pub no_chat_template: bool,
}

#[derive(Clone)]
struct AppState {
    model_id: String,
    max_tokens_limit: usize,
    work_tx: mpsc::Sender<WorkItem>,
    ids: Arc<AtomicU64>,
}

enum WorkItem {
    Chat {
        request_id: String,
        req: ChatCompletionRequest,
        response_tx: oneshot::Sender<Result<ChatCompletionResponse, ApiError>>,
    },
    ChatStream {
        request_id: String,
        req: ChatCompletionRequest,
        chunk_tx: tokio_mpsc::Sender<Result<StreamItem, ApiError>>,
    },
}

pub async fn serve(cfg: ServeConfig) -> Result<()> {
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port))
        .await
        .context("binding listener")?;
    let addr = listener.local_addr().context("reading listener addr")?;
    let model_id = cfg.model.clone();
    let max_tokens_limit = cfg.default_max_tokens;
    let (work_tx, work_rx) = mpsc::channel();

    spawn_model_worker(cfg, work_rx)?;

    let state = AppState {
        model_id,
        max_tokens_limit,
        work_tx,
        ids: Arc::new(AtomicU64::new(1)),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(create_chat_completion))
        .with_state(state);

    eprintln!("[mlx-lm-rs] OpenAI-compatible server listening on http://{addr}");
    axum::serve(listener, app).await.context("serving http")
}

fn spawn_model_worker(cfg: ServeConfig, work_rx: mpsc::Receiver<WorkItem>) -> Result<()> {
    let (init_tx, init_rx) = mpsc::channel::<std::result::Result<(), String>>();
    thread::Builder::new()
        .name("mlx-lm-rs-model-worker".into())
        .spawn(move || match init_worker(cfg) {
            Ok(mut worker) => {
                let _ = init_tx.send(Ok(()));
                run_worker_loop(&mut worker, work_rx);
            }
            Err(e) => {
                let message = format!("{e:#}");
                let _ = init_tx.send(Err(message.clone()));
                eprintln!("[mlx-lm-rs] model worker failed to initialize: {message}");
            }
        })
        .context("spawning model worker")?;

    match init_rx.recv().context("waiting for model worker startup")? {
        Ok(()) => Ok(()),
        Err(message) => anyhow::bail!("model worker failed to initialize: {message}"),
    }
}

fn init_worker(cfg: ServeConfig) -> Result<ModelWorker> {
    let model_dir = resolve_model_dir(&cfg.model).context("resolving model dir")?;
    eprintln!("[mlx-lm-rs] using model dir: {}", model_dir.display());

    let model_cfg = load_config(&model_dir).context("loading config.json")?;
    let tokenizer = load_tokenizer(&model_dir).context("loading tokenizer.json")?;
    let chat_template = if cfg.no_chat_template {
        None
    } else {
        ChatTemplate::load(&model_dir).context("loading chat template")?
    };

    let mut model = Model::new(model_cfg.clone()).context("constructing model")?;
    let shards = list_weight_files(&model_dir).context("listing weight files")?;
    eprintln!("[mlx-lm-rs] loading {} weight shard(s)...", shards.len());
    model.load_weights(&shards).context("loading weights")?;
    eprintln!("[mlx-lm-rs] weights loaded; accepting requests");

    Ok(ModelWorker {
        model,
        tokenizer,
        chat_template,
        eos_ids: model_cfg
            .eos_token_id
            .as_ref()
            .map(|x| x.ids())
            .unwrap_or_default(),
        model_id: cfg.model,
        max_tokens_limit: cfg.default_max_tokens,
        default_temperature: cfg.default_temperature,
        default_rep_penalty: cfg.default_rep_penalty,
        prefill_step_size: cfg.prefill_step_size,
    })
}

fn run_worker_loop(worker: &mut ModelWorker, work_rx: mpsc::Receiver<WorkItem>) {
    for item in work_rx {
        match item {
            WorkItem::Chat {
                request_id,
                req,
                response_tx,
            } => {
                let response = worker.complete_chat(&request_id, req);
                let _ = response_tx.send(response);
            }
            WorkItem::ChatStream {
                request_id,
                req,
                chunk_tx,
            } => {
                worker.stream_chat(&request_id, req, chunk_tx);
            }
        }
    }
}

struct ModelWorker {
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    chat_template: Option<ChatTemplate>,
    eos_ids: Vec<u32>,
    model_id: String,
    max_tokens_limit: usize,
    default_temperature: f32,
    default_rep_penalty: f32,
    prefill_step_size: NonZeroUsize,
}

impl ModelWorker {
    fn complete_chat(
        &mut self,
        request_id: &str,
        req: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        let created = unix_timestamp();
        let plan = self.prepare(req)?;
        let generation = self.generate_text(&plan)?;
        let finish_reason = finish_reason(generation.completion_tokens, plan.max_tokens);

        Ok(ChatCompletionResponse {
            id: request_id.to_string(),
            object: "chat.completion",
            created,
            model: self.model_id.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: generation.text,
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: plan.prompt_tokens,
                completion_tokens: generation.completion_tokens,
                total_tokens: plan.prompt_tokens + generation.completion_tokens,
            },
        })
    }

    fn stream_chat(
        &mut self,
        request_id: &str,
        req: ChatCompletionRequest,
        chunk_tx: tokio_mpsc::Sender<Result<StreamItem, ApiError>>,
    ) {
        let created = unix_timestamp();
        let result = self.prepare(req).and_then(|plan| {
            let mut completion_tokens = 0usize;
            let first = ChatCompletionChunk {
                id: request_id.to_string(),
                object: "chat.completion.chunk",
                created,
                model: self.model_id.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant"),
                        content: Some(String::new()),
                    },
                    finish_reason: None,
                }],
            };
            send_stream_item(&chunk_tx, Ok(StreamItem::Chunk(first)))?;

            let mut gen = Generator::new(
                &mut self.model,
                &plan.prompt_ids,
                plan.max_tokens,
                plan.temperature,
                plan.rep_penalty,
                self.eos_ids.clone(),
                self.prefill_step_size,
            )
            .map_err(ApiError::internal)?;
            let mut decode = self.tokenizer.decode_stream(false);

            for tok in &mut gen {
                let tok = tok.map_err(ApiError::internal)?;
                completion_tokens += 1;
                if let Some(text) = decode.step(tok).map_err(ApiError::internal)? {
                    let chunk = ChatCompletionChunk {
                        id: request_id.to_string(),
                        object: "chat.completion.chunk",
                        created,
                        model: self.model_id.clone(),
                        choices: vec![ChatCompletionChunkChoice {
                            index: 0,
                            delta: DeltaMessage {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    send_stream_item(&chunk_tx, Ok(StreamItem::Chunk(chunk)))?;
                }
            }

            let final_chunk = ChatCompletionChunk {
                id: request_id.to_string(),
                object: "chat.completion.chunk",
                created,
                model: self.model_id.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(finish_reason(completion_tokens, plan.max_tokens)),
                }],
            };
            send_stream_item(&chunk_tx, Ok(StreamItem::Chunk(final_chunk)))?;
            send_stream_item(&chunk_tx, Ok(StreamItem::Done))?;
            Ok(())
        });

        if let Err(e) = result {
            let _ = chunk_tx.blocking_send(Err(e));
        }
    }

    fn prepare(&self, req: ChatCompletionRequest) -> Result<GenerationPlan, ApiError> {
        req.validate_for_server(self.max_tokens_limit)?;
        let messages = req.template_messages()?;
        let prompt = if let Some(template) = &self.chat_template {
            template
                .render_messages(&messages, true)
                .map_err(ApiError::internal)?
        } else {
            render_plain_prompt(&messages)
        };
        let add_special_tokens = self.chat_template.is_none();
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), add_special_tokens)
            .map_err(ApiError::internal)?;
        let prompt_ids = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(ApiError::bad_request("empty prompt"));
        }

        Ok(GenerationPlan {
            prompt_tokens: prompt_ids.len(),
            prompt_ids,
            max_tokens: req.max_tokens().unwrap_or(self.max_tokens_limit),
            temperature: req.temperature.unwrap_or(self.default_temperature),
            rep_penalty: req
                .repetition_penalty
                .unwrap_or(self.default_rep_penalty),
        })
    }

    fn generate_text(&mut self, plan: &GenerationPlan) -> Result<GenerationResult, ApiError> {
        let mut gen = Generator::new(
            &mut self.model,
            &plan.prompt_ids,
            plan.max_tokens,
            plan.temperature,
            plan.rep_penalty,
            self.eos_ids.clone(),
            self.prefill_step_size,
        )
        .map_err(ApiError::internal)?;

        let mut decode = self.tokenizer.decode_stream(false);
        let mut text = String::new();
        let mut completion_tokens = 0usize;
        for tok in &mut gen {
            let tok = tok.map_err(ApiError::internal)?;
            completion_tokens += 1;
            if let Some(piece) = decode.step(tok).map_err(ApiError::internal)? {
                text.push_str(&piece);
            }
        }

        Ok(GenerationResult {
            text,
            completion_tokens,
        })
    }
}

fn send_stream_item(
    chunk_tx: &tokio_mpsc::Sender<Result<StreamItem, ApiError>>,
    item: Result<StreamItem, ApiError>,
) -> Result<(), ApiError> {
    chunk_tx
        .blocking_send(item)
        .map_err(|_| ApiError::internal("client disconnected"))
}

fn finish_reason(completion_tokens: usize, max_tokens: usize) -> &'static str {
    if completion_tokens >= max_tokens && max_tokens > 0 {
        "length"
    } else {
        "stop"
    }
}

fn render_plain_prompt(messages: &[ChatTemplateMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out.push_str("assistant: ");
    out
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": state.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "mlx-lm-rs"
            }
        ]
    }))
}

async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = req.validate_for_server(state.max_tokens_limit) {
        return e.into_response();
    }

    let request_id = format!(
        "chatcmpl-{}-{}",
        unix_timestamp(),
        state.ids.fetch_add(1, Ordering::Relaxed)
    );

    if req.stream.unwrap_or(false) {
        let (chunk_tx, chunk_rx) = tokio_mpsc::channel(16);
        if state
            .work_tx
            .send(WorkItem::ChatStream {
                request_id,
                req,
                chunk_tx,
            })
            .is_err()
        {
            return ApiError::internal("model worker is not running").into_response();
        }

        let stream = ReceiverStream::new(chunk_rx).map(|item| match item {
            Ok(StreamItem::Chunk(chunk)) => {
                Ok::<Event, Infallible>(Event::default().json_data(chunk).unwrap())
            }
            Ok(StreamItem::Done) => Ok::<Event, Infallible>(Event::default().data("[DONE]")),
            Err(e) => Ok::<Event, Infallible>(Event::default().json_data(e.error_body()).unwrap()),
        });
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let (response_tx, response_rx) = oneshot::channel();
        if state
            .work_tx
            .send(WorkItem::Chat {
                request_id,
                req,
                response_tx,
            })
            .is_err()
        {
            return ApiError::internal("model worker is not running").into_response();
        }

        match response_rx.await {
            Ok(Ok(response)) => Json(response).into_response(),
            Ok(Err(e)) => e.into_response(),
            Err(_) => ApiError::internal("model worker dropped request").into_response(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<RequestMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
}

impl ChatCompletionRequest {
    fn validate(&self) -> Result<(), ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::bad_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(ApiError::bad_request("messages must not be empty"));
        }
        if let Some(n) = self.n {
            if n != 1 {
                return Err(ApiError::bad_request("only n=1 is supported"));
            }
        }
        if self.max_tokens().is_some_and(|x| x == 0) {
            return Err(ApiError::bad_request(
                "max_tokens/max_completion_tokens must be greater than 0",
            ));
        }
        if let Some(temp) = self.temperature {
            if !(temp == 0.0 || (temp.is_finite() && temp > 0.0)) {
                return Err(ApiError::bad_request(
                    "temperature must be 0.0 or a finite positive value",
                ));
            }
        }
        if let Some(rp) = self.repetition_penalty {
            if !(rp.is_finite() && rp > 0.0) {
                return Err(ApiError::bad_request(
                    "repetition_penalty must be a finite positive value",
                ));
            }
        }
        if self.stop.is_some() {
            return Err(ApiError::bad_request(
                "stop sequences are not supported yet",
            ));
        }
        if self.tools.is_some() || self.tool_choice.is_some() {
            return Err(ApiError::bad_request("tool calling is not supported yet"));
        }
        if let Some(format) = &self.response_format {
            if format.kind != "text" {
                return Err(ApiError::bad_request(
                    "only response_format {\"type\":\"text\"} is supported",
                ));
            }
        }
        Ok(())
    }

    fn validate_for_server(&self, max_tokens_limit: usize) -> Result<(), ApiError> {
        self.validate()?;
        self.validate_content()?;
        if let Some(max_tokens) = self.max_tokens() {
            if max_tokens > max_tokens_limit {
                return Err(ApiError::bad_request(format!(
                    "max_tokens/max_completion_tokens must be less than or equal to server limit {max_tokens_limit}"
                )));
            }
        }
        Ok(())
    }

    fn max_tokens(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    fn validate_content(&self) -> Result<(), ApiError> {
        for message in &self.messages {
            message.validate_content()?;
        }
        Ok(())
    }

    fn template_messages(&self) -> Result<Vec<ChatTemplateMessage>, ApiError> {
        self.messages
            .iter()
            .map(|m| {
                Ok(ChatTemplateMessage {
                    role: m.role.clone(),
                    content: m.content_text()?,
                })
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct RequestMessage {
    role: String,
    content: MessageContent,
}

impl RequestMessage {
    fn validate_content(&self) -> Result<(), ApiError> {
        self.content_text().map(|_| ())
    }

    fn content_text(&self) -> Result<String, ApiError> {
        match &self.content {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part.kind.as_str() {
                        "text" => {
                            let Some(part_text) = &part.text else {
                                return Err(ApiError::bad_request(
                                    "text content parts require a text field",
                                ));
                            };
                            text.push_str(part_text);
                        }
                        other => {
                            return Err(ApiError::bad_request(format!(
                                "content part type {other:?} is not supported"
                            )));
                        }
                    }
                }
                Ok(text)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

#[derive(Debug, Deserialize)]
struct MessageContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

struct GenerationPlan {
    prompt_ids: Vec<u32>,
    prompt_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    rep_penalty: f32,
}

struct GenerationResult {
    text: String,
    completion_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

enum StreamItem {
    Chunk(ChatCompletionChunk),
    Done,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunkChoice {
    index: usize,
    delta: DeltaMessage,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }

    fn error_body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": if self.status == StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                "code": null
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.error_body())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("valid request json")
    }

    fn test_serve_config(model: String) -> ServeConfig {
        ServeConfig {
            model,
            host: "127.0.0.1".into(),
            port: 0,
            default_max_tokens: 16,
            default_temperature: 0.0,
            default_rep_penalty: 1.0,
            prefill_step_size: NonZeroUsize::new(2048).unwrap(),
            no_chat_template: false,
        }
    }

    fn test_app_state(work_tx: mpsc::Sender<WorkItem>) -> AppState {
        AppState {
            model_id: "qwen".into(),
            max_tokens_limit: 16,
            work_tx,
            ids: Arc::new(AtomicU64::new(1)),
        }
    }

    #[test]
    fn validates_minimal_chat_completion_request() {
        let req = request(json!({
            "model": "mlx-community/Qwen3-0.6B-bf16",
            "messages": [{"role": "user", "content": "hello"}]
        }));

        req.validate().expect("request should validate");
        assert_eq!(req.max_tokens(), None);

        let messages = req.template_messages().expect("template messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
    }

    #[test]
    fn max_completion_tokens_takes_precedence_over_max_tokens() {
        let req = request(json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
            "max_completion_tokens": 20
        }));

        assert_eq!(req.max_tokens(), Some(20));
    }

    #[test]
    fn accepts_text_content_parts() {
        let req = request(json!({
            "model": "qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": " world"}
                    ]
                }
            ]
        }));

        let messages = req.template_messages().expect("template messages");
        assert_eq!(messages[0].content, "hello world");
    }

    #[test]
    fn rejects_unsupported_multimodal_content_parts() {
        let req = request(json!({
            "model": "qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                    ]
                }
            ]
        }));

        let err = req
            .template_messages()
            .expect_err("image parts are unsupported");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("image_url"));
    }

    #[test]
    fn rejects_max_tokens_above_server_limit() {
        let req = request(json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 17
        }));

        let err = req
            .validate_for_server(16)
            .expect_err("request should exceed server limit");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("server limit 16"));
    }

    #[test]
    fn rejects_unsupported_openai_options_explicitly() {
        let cases = [
            (
                json!({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "n": 2
                }),
                "only n=1",
            ),
            (
                json!({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": -1
                }),
                "temperature",
            ),
            (
                json!({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stop": ["END"]
                }),
                "stop sequences",
            ),
            (
                json!({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": []
                }),
                "tool calling",
            ),
            (
                json!({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "response_format": {"type": "json_object"}
                }),
                "response_format",
            ),
        ];

        for (value, expected) in cases {
            let req = request(value);
            let err = req.validate().expect_err("request should be rejected");
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
            assert!(
                err.message.contains(expected),
                "expected {expected:?} in {:?}",
                err.message
            );
        }
    }

    #[test]
    fn renders_plain_prompt_when_template_is_disabled() {
        let prompt = render_plain_prompt(&[
            ChatTemplateMessage {
                role: "system".into(),
                content: "Be terse.".into(),
            },
            ChatTemplateMessage {
                role: "user".into(),
                content: "Ping".into(),
            },
        ]);

        assert_eq!(prompt, "system: Be terse.\nuser: Ping\nassistant: ");
    }

    #[test]
    fn finish_reason_reports_length_only_when_cap_is_hit() {
        assert_eq!(finish_reason(10, 10), "length");
        assert_eq!(finish_reason(9, 10), "stop");
        assert_eq!(finish_reason(0, 0), "stop");
    }

    #[test]
    fn serializes_chat_completion_response_shape() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion",
            created: 123,
            model: "qwen".into(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: "hello".into(),
                },
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
        };

        let value = serde_json::to_value(response).expect("serialize response");
        assert_eq!(value["id"], "chatcmpl-test");
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "hello");
        assert_eq!(value["usage"]["total_tokens"], 5);
    }

    #[test]
    fn model_worker_startup_reports_initialization_failure() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_dir = std::env::temp_dir().join(format!("mlx-lm-rs-empty-model-{unique}"));
        std::fs::create_dir_all(&model_dir).expect("create temp model dir");

        let (_work_tx, work_rx) = mpsc::channel();
        let err = spawn_model_worker(
            test_serve_config(model_dir.to_string_lossy().into_owned()),
            work_rx,
        )
        .expect_err("empty model dir should fail startup");

        std::fs::remove_dir_all(&model_dir).expect("remove temp model dir");

        let message = format!("{err:#}");
        assert!(message.contains("model worker failed to initialize"));
        assert!(message.contains("loading config.json"));
    }

    #[tokio::test]
    async fn invalid_streaming_request_returns_bad_request_before_sse() {
        let (work_tx, work_rx) = mpsc::channel();
        let state = test_app_state(work_tx.clone());
        let req = request(json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "n": 2
        }));

        let response = create_chat_completion(State(state), Json(req)).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(matches!(work_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn invalid_streaming_content_returns_bad_request_before_sse() {
        let (work_tx, work_rx) = mpsc::channel();
        let state = test_app_state(work_tx.clone());
        let req = request(json!({
            "model": "qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                    ]
                }
            ],
            "stream": true
        }));

        let response = create_chat_completion(State(state), Json(req)).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(matches!(work_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }
}
