//! Rank-0 chain client for ordered greedy pipeline generation.

use std::time::Duration;

use ax_engine_core::{ActivationFrame, PipelineTopology};
use reqwest::header::CONTENT_TYPE;
use thiserror::Error;

use crate::{CLUSTER_WORKER_TOKEN_HEADER, RankHealth, TokenStepRequest, TokenStepResponse};

const ACTIVATION_CONTENT_TYPE: &str = "application/x-ax-pipeline-frame";
const JSON_CONTENT_TYPE: &str = "application/json";
const MAX_ERROR_BYTES: usize = 16 * 1024;

pub struct PipelineChainClient {
    topology: PipelineTopology,
    endpoints: Vec<String>,
    worker_token: String,
    maximum_activation_bytes: u64,
    client: reqwest::Client,
}

impl PipelineChainClient {
    pub fn new(
        topology: PipelineTopology,
        endpoints: Vec<String>,
        worker_token: String,
        maximum_activation_bytes: u64,
    ) -> Result<Self, PipelineClientError> {
        topology.validate()?;
        if endpoints.len() != topology.ranks.len() {
            return Err(PipelineClientError::EndpointCount {
                expected: topology.ranks.len(),
                actual: endpoints.len(),
            });
        }
        if worker_token.len() < 16 {
            return Err(PipelineClientError::WeakWorkerToken);
        }
        let endpoints = endpoints
            .into_iter()
            .map(|endpoint| endpoint.trim_end_matches('/').to_string())
            .collect::<Vec<_>>();
        if endpoints
            .iter()
            .any(|endpoint| !endpoint.starts_with("http://") && !endpoint.starts_with("https://"))
        {
            return Err(PipelineClientError::InvalidEndpoint);
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(PipelineClientError::BuildClient)?;
        Ok(Self {
            topology,
            endpoints,
            worker_token,
            maximum_activation_bytes,
            client,
        })
    }

    /// Verify endpoint ordering and immutable generation identity before the
    /// gateway accepts traffic.
    pub async fn preflight(&self) -> Result<(), PipelineClientError> {
        for (index, endpoint) in self.endpoints.iter().enumerate() {
            let response = self
                .client
                .get(format!("{endpoint}/health"))
                .header(CLUSTER_WORKER_TOKEN_HEADER, &self.worker_token)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(PipelineClientError::Request)?;
            if !response.status().is_success() {
                return Err(self.http_status_error(response).await);
            }
            let bytes = read_bounded(response, 64 * 1024).await?;
            let health = serde_json::from_slice::<RankHealth>(&bytes)
                .map_err(PipelineClientError::HealthResponse)?;
            let expected_rank =
                u16::try_from(index).map_err(|_| PipelineClientError::EndpointCount {
                    expected: self.topology.ranks.len(),
                    actual: self.endpoints.len(),
                })?;
            if health.rank != expected_rank
                || health.generation != self.topology.generation
                || health.cluster_id != self.topology.cluster_id
                || health.manifest_digest != self.topology.manifest_digest
                || health.model_artifact_digest != self.topology.model_artifact_digest
            {
                return Err(PipelineClientError::RankIdentityMismatch {
                    expected_rank,
                    actual_rank: health.rank,
                });
            }
            if !health.ready {
                return Err(PipelineClientError::RankNotReady(expected_rank));
            }
        }
        Ok(())
    }

    /// Run one prompt or decode step through every rank.
    pub async fn step(
        &self,
        request: TokenStepRequest,
    ) -> Result<TokenStepResponse, PipelineClientError> {
        if request.token_ids.is_empty() {
            return Err(PipelineClientError::EmptyTokenStep);
        }
        let response = self
            .client
            .post(format!("{}/internal/pipeline/tokens", self.endpoints[0]))
            .header(CLUSTER_WORKER_TOKEN_HEADER, &self.worker_token)
            .json(&request)
            .send()
            .await
            .map_err(PipelineClientError::Request)?;
        let mut output = self.read_step_response(response).await?;

        for (rank, endpoint) in self.endpoints.iter().enumerate().skip(1) {
            let frame = match output {
                ChainStepOutput::Activation(frame) => frame,
                ChainStepOutput::Token(_) => {
                    return Err(PipelineClientError::EarlyToken { rank: rank - 1 });
                }
            };
            if frame.header.destination_rank as usize != rank {
                return Err(PipelineClientError::WrongActivationDestination {
                    expected: rank as u16,
                    actual: frame.header.destination_rank,
                });
            }
            let encoded = frame.encode(&self.topology)?;
            let response = self
                .client
                .post(format!("{endpoint}/internal/pipeline/activation"))
                .header(CLUSTER_WORKER_TOKEN_HEADER, &self.worker_token)
                .header(CONTENT_TYPE, ACTIVATION_CONTENT_TYPE)
                .body(encoded)
                .send()
                .await
                .map_err(PipelineClientError::Request)?;
            output = self.read_step_response(response).await?;
        }

        match output {
            ChainStepOutput::Token(token)
                if token.request_id == request.request_id
                    && token.request_sequence == request.request_sequence =>
            {
                Ok(token)
            }
            ChainStepOutput::Token(token) => Err(PipelineClientError::TokenIdentityMismatch {
                expected_request_id: request.request_id,
                actual_request_id: token.request_id,
                expected_sequence: request.request_sequence,
                actual_sequence: token.request_sequence,
            }),
            ChainStepOutput::Activation(_) => Err(PipelineClientError::MissingFinalToken),
        }
    }

    /// Greedy autoregressive generation over repeated ordered chain steps.
    pub async fn generate_greedy(
        &self,
        request_id: u64,
        prompt_tokens: &[u32],
        maximum_output_tokens: usize,
        stop_token_ids: &[u32],
    ) -> Result<Vec<u32>, PipelineClientError> {
        if prompt_tokens.is_empty() {
            return Err(PipelineClientError::EmptyTokenStep);
        }
        let mut output = Vec::with_capacity(maximum_output_tokens);
        let mut sequence = 1_u64;
        let mut token_offset = 0_u64;
        let mut input = prompt_tokens.to_vec();
        while output.len() < maximum_output_tokens {
            let token = self
                .step(TokenStepRequest {
                    request_id,
                    request_sequence: sequence,
                    token_offset,
                    token_ids: input,
                })
                .await?
                .token_id;
            output.push(token);
            if stop_token_ids.contains(&token) {
                break;
            }
            token_offset = token_offset
                .checked_add(if sequence == 1 {
                    prompt_tokens.len() as u64
                } else {
                    1
                })
                .ok_or(PipelineClientError::TokenOffsetOverflow)?;
            sequence = sequence
                .checked_add(1)
                .ok_or(PipelineClientError::TokenOffsetOverflow)?;
            input = vec![token];
        }
        Ok(output)
    }

    /// Best-effort close on every rank; returns the first failure after trying all.
    pub async fn close_request(&self, request_id: u64) -> Result<(), PipelineClientError> {
        let mut first_error = None;
        for endpoint in &self.endpoints {
            let result = self
                .client
                .post(format!(
                    "{endpoint}/internal/pipeline/requests/{request_id}/close"
                ))
                .header(CLUSTER_WORKER_TOKEN_HEADER, &self.worker_token)
                .send()
                .await;
            match result {
                Ok(response) if response.status().is_success() => {}
                Ok(response) => {
                    let error = self.http_status_error(response).await;
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(error) if first_error.is_none() => {
                    first_error = Some(PipelineClientError::Request(error));
                }
                Err(_) => {}
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn read_step_response(
        &self,
        response: reqwest::Response,
    ) -> Result<ChainStepOutput, PipelineClientError> {
        if !response.status().is_success() {
            return Err(self.http_status_error(response).await);
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("");
        match content_type {
            ACTIVATION_CONTENT_TYPE => {
                let maximum_frame_bytes = usize::try_from(self.maximum_activation_bytes)
                    .unwrap_or(usize::MAX)
                    .saturating_add(65_536);
                let bytes = read_bounded(response, maximum_frame_bytes).await?;
                let frame =
                    ActivationFrame::decode(&bytes, &self.topology, self.maximum_activation_bytes)?;
                Ok(ChainStepOutput::Activation(frame))
            }
            JSON_CONTENT_TYPE => {
                let bytes = read_bounded(response, 64 * 1024).await?;
                let token = serde_json::from_slice::<TokenStepResponse>(&bytes)
                    .map_err(PipelineClientError::TokenResponse)?;
                Ok(ChainStepOutput::Token(token))
            }
            other => Err(PipelineClientError::UnexpectedContentType(
                other.to_string(),
            )),
        }
    }

    async fn http_status_error(&self, response: reqwest::Response) -> PipelineClientError {
        let status = response.status();
        let body = read_bounded(response, MAX_ERROR_BYTES)
            .await
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_else(|error| format!("failed to read error body: {error}"));
        PipelineClientError::HttpStatus { status, body }
    }
}

enum ChainStepOutput {
    Activation(ActivationFrame),
    Token(TokenStepResponse),
}

async fn read_bounded(
    mut response: reqwest::Response,
    maximum_bytes: usize,
) -> Result<Vec<u8>, PipelineClientError> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(PipelineClientError::ResponseTooLarge {
            maximum: maximum_bytes,
        });
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(PipelineClientError::Request)?
    {
        let next_len =
            bytes
                .len()
                .checked_add(chunk.len())
                .ok_or(PipelineClientError::ResponseTooLarge {
                    maximum: maximum_bytes,
                })?;
        if next_len > maximum_bytes {
            return Err(PipelineClientError::ResponseTooLarge {
                maximum: maximum_bytes,
            });
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[derive(Debug, Error)]
pub enum PipelineClientError {
    #[error(transparent)]
    Contract(#[from] ax_engine_core::PipelineContractError),
    #[error("pipeline endpoint count mismatch: expected {expected}, got {actual}")]
    EndpointCount { expected: usize, actual: usize },
    #[error("pipeline endpoints must use explicit http:// or https:// URLs")]
    InvalidEndpoint,
    #[error("cluster worker token must contain at least 16 bytes")]
    WeakWorkerToken,
    #[error("pipeline token step must contain at least one token")]
    EmptyTokenStep,
    #[error("failed to construct pipeline HTTP client: {0}")]
    BuildClient(reqwest::Error),
    #[error("pipeline HTTP request failed: {0}")]
    Request(reqwest::Error),
    #[error("pipeline rank returned HTTP {status}: {body}")]
    HttpStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("rank {rank} returned a token before the final stage")]
    EarlyToken { rank: usize },
    #[error("activation destination mismatch: expected {expected}, got {actual}")]
    WrongActivationDestination { expected: u16, actual: u16 },
    #[error("final pipeline rank did not return a token")]
    MissingFinalToken,
    #[error(
        "token identity mismatch: expected request {expected_request_id}/{expected_sequence}, got {actual_request_id}/{actual_sequence}"
    )]
    TokenIdentityMismatch {
        expected_request_id: u64,
        actual_request_id: u64,
        expected_sequence: u64,
        actual_sequence: u64,
    },
    #[error("invalid token response JSON: {0}")]
    TokenResponse(serde_json::Error),
    #[error("invalid rank health JSON: {0}")]
    HealthResponse(serde_json::Error),
    #[error(
        "pipeline endpoint identity mismatch: expected rank {expected_rank}, endpoint reported rank {actual_rank}"
    )]
    RankIdentityMismatch {
        expected_rank: u16,
        actual_rank: u16,
    },
    #[error("pipeline rank {0} is not ready")]
    RankNotReady(u16),
    #[error("unexpected pipeline response content type {0:?}")]
    UnexpectedContentType(String),
    #[error("pipeline response exceeds {maximum} bytes")]
    ResponseTooLarge { maximum: usize },
    #[error("token offset or request sequence overflow")]
    TokenOffsetOverflow,
    #[error("pipeline generation deadline exceeded")]
    DeadlineExceeded,
    #[error("pipeline request cleanup deadline exceeded")]
    CloseDeadlineExceeded,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ax_engine_core::{PipelineLayerRange, PipelineRankAssignment};

    use super::*;
    use crate::{
        RankHealth, RankProcessor, RankServiceError, RankStepOutput, TokenStepRequest,
        TokenStepResponse, router,
    };

    fn topology() -> PipelineTopology {
        PipelineTopology {
            cluster_id: "cluster-a".into(),
            generation: 1,
            manifest_digest: "manifest-a".into(),
            model_artifact_digest: "model-a".into(),
            total_layers: 2,
            ranks: vec![
                PipelineRankAssignment {
                    rank: 0,
                    node_identity_digest: "node-a".into(),
                    layers: PipelineLayerRange { start: 0, end: 1 },
                    owns_embeddings: true,
                    owns_output_head: false,
                },
                PipelineRankAssignment {
                    rank: 1,
                    node_identity_digest: "node-b".into(),
                    layers: PipelineLayerRange { start: 1, end: 2 },
                    owns_embeddings: false,
                    owns_output_head: true,
                },
            ],
        }
    }

    #[test]
    fn client_rejects_endpoint_count_and_implicit_schemes() {
        assert!(matches!(
            PipelineChainClient::new(
                topology(),
                vec!["http://rank0".into()],
                "0123456789abcdef".into(),
                1024,
            ),
            Err(PipelineClientError::EndpointCount {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            PipelineChainClient::new(
                topology(),
                vec!["rank0".into(), "rank1".into()],
                "0123456789abcdef".into(),
                1024,
            ),
            Err(PipelineClientError::InvalidEndpoint)
        ));
    }

    struct ChainProcessor {
        topology: PipelineTopology,
        rank: u16,
        closes: AtomicUsize,
    }

    impl RankProcessor for ChainProcessor {
        fn health(&self) -> RankHealth {
            RankHealth {
                ready: true,
                rank: self.rank,
                generation: self.topology.generation,
                cluster_id: self.topology.cluster_id.clone(),
                manifest_digest: self.topology.manifest_digest.clone(),
                model_artifact_digest: self.topology.model_artifact_digest.clone(),
            }
        }

        fn process_tokens(
            &self,
            request: TokenStepRequest,
        ) -> Result<RankStepOutput, RankServiceError> {
            assert_eq!(self.rank, 0);
            let payload = vec![0_u8; request.token_ids.len() * 4];
            let frame = ActivationFrame {
                header: ax_engine_core::ActivationFrameHeader {
                    wire_version: ax_engine_core::PIPELINE_WIRE_VERSION,
                    cluster_id: self.topology.cluster_id.clone(),
                    generation: self.topology.generation,
                    manifest_digest: self.topology.manifest_digest.clone(),
                    model_artifact_digest: self.topology.model_artifact_digest.clone(),
                    request_id: request.request_id,
                    request_sequence: request.request_sequence,
                    source_rank: 0,
                    destination_rank: 1,
                    layer_boundary: 1,
                    token_offset: request.token_offset,
                    token_count: request.token_ids.len() as u32,
                    dtype: ax_engine_core::ActivationDtype::Float32,
                    shape: vec![1, request.token_ids.len() as u32, 1],
                    payload_bytes: payload.len() as u64,
                    payload_sha256: ax_engine_core::sha256_hex(&payload),
                },
                payload,
            };
            Ok(RankStepOutput::Activation(frame.encode(&self.topology)?))
        }

        fn process_activation(&self, bytes: &[u8]) -> Result<RankStepOutput, RankServiceError> {
            assert_eq!(self.rank, 1);
            let frame = ActivationFrame::decode(bytes, &self.topology, 1024)?;
            Ok(RankStepOutput::Token(TokenStepResponse {
                request_id: frame.header.request_id,
                request_sequence: frame.header.request_sequence,
                token_id: 77,
            }))
        }

        fn close_request(&self, _request_id: u64) -> Result<(), RankServiceError> {
            self.closes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn chain_client_crosses_two_authenticated_http_rank_services() {
        let topology = topology();
        let rank0 = Arc::new(ChainProcessor {
            topology: topology.clone(),
            rank: 0,
            closes: AtomicUsize::new(0),
        });
        let rank1 = Arc::new(ChainProcessor {
            topology: topology.clone(),
            rank: 1,
            closes: AtomicUsize::new(0),
        });
        let listener0 = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("rank 0 listener");
        let listener1 = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("rank 1 listener");
        let endpoint0 = format!("http://{}", listener0.local_addr().expect("rank 0 address"));
        let endpoint1 = format!("http://{}", listener1.local_addr().expect("rank 1 address"));
        let app0 = router(
            Arc::clone(&rank0) as Arc<dyn RankProcessor>,
            "0123456789abcdef".into(),
            4096,
        )
        .expect("rank 0 router");
        let app1 = router(
            Arc::clone(&rank1) as Arc<dyn RankProcessor>,
            "0123456789abcdef".into(),
            4096,
        )
        .expect("rank 1 router");
        let server0 = tokio::spawn(async move {
            axum::serve(listener0, app0).await.expect("rank 0 server");
        });
        let server1 = tokio::spawn(async move {
            axum::serve(listener1, app1).await.expect("rank 1 server");
        });
        let reversed = PipelineChainClient::new(
            topology.clone(),
            vec![endpoint1.clone(), endpoint0.clone()],
            "0123456789abcdef".into(),
            1024,
        )
        .expect("reversed chain client");
        assert!(matches!(
            reversed.preflight().await,
            Err(PipelineClientError::RankIdentityMismatch {
                expected_rank: 0,
                actual_rank: 1
            })
        ));
        let client = PipelineChainClient::new(
            topology,
            vec![endpoint0, endpoint1],
            "0123456789abcdef".into(),
            1024,
        )
        .expect("chain client");
        client.preflight().await.expect("rank identity preflight");
        let token = client
            .step(TokenStepRequest {
                request_id: 5,
                request_sequence: 1,
                token_offset: 0,
                token_ids: vec![1, 2],
            })
            .await
            .expect("chain step");
        assert_eq!(token.token_id, 77);
        client.close_request(5).await.expect("chain close");
        assert_eq!(rank0.closes.load(Ordering::Relaxed), 1);
        assert_eq!(rank1.closes.load(Ordering::Relaxed), 1);
        server0.abort();
        server1.abort();
    }
}
