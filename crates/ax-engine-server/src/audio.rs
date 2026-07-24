//! OpenAI-compatible native Whisper transcription and translation endpoints.

use ax_engine_sdk::SpeechTranscription;
use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::app_state::AppState;
use crate::errors::{
    ErrorResponse, admission_error_response, error_response, map_generation_service_error,
};
use crate::metadata::model_family_from_artifacts;
use crate::multimodal::decode_audio_waveform;
use crate::openai::validation::select_model;

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const MAX_AUDIO_SECONDS: usize = 60 * 60;
const MAX_AUDIO_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * MAX_AUDIO_SECONDS;

#[derive(Default)]
struct AudioRequest {
    file: Option<Vec<u8>>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
}

#[derive(Serialize)]
struct JsonTranscription {
    text: String,
}

#[derive(Serialize)]
struct VerboseTranscription {
    task: &'static str,
    language: String,
    duration: f32,
    text: String,
    segments: Vec<serde_json::Value>,
}

pub(crate) async fn audio_transcriptions(
    state: State<AppState>,
    multipart: Multipart,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    serve_audio(state, multipart, false).await
}

pub(crate) async fn audio_translations(
    state: State<AppState>,
    multipart: Multipart,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    serve_audio(state, multipart, true).await
}

async fn serve_audio(
    State(state): State<AppState>,
    multipart: Multipart,
    translate: bool,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let request = parse_audio_request(multipart).await?;
    let live = select_model(&state, request.model.as_deref())?;
    if model_family_from_artifacts(&live).as_deref() != Some("whisper") {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "audio transcription/translation requires a loaded native Whisper model".to_string(),
        ));
    }
    let file = request.file.ok_or_else(|| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "multipart field 'file' is required".to_string(),
        )
    })?;
    let samples =
        decode_audio_waveform(&file, WHISPER_SAMPLE_RATE, MAX_AUDIO_SAMPLES).map_err(|error| {
            error_response(StatusCode::BAD_REQUEST, "invalid_audio", error.to_string())
        })?;
    let duration = samples.len() as f32 / WHISPER_SAMPLE_RATE as f32;
    let language = request
        .language
        .filter(|language| !language.trim().is_empty());
    let response_format = request.response_format.as_deref().unwrap_or("json");
    if !matches!(response_format, "json" | "text" | "verbose_json") {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "response_format '{response_format}' is unsupported; use json, text, or verbose_json"
            ),
        ));
    }

    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let generation_service = live.generation_service.clone();
    let transcription = generation_service
        .execute(move |session| {
            let _permit = permit;
            session.transcribe_audio(&samples, language.as_deref(), translate)
        })
        .await
        .map_err(map_generation_service_error)?;

    Ok(render_response(
        transcription,
        response_format,
        duration,
        translate,
    ))
}

async fn parse_audio_request(
    mut multipart: Multipart,
) -> Result<AudioRequest, (StatusCode, Json<ErrorResponse>)> {
    let mut request = AudioRequest::default();
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                if request.file.is_some() {
                    return Err(error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "multipart field 'file' may only be provided once".to_string(),
                    ));
                }
                request.file = Some(field.bytes().await.map_err(multipart_error)?.to_vec());
            }
            "model" => request.model = Some(field.text().await.map_err(multipart_error)?),
            "language" => request.language = Some(field.text().await.map_err(multipart_error)?),
            "response_format" => {
                request.response_format = Some(field.text().await.map_err(multipart_error)?);
            }
            "temperature" => {
                let value = field.text().await.map_err(multipart_error)?;
                validate_temperature(&value)?;
            }
            "prompt" => {
                let value = field.text().await.map_err(multipart_error)?;
                if !value.is_empty() {
                    return Err(error_response(
                        StatusCode::BAD_REQUEST,
                        "unsupported_parameter",
                        "prompt is not yet supported by the native Whisper decoder".to_string(),
                    ));
                }
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                return Err(error_response(
                    StatusCode::BAD_REQUEST,
                    "unsupported_parameter",
                    "timestamp_granularities require timestamp decoding, which is not yet supported"
                        .to_string(),
                ));
            }
            _ => {
                // Preserve OpenAI-compatible forward compatibility: unknown
                // optional multipart fields do not invalidate the audio.
            }
        }
    }
    Ok(request)
}

fn validate_temperature(value: &str) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let temperature = value.parse::<f32>().map_err(|_| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "temperature must be a finite number".to_string(),
        )
    })?;
    if !temperature.is_finite() || temperature != 0.0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "native Whisper currently supports greedy decoding only (temperature=0)".to_string(),
        ));
    }
    Ok(())
}

fn multipart_error(
    error: axum::extract::multipart::MultipartError,
) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!("invalid multipart audio request: {error}"),
    )
}

fn render_response(
    transcription: SpeechTranscription,
    response_format: &str,
    duration: f32,
    translate: bool,
) -> Response {
    match response_format {
        "text" => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            transcription.text,
        )
            .into_response(),
        "verbose_json" => Json(VerboseTranscription {
            task: if translate { "translate" } else { "transcribe" },
            language: transcription
                .language
                .unwrap_or_else(|| "unknown".to_string()),
            duration,
            text: transcription.text,
            segments: Vec::new(),
        })
        .into_response(),
        _ => Json(JsonTranscription {
            text: transcription.text,
        })
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_temperature_is_the_only_supported_value() {
        assert!(validate_temperature("0").is_ok());
        assert!(validate_temperature("0.0").is_ok());
        assert!(validate_temperature("0.1").is_err());
        assert!(validate_temperature("NaN").is_err());
        assert!(validate_temperature("not-a-number").is_err());
    }

    #[test]
    fn verbose_response_carries_detected_language_and_task() {
        let response = render_response(
            SpeechTranscription {
                text: "hello".to_string(),
                language: Some("en".to_string()),
            },
            "verbose_json",
            1.5,
            false,
        );
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&header::HeaderValue::from_static("application/json"))
        );
    }
}
