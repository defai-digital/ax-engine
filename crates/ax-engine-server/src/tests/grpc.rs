use crate::grpc::AxEngineGrpcService;
use crate::grpc::proto;
use crate::grpc::proto::ax_engine_server::AxEngine;

use super::fixtures::llama_cpp_server_state;

#[tokio::test]
async fn grpc_health_reports_unavailable_when_session_is_busy() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let service = AxEngineGrpcService::new(state.clone());
    let live = state.snapshot();
    let session_guard = live.request_session.lock().await;

    let error = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect_err("busy session should not report healthy");

    assert_eq!(error.code(), tonic::Code::Unavailable);
    assert_eq!(
        error.message(),
        "ax-engine-server has not finished initialising its inference session"
    );

    drop(session_guard);
    let response = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect("unlocked session should report healthy")
        .into_inner();
    assert_eq!(response.status, "ok");
    assert_eq!(response.service, "ax-engine-server");
}

#[tokio::test]
async fn grpc_models_reports_stable_ax_engine_owner() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let service = AxEngineGrpcService::new(state);

    let response = service
        .models(tonic::Request::new(proto::ModelsRequest {}))
        .await
        .expect("models should respond")
        .into_inner();

    assert_eq!(response.object, "list");
    let model = response.data.first().expect("model card");
    assert_eq!(model.owned_by, "ax-engine");
}
