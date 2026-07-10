use crate::grpc::AxEngineGrpcService;
use crate::grpc::proto;
use crate::grpc::proto::ax_engine_server::AxEngine;
use std::sync::mpsc;
use std::time::Duration;

use super::fixtures::llama_cpp_server_state;

#[tokio::test]
async fn grpc_health_remains_available_while_session_is_busy() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let service = AxEngineGrpcService::new(state.clone());
    let live = state.snapshot();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    live.generation_service
        .submit(move |_| {
            entered_tx.send(()).expect("test should receive entry");
            release_rx.recv().expect("test should release worker");
        })
        .expect("test job should submit");
    entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("test job should start");

    let response = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect("busy live worker should report healthy")
        .into_inner();
    assert_eq!(response.status, "ok");
    assert_eq!(response.service, "ax-engine-server");

    release_tx.send(()).expect("test should release worker");
    tokio::time::timeout(Duration::from_secs(1), async {
        while live.generation_service.is_busy() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker should become idle");
    let response = service
        .health(tonic::Request::new(proto::HealthRequest {}))
        .await
        .expect("idle live worker should report healthy")
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
