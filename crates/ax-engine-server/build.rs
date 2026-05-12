fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &["../../proto/ax_engine/v1/ax_engine.proto"],
            &["../../proto"],
        )?;
    println!("cargo:rerun-if-changed=../../proto/ax_engine/v1/ax_engine.proto");
    Ok(())
}
