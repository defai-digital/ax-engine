fn main() {
    // Keep the rank service on the exact MLX dylib selected by mlx-sys.
    if let Ok(lib_dir) = std::env::var("DEP_MLX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
