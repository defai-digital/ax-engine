fn main() {
    // mlx-sys resolves which libmlx to link (MLX_LIB_DIR env, pip wheel, or
    // Homebrew) and exports the directory as DEP_MLX_LIB_DIR. The pip wheel's
    // libmlx.dylib carries an `@rpath/` install name, so every mlx-linking
    // executable this package produces (kernel probes in src/bin, test
    // binaries) must embed the matching rpath or it fails to load — or worse,
    // silently resolves a different MLX build than the one compiled against.
    if let Ok(lib_dir) = std::env::var("DEP_MLX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
