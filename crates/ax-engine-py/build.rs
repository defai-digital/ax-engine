fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    pyo3_build_config::add_extension_module_link_args();
    if let Some(lib_dir) = pyo3_build_config::get().lib_dir.as_deref() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
