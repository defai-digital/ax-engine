fn main() {
    // Link Apple Accelerate framework for BLAS (cblas_sgemm etc.)
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
