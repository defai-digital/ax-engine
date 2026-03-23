//! Matrix multiply via Apple Accelerate `cblas_sgemm`.
//!
//! C = alpha * A * B + beta * C
//!
//! All matrices are row-major. A is m x k, B is k x n, C is m x n.

use std::os::raw::c_int;

// cblas enums — Accelerate uses i32 values
const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// General matrix multiply: C = A * B  (row-major)
///
/// A: m x k, B: k x n, C: m x n (output, overwritten)
///
/// Uses Apple Accelerate `cblas_sgemm` which dispatches to AMX on M-series.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(
        a.len() >= m * k,
        "matmul: a.len()={} < m*k={}",
        a.len(),
        m * k
    );
    assert!(
        b.len() >= k * n,
        "matmul: b.len()={} < k*n={}",
        b.len(),
        k * n
    );
    assert!(
        c.len() >= m * n,
        "matmul: c.len()={} < m*n={}",
        c.len(),
        m * n
    );

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0, // alpha
            a.as_ptr(),
            k as c_int, // lda = k for row-major A (m x k)
            b.as_ptr(),
            n as c_int, // ldb = n for row-major B (k x n)
            0.0,        // beta
            c.as_mut_ptr(),
            n as c_int, // ldc = n for row-major C (m x n)
        );
    }
}

/// Matrix-vector multiply: y = A * x  (row-major)
///
/// A: m x k, x: k, y: m (output, overwritten)
///
/// This is matmul with n=1, optimized path through cblas_sgemm.
pub fn matvec_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    matmul_f32(a, x, y, m, 1, k);
}

/// Batched matrix multiply: C[i] = A[i] * B[i] for i in 0..batch
///
/// Each matrix is contiguous in memory:
///   A: batch * m * k, B: batch * k * n, C: batch * m * n
pub fn matmul_f32_batched(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) {
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    assert!(a.len() >= batch * a_stride);
    assert!(b.len() >= batch * b_stride);
    assert!(c.len() >= batch * c_stride);

    for i in 0..batch {
        matmul_f32(
            &a[i * a_stride..],
            &b[i * b_stride..],
            &mut c[i * c_stride..],
            m,
            n,
            k,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // A = 2x2 identity, B = 2x2 [[1,2],[3,4]]
        // C = A * B = B
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_simple() {
        // A = [[1,2,3],[4,5,6]] (2x3)
        // B = [[7,8],[9,10],[11,12]] (3x2)
        // C = [[58,64],[139,154]] (2x2)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_1x1() {
        // Scalar multiply: 3 * 4 = 12
        let a = [3.0];
        let b = [4.0];
        let mut c = [0.0f32; 1];
        matmul_f32(&a, &b, &mut c, 1, 1, 1);
        assert!((c[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec() {
        // A = [[1,2],[3,4]] (2x2), x = [5,6] (2x1)
        // y = [1*5+2*6, 3*5+4*6] = [17, 39]
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [5.0, 6.0];
        let mut y = [0.0f32; 2];
        matvec_f32(&a, &x, &mut y, 2, 2);
        assert!((y[0] - 17.0).abs() < 1e-4);
        assert!((y[1] - 39.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_rectangular() {
        // A = [[1,0,2]] (1x3), B = [[3],[1],[2]] (3x1)
        // C = [1*3 + 0*1 + 2*2] = [7] (1x1)
        let a = [1.0, 0.0, 2.0];
        let b = [3.0, 1.0, 2.0];
        let mut c = [0.0f32; 1];
        matmul_f32(&a, &b, &mut c, 1, 1, 3);
        assert!((c[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_zeros() {
        let a = [0.0f32; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [999.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        for &v in &c {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_batched() {
        // 2 batches of 2x2 matmul
        let a = [
            1.0, 0.0, 0.0, 1.0, // batch 0: identity
            2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
        ];
        let b = [
            1.0, 2.0, 3.0, 4.0, // batch 0
            1.0, 2.0, 3.0, 4.0, // batch 1
        ];
        let mut c = [0.0f32; 8];
        matmul_f32_batched(&a, &b, &mut c, 2, 2, 2, 2);

        // batch 0: I * B = B
        assert_eq!(&c[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // batch 1: 2I * B = 2B
        assert!((c[4] - 2.0).abs() < 1e-4);
        assert!((c[5] - 4.0).abs() < 1e-4);
        assert!((c[6] - 6.0).abs() < 1e-4);
        assert!((c[7] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_large() {
        // 64x64 identity * 64x64 sequential = sequential
        let n = 64;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let b: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; n * n];
        matmul_f32(&identity, &b, &mut c, n, n, n);
        for i in 0..n * n {
            assert!(
                (c[i] - b[i]).abs() < 1e-3,
                "c[{i}] = {}, expected {}",
                c[i],
                b[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "matmul: a.len()")]
    fn test_matmul_short_a() {
        let a = [1.0f32; 3]; // too short for 2x2
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "matmul: b.len()")]
    fn test_matmul_short_b() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 3]; // too short for 2x2
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "matmul: c.len()")]
    fn test_matmul_short_c() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 3]; // too short for 2x2
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
    }
}
