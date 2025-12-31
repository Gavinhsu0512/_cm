import numpy as np

def det_recursive(A: np.ndarray) -> float:
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square")

    if n == 0:
        return 1.0
    if n == 1:
        return float(A[0, 0])
    if n == 2:
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    det = 0.0
    # 展開第 0 列
    for j in range(n):
        a = A[0, j]
        if a == 0:
            continue
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        det += ((-1) ** j) * a * det_recursive(minor)
    return float(det)

def lu_decomposition_partial_pivot(A: np.ndarray, tol: float = 1e-12):
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swap_count = 0

    for k in range(n):
        # pivot row: k..n-1 找 |U[i,k]| 最大
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot, k]) < tol:
            # 近似奇異：這欄無法當 pivot
            continue

        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swap_count += 1

        # elimination
        for i in range(k + 1, n):
            if abs(U[k, k]) < tol:
                L[i, k] = 0.0
                continue
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U, swap_count


def det_via_lu(A: np.ndarray) -> float:
    P, L, U, swap_count = lu_decomposition_partial_pivot(A)
    detU = float(np.prod(np.diag(U)))
    detP = -1.0 if (swap_count % 2 == 1) else 1.0
    return detP * detU

def reconstruction_errors(A: np.ndarray):
    A = np.array(A, dtype=float)

    P, L, U, _ = lu_decomposition_partial_pivot(A)
    A_lu = P.T @ L @ U
    lu_err = float(np.linalg.norm(A - A_lu))

    if np.allclose(A, A.T, atol=1e-10):
        w, V = np.linalg.eigh(A)
        A_eig = V @ np.diag(w) @ V.T
        eig_err = float(np.linalg.norm(A - A_eig))
    else:
        w, V = np.linalg.eig(A)
        A_eig = V @ np.diag(w) @ np.linalg.inv(V)
        eig_err = float(np.linalg.norm(A.astype(complex) - A_eig))

    # SVD: A = U Σ V^T
    U_s, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_svd = U_s @ np.diag(s) @ Vt
    svd_err = float(np.linalg.norm(A - A_svd))

    return {"lu_error": lu_err, "eig_error": eig_err, "svd_error": svd_err}

def svd_via_eig(A: np.ndarray, tol: float = 1e-12):
    """
    由 A^T A = V Σ^2 V^T
    回傳 U, s, Vt 使得 A ≈ U diag(s) Vt
    注意：這方法教學用，數值上不如直接 np.linalg.svd 穩定。
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    B = A.T @ A  # n x n
    evals, V = np.linalg.eigh(B)  # 對稱半正定

    # 由大到小
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = V[:, idx]

    s = np.sqrt(np.clip(evals, 0, None))

    # U = A V Σ^{-1}
    U = np.zeros((m, n))
    for i in range(n):
        if s[i] > tol:
            U[:, i] = (A @ V[:, i]) / s[i]
        else:
            U[:, i] = 0.0

    return U, s, V.T


def svd_via_eig_error(A: np.ndarray) -> float:
    U, s, Vt = svd_via_eig(A)
    A_hat = U @ np.diag(s) @ Vt
    return float(np.linalg.norm(A - A_hat))

def pca_via_svd(X: np.ndarray, k: int):
    X = np.array(X, dtype=float)
    n, d = X.shape
    if not (1 <= k <= d):
        raise ValueError("k must be between 1 and d")

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    components = Vt[:k, :]       # k x d
    Z = Xc @ components.T        # n x k

    var = (s**2) / (n - 1)       # 每個奇異值對應的變異量
    explained_variance_ratio = var[:k] / var.sum()

    return {
        "mean": mean.ravel(),
        "components": components,
        "Z": Z,
        "explained_variance_ratio": explained_variance_ratio,
    }


def main():
    np.set_printoptions(precision=6, suppress=True)

    print("=== (A) 行列式：遞迴 vs LU vs numpy ===")
    A3 = np.array([[2, 1, 3],
                   [0, 4, 5],
                   [7, 2, 1]], dtype=float)
    dr = det_recursive(A3)
    dl = det_via_lu(A3)
    dn = float(np.linalg.det(A3))
    print("A3=\n", A3)
    print("det_recursive =", dr)
    print("det_via_lu   =", dl)
    print("numpy det    =", dn)
    print("abs diff(rec, np) =", abs(dr - dn))
    print("abs diff(lu , np) =", abs(dl - dn))
    print()

    print("=== (B) LU 分解驗證：P A = L U ===")
    P, L, U, swaps = lu_decomposition_partial_pivot(A3)
    print("swap_count =", swaps)
    print("||P@A - L@U|| =", float(np.linalg.norm(P @ A3 - L @ U)))
    print()

    print("=== (C) 分解重建誤差：LU / Eigen / SVD ===")
    np.random.seed(0)
    A = np.random.randn(5, 5)
    errs = reconstruction_errors(A)
    print("A=\n", A)
    print("errors =", errs)
    print()

    print("=== (D) 用特徵值分解做 SVD（由 A^T A）重建誤差 ===")
    B = np.random.randn(6, 4)
    print("||B - svd_via_eig(B)|| =", svd_via_eig_error(B))
    print()

    print("=== (E) PCA via SVD demo ===")
    X = np.random.randn(100, 5)
    out = pca_via_svd(X, k=2)
    print("mean =", out["mean"])
    print("components shape =", out["components"].shape)
    print("Z shape =", out["Z"].shape)
    print("explained_variance_ratio =", out["explained_variance_ratio"])


if __name__ == "__main__":
    main()
