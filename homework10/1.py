import numpy as np

def dft(fx, x, omega):
    x = np.asarray(x, dtype=float)
    omega = np.asarray(omega, dtype=float)
    fx = np.asarray(fx, dtype=complex)

    if len(x) < 2 or len(omega) < 2:
        raise ValueError("x 與 omega 至少需要 2 個點")
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x 必須等距取樣")
    dω = omega[1] - omega[0]
    if not np.allclose(np.diff(omega), dω):
        raise ValueError("omega 必須等距取樣")

    Fw = np.zeros(len(omega), dtype=complex)
    for i, w in enumerate(omega):
        Fw[i] = np.sum(fx * np.exp(-1j * w * x)) * dx
    return Fw


def idft(Fw, omega, x):
    x = np.asarray(x, dtype=float)
    omega = np.asarray(omega, dtype=float)
    Fw = np.asarray(Fw, dtype=complex)

    if len(x) < 2 or len(omega) < 2:
        raise ValueError("x 與 omega 至少需要 2 個點")
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x 必須等距取樣")
    dω = omega[1] - omega[0]
    if not np.allclose(np.diff(omega), dω):
        raise ValueError("omega 必須等距取樣")

    fx = np.zeros(len(x), dtype=complex)
    for i, xi in enumerate(x):
        fx[i] = np.sum(Fw * np.exp(1j * omega * xi)) * dω / (2 * np.pi)
    return fx


def verify_example():
    L = 10.0
    W = 10.0
    N = 2000

    x = np.linspace(-L, L, N)
    omega = np.linspace(-W, W, N)

    fx = np.exp(-x**2)

    Fw = dft(fx, x, omega)
    fx_rec = idft(Fw, omega, x)

    max_err = np.max(np.abs(fx - fx_rec.real))
    rms_err = np.sqrt(np.mean((fx - fx_rec.real)**2))

    print("max error =", max_err)
    print("rms error =", rms_err)

    return x, fx, omega, Fw, fx_rec


if __name__ == "__main__":
    verify_example()
