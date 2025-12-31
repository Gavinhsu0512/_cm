import math

def df(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def integral(f, a, b, n=10000):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

def theorem1(f, x):
    left = df(lambda t: integral(f, 0, t), x)
    right = f(x)
    print("x =", x)
    print("d/dx ∫₀ˣ f(t)dt ≈", left)
    print("f(x) =", right)
    print("誤差 =", abs(left - right))
    print("-" * 30)

def f(x):
    return x * x

for x in [0.5, 1.0, 2.0]:
    theorem1(f, x)
