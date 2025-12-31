import math

def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i + 1
        x = x - fx / df(x)
    raise ValueError("未在指定迭代次數內收斂")

f = lambda x: x**2 - 2
df = lambda x: 2*x

root, iterations = newton_method(f, df, x0=1.0)
print("近似根 =", root)
print("迭代次數 =", iterations)
print("驗證 f(root) =", f(root))
