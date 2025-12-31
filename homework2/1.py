import cmath

def root2(a, b, c):
    
    if a == 0:
        raise ValueError("a 不能為 0（否則不是二次多項式）")

    d = b**2 - 4*a*c             
    sqrt_d = cmath.sqrt(d)        
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)

    return x1, x2


def f(a, b, c, x):
    """多項式 f(x) = ax^2 + bx + c"""
    return a*x*x + b*x + c


if __name__ == "__main__":
    a, b, c = 1, 2, 5  

    r1, r2 = root2(a, b, c)

    print("根 1 =", r1)
    print("根 2 =", r2)

    v1 = f(a, b, c, r1)
    v2 = f(a, b, c, r2)

    print("\nf(r1) =", v1)
    print("f(r2) =", v2)

    print("\n驗證結果：")
    print("r1 是否為根：", cmath.isclose(v1, 0, abs_tol=1e-9))
    print("r2 是否為根：", cmath.isclose(v2, 0, abs_tol=1e-9))
