import cmath

def root3(a, b, c, d):
    
    if a == 0:
        raise ValueError("a 不能為 0")

    p = (3*a*c - b*b) / (3*a*a)
    q = (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27*a*a*a)

    delta = (q/2)**2 + (p/3)**3

    sqrt_delta = cmath.sqrt(delta)
    A = (-q/2 + sqrt_delta) ** (1/3)
    B = (-q/2 - sqrt_delta) ** (1/3)

    y1 = A + B
    y2 = -(A + B)/2 + (A - B)*cmath.sqrt(3)*1j/2
    y3 = -(A + B)/2 - (A - B)*cmath.sqrt(3)*1j/2

    shift = b / (3*a)
    x1 = y1 - shift
    x2 = y2 - shift
    x3 = y3 - shift

    return x1, x2, x3


if __name__ == "__main__":
    a, b, c, d = 1, 0, 0, 1   # x^3 + 1 = 0

    roots = root3(a, b, c, d)

    print("三個根：")
    for r in roots:
        print(r)

    print("\n驗證 f(x) ≈ 0：")
    for r in roots:
        value = a*r**3 + b*r**2 + c*r + d
        print(value, "->", cmath.isclose(value, 0, abs_tol=1e-9))
