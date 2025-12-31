import cmath
import random

def poly_eval(c, x):
    y = 0
    for a in reversed(c):
        y = y * x + a
    return y

def _trim(c, eps=0.0):
    c = list(c)
    while len(c) > 1 and abs(c[-1]) <= eps:
        c.pop()
    return c

def root(c, tol=1e-10, max_iter=3000, seed=0):
    c = _trim(c)
    n = len(c) - 1
    if n <= 0:
        raise ValueError("多項式次數必須 >= 1，且最高次係數不可為 0")

    if n == 1:
        # c0 + c1*x = 0
        return [(-c[0]) / c[1]]

    lead = c[-1]
    c = [a / lead for a in c]

    random.seed(seed)

    R = 1.0 + max(abs(a) for a in c[:-1])
    roots = []
    for k in range(n):
        ang = 2 * cmath.pi * (k / n)
        jitter = (random.random() - 0.5) * 1e-3 
        roots.append((R * (1 + jitter)) * cmath.exp(1j * ang))

    for _ in range(max_iter):
        new_roots = []
        max_move = 0.0

        for i in range(n):
            zi = roots[i]

            denom = 1.0 + 0j
            for j in range(n):
                if j != i:
                    denom *= (zi - roots[j])

            if abs(denom) < 1e-18:
                zi += (1e-6 + 1e-6j)
                denom = 1.0 + 0j
                for j in range(n):
                    if j != i:
                        denom *= (zi - roots[j])

            fz = poly_eval(c, zi)
            zi_new = zi - fz / denom

            move = abs(zi_new - zi)
            if move > max_move:
                max_move = move
            new_roots.append(zi_new)

        roots = new_roots
        if max_move < tol:
            break

    roots.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return roots


if __name__ == "__main__":
    c = [-1, 0, 0, 0, 0, 1]
    rs = root(c)

    print("roots:")
    for r in rs:
        val = poly_eval(c, r)
        print(r, " |P(r)|=", abs(val))
