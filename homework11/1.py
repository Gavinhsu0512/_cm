import numpy as np

def solve_ode_general(coefficients):
    
    coeffs = np.array(coefficients, dtype=float)
    if len(coeffs) < 2:
        raise ValueError("coefficients 長度至少要 2 (例如 [a0, a1])")

    # 數值容忍：用來把 -0、極小虛部、近似共軛/重根聚類起來
    tol_zero = 1e-8
    tol_cluster = 1e-6

    roots = np.roots(coeffs)

    # 先把接近 0 的實/虛部消掉，減少 e^(-0x) 或 cos(1.0000000001x) 這種輸出
    cleaned = []
    for r in roots:
        a = r.real
        b = r.imag
        if abs(a) < tol_zero:
            a = 0.0
        if abs(b) < tol_zero:
            b = 0.0
        cleaned.append(complex(a, b))

    # 聚類：把數值上很接近的根視為同一根，計入重根次數
    cleaned.sort(key=lambda z: (z.real, z.imag))
    clusters = []  # list of list[complex]
    for r in cleaned:
        placed = False
        for cl in clusters:
            rep = sum(cl) / len(cl)
            if abs(r - rep) < tol_cluster:
                cl.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])

    # 代表根 + 重根次數
    reps = []
    for cl in clusters:
        rep = sum(cl) / len(cl)
        a = rep.real
        b = rep.imag
        if abs(a) < tol_zero:
            a = 0.0
        if abs(b) < tol_zero:
            b = 0.0
        reps.append((complex(a, b), len(cl)))

    # 輸出格式化
    def fmt_num(x):
        # 盡量去掉浮點雜訊：-0、尾端 0
        if abs(x) < tol_zero:
            x = 0.0
        s = f"{x:.10g}"  # 足夠穩定且不會太長
        if s == "-0":
            s = "0"
        return s

    def poly_x(k):
        if k == 0:
            return ""
        if k == 1:
            return "x"
        return f"x^{k}"

    def exp_part(alpha):
        if abs(alpha) < tol_zero:
            return ""
        return f"e^({fmt_num(alpha)}x)"

    terms = []
    C_idx = 1

    # 先處理實根，再處理複根(只取 beta>0 的那一半，避免重複)
    real_roots = []
    complex_roots = []
    for r, m in reps:
        if abs(r.imag) < tol_zero:
            real_roots.append((r.real, m))
        else:
            complex_roots.append((r, m))

    # 實根： (D-r)^m -> e^(rx), x e^(rx), ..., x^(m-1) e^(rx)
    real_roots.sort(key=lambda t: t[0])
    for r, m in real_roots:
        for k in range(m):
            px = poly_x(k)
            ep = exp_part(r)
            if ep == "":
                # r=0: 只剩多項式
                if px == "":
                    term = f"C_{C_idx}"
                else:
                    term = f"C_{C_idx}{px}"
            else:
                term = f"C_{C_idx}{px}{ep}"
            terms.append(term)
            C_idx += 1

    # 複根： alpha ± i beta，重根 m
    # -> x^k e^(alpha x) cos(beta x), x^k e^(alpha x) sin(beta x) (k=0..m-1)
    # 只輸出 beta>0 的代表，避免把共軛各輸一次
    # 先把接近共軛者聚成「同一對」(用 alpha, |beta| 分組)
    pair_buckets = []
    for r, m in complex_roots:
        a = r.real
        b = r.imag
        if abs(a) < tol_zero:
            a = 0.0
        if abs(b) < tol_zero:
            continue
        key = (round(a / tol_cluster) * tol_cluster, round(abs(b) / tol_cluster) * tol_cluster)

        placed = False
        for bucket in pair_buckets:
            if bucket["key"] == key:
                bucket["items"].append((r, m))
                placed = True
                break
        if not placed:
            pair_buckets.append({"key": key, "items": [(r, m)]})

    # 對每一對共軛，取 beta>0 的那個作為代表；重根次數用該對中最大 m（數值上兩邊應相同）
    pair_buckets.sort(key=lambda d: (d["key"][0], d["key"][1]))
    for bucket in pair_buckets:
        items = bucket["items"]
        # 找 beta>0 的代表，沒有就用 abs(beta) 當 beta
        rep = None
        mult = 0
        for r, m in items:
            mult = max(mult, m)
            if r.imag > 0:
                rep = r
        if rep is None:
            r0, m0 = items[0]
            rep = complex(r0.real, abs(r0.imag))
            mult = max(mult, m0)

        alpha = rep.real
        beta = abs(rep.imag)
        if abs(alpha) < tol_zero:
            alpha = 0.0

        for k in range(mult):
            px = poly_x(k)
            ep = exp_part(alpha)
            # cos
            t1 = f"C_{C_idx}{px}{ep}cos({fmt_num(beta)}x)"
            terms.append(t1)
            C_idx += 1
            # sin
            t2 = f"C_{C_idx}{px}{ep}sin({fmt_num(beta)}x)"
            terms.append(t2)
            C_idx += 1

    return "y(x) = " + " + ".join(terms)
