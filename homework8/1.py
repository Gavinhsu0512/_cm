import math
from typing import List, Tuple

def prob_all_heads_fair_coin(n: int = 10000) -> float:
    """公平銅板連續投擲 n 次都正面的機率：p^n，p=0.5"""
    p = 0.5
    return p ** n 


def log_prob_power(p: float, n: int) -> float:
    """用 log(p^n) = n log(p) 計算 log(p^n)"""
    return n * math.log(p)


EPS = 1e-15  

def _clip_prob(x: float) -> float:
    return min(1.0, max(EPS, x))

def normalize(dist: List[float]) -> List[float]:
    s = sum(dist)
    if s <= 0:
        raise ValueError("distribution sum must be > 0")
    return [x / s for x in dist]

def entropy(p: List[float]) -> float:
    """H(p) = - sum p_i log p_i"""
    p = normalize(p)
    return -sum(pi * math.log(_clip_prob(pi)) for pi in p)

def cross_entropy(p: List[float], q: List[float]) -> float:
    """H(p, q) = - sum p_i log q_i"""
    p = normalize(p)
    q = normalize(q)
    if len(p) != len(q):
        raise ValueError("p and q must have same length")
    return -sum(pi * math.log(_clip_prob(qi)) for pi, qi in zip(p, q))

def kl_divergence(p: List[float], q: List[float]) -> float:
    """KL(p||q) = sum p_i log(p_i/q_i)"""
    p = normalize(p)
    q = normalize(q)
    if len(p) != len(q):
        raise ValueError("p and q must have same length")
    return sum(pi * (math.log(_clip_prob(pi)) - math.log(_clip_prob(qi))) for pi, qi in zip(p, q))

def mutual_information_from_joint(joint: List[List[float]]) -> float:
    """
    I(X;Y) = sum_{x,y} p(x,y) log ( p(x,y) / (p(x)p(y)) )
    joint: 2D list, joint[x][y] >=0，會自動正規化
    """
    total = sum(sum(row) for row in joint)
    if total <= 0:
        raise ValueError("joint sum must be > 0")
    Pxy = [[v / total for v in row] for row in joint]

    Px = [sum(row) for row in Pxy]
    Py = [sum(Pxy[x][y] for x in range(len(Pxy))) for y in range(len(Pxy[0]))]

    mi = 0.0
    for x in range(len(Pxy)):
        for y in range(len(Pxy[0])):
            pxy = Pxy[x][y]
            if pxy <= 0:
                continue
            denom = _clip_prob(Px[x]) * _clip_prob(Py[y])
            mi += pxy * (math.log(pxy) - math.log(denom))
    return mi


def verify_cross_entropy_inequality(p: List[float], q: List[float]) -> Tuple[float, float, float]:
    """
    回傳 (Hpp, Hpq, KLpq) 並做簡單檢查
    """
    Hpp = cross_entropy(p, p)
    Hpq = cross_entropy(p, q)
    KLpq = kl_divergence(p, q)

    if abs(Hpq - (Hpp + KLpq)) > 1e-9:
        raise AssertionError("Hpq != Hpp + KL(p||q) (numerical issue too large)")

    if Hpp - Hpq > 1e-12:
        raise AssertionError("Expected H(p,p) <= H(p,q), but got opposite. Check input.")

    return Hpp, Hpq, KLpq



def hamming74_encode(data4: str) -> str:
    """
    data4: 長度 4 的 '0'/'1' 字串，例如 "1011"
    回傳長度 7 的碼字字串
    """
    if len(data4) != 4 or any(c not in "01" for c in data4):
        raise ValueError("data4 must be 4 bits string of '0'/'1'")

    d1, d2, d3, d4 = (int(b) for b in data4)

    c = [0] * 8  
    c[3] = d1
    c[5] = d2
    c[6] = d3
    c[7] = d4

    p1 = (c[3] ^ c[5] ^ c[7])  
    p2 = (c[3] ^ c[6] ^ c[7]) 
    p4 = (c[5] ^ c[6] ^ c[7])  

    c[1] = p1
    c[2] = p2
    c[4] = p4

    return "".join(str(c[i]) for i in range(1, 8))


def hamming74_decode(code7: str) -> Tuple[str, int, str]:
    """
    code7: 長度 7 的 '0'/'1' 字串
    回傳 (decoded_data4, error_position, corrected_code7)
      - error_position: 0 表示無錯；1..7 表示更正了該位置的單一位元錯誤
    """
    if len(code7) != 7 or any(c not in "01" for c in code7):
        raise ValueError("code7 must be 7 bits string of '0'/'1'")

    c = [0] * 8
    for i, ch in enumerate(code7, start=1):
        c[i] = int(ch)

    s1 = c[1] ^ c[3] ^ c[5] ^ c[7]
    s2 = c[2] ^ c[3] ^ c[6] ^ c[7]
    s4 = c[4] ^ c[5] ^ c[6] ^ c[7]

    err_pos = s1 * 1 + s2 * 2 + s4 * 4  # 0..7

    if err_pos != 0:
        c[err_pos] ^= 1  

    data4 = f"{c[3]}{c[5]}{c[6]}{c[7]}"
    corrected = "".join(str(c[i]) for i in range(1, 8))
    return data4, err_pos, corrected


def main():
    p_pow = prob_all_heads_fair_coin(10000)
    print("0.5^10000 (float) =", p_pow)  


    log_val = log_prob_power(0.5, 10000)
    print("log(0.5^10000) =", log_val)
    print("log2(0.5^10000) =", log_val / math.log(2))  


    p = [0.1, 0.2, 0.7]
    q = [0.2, 0.2, 0.6]
    print("\nEntropy H(p) =", entropy(p))
    print("Cross-entropy H(p,p) =", cross_entropy(p, p))
    print("Cross-entropy H(p,q) =", cross_entropy(p, q))
    print("KL(p||q) =", kl_divergence(p, q))

    joint = [
        [0.10, 0.10],
        [0.20, 0.60],
    ]
    print("Mutual information I(X;Y) from joint =", mutual_information_from_joint(joint))

    Hpp, Hpq, KLpq = verify_cross_entropy_inequality(p, q)
    print("\nVerify: H(p,q) = H(p,p) + KL(p||q)")
    print("H(p,p) =", Hpp)
    print("H(p,q) =", Hpq)
    print("KL(p||q) =", KLpq)
    print("Check inequality: H(p,p) <= H(p,q) :", Hpp <= Hpq)

    data = "1011"
    code = hamming74_encode(data)
    print("\nHamming(7,4) encode:", data, "->", code)

    tampered = list(code)
    tampered[5] = "1" if tampered[5] == "0" else "0" 
    tampered = "".join(tampered)
    print("Tampered code (bit6 flipped):", tampered)

    dec, pos, corrected = hamming74_decode(tampered)
    print("Decoded data =", dec)
    print("Error position corrected =", pos)
    print("Corrected code =", corrected)


if __name__ == "__main__":
    main()
