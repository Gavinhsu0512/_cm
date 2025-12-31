from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple


Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float
    df: Optional[float]
    n1: int
    n2: Optional[int]
    alternative: Alternative

    def __str__(self) -> str:
        df_str = "None" if self.df is None else f"{self.df:.6g}"
        n2_str = "None" if self.n2 is None else str(self.n2)
        return (
            f"TestResult(test={self.test}, statistic={self.statistic:.6g}, "
            f"p_value={self.p_value:.6g}, df={df_str}, n1={self.n1}, n2={n2_str}, "
            f"alternative={self.alternative})"
        )

def _to_list(x: Iterable[float]) -> List[float]:
    xs = list(x)
    if len(xs) == 0:
        raise ValueError("sample is empty")
    for v in xs:
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"invalid numeric value in sample: {v!r}")
    return [float(v) for v in xs]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sample_var(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations to compute sample variance")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def sample_std(xs: List[float]) -> float:
    return math.sqrt(sample_var(xs))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_sf(z: float) -> float:
    return 1.0 - _normal_cdf(z)



def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    am = 1.0
    bm = 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap

    if abs(bz) < 1e-300:
        bz = 1e-300

    em2 = 0.0
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm

        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz

        am, bm = az, bz
        az, bz = app, bpp

        if abs(bz) < 1e-300:
            bz = 1e-300

        if m == 1:
            em2 = 2.0

        if abs(az - am) < eps * abs(az):
            return az / bz

    return az / bz 


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be in [0,1]")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a,b must be > 0")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    ln_bt = a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    bt = math.exp(ln_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(a, b, x)
        return bt * cf / a
    else:
        cf = _betacf(b, a, 1.0 - x)
        return 1.0 - bt * cf / b


def _t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be > 0")
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    ib = _reg_incomplete_beta(x, a, b)
    if t >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib


def _p_value_from_cdf(cdf_at_stat: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return max(0.0, min(1.0, 2.0 * min(cdf_at_stat, 1.0 - cdf_at_stat)))
    elif alternative == "greater":
        return max(0.0, min(1.0, 1.0 - cdf_at_stat))
    elif alternative == "less":
        return max(0.0, min(1.0, cdf_at_stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

def z_test_1sample(
    sample: Iterable[float],
    mu0: float,
    sigma: float,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """
    One-sample z-test: population mean mu0 known, population std sigma known.
    """
    xs = _to_list(sample)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    n = len(xs)
    xbar = mean(xs)
    se = sigma / math.sqrt(n)
    z = (xbar - mu0) / se

    cdf = _normal_cdf(z)
    p = _p_value_from_cdf(cdf, alternative)

    return TestResult(
        test="z_test_1sample",
        statistic=z,
        p_value=p,
        df=None,
        n1=n,
        n2=None,
        alternative=alternative,
    )


def t_test_1sample(
    sample: Iterable[float],
    mu0: float,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """
    One-sample t-test: population mean mu0 known, population std unknown.
    """
    xs = _to_list(sample)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations for 1-sample t-test")

    xbar = mean(xs)
    s = sample_std(xs)
    se = s / math.sqrt(n)
    t = (xbar - mu0) / se
    df = n - 1

    cdf = _t_cdf(t, df)
    p = _p_value_from_cdf(cdf, alternative)

    return TestResult(
        test="t_test_1sample",
        statistic=t,
        p_value=p,
        df=float(df),
        n1=n,
        n2=None,
        alternative=alternative,
    )


def t_test_independent(
    sample1: Iterable[float],
    sample2: Iterable[float],
    alternative: Alternative = "two-sided",
    equal_var: bool = False,
) -> TestResult:
    """
    Two-sample independent t-test.
    - equal_var=False: Welch's t-test (default)
    - equal_var=True : pooled variance (classical Student t-test)
    """
    x1 = _to_list(sample1)
    x2 = _to_list(sample2)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")

    m1, m2 = mean(x1), mean(x2)
    v1, v2 = sample_var(x1), sample_var(x2)

    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        t = (m1 - m2) / se
        cdf = _t_cdf(t, df)
        p = _p_value_from_cdf(cdf, alternative)
        return TestResult(
            test="t_test_independent_pooled",
            statistic=t,
            p_value=p,
            df=float(df),
            n1=n1,
            n2=n2,
            alternative=alternative,
        )
    else:
        se2 = v1 / n1 + v2 / n2
        se = math.sqrt(se2)
        t = (m1 - m2) / se

        num = se2 * se2
        den = (v1 * v1) / (n1 * n1 * (n1 - 1)) + (v2 * v2) / (n2 * n2 * (n2 - 1))
        df = num / den

        cdf = _t_cdf(t, df)
        p = _p_value_from_cdf(cdf, alternative)
        return TestResult(
            test="t_test_independent_welch",
            statistic=t,
            p_value=p,
            df=float(df),
            n1=n1,
            n2=n2,
            alternative=alternative,
        )


def t_test_paired(
    before: Iterable[float],
    after: Iterable[float],
    alternative: Alternative = "two-sided",
) -> TestResult:
    """
    Paired t-test: same individuals measured twice.
    Performs 1-sample t-test on differences (after - before).
    """
    b = _to_list(before)
    a = _to_list(after)
    if len(b) != len(a):
        raise ValueError("paired samples must have the same length")
    if len(b) < 2:
        raise ValueError("need at least 2 pairs")

    diffs = [ai - bi for ai, bi in zip(a, b)]
    res = t_test_1sample(diffs, mu0=0.0, alternative=alternative)
    return TestResult(
        test="t_test_paired",
        statistic=res.statistic,
        p_value=res.p_value,
        df=res.df,
        n1=res.n1,
        n2=None,
        alternative=alternative,
    )


if __name__ == "__main__":
    x = [102, 100, 98, 101, 99, 103, 100, 97, 104, 99]

    print(z_test_1sample(x, mu0=100, sigma=5, alternative="two-sided"))
    print(t_test_1sample(x, mu0=100, alternative="two-sided"))

    g1 = [10, 12, 9, 11, 10, 13]
    g2 = [8, 7, 9, 6, 8, 7]
    print(t_test_independent(g1, g2, alternative="greater", equal_var=False))  # Welch

    before = [50, 52, 49, 51, 50]
    after = [53, 54, 50, 52, 55]
    print(t_test_paired(before, after, alternative="greater"))
