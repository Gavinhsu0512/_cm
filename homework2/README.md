#  [AI對話](https://chatgpt.com/c/6954c2ca-9fbc-8322-907c-f924c2b769cd)
# 二次多項式求根（root2）

## 一、目標

使用 Python 撰寫一個函數 `root2(a, b, c)`，求解二次多項式：

\[
f(x) = ax^2 + bx + c
\]

並且滿足以下條件：

- 正確計算兩個根
- 當判別式為負時，能回傳 **複數根**
- 將求得的根代回原多項式，驗證結果應非常接近 0

---

## 二、解題想法

二次多項式的根可由標準公式求得：

\[
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\]

然而，當判別式 \(b^2 - 4ac < 0\) 時，根會是複數。因此，使用 Python 的 `cmath` 模組來處理平方根運算，以同時支援實數與複數情況。

---

## 三、程式設計說明

### 1. root2(a, b, c)：計算二次多項式的根

```python
def root2(a, b, c):
    if a == 0:
        raise ValueError("a 不能為 0（否則不是二次多項式）")

    d = b**2 - 4*a*c
    sqrt_d = cmath.sqrt(d)
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)

    return x1, x2
