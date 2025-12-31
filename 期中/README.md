#  [AI對話](https://chatgpt.com/c/69556d4a-2abc-8322-9ef4-310a0948f8f2)

## 1. 牛頓法在做什麼

牛頓法是一個「用切線逼近根」的方法：

- 先猜一個初始值 x0
- 在 x0 位置對 f(x) 畫切線
- 切線與 x 軸的交點會得到下一個更好的近似值 x1
- 重複迭代直到 f(x) 足夠接近 0

直覺：曲線不好解，但切線（直線）很好解，所以每一步用切線近似曲線，快速逼近根。

---

## 2. 數學公式

牛頓法的更新公式：

x_{n+1} = x_n - f(x_n) / f'(x_n)

其中：
- f'(x) 是 f(x) 的導數
- x_n 是第 n 次近似
- x_{n+1} 是下一次近似

---

## 3. 程式在做什麼

程式提供 `newton_method(f, df, x0, tol, max_iter)`：

輸入：
- f：目標函數 f(x)
- df：導數函數 f'(x)
- x0：初始猜測
- tol：停止門檻（當 |f(x)| < tol 視為收斂）
- max_iter：最大迭代次數（避免無限迴圈）

輸出：
- root：近似根
- iterations：迭代次數

流程：
1. 計算 fx = f(x)
2. 若 |fx| < tol，停止並回傳
3. 計算 dfx = df(x)，若 dfx == 0 則無法更新（會除以 0）
4. 用公式更新：x = x - fx / dfx
5. 重複直到收斂或超過 max_iter

---

## 4. 程式碼（newton.py）

```python
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

5. 範例：求 sqrt(2)（或 √2）

想求 √2，可以把它改寫成方程式：

f(x) = x^2 - 2

根就是使得 x^2 - 2 = 0 的 x，也就是 x = √2。

導數為：

f'(x) = 2x

6. 執行方式

建立 newton.py，在命令列執行：
 python newton.py
 
7. 注意事項（為什麼有時會失敗）

牛頓法雖然快，但不保證一定成功：

初始值 x0 選太差可能發散或跑到別的根

若 f'(x_n) = 0 會除以 0

函數在根附近若非常平坦，可能收斂慢或震盪

因此程式用：

max_iter 防止無限迴圈

檢查 dfx == 0 避免除以 0



