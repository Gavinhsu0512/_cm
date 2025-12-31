# [AI對話](https://chatgpt.com/c/69556b42-7634-8323-9e27-0e0ff9abc714)

# 目的
`solve_ode_general(coefficients)` 用來解「常係數齊次線性常微分方程」的一般解：

\[
a_0 y^{(n)} + a_1 y^{(n-1)} + \cdots + a_n y = 0
\]

輸入 `coefficients = [a0, a1, ..., an]`，回傳一般解的字串 `y(x) = ...`。

---

# 數學背景

## (1) 特徵方程
令解為 \(y=e^{\lambda x}\) 代入可得特徵多項式：

\[
a_0 \lambda^n + a_1 \lambda^{n-1} + \cdots + a_n = 0
\]

只要找到所有根 \(\lambda\)（含重根、複數根），就能寫出一般解。

程式用 `np.roots(coeffs)` 直接求多項式根（數值法），因此可以處理任意階 \(n\)。

---

## (2) 根 → 一般解形式

### (A) 實數單根 \(r\)
\[
\lambda=r \Rightarrow y = C e^{rx}
\]

### (B) 實數重根 \(r\)，重根次數 \(m\)
\[
(\lambda-r)^m \Rightarrow y = (C_1 + C_2 x + \cdots + C_m x^{m-1}) e^{rx}
\]

程式對每個實根重根次數 `m`，產生 \(k=0..m-1\) 的項：
- \(x^k e^{rx}\)

若 \(r=0\)，則 \(e^{0x}=1\)，項就退化成純多項式 \(C_1 + C_2 x + \cdots\)。

### (C) 複數根 \(\alpha \pm i\beta\)
\[
\lambda=\alpha\pm i\beta \Rightarrow
y = e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x))
\]

### (D) 複數重根（重根次數 \(m\)）
\[
(\lambda-(\alpha+i\beta))^m \Rightarrow
y = e^{\alpha x}\sum_{k=0}^{m-1} x^k\left(C_{k,1}\cos(\beta x)+C_{k,2}\sin(\beta x)\right)
\]

程式對每一對共軛根輸出：
- \(x^k e^{\alpha x}\cos(\beta x)\)
- \(x^k e^{\alpha x}\sin(\beta x)\)
其中 \(k=0..m-1\)。

---

## 為什麼要做「清理與聚類」（tol_zero / tol_cluster）

`np.roots` 給的是浮點近似根，常見問題：
- 理論上 0 會變成 `-1e-14`
- 理論上純實根會帶極小虛部 `2e-12j`
- 理論上重根會被算成很接近但不完全相同的多個根
- 複數根應成共軛對，但數值誤差可能導致「看起來不完全共軛」

因此程式加入兩個容忍參數：

### (1) `tol_zero = 1e-8`：去除接近 0 的雜訊
把根的實部/虛部若 `abs(value) < tol_zero` 直接設為 `0.0`，避免輸出：
- `e^(-0x)`
- `cos(1.00000000003x)` 這種雜訊

### (2) `tol_cluster = 1e-6`：把近似相同的根視為同一根（重根處理）
做法：
1. 先排序 `cleaned.sort(...)`
2. 逐個根丟進 cluster
3. 若與某 cluster 的代表（用平均值當 rep）距離小於 `tol_cluster`，就視為同一群

這樣才能把「數值上分裂的重根」重新合併，正確得到重根次數 `m`。

---

## 為什麼複數根要「只輸出 beta>0」
複數根會成對出現：\(\alpha+i\beta\) 與 \(\alpha-i\beta\)。

但一般解寫成 `cos/sin` 形式時，只需要其中一對即可，因為：
\[
e^{(\alpha+i\beta)x}, e^{(\alpha-i\beta)x}
\Rightarrow e^{\alpha x}\cos(\beta x),\; e^{\alpha x}\sin(\beta x)
\]

若兩邊都輸出會重複，因此程式：
- 只挑 `imag > 0` 的根當代表
- 用 `(alpha, |beta|)` 分組（pair bucket）避免把共軛當成兩組

---

## 輸出格式設計（fmt_num / poly_x / exp_part）

### (1) `fmt_num(x)`
用 `"{x:.10g}"` 控制浮點輸出長度，並把 `-0` 改成 `0`，避免難看字串。

### (2) `poly_x(k)`
用來輸出乘上的多項式部分：
- k=0 → `""`
- k=1 → `"x"`
- k>=2 → `"x^k"`

### (3) `exp_part(alpha)`
若 `alpha=0`，則不輸出 `e^(0x)`，直接省略，讓字串更乾淨。

---

## 程式結構對應關係（讀程式時的地圖）

1. **輸入檢查**  
   - `len(coeffs) < 2` 代表不是有效 ODE 多項式

2. **求根**  
   - `roots = np.roots(coeffs)`

3. **清理小數誤差**  
   - `tol_zero` 把接近 0 的實/虛部歸零

4. **根聚類 → 得到 (代表根, 重根次數)**  
   - `clusters` → `reps = [(rep, multiplicity)]`

5. **分類實根 / 複根**  
   - `real_roots` vs `complex_roots`

6. **產生一般解項**
   - 實根：`x^k e^(rx)`
   - 複根：`x^k e^(alpha x) cos(beta x)` + `sin(...)`

7. **回傳字串**
   - `"y(x) = " + " + ".join(terms)`

---

## 限制與注意事項
- 這是「數值求根」：高階多項式或病態係數可能造成根誤差，`tol_cluster` 只是補救，不是嚴格符號運算。
- `tol_zero` / `tol_cluster` 在不同尺度的係數下可能需要調整（係數很大或很小時，誤差尺度也會變）。

---

## 簡短使用範例

```python
# y'' - 3y' + 2y = 0 -> roots 1, 2
print(solve_ode_general([1, -3, 2]))
# y(x) = C_1e^(1x) + C_2e^(2x)

# y'' - 4y' + 4y = 0 -> root 2 重根
print(solve_ode_general([1, -4, 4]))
# y(x) = C_1e^(2x) + C_2xe^(2x)

# y'' + y = 0 -> roots ± i
print(solve_ode_general([1, 0, 1]))
# y(x) = C_1cos(1x) + C_2sin(1x)
