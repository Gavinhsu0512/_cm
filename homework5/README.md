# [AI對話](https://chatgpt.com/c/6954c27a-e820-8321-b4dd-3e342d342a44)
# Finite Field（有限體）實作說明

實作一個 **質數階有限體** \( \mathbb{F}_p \)（又稱 GF(p)），並符合以下需求：

1. **有限體的加法形成一個群**
   - 對應 `field_rational.py` 中的 `RationalAddGroup`
   - 可通過 `group_axioms.py` 的群公理檢驗
2. **有限體的乘法（去除 0）形成一個群**
   - 對應 `field_rational.py` 中的 `RationalMulGroup`
   - 可通過 `group_axioms.py` 的群公理檢驗
3. **加法對乘法滿足分配律**
   - 可通過 `field_axioms.py` 中的 `check_distributivity()` 驗證
4. **有限體以類別方式封裝**
   - 可建立有限體物件與其元素
5. **支援運算子重載**
   - 可像整數、浮點數一樣使用 `+ - * / **`

---

## 檔案說明

- `FiniteField`  
  代表一個有限體 \( \mathbb{F}_p \)，其中 `p` 必須是質數
- `FFElement`  
  有限體中的元素，支援加減乘除與冪次運算
- `FiniteFieldAddGroup`  
  有限體加法群（所有元素）
- `FiniteFieldMulGroup`  
  有限體乘法群（非 0 元素）
- `demo()`  
  示範有限體與群的基本用法

---

## 數學背景

對於質數 \( p \)：

- 有限體元素集合為  
  \[
  \mathbb{F}_p = \{0, 1, 2, \dots, p-1\}
  \]
- 加法與乘法皆以 **mod p** 定義
- 乘法反元素使用 **費馬小定理**  
  \[
  a^{-1} \equiv a^{p-2} \pmod{p}
  \]

---

## 基本使用方式

### 建立有限體

```python
from finite_field import FiniteField

F = FiniteField(7)   # 建立 F_7

a = F(3)
b = F(5)

a + b    # 1 (mod 7)
a - b    # 5 (mod 7)
a * b    # 1 (mod 7)
a / b    # 2 (mod 7)
b ** 3   # 6 (mod 7)

G_add = F.add_group

G_add.e()        # 加法單位元 0
G_add.inv(a)     # a 的加法反元素
G_add.op(a, b)   # a + b

G_mul = F.mul_group

G_mul.e()        # 乘法單位元 1
G_mul.inv(a)     # a 的乘法反元素
G_mul.op(a, b)   # a * b

from finite_field import FiniteField
from group_axioms import check_group

F = FiniteField(7)

check_group(F.add_group)   # 檢驗加法群
check_group(F.mul_group)   # 檢驗乘法群（非 0）

from field_axioms import check_distributivity

check_distributivity(F)

