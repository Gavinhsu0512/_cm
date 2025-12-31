from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Union


def _is_prime(p: int) -> bool:
    if p <= 1:
        return False
    if p <= 3:
        return True
    if p % 2 == 0 or p % 3 == 0:
        return False
    i = 5
    while i * i <= p:
        if p % i == 0 or p % (i + 2) == 0:
            return False
        i += 6
    return True


class FiniteField:
    """Prime field F_p."""

    def __init__(self, p: int):
        if not isinstance(p, int):
            raise TypeError("p must be int")
        if not _is_prime(p):
            raise ValueError("For this implementation, p must be prime (GF(p)).")
        self.p = p
        self._add_group = FiniteFieldAddGroup(self)
        self._mul_group = FiniteFieldMulGroup(self)

    def element(self, value: Union[int, 'FFElement']) -> 'FFElement':
        if isinstance(value, FFElement):
            if value.field is not self:
                raise TypeError("Cannot mix elements from different fields")
            return value
        if not isinstance(value, int):
            raise TypeError("Field elements must be created from int or FFElement")
        return FFElement(self, value % self.p)

    def __call__(self, value: Union[int, 'FFElement']) -> 'FFElement':
        return self.element(value)

    @property
    def zero(self) -> 'FFElement':
        return self.element(0)

    @property
    def one(self) -> 'FFElement':
        return self.element(1)

    @property
    def elements(self) -> List['FFElement']:
        return [self.element(i) for i in range(self.p)]

    @property
    def nonzero_elements(self) -> List['FFElement']:
        return [self.element(i) for i in range(1, self.p)]

    @property
    def add_group(self) -> 'FiniteFieldAddGroup':
        return self._add_group

    @property
    def mul_group(self) -> 'FiniteFieldMulGroup':
        return self._mul_group


class FiniteField:
    """Prime field F_p."""

    def __init__(self, p: int):
        if not isinstance(p, int):
            raise TypeError("p must be int")
        if not _is_prime(p):
            raise ValueError("For this implementation, p must be prime (GF(p)).")
        self.p = p
        self._add_group = FiniteFieldAddGroup(self)
        self._mul_group = FiniteFieldMulGroup(self)

    def element(self, value: Union[int, "FFElement"]) -> "FFElement":
        if isinstance(value, FFElement):
            if value.field is not self:
                raise ValueError("Cannot mix elements from different fields")
            return value
        if not isinstance(value, int):
            raise TypeError("Finite field elements must be constructed from int")
        return FFElement(self, value % self.p)

    def __call__(self, value: Union[int, "FFElement"]) -> "FFElement":
        return self.element(value)

    @property
    def zero(self) -> "FFElement":
        return self.element(0)

    @property
    def one(self) -> "FFElement":
        return self.element(1)

    @property
    def elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(self.p)]

    @property
    def nonzero_elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(1, self.p)]

    @property
    def add_group(self) -> "FiniteFieldAddGroup":
        return self._add_group

    @property
    def mul_group(self) -> "FiniteFieldMulGroup":
        return self._mul_group

    def __repr__(self) -> str:
        return f"GF({self.p})"

    def element(self, value: Union[int, "FFElement"]) -> "FFElement":
        if isinstance(value, FFElement):
            if value.field is not self:
                raise ValueError("Cannot mix elements from different fields")
            return value
        if not isinstance(value, int):
            raise TypeError("Finite field elements must be constructed from int")
        return FFElement(self, value % self.p)

    def __call__(self, value: Union[int, "FFElement"]) -> "FFElement":
        return self.element(value)

    @property
    def zero(self) -> "FFElement":
        return self.element(0)

    @property
    def one(self) -> "FFElement":
        return self.element(1)

    def elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(self.p)]

    def nonzero_elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(1, self.p)]

    @property
    def add_group(self) -> "FiniteFieldAddGroup":
        return self._add_group

    @property
    def mul_group(self) -> "FiniteFieldMulGroup":
        return self._mul_group

    @property
    def zero(self) -> "FFElement":
        return self.element(0)

    @property
    def one(self) -> "FFElement":
        return self.element(1)

    def elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(self.p)]

    def nonzero_elements(self) -> List["FFElement"]:
        return [self.element(i) for i in range(1, self.p)]

    @property
    def add_group(self) -> "FiniteFieldAddGroup":
        return self._add_group

    @property
    def mul_group(self) -> "FiniteFieldMulGroup":
        return self._mul_group

    def __repr__(self) -> str:
        return f"F_{self.p}"


@dataclass(frozen=True)
class FFElement:
    field: FiniteField
    value: int

    def _coerce(self, other: Union[int, "FFElement"]) -> "FFElement":
        return self.field.element(other)

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value} (mod {self.field.p})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (FFElement, int)):
            return False
        try:
            o = self._coerce(other)  
        except Exception:
            return False
        return self.value == o.value and self.field is o.field

    def __hash__(self) -> int:
        return hash((id(self.field), self.value))


@dataclass(frozen=True)
class FFElement:
    field: FiniteField
    value: int

    def _coerce(self, other: Union[int, "FFElement"]) -> "FFElement":
        return self.field.element(other)

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value} (mod {self.field.p})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FFElement):
            return False
        return self.field is other.field and self.value == other.value

    def __hash__(self) -> int:
        return hash((id(self.field), self.value))

    def __neg__(self) -> "FFElement":
        return FFElement(self.field, (-self.value) % self.field.p)

    def __add__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (self.value + o.value) % self.field.p)

    def __radd__(self, other: Union[int, "FFElement"]) -> "FFElement":
        return self.__add__(other)

    def __sub__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (self.value - o.value) % self.field.p)

    def __rsub__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (o.value - self.value) % self.field.p)

    def __sub__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (self.value - o.value) % self.field.p)

    def __rsub__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (o.value - self.value) % self.field.p)

    def __mul__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return FFElement(self.field, (self.value * o.value) % self.field.p)

    def __rmul__(self, other: Union[int, "FFElement"]) -> "FFElement":
        return self.__mul__(other)

    def inverse(self) -> "FFElement":
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse")
        inv = pow(self.value, self.field.p - 2, self.field.p)
        return FFElement(self.field, inv)

    def __truediv__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return self * o.inverse()

    def __rtruediv__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return o * self.inverse()

    def __pow__(self, n: int) -> "FFElement":
        if not isinstance(n, int):
            raise TypeError("exponent must be int")
        return FFElement(self.field, pow(self.value, n, self.field.p))

    def inverse(self) -> "FFElement":
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse")
        return FFElement(self.field, pow(self.value, self.field.p - 2, self.field.p))

    def __truediv__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return self * o.inverse()

    def __rtruediv__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return o * self.inverse()

    def __pow__(self, n: int) -> "FFElement":
        if not isinstance(n, int):
            raise TypeError("Exponent must be int")
        return FFElement(self.field, pow(self.value, n, self.field.p))

    def __rtruediv__(self, other: Union[int, "FFElement"]) -> "FFElement":
        o = self._coerce(other)
        return o * self.inverse()

    def __pow__(self, n: int) -> "FFElement":
        if not isinstance(n, int):
            raise TypeError("Exponent must be int")
        if n < 0:
            return (self.inverse()) ** (-n)
        return FFElement(self.field, pow(self.value, n, self.field.p))


class _BaseGroup:
    """Tiny adapter for common group-axiom checker conventions."""

    def __iter__(self) -> Iterator[FFElement]:
        return iter(self.elements)

    @property
    def elems(self) -> List[FFElement]:
        return list(self.elements)

    def identity(self) -> FFElement:
        return self.e()

    def inverse(self, a: FFElement) -> FFElement:
        return self.inv(a)


class FiniteFieldAddGroup(_BaseGroup):
    def __init__(self, field: FiniteField):
        self.field = field
        self.elements: List[FFElement] = field.elements()

    def op(self, a: FFElement, b: FFElement) -> FFElement:
        return a + b

    def e(self) -> FFElement:
        return self.field.zero

    def inv(self, a: FFElement) -> FFElement:
        return -a


class FiniteFieldMulGroup(_BaseGroup):
    def __init__(self, field: FiniteField):
        self.field = field
        self.elements: List[FFElement] = field.nonzero_elements()

    def op(self, a: FFElement, b: FFElement) -> FFElement:
        return a * b

    def e(self) -> FFElement:
        return self.field.one

    def inv(self, a: FFElement) -> FFElement:
        return a.inverse()


if __name__ == "__main__":
    F7 = FiniteField(7)
    a = F7(3)
    b = F7(5)
    print("Field:", F7)
    print("a, b:", a, b)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("b**3 =", b ** 3)

    G_add = F7.add_group
    G_mul = F7.mul_group
    print("|Add group| =", len(G_add.elements))
    print("|Mul group| =", len(G_mul.elements))
    print("Add identity:", G_add.e())
    print("Mul identity:", G_mul.e())
    print("Add inverse of a:", G_add.inv(a))
    print("Mul inverse of a:", G_mul.inv(a))

class _BaseGroup:
    """Tiny adapter for common group-axiom checker conventions."""

    def __iter__(self) -> Iterator[FFElement]:
        return iter(self.elements)

    @property
    def elems(self) -> List[FFElement]:
        return list(self.elements)

    def identity(self) -> FFElement:
        return self.e()

    def inverse(self, a: FFElement) -> FFElement:
        return self.inv(a)


class _BaseGroup:
    """Tiny adapter for common group-axiom checker conventions."""

    def __iter__(self) -> Iterator[FFElement]:
        return iter(self.elements)

    @property
    def elems(self) -> List[FFElement]:
        return list(self.elements)

    def identity(self) -> FFElement:
        return self.e()


class FiniteFieldAddGroup(_BaseGroup):
    def __init__(self, field: FiniteField):
        self.field = field
        self.elements = field.elements()

    def op(self, a: FFElement, b: FFElement) -> FFElement:
        return a + b

    def e(self) -> FFElement:
        return self.field.zero

    def inv(self, a: FFElement) -> FFElement:
        return -a


class FiniteFieldMulGroup(_BaseGroup):
    def __init__(self, field: FiniteField):
        self.field = field
        self.elements = field.nonzero_elements()

    def op(self, a: FFElement, b: FFElement) -> FFElement:
        return a * b

    def e(self) -> FFElement:
        return self.field.one

    def inv(self, a: FFElement) -> FFElement:
        return a.inverse()


def demo() -> None:
    F7 = FiniteField(7)
    a = F7(3)
    b = F7(5)

    print("Field:", F7)
    print("a =", a)
    print("b =", b)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("b ** 4 =", b ** 4)

    Gadd = F7.add_group
    Gmul = F7.mul_group
    print("Additive identity:", Gadd.e())
    print("Additive inverse of a:", Gadd.inv(a))
    print("Multiplicative identity:", Gmul.e())
    print("Multiplicative inverse of a:", Gmul.inv(a))


if __name__ == "__main__":
    demo()
