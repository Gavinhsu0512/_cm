from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

EPS = 1e-9

def is_close(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Point":  # scalar
        return Point(self.x * k, self.y * k)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def norm2(self) -> float:
        return self.dot(self)

    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    # 幾何變換（對點）
    def translate(self, dx: float, dy: float) -> "Point":
        return Point(self.x + dx, self.y + dy)

    def scale(self, s: float, about: "Point" = None) -> "Point":
        # 以 about 為縮放中心：P' = about + s*(P-about)
        if about is None:
            about = Point(0.0, 0.0)
        v = self - about
        return about + v * s

    def rotate(self, theta: float, about: "Point" = None) -> "Point":
        # 以 about 為旋轉中心：P' = about + R(theta)*(P-about)
        if about is None:
            about = Point(0.0, 0.0)
        v = self - about
        c = math.cos(theta)
        s = math.sin(theta)
        return about + Point(c * v.x - s * v.y, s * v.x + c * v.y)

@dataclass(frozen=True)
class Line:
    # 一般式 ax + by + c = 0
    a: float
    b: float
    c: float

    @staticmethod
    def from_points(p1: Point, p2: Point) -> "Line":
        # 通過兩點的直線：令法向量為 (dy, -dx)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        a = dy
        b = -dx
        c = -(a * p1.x + b * p1.y)
        if abs(a) < EPS and abs(b) < EPS:
            raise ValueError("兩點重合，無法定義直線")
        return Line(a, b, c)

    def direction(self) -> Point:
        # 直線方向向量可取 (b, -a)
        return Point(self.b, -self.a)

    def normal(self) -> Point:
        return Point(self.a, self.b)

    def eval(self, p: Point) -> float:
        return self.a * p.x + self.b * p.y + self.c

    def intersection_line(self, other: "Line") -> Optional[Point]:
        # 解聯立：
        # a1 x + b1 y + c1 = 0
        # a2 x + b2 y + c2 = 0
        d = self.a * other.b - other.a * self.b
        if abs(d) < EPS:
            return None  # 平行或重合：此處回傳 None
        x = (self.b * other.c - other.b * self.c) / d
        y = (other.a * self.c - self.a * other.c) / d
        return Point(x, y)

    def perpendicular_through(self, p: Point) -> "Line":
        # 原線法向量 n=(a,b)，垂線方向與 n 同向
        # 垂線一般式：b x - a y + c' = 0 (其法向量為 (b,-a) 即原線方向)
        A = self.b
        B = -self.a
        C = -(A * p.x + B * p.y)
        if abs(A) < EPS and abs(B) < EPS:
            raise ValueError("無法建立垂線（原線係數異常）")
        return Line(A, B, C)

    def foot_of_perpendicular(self, p: Point) -> Point:
        # 垂足 = p 在直線上的正交投影
        # 公式：t = (a x0 + b y0 + c) / (a^2 + b^2)
        # (x',y') = (x0 - a t, y0 - b t)
        denom = self.a * self.a + self.b * self.b
        if denom < EPS:
            raise ValueError("直線係數無效")
        t = (self.a * p.x + self.b * p.y + self.c) / denom
        return Point(p.x - self.a * t, p.y - self.b * t)

    # 幾何變換（對線）：用「兩點取樣再重建」最穩健直觀
    def transform(self, f) -> "Line":
        # 找兩個在線上的點：
        # 若 b != 0, 取 x=0 得 y = -c/b；取 x=1 得 y = -(a+c)/b
        # 否則 a != 0, 取 y=0 得 x = -c/a；取 y=1 得 x = -(b+c)/a
        if abs(self.b) > EPS:
            p1 = Point(0.0, -self.c / self.b)
            p2 = Point(1.0, -(self.a * 1.0 + self.c) / self.b)
        else:
            p1 = Point(-self.c / self.a, 0.0)
            p2 = Point(-(self.b * 1.0 + self.c) / self.a, 1.0)
        q1 = f(p1)
        q2 = f(p2)
        return Line.from_points(q1, q2)

    def translate(self, dx: float, dy: float) -> "Line":
        return self.transform(lambda p: p.translate(dx, dy))

    def scale(self, s: float, about: Point = None) -> "Line":
        return self.transform(lambda p: p.scale(s, about))

    def rotate(self, theta: float, about: Point = None) -> "Line":
        return self.transform(lambda p: p.rotate(theta, about))

@dataclass(frozen=True)
class Circle:
    center: Point
    r: float

    def intersection_circle(self, other: "Circle") -> List[Point]:
        # 圓圓交點（幾何解）：
        # 令 d = |C2-C1|
        # 若 d>r1+r2 或 d<|r1-r2| 或 d=0且r1=r2 => 無（或無限多，這裡回空）
        c1, c2 = self.center, other.center
        r1, r2 = self.r, other.r
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        d = math.hypot(dx, dy)

        if d < EPS and abs(r1 - r2) < EPS:
            return []  # 重合：無限多交點，這裡不處理
        if d > r1 + r2 + EPS:
            return []
        if d < abs(r1 - r2) - EPS:
            return []
        if d < EPS:
            return []

        # a = (r1^2 - r2^2 + d^2) / (2d)
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        # h^2 = r1^2 - a^2
        h2 = r1*r1 - a*a
        if h2 < -EPS:
            return []
        h = math.sqrt(max(0.0, h2))

        # P0 = C1 + a*(C2-C1)/d
        ux = dx / d
        uy = dy / d
        p0 = Point(c1.x + a*ux, c1.y + a*uy)

        # 交點 = p0 ± h * (-uy, ux)
        rx = -uy * h
        ry = ux * h

        pA = Point(p0.x + rx, p0.y + ry)
        pB = Point(p0.x - rx, p0.y - ry)

        if pA.dist(pB) < 1e-7:
            return [pA]  # 相切
        return [pA, pB]

    def intersection_line(self, line: Line) -> List[Point]:
        # 線圓交點：把直線參數化並代回圓，解二次
        # 取線上一點 p0 與方向向量 v，點為 p(t)=p0 + t v
        # 代入 |p(t)-center|^2 = r^2 => At^2 + Bt + C = 0
        if abs(line.b) > EPS:
            p0 = Point(0.0, -line.c / line.b)
        else:
            p0 = Point(-line.c / line.a, 0.0)
        v = line.direction()  # (b,-a)

        cx, cy = self.center.x, self.center.y
        x0, y0 = p0.x - cx, p0.y - cy
        vx, vy = v.x, v.y

        A = vx*vx + vy*vy
        B = 2*(x0*vx + y0*vy)
        C = x0*x0 + y0*y0 - self.r*self.r

        disc = B*B - 4*A*C
        if disc < -EPS:
            return []
        if abs(disc) <= EPS:
            t = -B / (2*A)
            p = Point(p0.x + t*vx, p0.y + t*vy)
            return [p]
        sqrt_disc = math.sqrt(max(0.0, disc))
        t1 = (-B + sqrt_disc) / (2*A)
        t2 = (-B - sqrt_disc) / (2*A)
        p1 = Point(p0.x + t1*vx, p0.y + t1*vy)
        p2 = Point(p0.x + t2*vx, p0.y + t2*vy)
        return [p1, p2]

    # 幾何變換（對圓）
    def translate(self, dx: float, dy: float) -> "Circle":
        return Circle(self.center.translate(dx, dy), self.r)

    def scale(self, s: float, about: Point = None) -> "Circle":
        # 半徑也乘以 |s|
        return Circle(self.center.scale(s, about), abs(s) * self.r)

    def rotate(self, theta: float, about: Point = None) -> "Circle":
        # 半徑不變，中心旋轉
        return Circle(self.center.rotate(theta, about), self.r)

@dataclass(frozen=True)
class Triangle:
    a: Point
    b: Point
    c: Point

    def translate(self, dx: float, dy: float) -> "Triangle":
        return Triangle(self.a.translate(dx, dy), self.b.translate(dx, dy), self.c.translate(dx, dy))

    def scale(self, s: float, about: Point = None) -> "Triangle":
        return Triangle(self.a.scale(s, about), self.b.scale(s, about), self.c.scale(s, about))

    def rotate(self, theta: float, about: Point = None) -> "Triangle":
        return Triangle(self.a.rotate(theta, about), self.b.rotate(theta, about), self.c.rotate(theta, about))

    def side_lengths(self) -> Tuple[float, float, float]:
        return (self.a.dist(self.b), self.b.dist(self.c), self.c.dist(self.a))

def verify_pythagorean(line: Line, point_on_line: Point, point_off_line: Point) -> Tuple[Point, float, float, float, float]:
    """
    用：線上一點 A、線外點 P、垂足 H
    檢查：AP^2 ≈ AH^2 + PH^2 （直角在 H）
    回傳：(H, AP2, AH2, PH2, (AH2+PH2)-AP2)
    """
    H = line.foot_of_perpendicular(point_off_line)
    AP2 = (point_on_line - point_off_line).norm2()
    AH2 = (point_on_line - H).norm2()
    PH2 = (point_off_line - H).norm2()
    return H, AP2, AH2, PH2, (AH2 + PH2) - AP2

if __name__ == "__main__":
    # 兩直線交點
    L1 = Line.from_points(Point(0, 0), Point(2, 2))     # y = x
    L2 = Line.from_points(Point(0, 2), Point(2, 0))     # y = -x + 2
    I = L1.intersection_line(L2)
    print("Line-Line intersection:", I)

    # 兩圓交點
    C1 = Circle(Point(0, 0), 2)
    C2 = Circle(Point(2, 0), 2)
    print("Circle-Circle intersections:", C1.intersection_circle(C2))

    # 線圓交點
    print("Line-Circle intersections:", C1.intersection_line(L2))

    # 垂足與畢氏定理驗證
    A = Point(1, 1)           # 線上點（在 y=x 上）
    P = Point(2, 0)           # 線外點
    H, AP2, AH2, PH2, err = verify_pythagorean(L1, A, P)
    print("Foot H:", H)
    print("AP^2, AH^2, PH^2:", AP2, AH2, PH2)
    print("Error (AH^2+PH^2-AP^2):", err)

    # 變換示例：把三角形旋轉 30 度、再平移
    T = Triangle(Point(0, 0), Point(2, 0), Point(0, 1))
    T2 = T.rotate(math.radians(30), about=Point(0, 0)).translate(3, 1)
    print("Triangle transformed:", T2)
