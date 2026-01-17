from __future__ import annotations

from typing import Iterable


def apply_translation(points: Iterable[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in points]


def apply_affine(points: Iterable[tuple[float, float]], matrix: list[list[float]] | tuple[tuple[float, float, float], tuple[float, float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    a00, a01, a02 = matrix[0]
    a10, a11, a12 = matrix[1]
    for x, y in points:
        nx = a00 * x + a01 * y + a02
        ny = a10 * x + a11 * y + a12
        result.append((nx, ny))
    return result
