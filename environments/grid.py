from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

Wall = tuple[tuple[int, int], tuple[int, int]]


def two_room_walls(rows: int = 9, wall_col: int = 4, hole_row: int = 4) -> list[Wall]:
    """Return blocked edges between wall_col and wall_col+1, except at hole_row."""
    return [((r, wall_col), (r, wall_col + 1)) for r in range(rows) if r != hole_row]


@dataclass
class GridEnv:
    """Grid world whose ``walls`` block moves between adjacent cells.

    With probability ``slip`` the intended action is replaced by a uniformly
    random one; moves off the grid or into a wall leave the agent in place.
    """

    rows: int = 9
    cols: int = 9
    slip: float = 0.0
    actions: Sequence[str] = ("R", "D", "L", "U", "S")
    walls: list[Wall] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.actions = tuple(self.actions)
        self.nS = self.rows * self.cols
        self.nA = len(self.actions)
        self.blocked_edges = {edge for a, b in self.walls for edge in ((a, b), (b, a))}
        # P[s,a,s'] = Pr(S_{t+1}=s' | S_t=s, A_t=a), shape (nS,nA,nS).
        self.P = self._build_transition_matrix()

    def to_rc(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.cols)

    def to_s(self, r: int, c: int) -> int:
        return int(self.cols * r + c)

    def _move(self, r: int, c: int, action: str) -> tuple[int, int]:
        dr, dc = {"R": (0, 1), "L": (0, -1), "D": (1, 0), "U": (-1, 0), "S": (0, 0)}[action]
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        if ((r, c), (nr, nc)) in self.blocked_edges:
            return r, c
        return nr, nc

    def _build_transition_matrix(self) -> np.ndarray:
        P = np.zeros((self.nS, self.nA, self.nS), dtype=np.float32)
        for s in range(self.nS):
            r, c = self.to_rc(s)
            for a, action in enumerate(self.actions):
                for slip_action in self.actions:
                    P[s, a, self.to_s(*self._move(r, c, slip_action))] += self.slip / self.nA
                P[s, a, self.to_s(*self._move(r, c, action))] += 1.0 - self.slip
        return P
