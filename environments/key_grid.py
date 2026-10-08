from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class KeyGridEnv:
    """Grid where entering side-room cells requires holding the key.

    A state is ``(cell, has_key)``; actions move, stay (S), pick up (P) or drop
    (O) the key at ``key_pos``. With probability ``slip`` the agent stays put.
    """

    rows: int = 5
    cols: int = 5
    slip: float = 0.0
    actions: Sequence[str] = ("R", "D", "L", "U", "S", "P", "O")
    hallway_col: int = 2
    key_pos: tuple[int, int] = (0, 2)

    def __post_init__(self) -> None:
        self.actions = tuple(self.actions)
        self.nA = len(self.actions)
        self.n_cells = self.rows * self.cols
        self.nS = 2 * self.n_cells
        self.locked_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols) if c != self.hallway_col
        ]
        # P[s,a,s'] = Pr(S_{t+1}=s' | S_t=s, A_t=a), shape (nS,nA,nS).
        # Both state indices encode cell + has_key * n_cells.
        self.P = self._build_transition_matrix()

    def encode_state(self, cell: int, has_key: int) -> int:
        return int(cell + has_key * self.n_cells)

    def decode_state(self, s: int) -> tuple[int, int]:
        """Return ``(cell, has_key)``."""
        return int(s) % self.n_cells, int(s) // self.n_cells

    def to_s(self, r: int, c: int, has_key: int = 0) -> int:
        return self.encode_state(r * self.cols + c, has_key)

    def to_rc(self, s: int) -> tuple[int, int]:
        return divmod(self.decode_state(s)[0], self.cols)

    def valid_starts(self) -> list[int]:
        """Hallway cells without the key, plus every cell with it."""
        hallway = [self.to_s(r, self.hallway_col) for r in range(self.rows)]
        return hallway + list(range(self.n_cells, self.nS))

    def _move(self, r: int, c: int, action: str) -> tuple[int, int]:
        dr, dc = {"R": (0, 1), "L": (0, -1), "D": (1, 0), "U": (-1, 0)}[action]
        return max(0, min(self.rows - 1, r + dr)), max(0, min(self.cols - 1, c + dc))

    def _build_transition_matrix(self) -> np.ndarray:
        P = np.zeros((self.nS, self.nA, self.nS), dtype=np.float32)
        for s in range(self.nS):
            cell, has_key = self.decode_state(s)
            r, c = divmod(cell, self.cols)
            for a, action in enumerate(self.actions):
                next_r, next_c, next_key = r, c, has_key
                if action in {"R", "L", "U", "D"}:
                    next_r, next_c = self._move(r, c, action)
                    if next_c != self.hallway_col and not has_key:
                        next_r, next_c = r, c
                elif action == "P" and (r, c) == self.key_pos:
                    next_key = 1
                elif action == "O" and (r, c) == self.key_pos:
                    next_key = 0
                P[s, a, s] += self.slip
                P[s, a, self.to_s(next_r, next_c, next_key)] += 1.0 - self.slip
        return P
