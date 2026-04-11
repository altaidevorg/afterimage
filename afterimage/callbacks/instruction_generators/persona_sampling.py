import random
import threading
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PersonaCandidate:
    text: str
    generation_depth: int


@dataclass
class PersonaSelectionState:
    mode: Literal["cycle", "weighted"]
    active_pool: list[PersonaCandidate] = field(default_factory=list)
    population: list[PersonaCandidate] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    next_index: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def next_candidate(self) -> PersonaCandidate | None:
        if self.mode == "weighted":
            if not self.population:
                return None
            return random.choices(self.population, weights=self.weights, k=1)[0]

        if not self.active_pool:
            return None

        with self.lock:
            candidate = self.active_pool[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.active_pool)
            return candidate
