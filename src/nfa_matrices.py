"""
NFA → Boolean transition matrices for parallel regex matching.

Encodes an NFA (with epsilon transitions) as a set of boolean N×N matrices,
one per alphabet symbol. The matrix convention matches DFAMatrices:

    T[dest, src] = 1  iff  src can reach dest via char + epsilon closure

Simulation:
    state_vec = T[c] @ state_vec   (then threshold to {0,1})

The epsilon closure is folded into each per-character matrix so that no
separate epsilon-closure pass is needed at simulation time.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from src.regex_to_dfa import NFA, RegexParser, EPSILON


# ─── Epsilon Closure ────────────────────────────────────────────────────────

def _epsilon_closure_set(nfa: NFA, states: set[int]) -> set[int]:
    """Compute epsilon closure over a set of NFA states."""
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for dst in nfa.states[s].transitions.get(EPSILON, []):
            if dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return closure


# ─── NFAMatrices ─────────────────────────────────────────────────────────────

class NFAMatrices:
    """
    Encodes an NFA as a set of boolean int8 transition matrices.

    For each symbol c ∈ Σ, T[c] ∈ {0,1}^{N×N} where:
        T[c][dest, src] = 1  iff  there exists a path from src
                                  via c followed by zero or more epsilon
                                  transitions to dest.

    Convention: column = source state, row = destination state.
    Multiply column-vectors on the right:  next_vec = T[c] @ cur_vec
    """

    def __init__(self, nfa: NFA, pad_to: Optional[int] = None):
        self.nfa = nfa
        self.alphabet = sorted(nfa.alphabet)
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.n_states_raw = len(nfa.states)

        if pad_to is not None:
            self.n_states = max(pad_to, self.n_states_raw)
        else:
            # Round up to next multiple of 16 (tensor-core tile alignment)
            self.n_states = ((self.n_states_raw + 15) // 16) * 16

        self._build_matrices()
        self._build_state_vectors()

    # ── Matrix construction ──────────────────────────────────────────────────

    def _build_matrices(self):
        """Build per-character boolean transition matrices."""
        N = self.n_states
        nfa = self.nfa
        self.matrices: dict[str, np.ndarray] = {}

        for ch in self.alphabet:
            T = np.zeros((N, N), dtype=np.int8)

            for src_id in nfa.states:
                # Collect all states reachable from src_id via ch
                direct = set(nfa.states[src_id].transitions.get(ch, []))
                if not direct:
                    continue
                # Apply epsilon closure after the character transition
                reachable = _epsilon_closure_set(nfa, direct)
                for dst in reachable:
                    if dst < N:
                        T[dst, src_id] = 1

            self.matrices[ch] = T

        # 3-D stack: shape (|Σ|, N, N) — indexed by char_to_idx
        self.matrix_stack = np.stack(
            [self.matrices[ch] for ch in self.alphabet], axis=0
        )  # dtype int8

    def _build_state_vectors(self):
        """Build start state column-vector and accept mask."""
        N = self.n_states
        nfa = self.nfa

        # Start vector: epsilon closure of the NFA start state
        start_closure = _epsilon_closure_set(nfa, {nfa.start})
        self.start_vec = np.zeros(N, dtype=np.int8)
        for s in start_closure:
            if s < N:
                self.start_vec[s] = 1

        # Accept mask: 1 at every accept state
        self.accept_mask = np.zeros(N, dtype=np.int8)
        for sid, state in nfa.states.items():
            if state.is_accept and sid < N:
                self.accept_mask[sid] = 1

    # ── Acceptance check ─────────────────────────────────────────────────────

    def check_accept(self, state_vec: np.ndarray) -> bool:
        """Return True if any active state in state_vec is an accept state."""
        return bool(np.any(
            state_vec[:self.n_states_raw].astype(np.int8)
            & self.accept_mask[:self.n_states_raw]
        ))


# ─── Public API ──────────────────────────────────────────────────────────────

def compile_nfa_matrices(regex: str, pad_to: Optional[int] = None) -> NFAMatrices:
    """Parse regex into an NFA and wrap it in an NFAMatrices object."""
    nfa = RegexParser(regex).parse()
    return NFAMatrices(nfa, pad_to=pad_to)


def simulate_nfa(nm: NFAMatrices, input_str: str) -> bool:
    """
    Sequential NFA simulation using matrix-vector products.

    For each character c in input_str:
        state = T[c] @ state
        state = min(state, 1)    # Boolean threshold (prevents overflow)

    Empty string: check start_vec directly against accept mask.
    """
    if not input_str:
        return nm.check_accept(nm.start_vec)

    state = nm.start_vec.astype(np.int32)

    for ch in input_str:
        idx = nm.char_to_idx.get(ch)
        if idx is None:
            # Character not in NFA alphabet → no valid transitions → reject
            return False
        T = nm.matrix_stack[idx].astype(np.int32)
        state = T @ state
        # Boolean threshold: collapse multiple active paths to {0,1}
        np.minimum(state, 1, out=state)

    return nm.check_accept(state.astype(np.int8))
