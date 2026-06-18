"""
DFA → int8 transition matrices, plus CPU simulation backends.

Three simulation modes:
  1. Sequential:     O(L) character-at-a-time loop (baseline).
  2. MatSeq:         O(L) matrix–vector multiplies (validates matrix encoding).
  3. PrefixScan:     O(log L) depth parallel prefix scan over matrix products.

All matrix operations use int8/int32 numpy to mirror tensor-core arithmetic.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.regex_to_dfa import DFA


# ─── DFA → Transition Matrices ─────────────────────────────────────────────

class DFAMatrices:
    """
    Encodes a complete DFA as a set of int8 transition matrices.

    For a DFA with N states and alphabet Σ:
      T[c] ∈ {0,1}^{N×N}  where T[c][i][j] = 1 iff δ(j, c) = i

    Convention: column j = source state, row i = destination state.
    So next_state_vec = T[c] @ current_state_vec.
    """

    def __init__(self, dfa: DFA, pad_to: Optional[int] = None):
        self.dfa = dfa
        self.alphabet = sorted(dfa.alphabet)
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

        # Pad state dimension to multiple of tile_size for tensor-core alignment
        self.n_states_raw = dfa.n_states
        if pad_to is not None:
            self.n_states = max(pad_to, dfa.n_states)
        else:
            # Pad to next multiple of 16 (tensor core tile)
            self.n_states = ((dfa.n_states + 15) // 16) * 16

        self._build_matrices()
        self._build_state_vectors()

    def _build_matrices(self):
        """Build per-character transition matrices."""
        N = self.n_states
        # Identity for padded "extra" states (they self-loop)
        self.matrices = {}
        for ch in self.alphabet:
            T = np.zeros((N, N), dtype=np.int8)
            for src in range(self.n_states_raw):
                dst = self.dfa.transitions.get(src, {}).get(ch)
                if dst is not None:
                    T[dst, src] = 1
            # Padded states self-loop (so they don't pollute results)
            for s in range(self.n_states_raw, N):
                T[s, s] = 1
            self.matrices[ch] = T

        # Also store as a 3D array indexed by char index for fast lookup
        self.matrix_stack = np.stack(
            [self.matrices[ch] for ch in self.alphabet], axis=0
        )  # shape: (|Σ|, N, N), dtype int8

    def _build_state_vectors(self):
        """Build start state vector and accept mask."""
        N = self.n_states
        self.start_vec = np.zeros(N, dtype=np.int8)
        self.start_vec[self.dfa.start] = 1
        self.accept_mask = np.zeros(N, dtype=np.int8)
        for s in self.dfa.accept_states:
            self.accept_mask[s] = 1

    def get_matrix_for_char(self, ch: str) -> np.ndarray:
        return self.matrices.get(ch)

    def get_matrix_sequence(self, input_str: str) -> np.ndarray:
        """Return (L, N, N) array of transition matrices for input string."""
        N = self.n_states
        L = len(input_str)
        seq = np.zeros((L, N, N), dtype=np.int8)
        for i, ch in enumerate(input_str):
            idx = self.char_to_idx.get(ch)
            if idx is not None:
                seq[i] = self.matrix_stack[idx]
            else:
                # Unknown char → identity (stays in same state; will hit dead state
                # only if DFA is complete with dead state)
                np.fill_diagonal(seq[i], 1)
        return seq

    def check_accept(self, state_vec: np.ndarray) -> bool:
        """Check if any accept state is active in state_vec."""
        return bool(np.any(state_vec[:self.n_states_raw] & self.accept_mask[:self.n_states_raw]))

    def identity_matrix(self) -> np.ndarray:
        return np.eye(self.n_states, dtype=np.int8)


# ─── Simulation Backend 1: Sequential (baseline) ───────────────────────────

def simulate_sequential(dfa: DFA, input_str: str) -> bool:
    """Standard O(L) sequential DFA simulation."""
    return dfa.simulate(input_str)


# ─── Simulation Backend 2: Matrix-Vector Sequential ────────────────────────

def simulate_matrix_sequential(dm: DFAMatrices, input_str: str) -> bool:
    """
    O(L) simulation via matrix–vector multiply each step.
    Validates the matrix encoding is correct.
    """
    state = dm.start_vec.astype(np.int32)
    for ch in input_str:
        T = dm.get_matrix_for_char(ch)
        if T is None:
            return False
        state = T.astype(np.int32) @ state
    return dm.check_accept(state.astype(np.int8))


# ─── Simulation Backend 3: Parallel Prefix Scan (CPU emulation) ────────────

def _matmul_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    int8 matrix multiply with int32 accumulation, then clamp to int8.
    Mirrors tensor-core semantics: D = A * B  (int8×int8 → int32 → int8).
    For DFA (one-hot columns), results naturally stay in {0,1}.
    """
    return (A.astype(np.int32) @ B.astype(np.int32)).astype(np.int8)


def _matmul_int32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """int8 matrix multiply with int32 result (no truncation)."""
    return A.astype(np.int32) @ B.astype(np.int32)


def prefix_scan_sequential(matrices: np.ndarray) -> np.ndarray:
    """
    Sequential prefix products for reference.
    Input:  (L, N, N) array of transition matrices
    Output: (L, N, N) array where out[i] = matrices[i] @ ... @ matrices[0]
    """
    L, N, _ = matrices.shape
    result = np.zeros_like(matrices)
    result[0] = matrices[0]
    for i in range(1, L):
        result[i] = _matmul_int8(matrices[i], result[i - 1])
    return result


def prefix_scan_parallel(matrices: np.ndarray) -> np.ndarray:
    """
    Hillis-Steele inclusive parallel prefix scan over matrix products.
    Simulates O(log L) depth on CPU using numpy.

    This is the core algorithm that maps to tensor-core execution:
    each "parallel step" is a batch of independent N×N matmuls.

    Input:  (L, N, N) array of transition matrices
    Output: (L, N, N) array of prefix products
            out[i] = matrices[i] @ matrices[i-1] @ ... @ matrices[0]

    Work: O(L log L) matmuls (work-inefficient but simple and correct).
    Depth: O(log L) steps.
    On tensor cores, each step is a batched MMA kernel launch.
    (A work-efficient Brent-Kung variant reduces total work to O(L).)
    """
    L, N, _ = matrices.shape
    if L == 0:
        return matrices.copy()

    result = matrices.copy()

    stride = 1
    while stride < L:
        new_result = result.copy()
        for i in range(stride, L):
            # result[i] = result[i] @ result[i - stride]
            # Each of these is independent → parallelizable
            new_result[i] = _matmul_int8(result[i], result[i - stride])
        result = new_result
        stride *= 2

    return result


def simulate_prefix_scan(dm: DFAMatrices, input_str: str,
                         use_parallel: bool = True) -> bool:
    """
    O(log L) depth simulation via parallel prefix scan over transition matrices.
    The final prefix product × start_vec gives the final state.
    """
    if not input_str:
        return dm.check_accept(dm.start_vec)

    matrices = dm.get_matrix_sequence(input_str)

    if use_parallel:
        prefixes = prefix_scan_parallel(matrices)
    else:
        prefixes = prefix_scan_sequential(matrices)

    # Final state = last_prefix_product @ start_vec
    final_matrix = prefixes[-1]  # = T[L-1] @ ... @ T[0]
    final_state = final_matrix.astype(np.int32) @ dm.start_vec.astype(np.int32)
    return dm.check_accept(final_state.astype(np.int8))


# ─── Batch simulation (multiple strings, one DFA) ──────────────────────────

def simulate_batch_sequential(dfa: DFA, strings: list[str]) -> list[bool]:
    """Batch sequential simulation."""
    return [dfa.simulate(s) for s in strings]


def simulate_batch_matrix(dm: DFAMatrices, strings: list[str]) -> list[bool]:
    """
    Batch simulation via matrix–vector multiply.
    Packs state vectors into columns of a matrix for simultaneous update.

    For strings of equal length, this is a single matmul per position.
    For variable-length strings, we group by length.
    """
    results = [False] * len(strings)

    # Group by length for efficient batching
    from collections import defaultdict
    length_groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i, s in enumerate(strings):
        length_groups[len(s)].append((i, s))

    for length, group in length_groups.items():
        if length == 0:
            is_accept = dm.check_accept(dm.start_vec)
            for idx, _ in group:
                results[idx] = is_accept
            continue

        batch_size = len(group)
        N = dm.n_states

        # State matrix: columns are state vectors for each string
        states = np.tile(dm.start_vec.astype(np.int32), (batch_size, 1)).T  # (N, batch)

        for pos in range(length):
            # Group strings by their character at this position
            char_groups: dict[str, list[int]] = defaultdict(list)
            for batch_idx, (_, s) in enumerate(group):
                char_groups[s[pos]].append(batch_idx)

            new_states = np.zeros_like(states)
            for ch, batch_indices in char_groups.items():
                T = dm.get_matrix_for_char(ch)
                if T is None:
                    continue
                cols = states[:, batch_indices]
                new_states[:, batch_indices] = T.astype(np.int32) @ cols

            states = new_states

        # Check acceptance
        for batch_idx, (orig_idx, _) in enumerate(group):
            sv = states[:, batch_idx].astype(np.int8)
            results[orig_idx] = dm.check_accept(sv)

    return results


def precompute_tmap(dm: DFAMatrices) -> np.ndarray:
    """Build fused transition map: tmap[byte_val * N + state] = dest_state.

    For each raw byte value (0-255), maps each DFA state to its destination.
    Unmapped characters act as identity (state maps to itself).
    Table size: 256 * N bytes (4 KB for N=16).
    """
    N = dm.n_states
    tmap = np.zeros(256 * N, dtype=np.uint8)

    # Default: identity for all bytes and all states
    for s in range(N):
        for byte_val in range(256):
            tmap[byte_val * N + s] = s

    # Overwrite with actual transitions for alphabet characters
    for ch_name in dm.alphabet:
        byte_val = ord(ch_name)
        T = dm.matrices[ch_name]
        for s in range(dm.n_states_raw):
            dst = int(np.argmax(T[:, s]))
            tmap[byte_val * N + s] = dst

    return tmap


if __name__ == '__main__':
    from src.regex_to_dfa import compile_regex

    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    print(f"Padded states: {dm.n_states} (raw: {dm.n_states_raw})")
    print(f"Alphabet size: {len(dm.alphabet)}")

    tests = ["abb", "aabb", "babb", "ababb", "ab", "ba", ""]
    expected = [True, True, True, True, False, False, False]

    for s, exp in zip(tests, expected):
        r1 = simulate_sequential(dfa, s)
        r2 = simulate_matrix_sequential(dm, s)
        r3 = simulate_prefix_scan(dm, s, use_parallel=False)
        r4 = simulate_prefix_scan(dm, s, use_parallel=True)
        ok = r1 == r2 == r3 == r4 == exp
        print(f"  '{s}': seq={r1} mat={r2} scan_seq={r3} scan_par={r4} expected={exp} {'✓' if ok else '✗'}")
