"""
Batched state-vector evolution: process B strings simultaneously.

Instead of one string at a time, maintains a state matrix S[N, B] where
column j is the state vector of string j. At each position t, strings are
grouped by their character and the corresponding transition matrix is applied
via matmul.

This is the CPU reference implementation that will later be mirrored on GPU
with tensor-core int8 matmuls.
"""

from __future__ import annotations
import numpy as np
from src.simulation import DFAMatrices


def simulate_batched_cpu(dm: DFAMatrices, strings: list[str]) -> list[bool]:
    """
    Batched state-vector evolution on CPU.

    Algorithm:
      1. Pad all strings to L_max using an identity character index.
      2. Build input_arr[t, j] = char_to_idx for position t of string j.
      3. Extend matrix_stack with identity at index |Sigma|.
      4. Initialize S[N, B] with start state.
      5. For each position t, group columns by character, apply T[c] via matmul.
      6. Clamp to {0,1} after each step (boolean threshold for NFA safety).
      7. Check acceptance via accept_mask dot product.

    Args:
        dm: DFAMatrices encoding the DFA.
        strings: List of input strings to match.

    Returns:
        List of booleans, one per string.
    """
    B = len(strings)
    if B == 0:
        return []

    N = dm.n_states
    sigma_size = len(dm.alphabet)  # |Sigma|
    identity_idx = sigma_size      # index for identity (padding) matrix

    # ── Step 1: Find L_max ──
    L_max = max(len(s) for s in strings) if strings else 0

    if L_max == 0:
        # All strings are empty; check if start state is accepting
        is_accept = dm.check_accept(dm.start_vec)
        return [is_accept] * B

    # ── Step 2: Build position-contiguous input array ──
    # input_arr[t, j] = char index at position t of string j
    # Use identity_idx for positions past end of string
    input_arr = np.full((L_max, B), identity_idx, dtype=np.int32)
    for j, s in enumerate(strings):
        for t, ch in enumerate(s):
            idx = dm.char_to_idx.get(ch)
            if idx is not None:
                input_arr[t, j] = idx
            # Characters not in alphabet get identity (stay in same state)

    # ── Step 3: Extend matrix_stack with identity ──
    # dm.matrix_stack shape: (|Sigma|, N, N)
    identity = np.eye(N, dtype=np.int8)
    matrices = np.concatenate(
        [dm.matrix_stack, identity.reshape(1, N, N)], axis=0
    )  # shape: (|Sigma|+1, N, N)

    # ── Step 4: Initialize state matrix S[N, B] ──
    S = np.zeros((N, B), dtype=np.int32)
    S[dm.dfa.start, :] = 1

    # ── Step 5-6: Evolve through each position ──
    for t in range(L_max):
        S_new = np.zeros((N, B), dtype=np.int32)
        chars_at_t = input_arr[t, :]  # shape (B,)

        # Group columns by character index for batched matmul
        for c in range(sigma_size + 1):
            col_mask = (chars_at_t == c)
            if not np.any(col_mask):
                continue
            col_indices = np.where(col_mask)[0]
            # T[c] is (N, N), S[:, cols] is (N, |cols|)
            T_c = matrices[c].astype(np.int32)
            S_new[:, col_indices] = T_c @ S[:, col_indices]

        # Boolean threshold: clamp to [0, 1] to prevent int8 overflow for NFA
        np.minimum(S_new, 1, out=S_new)
        S = S_new

    # ── Step 7: Check acceptance ──
    accept = dm.accept_mask.astype(np.int32)  # shape (N,)
    # For each column j: accepted if any(S[:, j] * accept_mask > 0)
    # Vectorized: accept @ S gives shape (B,), nonzero means accepted
    accept_scores = accept @ S  # shape (B,)
    results = [bool(accept_scores[j] > 0) for j in range(B)]

    return results
