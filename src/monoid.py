"""
Transition Monoid — precomputed algebraic structure for DFA simulation.

The transition monoid of a DFA is the set of all distinct functions
(transition matrices) reachable as products of per-character matrices,
closed under composition.  For small DFAs (N ≤ 16), the monoid is
typically tiny (10–200 elements), so the "online" simulation phase
reduces to O(1) table lookups instead of O(N³) matmuls.

Public API
----------
MonoidData   — dataclass holding all precomputed tables
compute_monoid(dm, max_size) -> MonoidData | None
simulate_monoid(md, dm, input_str) -> bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np

from src.simulation import DFAMatrices, _matmul_int8


# ─── Data Container ────────────────────────────────────────────────────────

@dataclass
class MonoidData:
    """Precomputed transition monoid for a DFA.

    Fields
    ------
    elements        list of (N, N) int8 numpy arrays — the distinct matrices.
    compose_table   (M, M) uint16 ndarray where compose_table[i, j] is the
                    index of elements[i] @ elements[j].
    char_to_monoid  dict[str, int] — per-character monoid index (pre-seeded
                    from DFAMatrices.matrices).
    accept_table    (M,) bool ndarray — True iff the corresponding matrix,
                    when applied to the start vector, reaches an accept state.
    identity_idx    int — index of the identity matrix in elements.
    size            int — number of elements (== len(elements)).
    """

    elements: list          # list[np.ndarray], each (N, N) int8
    compose_table: np.ndarray   # (M, M) uint16
    char_to_monoid: dict        # str -> int
    accept_table: np.ndarray    # (M,) bool
    identity_idx: int
    size: int


# ─── BFS Closure ────────────────────────────────────────────────────────────

def compute_monoid(dm: DFAMatrices, max_size: int = 65536) -> Optional[MonoidData]:
    """Compute the transition monoid via BFS closure over matrix products.

    Starts from the identity matrix and all per-character matrices, then
    repeatedly multiplies every pair until no new element is discovered.

    Returns None if the monoid exceeds *max_size* distinct elements.

    Composition convention (matching prefix-scan / simulate_monoid):
        compose_table[i, j]  ==  index of  elements[i] @ elements[j]
    i.e. element i is "newer" (applied after j in left-to-right reading).
    """
    # ── Seed elements: identity + one matrix per alphabet character ──────────
    I = dm.identity_matrix()

    # bytes -> index map for dedup
    key_to_idx: dict[bytes, int] = {}
    elements: list[np.ndarray] = []

    def _add(mat: np.ndarray) -> int:
        """Add matrix if new; return its index either way."""
        key = mat.tobytes()
        if key in key_to_idx:
            return key_to_idx[key]
        idx = len(elements)
        key_to_idx[key] = idx
        elements.append(mat)
        return idx

    identity_idx = _add(I)

    # Map each alphabet character to its monoid index
    char_to_monoid: dict[str, int] = {}
    for ch in dm.alphabet:
        mat = dm.matrices[ch]
        char_to_monoid[ch] = _add(mat)

    # ── BFS: keep a frontier of newly discovered elements ───────────────────
    # We need to try products of every ordered pair (new × all) and (all × new)
    # to ensure closure.
    #
    # Strategy: maintain a set of "pending" indices that haven't yet been
    # multiplied against all existing elements.  Each time we pop an index,
    # we multiply it (left and right) against ALL current elements.  Products
    # that are truly new get added to pending.
    #
    # Key invariant: an index is only ever added to `pending` exactly once
    # (when it first appears in `elements`).  We track `processed` to avoid
    # redundant work.

    pending = deque(range(len(elements)))   # newly found, not yet expanded
    processed: set[int] = set()              # indices fully expanded

    while pending:
        if len(elements) > max_size:
            return None

        i = pending.popleft()
        if i in processed:
            continue
        processed.add(i)

        # Snapshot current size; we'll check for new elements after each add
        before = len(elements)

        # Pair i (left) against every element that exists right now
        n_current = len(elements)
        for j in range(n_current):
            product = _matmul_int8(elements[i], elements[j])
            _add(product)

        # Pair every previously-processed element (left) against i (right)
        for j in sorted(processed - {i}):
            product = _matmul_int8(elements[j], elements[i])
            _add(product)

        # Also pair i (right) against every element (left) — catches pairs
        # where i hasn't been used on the right yet by earlier iterations
        for j in range(n_current):
            product = _matmul_int8(elements[j], elements[i])
            _add(product)

        # Any element added since `before` is newly discovered → enqueue
        for new_idx in range(before, len(elements)):
            if len(elements) > max_size:
                return None
            pending.append(new_idx)

    M = len(elements)

    # ── Build compose table ──────────────────────────────────────────────────
    compose_table = np.zeros((M, M), dtype=np.uint16)
    for i in range(M):
        for j in range(M):
            product = _matmul_int8(elements[i], elements[j])
            compose_table[i, j] = key_to_idx[product.tobytes()]

    # ── Build accept table ───────────────────────────────────────────────────
    # element k "accepts" iff  element_k @ start_vec  has an accept state set.
    accept_table = np.zeros(M, dtype=bool)
    start = dm.start_vec.astype(np.int32)
    for k, mat in enumerate(elements):
        final = mat.astype(np.int32) @ start
        accept_table[k] = dm.check_accept(final.astype(np.int8))

    return MonoidData(
        elements=elements,
        compose_table=compose_table,
        char_to_monoid=char_to_monoid,
        accept_table=accept_table,
        identity_idx=identity_idx,
        size=M,
    )


# ─── Monoid-Based Simulation ─────────────────────────────────────────────────

def simulate_monoid(md: MonoidData, dm: DFAMatrices, input_str: str) -> bool:
    """Simulate DFA acceptance using precomputed monoid tables.

    Scans left-to-right, accumulating a single monoid index.
    Convention (matching prefix-scan): the *newer* character's matrix
    goes on the LEFT, so we compose as::

        acc = compose_table[c_new_idx, acc_old]

    For an empty string we check the identity element.

    Characters not in the DFA alphabet are treated as «stay in current
    state» — i.e. they contribute the identity matrix.
    """
    acc = md.identity_idx
    for ch in input_str:
        c_idx = md.char_to_monoid.get(ch, md.identity_idx)
        acc = int(md.compose_table[c_idx, acc])
    return bool(md.accept_table[acc])


def precompute_batch_tables(md: MonoidData, dm: DFAMatrices) -> dict:
    """Build GPU-friendly tables for the monoid batch kernel.

    Returns a dict with keys:
        char_compose   (M * sigma_ext,) uint8 — fused compose table
        raw_char_map   (256,) uint8 — maps raw ASCII byte to DFA char index
        accept         (M,) uint8 — 1 if monoid element is accepting
        monoid_compose (M * M,) uint8 — M×M compose table for tree reduce
        fused_compose  (M * 256,) uint8 — merged charmap+compose in one lookup
        kgram_compose  (M * sigma_k,) uint8 or None — k-gram compose table
        M, sigma, sigma_ext, sigma_k, kgram_k, identity_idx
    """
    import math

    M = md.size
    if M > 255:
        raise ValueError(f"Monoid size {M} exceeds uint8 limit (255)")

    sigma = len(dm.alphabet)
    sigma_ext = sigma + 1

    char_compose = np.zeros((M, sigma_ext), dtype=np.uint8)
    for ch_name, ch_dfa_idx in dm.char_to_idx.items():
        ch_monoid = md.char_to_monoid[ch_name]
        for m in range(M):
            char_compose[m, ch_dfa_idx] = int(md.compose_table[ch_monoid, m])
    for m in range(M):
        char_compose[m, sigma] = m

    raw_char_map = np.full(256, sigma, dtype=np.uint8)
    for ch_name, ch_dfa_idx in dm.char_to_idx.items():
        raw_char_map[ord(ch_name)] = ch_dfa_idx

    accept = np.ascontiguousarray(md.accept_table.astype(np.uint8))

    monoid_compose = np.ascontiguousarray(
        md.compose_table.astype(np.uint8).reshape(-1)
    )

    fused_compose = np.zeros(M * 256, dtype=np.uint8)
    for m in range(M):
        for byte_val in range(256):
            ch_dfa_idx = raw_char_map[byte_val]
            fused_compose[m * 256 + byte_val] = char_compose[m, ch_dfa_idx]

    if sigma >= 2:
        kgram_k = int(math.log(256) / math.log(sigma))
        kgram_k = max(kgram_k, 1)
        sigma_k = sigma ** kgram_k
        while sigma_k > 256:
            kgram_k -= 1
            sigma_k = sigma ** kgram_k
    else:
        kgram_k = 1
        sigma_k = 1

    if kgram_k >= 2:
        char_monoid_by_dfa = [md.identity_idx] * sigma
        for ch_name, ch_dfa_idx in dm.char_to_idx.items():
            char_monoid_by_dfa[ch_dfa_idx] = md.char_to_monoid[ch_name]

        kgram_compose = np.zeros(M * sigma_k, dtype=np.uint8)
        for m in range(M):
            for gram in range(sigma_k):
                chars = []
                g = gram
                for _ in range(kgram_k):
                    chars.append(g % sigma)
                    g //= sigma
                chars.reverse()
                curr = m
                for dfa_idx in chars:
                    ch_monoid = char_monoid_by_dfa[dfa_idx]
                    curr = int(md.compose_table[ch_monoid, curr])
                kgram_compose[m * sigma_k + gram] = curr
    else:
        kgram_compose = None

    return {
        'char_compose': np.ascontiguousarray(char_compose.reshape(-1)),
        'raw_char_map': raw_char_map,
        'accept': accept,
        'monoid_compose': monoid_compose,
        'fused_compose': np.ascontiguousarray(fused_compose),
        'kgram_compose': (np.ascontiguousarray(kgram_compose)
                          if kgram_compose is not None else None),
        'M': M,
        'sigma': sigma,
        'sigma_ext': sigma_ext,
        'sigma_k': sigma_k,
        'kgram_k': kgram_k,
        'identity_idx': md.identity_idx,
    }
