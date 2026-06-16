"""
k-Gram Precomputation for DFA simulation.

Precomputes composed transition monoid elements (or matrices) for all
possible k-grams over the DFA alphabet.  During simulation, each k-gram
is looked up in O(1) instead of composing k individual characters.

Public API
----------
auto_k(alphabet_size, max_entries) -> int
KGramTable                           — holds precomputed lookup tables
precompute_kgrams(dm, k, monoid)     -> KGramTable
simulate_kgram_monoid(kg, md, dm, input_str) -> bool
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from src.simulation import DFAMatrices, _matmul_int8
from src.monoid import MonoidData


# ─── auto_k ─────────────────────────────────────────────────────────────────

def auto_k(alphabet_size: int, max_entries: int = 65536) -> int:
    """Return the largest k such that alphabet_size^k <= max_entries.

    Special case: if alphabet_size <= 1, returns 1.
    """
    if alphabet_size <= 1:
        return 1
    k = 1
    while True:
        if alphabet_size ** (k + 1) > max_entries:
            return k
        k += 1


# ─── KGramTable ─────────────────────────────────────────────────────────────

class KGramTable:
    """Precomputed lookup table for k-grams.

    Supports two modes:
      - Monoid mode:  lookup(gram) -> int  (monoid index)
      - Matrix mode:  lookup_matrix(gram) -> np.ndarray  (composed matrix)

    Internal key encoding (mixed-radix):
        key = sum(char_idx[i] * sigma^i for i in range(k))
    where i=0 is the leftmost (first) character of the gram.
    """

    def __init__(
        self,
        k: int,
        alphabet: list[str],
        monoid_table: Optional[dict] = None,
        matrix_table: Optional[dict] = None,
    ):
        self.k = k
        self.alphabet = alphabet
        self.sigma = len(alphabet)
        self.char_to_idx: dict[str, int] = {ch: i for i, ch in enumerate(alphabet)}
        self._monoid_table = monoid_table   # int key -> int (monoid index)
        self._matrix_table = matrix_table   # int key -> np.ndarray

    def _gram_to_key(self, gram: tuple) -> int:
        """Convert a k-gram tuple of chars to a mixed-radix integer key."""
        key = 0
        sigma = self.sigma
        for ch in gram:
            key = key * sigma + self.char_to_idx[ch]
        return key

    def lookup(self, gram: tuple) -> int:
        """Return the monoid index for this k-gram (monoid mode)."""
        return self._monoid_table[self._gram_to_key(gram)]

    def lookup_matrix(self, gram: tuple) -> np.ndarray:
        """Return the composed transition matrix for this k-gram (matrix mode)."""
        return self._matrix_table[self._gram_to_key(gram)]


# ─── precompute_kgrams ───────────────────────────────────────────────────────

def precompute_kgrams(
    dm: DFAMatrices,
    k: int,
    monoid: Optional[MonoidData] = None,
) -> KGramTable:
    """Precompute composed monoid elements (or matrices) for all k-grams.

    Mode A (monoid is not None):
        For each k-gram, compose the k per-character monoid indices via
        compose_table.  The composition convention is left-to-right reading:
            acc = compose_table[c_new, acc_old]
        starting from identity.

    Mode B (monoid is None):
        For each k-gram, compose the k per-character transition matrices via
        _matmul_int8.  Convention (matching prefix-scan):
            acc = _matmul_int8(dm.matrices[ch], acc)
        starting from the identity matrix.

    The integer key for a k-gram (c0, c1, ..., c_{k-1}) is:
        key = c0_idx * sigma^(k-1) + c1_idx * sigma^(k-2) + ... + c_{k-1}_idx
    which is equivalent to the loop:
        key = 0
        for ch in gram:
            key = key * sigma + char_to_idx[ch]
    """
    alphabet = dm.alphabet  # already sorted list
    sigma = len(alphabet)
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    if monoid is not None:
        # Mode A: monoid index composition
        monoid_table: dict[int, int] = {}
        for gram_indices in itertools.product(range(sigma), repeat=k):
            # Compute mixed-radix key
            key = 0
            for idx in gram_indices:
                key = key * sigma + idx

            # Compose monoid indices left-to-right
            acc = monoid.identity_idx
            for idx in gram_indices:
                ch = alphabet[idx]
                c_monoid = monoid.char_to_monoid[ch]
                acc = int(monoid.compose_table[c_monoid, acc])

            monoid_table[key] = acc

        return KGramTable(
            k=k,
            alphabet=alphabet,
            monoid_table=monoid_table,
            matrix_table=None,
        )

    else:
        # Mode B: matrix composition
        matrix_table: dict[int, np.ndarray] = {}
        identity = dm.identity_matrix()

        for gram_indices in itertools.product(range(sigma), repeat=k):
            # Compute mixed-radix key
            key = 0
            for idx in gram_indices:
                key = key * sigma + idx

            # Compose matrices left-to-right:
            # acc starts as identity; each new char's matrix goes on the LEFT
            acc = identity.copy()
            for idx in gram_indices:
                ch = alphabet[idx]
                acc = _matmul_int8(dm.matrices[ch], acc)

            matrix_table[key] = acc

        return KGramTable(
            k=k,
            alphabet=alphabet,
            monoid_table=None,
            matrix_table=matrix_table,
        )


# ─── simulate_kgram_monoid ───────────────────────────────────────────────────

def simulate_kgram_monoid(
    kg: KGramTable,
    md: MonoidData,
    dm: DFAMatrices,
    input_str: str,
) -> bool:
    """Simulate DFA acceptance using k-gram precomputed monoid table.

    Chunks input into non-overlapping k-grams and looks up each chunk's
    precomputed monoid index.  Any tail shorter than k is composed
    character-by-character using the monoid's char_to_monoid and
    compose_table.

    Empty string → md.accept_table[md.identity_idx].
    """
    if not input_str:
        return bool(md.accept_table[md.identity_idx])

    acc = md.identity_idx
    L = len(input_str)
    k = kg.k

    # Process full k-grams
    pos = 0
    while pos + k <= L:
        gram = tuple(input_str[pos:pos + k])
        gram_monoid = kg.lookup(gram)
        acc = int(md.compose_table[gram_monoid, acc])
        pos += k

    # Handle tail (L % k != 0)
    while pos < L:
        ch = input_str[pos]
        c_idx = md.char_to_monoid.get(ch, md.identity_idx)
        acc = int(md.compose_table[c_idx, acc])
        pos += 1

    return bool(md.accept_table[acc])
