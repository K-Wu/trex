"""
OptimizedEngine — Unified API for DFA/NFA regex matching with auto-selected backend.

Backends:
  - sequential  (DFA + char-by-char, baseline)
  - monoid      (DFA + transition monoid)
  - monoid+kgram(DFA + transition monoid + k-gram precomputation)
  - nfa         (NFA matrix simulation)

Auto-selection heuristic:
  1. Try DFA construction. If n_states > dfa_state_cap → NFA path.
  2. If DFA is small enough, compute monoid (cap monoid_cap).
     If monoid fits → monoid backend + kgram.
  3. If monoid too large → sequential (matrix scan) backend.
"""

from __future__ import annotations

import time
from typing import Optional

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, simulate_sequential
from src.monoid import compute_monoid, simulate_monoid
from src.kgram import precompute_kgrams, simulate_kgram_monoid, auto_k
from src.nfa_matrices import compile_nfa_matrices, simulate_nfa


class OptimizedEngine:
    """Unified regex matching engine with automatic backend selection.

    Parameters
    ----------
    regex : str
        Regular expression to compile.
    config : str | None
        Backend selection:
          None            — auto-select (try DFA → monoid → kgram, fall back to NFA)
          "monoid"        — force DFA + monoid scan
          "monoid+kgram"  — force DFA + monoid + k-gram
          "baseline"      — force DFA + sequential simulation
          "nfa"           — force NFA path
    dfa_state_cap : int
        Maximum DFA states before falling back to NFA in auto mode (default 64).
    monoid_cap : int
        Maximum monoid elements before falling back to matrix scan (default 65536).
    """

    def __init__(
        self,
        regex: str,
        config: Optional[str] = None,
        dfa_state_cap: int = 64,
        monoid_cap: int = 65536,
    ):
        self._regex = regex
        self._config = config
        self._dfa_state_cap = dfa_state_cap
        self._monoid_cap = monoid_cap

        # Backend objects — only the active ones will be set.
        self._dfa = None      # DFA object
        self._dm = None       # DFAMatrices
        self._md = None       # MonoidData
        self._kg = None       # KGramTable
        self._nm = None       # NFAMatrices

        self._representation = None   # "dfa" | "nfa"
        self._scan_backend = None     # "sequential" | "monoid" | "monoid+kgram" | "nfa"
        self._selection_reason = None
        self._kgram_k = None

        if config is None:
            self._auto_select()
        elif config == "monoid":
            self._force_monoid()
        elif config == "monoid+kgram":
            self._force_monoid_kgram()
        elif config == "baseline":
            self._force_baseline()
        elif config == "nfa":
            self._force_nfa()
        else:
            raise ValueError(f"Unknown config: {config!r}. "
                             f"Choose from None, 'monoid', 'monoid+kgram', 'baseline', 'nfa'.")

    # ── Private setup helpers ────────────────────────────────────────────────

    def _build_dfa(self):
        """Compile regex → DFA → DFAMatrices (cached)."""
        if self._dfa is None:
            self._dfa = compile_regex(self._regex)
            self._dm = DFAMatrices(self._dfa)

    def _build_nfa(self):
        """Compile regex → NFAMatrices (cached)."""
        if self._nm is None:
            self._nm = compile_nfa_matrices(self._regex)

    def _auto_select(self):
        """Heuristic selection: DFA size → monoid size → backend choice."""
        self._build_dfa()
        n_states = self._dfa.n_states
        alphabet_size = len(self._dfa.alphabet)

        if n_states > self._dfa_state_cap:
            # DFA too large — fall back to NFA
            self._build_nfa()
            self._representation = "nfa"
            self._scan_backend = "nfa"
            self._selection_reason = (
                f"DFA has {n_states} states > cap {self._dfa_state_cap}; using NFA"
            )
            return

        # Try monoid
        md = compute_monoid(self._dm, max_size=self._monoid_cap)
        if md is not None:
            self._md = md
            k = auto_k(alphabet_size)
            self._kg = precompute_kgrams(self._dm, k, monoid=self._md)
            self._kgram_k = k
            self._representation = "dfa"
            self._scan_backend = "monoid+kgram"
            self._selection_reason = (
                f"DFA has {n_states} states; monoid size {md.size} fits; using monoid+kgram"
            )
        else:
            # Monoid too large — fall back to sequential
            self._representation = "dfa"
            self._scan_backend = "sequential"
            self._selection_reason = (
                f"DFA has {n_states} states; monoid exceeds cap {self._monoid_cap}; "
                f"using sequential"
            )

    def _force_monoid(self):
        self._build_dfa()
        self._md = compute_monoid(self._dm, max_size=self._monoid_cap)
        if self._md is None:
            raise RuntimeError(
                "compute_monoid returned None (monoid too large). "
                "Increase monoid_cap or use a different config."
            )
        self._representation = "dfa"
        self._scan_backend = "monoid"
        self._selection_reason = "forced config='monoid'"

    def _force_monoid_kgram(self):
        self._build_dfa()
        self._md = compute_monoid(self._dm, max_size=self._monoid_cap)
        if self._md is None:
            raise RuntimeError(
                "compute_monoid returned None (monoid too large). "
                "Increase monoid_cap or use a different config."
            )
        alphabet_size = len(self._dfa.alphabet)
        k = auto_k(alphabet_size)
        self._kg = precompute_kgrams(self._dm, k, monoid=self._md)
        self._kgram_k = k
        self._representation = "dfa"
        self._scan_backend = "monoid+kgram"
        self._selection_reason = "forced config='monoid+kgram'"

    def _force_baseline(self):
        self._build_dfa()
        self._representation = "dfa"
        self._scan_backend = "sequential"
        self._selection_reason = "forced config='baseline'"

    def _force_nfa(self):
        self._build_nfa()
        self._representation = "nfa"
        self._scan_backend = "nfa"
        self._selection_reason = "forced config='nfa'"

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def config_info(self) -> dict:
        """Return a dict describing the selected backend configuration.

        Required keys: representation, scan_backend, alphabet_size,
                       kgram_k, selection_reason.
        Optional keys: dfa_states, monoid_size, nfa_states.
        """
        info: dict = {
            "representation": self._representation,
            "scan_backend": self._scan_backend,
            "kgram_k": self._kgram_k,
            "selection_reason": self._selection_reason,
        }

        if self._dfa is not None:
            info["alphabet_size"] = len(self._dfa.alphabet)
            info["dfa_states"] = self._dfa.n_states
        elif self._nm is not None:
            info["alphabet_size"] = len(self._nm.alphabet)
            info["nfa_states"] = self._nm.n_states_raw

        if self._md is not None:
            info["monoid_size"] = self._md.size

        if self._nm is not None and "nfa_states" not in info:
            info["nfa_states"] = self._nm.n_states_raw

        return info

    def _match_one(self, s: str) -> bool:
        """Dispatch a single string to the active backend."""
        if self._nm is not None:
            return simulate_nfa(self._nm, s)
        if self._md is not None and self._kg is not None:
            return simulate_kgram_monoid(self._kg, self._md, self._dm, s)
        if self._md is not None:
            return simulate_monoid(self._md, self._dm, s)
        # Fallback: sequential DFA
        return simulate_sequential(self._dfa, s)

    def match_batch(self, strings: list) -> list:
        """Match a list of strings. Returns list[bool]."""
        return [self._match_one(s) for s in strings]

    def match_batch_timed(self, strings: list) -> tuple:
        """Match a list of strings and return (results, timing_dict).

        timing_dict keys: total_seconds, per_string_seconds, n_strings.
        """
        t0 = time.perf_counter()
        results = self.match_batch(strings)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timing = {
            "total_seconds": elapsed,
            "per_string_seconds": elapsed / len(strings) if strings else 0.0,
            "n_strings": len(strings),
        }
        return results, timing
