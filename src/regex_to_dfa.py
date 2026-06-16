"""
Regex → NFA → DFA → Minimized DFA pipeline.

Supports: concatenation, alternation (|), Kleene star (*), plus (+),
          optional (?), character classes [abc], [a-z], dot (.), escapes.
Does NOT support: backreferences, lookahead/behind, counted repetition {n,m}.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import string

EPSILON = None

@dataclass
class NFAState:
    id: int
    transitions: dict  = field(default_factory=dict)  # symbol → list[int]
    is_accept: bool = False

class NFA:
    def __init__(self):
        self.states: dict[int, NFAState] = {}
        self.start: int = -1
        self.accept: int = -1
        self.alphabet: set[str] = set()
        self._next_id = 0

    def new_state(self, is_accept=False) -> int:
        sid = self._next_id
        self._next_id += 1
        self.states[sid] = NFAState(id=sid, is_accept=is_accept)
        return sid

    def add_trans(self, src: int, sym: Optional[str], dst: int):
        self.states[src].transitions.setdefault(sym, []).append(dst)
        if sym is not None:
            self.alphabet.add(sym)


def _build_literal(nfa: NFA, ch: str) -> tuple[int, int]:
    s = nfa.new_state()
    a = nfa.new_state()
    nfa.add_trans(s, ch, a)
    return s, a

def _build_concat(nfa: NFA, parts: list[tuple[int,int]]) -> tuple[int, int]:
    if not parts:
        s = nfa.new_state()
        a = nfa.new_state()
        nfa.add_trans(s, EPSILON, a)
        return s, a
    start, end = parts[0]
    for ps, pe in parts[1:]:
        nfa.add_trans(end, EPSILON, ps)
        end = pe
    return start, end

def _build_alt(nfa: NFA, branches: list[tuple[int,int]]) -> tuple[int, int]:
    s = nfa.new_state()
    a = nfa.new_state()
    for bs, be in branches:
        nfa.add_trans(s, EPSILON, bs)
        nfa.add_trans(be, EPSILON, a)
    return s, a

def _build_star(nfa: NFA, inner: tuple[int,int]) -> tuple[int, int]:
    s = nfa.new_state()
    a = nfa.new_state()
    i_s, i_a = inner
    nfa.add_trans(s, EPSILON, i_s)
    nfa.add_trans(s, EPSILON, a)
    nfa.add_trans(i_a, EPSILON, i_s)
    nfa.add_trans(i_a, EPSILON, a)
    return s, a

def _build_plus(nfa: NFA, inner: tuple[int,int]) -> tuple[int, int]:
    s = nfa.new_state()
    a = nfa.new_state()
    i_s, i_a = inner
    nfa.add_trans(s, EPSILON, i_s)
    nfa.add_trans(i_a, EPSILON, i_s)
    nfa.add_trans(i_a, EPSILON, a)
    return s, a

def _build_optional(nfa: NFA, inner: tuple[int,int]) -> tuple[int, int]:
    s = nfa.new_state()
    a = nfa.new_state()
    i_s, i_a = inner
    nfa.add_trans(s, EPSILON, i_s)
    nfa.add_trans(s, EPSILON, a)
    nfa.add_trans(i_a, EPSILON, a)
    return s, a


# ─── Regex Parser ───────────────────────────────────────────────────────────

class RegexParser:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0
        self.nfa = NFA()

    def _peek(self) -> Optional[str]:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def _consume(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def parse(self) -> NFA:
        s, a = self._parse_alt()
        self.nfa.start = s
        self.nfa.accept = a
        self.nfa.states[a].is_accept = True
        return self.nfa

    def _parse_alt(self) -> tuple[int, int]:
        branches = [self._parse_concat()]
        while self._peek() == '|':
            self._consume()
            branches.append(self._parse_concat())
        if len(branches) == 1:
            return branches[0]
        return _build_alt(self.nfa, branches)

    def _parse_concat(self) -> tuple[int, int]:
        parts = []
        while self._peek() is not None and self._peek() not in ('|', ')'):
            parts.append(self._parse_repeat())
        if not parts:
            s = self.nfa.new_state()
            a = self.nfa.new_state()
            self.nfa.add_trans(s, EPSILON, a)
            return s, a
        return _build_concat(self.nfa, parts)

    def _parse_repeat(self) -> tuple[int, int]:
        inner = self._parse_atom()
        if self._peek() == '*':
            self._consume()
            return _build_star(self.nfa, inner)
        elif self._peek() == '+':
            self._consume()
            return _build_plus(self.nfa, inner)
        elif self._peek() == '?':
            self._consume()
            return _build_optional(self.nfa, inner)
        return inner

    def _parse_atom(self) -> tuple[int, int]:
        ch = self._peek()
        if ch == '(':
            self._consume()
            result = self._parse_alt()
            if self._peek() == ')':
                self._consume()
            return result
        elif ch == '[':
            return self._parse_charclass()
        elif ch == '.':
            self._consume()
            chars = [c for c in string.printable[:95]]
            lits = [_build_literal(self.nfa, c) for c in chars]
            return _build_alt(self.nfa, lits)
        elif ch == '\\':
            self._consume()
            esc = self._consume()
            if esc == 'd':
                lits = [_build_literal(self.nfa, c) for c in '0123456789']
                return _build_alt(self.nfa, lits)
            elif esc == 'w':
                chars = string.ascii_letters + string.digits + '_'
                lits = [_build_literal(self.nfa, c) for c in chars]
                return _build_alt(self.nfa, lits)
            elif esc == 's':
                lits = [_build_literal(self.nfa, c) for c in ' \t\n\r']
                return _build_alt(self.nfa, lits)
            return _build_literal(self.nfa, esc)
        else:
            self._consume()
            return _build_literal(self.nfa, ch)

    def _parse_charclass(self) -> tuple[int, int]:
        self._consume()  # '['
        chars = []
        negate = False
        if self._peek() == '^':
            negate = True
            self._consume()
        while self._peek() != ']' and self._peek() is not None:
            c = self._consume()
            if self._peek() == '-' and self.pos + 1 < len(self.pattern) and self.pattern[self.pos + 1] != ']':
                self._consume()
                end = self._consume()
                chars.extend(chr(i) for i in range(ord(c), ord(end) + 1))
            else:
                chars.append(c)
        if self._peek() == ']':
            self._consume()
        if negate:
            chars = list(set(string.printable[:95]) - set(chars))
        if not chars:
            chars = ['\x00']
        lits = [_build_literal(self.nfa, c) for c in chars]
        if len(lits) == 1:
            return lits[0]
        return _build_alt(self.nfa, lits)


# ─── NFA → DFA (Subset Construction) ───────────────────────────────────────

@dataclass
class DFA:
    n_states: int
    start: int
    accept_states: set[int]
    transitions: dict[int, dict[str, int]]
    alphabet: set[str]
    dead_state: Optional[int] = None

    def simulate(self, input_str: str) -> bool:
        state = self.start
        for ch in input_str:
            if ch in self.transitions.get(state, {}):
                state = self.transitions[state][ch]
            elif self.dead_state is not None:
                state = self.dead_state
            else:
                return False
        return state in self.accept_states


def epsilon_closure(nfa: NFA, states: frozenset[int]) -> frozenset[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for dst in nfa.states[s].transitions.get(EPSILON, []):
            if dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return frozenset(closure)


def nfa_to_dfa(nfa: NFA) -> DFA:
    start_set = epsilon_closure(nfa, frozenset([nfa.start]))
    unmarked = [start_set]
    dfa_states = {start_set: 0}
    dfa_transitions: dict[int, dict[str, int]] = {}
    accept_states = set()
    next_id = 1
    alphabet = nfa.alphabet

    while unmarked:
        T = unmarked.pop()
        t_id = dfa_states[T]
        dfa_transitions[t_id] = {}
        if nfa.accept in T:
            accept_states.add(t_id)
        for ch in alphabet:
            move = set()
            for s in T:
                move.update(nfa.states[s].transitions.get(ch, []))
            if move:
                U = epsilon_closure(nfa, frozenset(move))
                if U not in dfa_states:
                    dfa_states[U] = next_id
                    next_id += 1
                    unmarked.append(U)
                dfa_transitions[t_id][ch] = dfa_states[U]

    return DFA(n_states=next_id, start=0, accept_states=accept_states,
               transitions=dfa_transitions, alphabet=alphabet)


def complete_dfa(dfa: DFA) -> DFA:
    needs_dead = any(
        ch not in dfa.transitions.get(s, {})
        for s in range(dfa.n_states)
        for ch in dfa.alphabet
    )
    if not needs_dead:
        dfa.dead_state = None
        return dfa

    dead = dfa.n_states
    dfa.n_states += 1
    dfa.dead_state = dead
    dfa.transitions[dead] = {ch: dead for ch in dfa.alphabet}
    for s in range(dfa.n_states):
        if s not in dfa.transitions:
            dfa.transitions[s] = {}
        for ch in dfa.alphabet:
            if ch not in dfa.transitions[s]:
                dfa.transitions[s][ch] = dead
    return dfa


def minimize_dfa(dfa: DFA) -> DFA:
    non_accept = set(range(dfa.n_states)) - dfa.accept_states
    if not non_accept or not dfa.accept_states:
        return dfa

    P = [dfa.accept_states.copy(), non_accept.copy()]
    W = [dfa.accept_states.copy(), non_accept.copy()]

    while W:
        A = W.pop()
        for ch in dfa.alphabet:
            X = {s for s in range(dfa.n_states) if dfa.transitions.get(s, {}).get(ch) in A}
            if not X:
                continue
            new_P = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    new_P.append(inter)
                    new_P.append(diff)
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    new_P.append(Y)
            P = new_P

    state_to_part = {}
    for i, part in enumerate(P):
        for s in part:
            state_to_part[s] = i

    new_transitions: dict[int, dict[str, int]] = {}
    for i, part in enumerate(P):
        rep = next(iter(part))
        new_transitions[i] = {}
        for ch in dfa.alphabet:
            if ch in dfa.transitions.get(rep, {}):
                new_transitions[i][ch] = state_to_part[dfa.transitions[rep][ch]]

    return DFA(
        n_states=len(P),
        start=state_to_part[dfa.start],
        accept_states={state_to_part[s] for s in dfa.accept_states},
        transitions=new_transitions,
        alphabet=dfa.alphabet,
        dead_state=state_to_part.get(dfa.dead_state) if dfa.dead_state is not None else None,
    )


def compile_regex(pattern: str, minimize: bool = True) -> DFA:
    parser = RegexParser(pattern)
    nfa = parser.parse()
    dfa = nfa_to_dfa(nfa)
    dfa = complete_dfa(dfa)
    if minimize:
        dfa = minimize_dfa(dfa)
        dfa = complete_dfa(dfa)
    return dfa


if __name__ == '__main__':
    dfa = compile_regex("(a|b)*abb")
    print(f"States: {dfa.n_states}, Alphabet: {len(dfa.alphabet)} chars")
    print(f"Accept: {dfa.accept_states}")
    tests = [("abb", True), ("aabb", True), ("babb", True),
             ("ab", False), ("abc", False), ("", False)]
    for s, exp in tests:
        got = dfa.simulate(s)
        ok = "✓" if got == exp else "✗"
        print(f"  '{s}': {got} (expected {exp}) {ok}")
