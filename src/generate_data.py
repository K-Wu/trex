"""
Generate test data for benchmarking tensor-core regex matching.

Workloads:
  1. Synthetic regex patterns with known DFA sizes
  2. Realistic patterns (email, IP, log timestamps, identifiers)
  3. Input strings: random, adversarial, real-world-like
  4. Scaling axes: string length, batch size, DFA state count
"""

from __future__ import annotations
import json
import random
import string
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TestPattern:
    name: str
    regex: str
    description: str
    expected_dfa_states: Optional[int] = None  # estimate, verified at runtime


@dataclass
class TestWorkload:
    pattern: TestPattern
    strings: list[str]
    expected_results: list[bool]  # ground truth from sequential simulation
    category: str  # 'correctness', 'throughput', 'scaling'


# ─── Pattern Library ────────────────────────────────────────────────────────

PATTERNS = {
    # ── Small DFAs (≤ 8 states) — fit comfortably in one tensor core tile ──
    'abb': TestPattern(
        name='abb',
        regex='(a|b)*abb',
        description='Classic textbook: strings over {a,b} ending in "abb"',
        expected_dfa_states=5,
    ),
    'binary_div3': TestPattern(
        name='binary_div3',
        regex='(0|(1(01*0)*1))*',
        description='Binary strings divisible by 3',
        expected_dfa_states=4,
    ),
    'even_a': TestPattern(
        name='even_a',
        regex='(b*ab*ab*)*b*',
        description='Even number of a\'s over {a,b}',
        expected_dfa_states=3,
    ),
    'ab_star': TestPattern(
        name='ab_star',
        regex='(ab)*',
        description='Alternating ab pairs',
        expected_dfa_states=4,
    ),

    # ── Medium DFAs (5-12 states) ──
    'email_simple': TestPattern(
        name='email_simple',
        regex='[a-z]+@[a-z]+\\.[a-z]+',
        description='Simplified email-like pattern',
        expected_dfa_states=8,
    ),
    'hex_number': TestPattern(
        name='hex_number',
        regex='0x[0-9a-f]+',
        description='Hex literal (e.g. 0x1a2f)',
        expected_dfa_states=5,
    ),
    'identifier': TestPattern(
        name='identifier',
        regex='[a-z][a-z0-9]*',
        description='C-style identifier (lowercase)',
        expected_dfa_states=4,
    ),
    'fixed_keyword': TestPattern(
        name='fixed_keyword',
        regex='(if|else|while|for|return)',
        description='Match any of 5 keywords',
        expected_dfa_states=16,
    ),

    # ── Larger DFAs (state-explosion stress) ──
    'three_char_end': TestPattern(
        name='three_char_end',
        regex='[a-c]*abc',
        description='Strings over {a,b,c} ending in "abc"',
        expected_dfa_states=8,
    ),
    'nested_alt': TestPattern(
        name='nested_alt',
        regex='((ab|cd)(ef|gh))+',
        description='Nested alternation pairs',
        expected_dfa_states=10,
    ),
}


# ─── String Generators ─────────────────────────────────────────────────────

def gen_random_string(alphabet: str, length: int, rng: random.Random) -> str:
    """Uniform random string over alphabet."""
    return ''.join(rng.choice(alphabet) for _ in range(length))


def gen_matching_string_abb(length: int, rng: random.Random) -> str:
    """Generate a string that matches (a|b)*abb."""
    if length < 3:
        return 'abb'[:length]
    prefix = ''.join(rng.choice('ab') for _ in range(length - 3))
    return prefix + 'abb'


def gen_adversarial_string(alphabet: str, length: int, rng: random.Random) -> str:
    """String designed to keep DFA in non-trivial states (near-match then diverge)."""
    # Build partial matches that reset
    s = []
    while len(s) < length:
        # Add a near-matching prefix then break it
        chunk_len = rng.randint(2, min(10, length - len(s)))
        for _ in range(chunk_len - 1):
            s.append(rng.choice(alphabet[:2]))  # bias toward "matching" chars
        s.append(rng.choice(alphabet))  # random char to potentially break
    return ''.join(s[:length])


def gen_strings_for_pattern(
    pattern_name: str,
    n_strings: int,
    length: int,
    match_ratio: float = 0.5,
    rng: Optional[random.Random] = None,
) -> tuple[list[str], str]:
    """
    Generate n_strings of given length for a named pattern.
    Returns (strings, alphabet_hint).
    """
    if rng is None:
        rng = random.Random(42)

    pat = PATTERNS[pattern_name]
    strings = []

    # Determine alphabet
    if pattern_name in ('abb', 'even_a', 'ab_star'):
        alphabet = 'ab'
    elif pattern_name == 'binary_div3':
        alphabet = '01'
    elif pattern_name == 'three_char_end':
        alphabet = 'abc'
    elif pattern_name == 'nested_alt':
        alphabet = 'abcdefgh'
    elif pattern_name in ('email_simple', 'identifier'):
        alphabet = string.ascii_lowercase + '@.'
    elif pattern_name == 'hex_number':
        alphabet = '0123456789abcdefx'
    elif pattern_name == 'fixed_keyword':
        alphabet = string.ascii_lowercase
    else:
        alphabet = string.ascii_lowercase

    n_match = int(n_strings * match_ratio)
    for i in range(n_strings):
        if i < n_match and pattern_name == 'abb':
            s = gen_matching_string_abb(length, rng)
        elif i < n_match and pattern_name == 'hex_number':
            body = ''.join(rng.choice('0123456789abcdef') for _ in range(max(1, length - 2)))
            s = ('0x' + body)[:length]
        else:
            s = gen_random_string(alphabet, length, rng)
        strings.append(s)

    return strings, alphabet


# ─── Workload Generators ───────────────────────────────────────────────────

def generate_correctness_workloads() -> list[dict]:
    """
    Small workloads for verifying all simulation backends agree.
    """
    workloads = []

    # Hand-crafted tests for (a|b)*abb
    workloads.append({
        'name': 'abb_handcrafted',
        'pattern': 'abb',
        'strings': ['abb', 'aabb', 'babb', 'ababb', 'ab', 'ba', 'a', 'b', '',
                     'aababb', 'bbbabb', 'ababab', 'abba'],
        'expected': [True, True, True, True, False, False, False, False, False,
                     True, True, False, False],
        'category': 'correctness',
    })

    # Various patterns with short strings
    for pname in ['binary_div3', 'even_a', 'ab_star', 'hex_number', 'identifier']:
        strings, _ = gen_strings_for_pattern(pname, 50, 20, match_ratio=0.5)
        workloads.append({
            'name': f'{pname}_short',
            'pattern': pname,
            'strings': strings,
            'expected': None,  # computed at runtime
            'category': 'correctness',
        })

    return workloads


def generate_throughput_workloads() -> list[dict]:
    """
    Large workloads for measuring throughput (GB/s).
    Vary string length on a fixed pattern.
    """
    workloads = []
    rng = random.Random(42)

    lengths = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    for length in lengths:
        # Fewer strings for longer lengths to keep memory reasonable
        n_strings = max(10, min(1000, 10_000_000 // length))
        strings, _ = gen_strings_for_pattern('abb', n_strings, length, 0.5, rng)
        workloads.append({
            'name': f'abb_len{length}',
            'pattern': 'abb',
            'strings': strings,
            'expected': None,
            'category': 'throughput',
            'length': length,
            'n_strings': n_strings,
        })

    return workloads


def generate_scaling_workloads() -> list[dict]:
    """
    Workloads that vary DFA state count to measure the effect of matrix size.
    """
    workloads = []
    rng = random.Random(42)
    length = 4096
    n_strings = 100

    for pname in ['even_a', 'abb', 'hex_number', 'three_char_end',
                   'nested_alt', 'fixed_keyword']:
        strings, _ = gen_strings_for_pattern(pname, n_strings, length, 0.3, rng)
        workloads.append({
            'name': f'scale_{pname}',
            'pattern': pname,
            'strings': strings,
            'expected': None,
            'category': 'scaling',
        })

    return workloads


def generate_batch_workloads() -> list[dict]:
    """
    Fixed-length strings in varying batch sizes to measure batch parallelism.
    """
    workloads = []
    rng = random.Random(42)
    length = 1024

    for batch_size in [16, 64, 256, 1024, 4096, 16384]:
        strings, _ = gen_strings_for_pattern('abb', batch_size, length, 0.5, rng)
        workloads.append({
            'name': f'batch_{batch_size}',
            'pattern': 'abb',
            'strings': strings,
            'expected': None,
            'category': 'batch_scaling',
            'batch_size': batch_size,
        })

    return workloads


# ─── Save / Load ────────────────────────────────────────────────────────────

def save_workloads(workloads: list[dict], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(workloads, f, indent=2)


def load_workloads(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("Generating workloads...")

    cw = generate_correctness_workloads()
    save_workloads(cw, os.path.join(data_dir, 'correctness_workloads.json'))
    print(f"  Correctness: {len(cw)} workloads")

    tw = generate_throughput_workloads()
    save_workloads(tw, os.path.join(data_dir, 'throughput_workloads.json'))
    total_bytes = sum(w['length'] * w['n_strings'] for w in tw)
    print(f"  Throughput:  {len(tw)} workloads, {total_bytes/1e6:.1f} MB total input")

    sw = generate_scaling_workloads()
    save_workloads(sw, os.path.join(data_dir, 'scaling_workloads.json'))
    print(f"  Scaling:     {len(sw)} workloads")

    bw = generate_batch_workloads()
    save_workloads(bw, os.path.join(data_dir, 'batch_workloads.json'))
    print(f"  Batch:       {len(bw)} workloads")

    # Save pattern library
    pl = {k: asdict(v) for k, v in PATTERNS.items()}
    save_workloads(pl, os.path.join(data_dir, 'patterns.json'))
    print(f"\n  Pattern library: {len(pl)} patterns saved")
    print("Done.")
