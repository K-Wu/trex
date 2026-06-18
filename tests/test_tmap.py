import numpy as np
from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices, precompute_tmap


def test_tmap_matches_matrices():
    """For each alphabet char, tmap should match the transition matrix."""
    dfa = compile_regex("(a|b)*a(a|b)")
    dm = DFAMatrices(dfa)
    N = dm.n_states

    tmap = precompute_tmap(dm)
    assert tmap.dtype == np.uint8
    assert tmap.shape == (256 * N,)

    for ch in dm.alphabet:
        byte_val = ord(ch)
        T = dm.matrices[ch]
        for s in range(dm.n_states_raw):
            dst = int(np.argmax(T[:, s]))
            assert tmap[byte_val * N + s] == dst, \
                f"tmap mismatch: char={ch!r}, state={s}, got={tmap[byte_val * N + s]}, expected={dst}"


def test_tmap_unmapped_identity():
    """Unmapped characters should act as identity."""
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    N = dm.n_states
    tmap = precompute_tmap(dm)

    unmapped_byte = 0  # null byte
    if chr(unmapped_byte) not in dm.char_to_idx:
        for s in range(N):
            assert tmap[unmapped_byte * N + s] == s


def test_tmap_padded_states_identity():
    """Padded states beyond n_states_raw should self-loop."""
    dfa = compile_regex("ab")
    dm = DFAMatrices(dfa)
    N = dm.n_states
    tmap = precompute_tmap(dm)

    for byte_val in range(256):
        for s in range(dm.n_states_raw, N):
            assert tmap[byte_val * N + s] == s


def test_tmap_compose_matches_matrix():
    """Composing two tmaps should match matrix multiplication."""
    dfa = compile_regex("(a|b)*abb")
    dm = DFAMatrices(dfa)
    N = dm.n_states
    tmap = precompute_tmap(dm)

    a_byte = ord('a')
    b_byte = ord('b')
    T_ab = dm.matrices['b'].astype(np.int32) @ dm.matrices['a'].astype(np.int32)
    for s in range(dm.n_states_raw):
        intermediate = int(tmap[a_byte * N + s])
        composed = int(tmap[b_byte * N + intermediate])
        expected = int(np.argmax(T_ab[:, s]))
        assert composed == expected, \
            f"compose mismatch: state={s}, got={composed}, expected={expected}"
