"""
K-gram TC engine performance benchmarks with figure generation.

Measures:
  1. Throughput vs k value (k=1,2,4,8,16) for binary alphabet
  2. Throughput vs batch size across backends (monoid, batched TC, k-gram TC)
  3. Throughput vs string length across backends
  4. Kernel-only vs end-to-end throughput breakdown
  5. TC utilization estimates

Generates figures to results/kgram_*.png
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.regex_to_dfa import compile_regex
from src.simulation import DFAMatrices
from src.generate_data import PATTERNS
from src.kgram import auto_k_for_gpu

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _random_strings(alphabet, n, length, seed=42):
    rng = random.Random(seed)
    alpha = sorted(alphabet)
    return [''.join(rng.choice(alpha) for _ in range(length)) for _ in range(n)]


def _robust_median(times):
    """Median after dropping outliers > 10x the minimum."""
    if not times:
        return 0.0
    mn = min(times)
    filtered = [t for t in times if t < mn * 10]
    if not filtered:
        filtered = times
    return float(np.median(filtered))


def _load_backends(dm, md=None):
    backends = {}

    try:
        from src.gpu_bridge_kgram import KGramGPUSimulator
        backends['kgram_sim'] = KGramGPUSimulator()
    except Exception:
        pass

    try:
        from src.gpu_bridge_batched import BatchedGPUSimulator
        backends['batched_sim'] = BatchedGPUSimulator()
    except Exception:
        pass

    try:
        from src.gpu_bridge_monoid import MonoidGPUSimulator
        backends['monoid_sim'] = MonoidGPUSimulator()
    except Exception:
        pass

    return backends


def bench_k_sweep(dm, backends, alphabet, n_warmup=5, n_reps=10):
    """Throughput as a function of k for the k-gram kernel."""
    if 'kgram_sim' not in backends:
        print("  [skipped: kgram GPU not available]")
        return None

    sigma = len(dm.alphabet)
    N = dm.n_states

    B = 16384
    L = 512
    strings = _random_strings(alphabet, B, L, seed=42)
    total_chars = B * L

    k_values = [1, 2, 4, 8, 16]
    max_k = auto_k_for_gpu(sigma, N)
    k_values = [k for k in k_values if k <= max_k]

    results = []
    for k in k_values:
        eng = backends['kgram_sim'].create_engine(dm, k=k, max_B=B + 128, max_L=L + 64)

        for _ in range(n_warmup):
            eng.simulate_batch(strings)

        kern_times = []
        total_times = []
        for _ in range(n_reps):
            _, kern_ms, total_ms = eng.simulate_batch_timed(strings)
            kern_times.append(kern_ms)
            total_times.append(total_ms)

        kern_ms = _robust_median(kern_times)
        total_ms = _robust_median(total_times)

        kern_gcs = total_chars / (kern_ms * 1e6) if kern_ms > 0 else 0
        total_gcs = total_chars / (total_ms * 1e6) if total_ms > 0 else 0

        results.append({
            'k': k, 'kern_ms': kern_ms, 'total_ms': total_ms,
            'kern_gcs': kern_gcs, 'total_gcs': total_gcs,
            'B': B, 'L': L, 'sigma': sigma, 'N': N,
        })
        print(f"    k={k:>2}: kernel {kern_gcs:.2f} Gc/s, total {total_gcs:.2f} Gc/s  "
              f"(kern {kern_ms:.3f} ms, total {total_ms:.3f} ms)")
        eng.destroy()

    return results


def bench_batch_scaling(dm, md, backends, alphabet, n_warmup=5, n_reps=10):
    """Throughput vs batch size across all GPU backends."""
    L = 512
    batch_sizes = [64, 256, 1024, 4096, 16384, 65536]

    sigma = len(dm.alphabet)
    k_gpu = auto_k_for_gpu(sigma, dm.n_states)

    results = {'batched': [], 'kgram': [], 'monoid': []}

    for B in batch_sizes:
        strings = _random_strings(alphabet, B, L, seed=42)
        total_chars = B * L

        for name, sim_key, create_fn in [
            ('batched', 'batched_sim', lambda: backends['batched_sim'].create_engine(
                dm, max_B=B + 128, max_L=L + 64)),
            ('kgram', 'kgram_sim', lambda: backends['kgram_sim'].create_engine(
                dm, k=k_gpu, max_B=B + 128, max_L=L + 64)),
            ('monoid', 'monoid_sim', lambda: backends['monoid_sim'].create_engine(
                md, dm, max_total_chars=total_chars + 4096, max_batch=B + 128) if md else None),
        ]:
            if sim_key not in backends:
                continue
            try:
                eng = create_fn()
                if eng is None:
                    continue
            except Exception as e:
                print(f"    [{name} B={B}: create failed: {e}]")
                continue

            for _ in range(n_warmup):
                eng.simulate_batch(strings)

            kern_times = []
            total_times = []
            for _ in range(n_reps):
                _, kern_ms, total_ms = eng.simulate_batch_timed(strings)
                kern_times.append(kern_ms)
                total_times.append(total_ms)

            kern_ms = _robust_median(kern_times)
            total_ms = _robust_median(total_times)
            kern_gcs = total_chars / (kern_ms * 1e6) if kern_ms > 0 else 0
            total_gcs = total_chars / (total_ms * 1e6) if total_ms > 0 else 0

            results[name].append({
                'B': B, 'L': L, 'kern_gcs': kern_gcs, 'total_gcs': total_gcs,
                'kern_ms': kern_ms, 'total_ms': total_ms,
            })
            eng.destroy()

        print(f"    B={B:>6}: " + "  ".join(
            f"{name}={results[name][-1]['kern_gcs']:.2f}" if results[name] and results[name][-1]['B'] == B else f"{name}=N/A"
            for name in ['monoid', 'batched', 'kgram']
        ) + " Gc/s (kernel)")

    return results


def bench_length_scaling(dm, md, backends, alphabet, n_warmup=5, n_reps=10):
    """Throughput vs string length across all GPU backends."""
    B = 16384
    lengths = [32, 64, 128, 256, 512, 1024, 2048]

    sigma = len(dm.alphabet)
    k_gpu = auto_k_for_gpu(sigma, dm.n_states)

    results = {'batched': [], 'kgram': [], 'monoid': []}

    for L in lengths:
        strings = _random_strings(alphabet, B, L, seed=42)
        total_chars = B * L

        for name, sim_key, create_fn in [
            ('batched', 'batched_sim', lambda: backends['batched_sim'].create_engine(
                dm, max_B=B + 128, max_L=L + 64)),
            ('kgram', 'kgram_sim', lambda: backends['kgram_sim'].create_engine(
                dm, k=k_gpu, max_B=B + 128, max_L=L + 64)),
            ('monoid', 'monoid_sim', lambda: backends['monoid_sim'].create_engine(
                md, dm, max_total_chars=total_chars + 4096, max_batch=B + 128) if md else None),
        ]:
            if sim_key not in backends:
                continue
            try:
                eng = create_fn()
                if eng is None:
                    continue
            except Exception as e:
                continue

            for _ in range(n_warmup):
                eng.simulate_batch(strings)

            kern_times = []
            total_times = []
            for _ in range(n_reps):
                _, kern_ms, total_ms = eng.simulate_batch_timed(strings)
                kern_times.append(kern_ms)
                total_times.append(total_ms)

            kern_ms = _robust_median(kern_times)
            total_ms = _robust_median(total_times)
            kern_gcs = total_chars / (kern_ms * 1e6) if kern_ms > 0 else 0
            total_gcs = total_chars / (total_ms * 1e6) if total_ms > 0 else 0

            results[name].append({
                'B': B, 'L': L, 'kern_gcs': kern_gcs, 'total_gcs': total_gcs,
                'kern_ms': kern_ms, 'total_ms': total_ms,
            })
            eng.destroy()

        print(f"    L={L:>5}: " + "  ".join(
            f"{name}={results[name][-1]['kern_gcs']:.2f}" if results[name] and results[name][-1]['L'] == L else f"{name}=N/A"
            for name in ['monoid', 'batched', 'kgram']
        ) + " Gc/s (kernel)")

    return results


def bench_tc_utilization(dm, backends, alphabet, n_warmup=5, n_reps=10):
    """Estimate TC utilization for k-gram kernel at different k."""
    if 'kgram_sim' not in backends:
        return None

    sigma = len(dm.alphabet)
    N = dm.n_states
    B = 65536
    L = 512
    strings = _random_strings(alphabet, B, L, seed=42)
    total_chars = B * L

    max_k = auto_k_for_gpu(sigma, N)
    k_values = [k for k in [1, 2, 4, 8, 16] if k <= max_k]

    H200_INT8_TOPS = 3958
    results = []

    for k in k_values:
        eng = backends['kgram_sim'].create_engine(dm, k=k, max_B=B + 128, max_L=L + 64)

        for _ in range(n_warmup):
            eng.simulate_batch(strings)

        kern_times = []
        for _ in range(n_reps):
            _, kern_ms, _ = eng.simulate_batch_timed(strings)
            kern_times.append(kern_ms)

        kern_ms = _robust_median(kern_times)

        mmas_per_string = L // k + (1 if L % k else 0)
        ops_per_mma = 2 * 16 * 16 * 16
        total_ops = B * mmas_per_string * ops_per_mma
        effective_ops = B * L * ops_per_mma
        tflops_actual = total_ops / (kern_ms * 1e-3) / 1e12
        tflops_effective = effective_ops / (kern_ms * 1e-3) / 1e12

        ai_effective = k * 16
        tc_util = ai_effective / 805 * 100

        results.append({
            'k': k, 'kern_ms': kern_ms,
            'tflops_actual': tflops_actual,
            'tflops_effective': tflops_effective,
            'ai_effective': ai_effective,
            'tc_util_roofline': tc_util,
            'tc_util_measured': tflops_actual / H200_INT8_TOPS * 100,
        })
        print(f"    k={k:>2}: {tflops_actual:.1f} actual TOPS, "
              f"AI_eff={ai_effective}, roofline util={tc_util:.1f}%, "
              f"measured util={tflops_actual / H200_INT8_TOPS * 100:.2f}%")

        eng.destroy()

    return results


# ── Figure generation ──────────────────────────────────────────────────────

def plot_k_sweep(k_results, path):
    """Fig 1: Throughput vs k value."""
    if not k_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ks = [r['k'] for r in k_results]
    kern = [r['kern_gcs'] for r in k_results]
    total = [r['total_gcs'] for r in k_results]

    ax.bar([k - 0.2 for k in ks], kern, width=0.35, label='Kernel only', color='#2196F3', zorder=3)
    ax.bar([k + 0.2 for k in ks], total, width=0.35, label='End-to-end', color='#FF9800', zorder=3)

    for i, (k, v) in enumerate(zip(ks, kern)):
        ax.text(k - 0.2, v + max(kern) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for i, (k, v) in enumerate(zip(ks, total)):
        ax.text(k + 0.2, v + max(kern) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('k (characters per WMMA MMA)', fontsize=12)
    ax.set_ylabel('Throughput (Gchar/s)', fontsize=12)
    ax.set_title(f'K-gram TC Throughput vs k\n'
                 f'(B={k_results[0]["B"]}, L={k_results[0]["L"]}, '
                 f'N={k_results[0]["N"]}, |Σ|={k_results[0]["sigma"]})',
                 fontsize=13)
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_batch_scaling(batch_results, path):
    """Fig 2: Throughput vs batch size, all backends."""
    if not batch_results:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {'monoid': '#4CAF50', 'batched': '#9C27B0', 'kgram': '#2196F3'}
    markers = {'monoid': 's', 'batched': '^', 'kgram': 'o'}
    labels = {'monoid': 'Monoid R1 (GPU)', 'batched': 'TC Batched', 'kgram': 'K-gram TC'}

    for name in ['monoid', 'kgram', 'batched']:
        data = batch_results[name]
        if not data:
            continue
        bs = [r['B'] for r in data]
        kern = [r['kern_gcs'] for r in data]
        ax.plot(bs, kern, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=7, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch size (B)', fontsize=12)
    ax.set_ylabel('Kernel throughput (Gchar/s)', fontsize=12)
    ax.set_title('GPU Backend Throughput vs Batch Size\n(L=512, pattern=(a|b)*abb)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_length_scaling(length_results, path):
    """Fig 3: Throughput vs string length, all backends."""
    if not length_results:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {'monoid': '#4CAF50', 'batched': '#9C27B0', 'kgram': '#2196F3'}
    markers = {'monoid': 's', 'batched': '^', 'kgram': 'o'}
    labels = {'monoid': 'Monoid R1 (GPU)', 'batched': 'TC Batched', 'kgram': 'K-gram TC'}

    for name in ['monoid', 'kgram', 'batched']:
        data = length_results[name]
        if not data:
            continue
        ls = [r['L'] for r in data]
        kern = [r['kern_gcs'] for r in data]
        ax.plot(ls, kern, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=7, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('String length (L)', fontsize=12)
    ax.set_ylabel('Kernel throughput (Gchar/s)', fontsize=12)
    ax.set_title('GPU Backend Throughput vs String Length\n(B=16384, pattern=(a|b)*abb)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tc_utilization(tc_results, path):
    """Fig 4: TC utilization (roofline vs measured) as a function of k."""
    if not tc_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ks = [r['k'] for r in tc_results]
    ai = [r['ai_effective'] for r in tc_results]
    roofline = [r['tc_util_roofline'] for r in tc_results]
    measured = [r['tc_util_measured'] for r in tc_results]
    tflops = [r['tflops_actual'] for r in tc_results]

    # Left: AI and TC utilization
    ax1.bar([k - 0.2 for k in ks], roofline, width=0.35, label='Roofline estimate',
            color='#90CAF9', edgecolor='#1565C0', zorder=3)
    ax1.bar([k + 0.2 for k in ks], measured, width=0.35, label='Measured',
            color='#A5D6A7', edgecolor='#2E7D32', zorder=3)

    for k, v in zip(ks, roofline):
        ax1.text(k - 0.2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for k, v in zip(ks, measured):
        ax1.text(k + 0.2, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)

    ax1.axhline(100, color='red', linestyle='--', alpha=0.5, label='H200 INT8 peak')
    ax1.set_xlabel('k (characters per MMA)', fontsize=11)
    ax1.set_ylabel('TC utilization (%)', fontsize=11)
    ax1.set_title('Tensor Core Utilization vs k', fontsize=12)
    ax1.set_xticks(ks)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Right: effective AI on roofline
    ai_range = np.linspace(1, 1000, 500)
    H200_BW = 4915  # GB/s
    H200_TOPS = 3958
    bw_bound = ai_range * H200_BW  # TOPS
    compute_bound = np.full_like(ai_range, H200_TOPS)
    achieved = np.minimum(bw_bound, compute_bound)

    ax2.loglog(ai_range, achieved, 'k-', linewidth=2, label='H200 roofline')
    ax2.fill_between(ai_range, achieved, alpha=0.05, color='gray')

    for r in tc_results:
        ax2.plot(r['ai_effective'], r['tflops_actual'], 'o', markersize=10,
                 label=f"k={r['k']} ({r['tflops_actual']:.1f} TOPS)", zorder=5)

    ax2.axvline(805, color='red', linestyle=':', alpha=0.4, label='Balance AI=805')
    ax2.set_xlabel('Arithmetic Intensity (ops/byte)', fontsize=11)
    ax2.set_ylabel('Performance (TOPS)', fontsize=11)
    ax2.set_title('INT8 Roofline (H200 SXM)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_kernel_breakdown(k_results, batch_results, path):
    """Fig 5: Kernel vs transfer overhead breakdown."""
    if not k_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: kernel vs overhead by k
    ks = [r['k'] for r in k_results]
    kern = [r['kern_ms'] for r in k_results]
    overhead = [r['total_ms'] - r['kern_ms'] for r in k_results]

    ax1.bar(ks, kern, width=0.6, label='Kernel', color='#2196F3', zorder=3)
    ax1.bar(ks, overhead, width=0.6, bottom=kern, label='H2D/D2H transfer', color='#FFCC80', zorder=3)

    for k, kv, ov in zip(ks, kern, overhead):
        pct = kv / (kv + ov) * 100 if (kv + ov) > 0 else 0
        ax1.text(k, kv + ov + 0.05, f'{pct:.0f}%\nkern', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('k', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title(f'Time Breakdown by k\n(B={k_results[0]["B"]}, L={k_results[0]["L"]})', fontsize=12)
    ax1.set_xticks(ks)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Right: kernel vs overhead by backend at B=16384
    if batch_results:
        names = []
        kern_vals = []
        overhead_vals = []
        for name in ['monoid', 'kgram', 'batched']:
            data = batch_results[name]
            row = next((r for r in data if r['B'] == 16384), None)
            if row:
                names.append({'monoid': 'Monoid R1', 'kgram': 'K-gram TC', 'batched': 'TC Batched'}[name])
                kern_vals.append(row['kern_ms'])
                overhead_vals.append(row['total_ms'] - row['kern_ms'])

        if names:
            x = np.arange(len(names))
            colors_k = ['#4CAF50', '#2196F3', '#9C27B0']
            ax2.bar(x, kern_vals, width=0.5, label='Kernel', color=colors_k[:len(names)], zorder=3)
            ax2.bar(x, overhead_vals, width=0.5, bottom=kern_vals, label='Transfer',
                    color='#FFCC80', zorder=3)

            for i, (kv, ov) in enumerate(zip(kern_vals, overhead_vals)):
                gcs = (16384 * 512) / (kv * 1e6) if kv > 0 else 0
                ax2.text(i, kv + ov + 0.05, f'{gcs:.1f}\nGc/s', ha='center', va='bottom', fontsize=9)

            ax2.set_xticks(x)
            ax2.set_xticklabels(names)
            ax2.set_ylabel('Time (ms)', fontsize=11)
            ax2.set_title('Backend Comparison\n(B=16384, L=512)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    pat = PATTERNS['abb']
    dfa = compile_regex(pat.regex)
    dm = DFAMatrices(dfa)
    alphabet = 'ab'

    from src.monoid import compute_monoid
    md = compute_monoid(dm)

    print(f"Pattern: {pat.regex}, N={dm.n_states}, |Sigma|={len(dm.alphabet)}")
    print(f"Monoid size: {md.size if md else 'N/A'}")
    print(f"auto_k_for_gpu: {auto_k_for_gpu(len(dm.alphabet), dm.n_states)}")
    print()

    backends = _load_backends(dm, md)
    print(f"Available backends: {list(backends.keys())}")
    print()

    all_data = {}

    print("=" * 70)
    print("P1: K-gram throughput vs k")
    print("=" * 70)
    k_results = bench_k_sweep(dm, backends, alphabet)
    all_data['k_sweep'] = k_results

    print()
    print("=" * 70)
    print("P2: Batch size scaling (all backends)")
    print("=" * 70)
    batch_results = bench_batch_scaling(dm, md, backends, alphabet)
    all_data['batch_scaling'] = batch_results

    print()
    print("=" * 70)
    print("P3: String length scaling (all backends)")
    print("=" * 70)
    length_results = bench_length_scaling(dm, md, backends, alphabet)
    all_data['length_scaling'] = length_results

    print()
    print("=" * 70)
    print("P4: TC utilization estimates")
    print("=" * 70)
    tc_results = bench_tc_utilization(dm, backends, alphabet)
    all_data['tc_utilization'] = tc_results

    # Save raw data
    data_path = os.path.join(RESULTS_DIR, 'kgram_benchmark_results.json')
    with open(data_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nRaw data saved to {data_path}")

    # Generate figures
    print("\nGenerating figures...")
    plot_k_sweep(k_results, os.path.join(RESULTS_DIR, 'kgram_throughput_vs_k.png'))
    plot_batch_scaling(batch_results, os.path.join(RESULTS_DIR, 'kgram_batch_scaling.png'))
    plot_length_scaling(length_results, os.path.join(RESULTS_DIR, 'kgram_length_scaling.png'))
    plot_tc_utilization(tc_results, os.path.join(RESULTS_DIR, 'kgram_tc_utilization.png'))
    plot_kernel_breakdown(k_results, batch_results, os.path.join(RESULTS_DIR, 'kgram_time_breakdown.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
