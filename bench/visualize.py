"""
Generate benchmark visualization charts.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_results(path='results/benchmark_results.json'):
    with open(path) as f:
        return json.load(f)


def plot_throughput_vs_length(results, outdir='results'):
    """Plot throughput (MB/s) vs string length for each method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods_main = ['sequential', 'matrix_vec', 'batch_matrix']
    methods_scan = ['scan_sequential', 'scan_parallel']
    colors = {
        'sequential': '#2196F3',
        'matrix_vec': '#FF9800',
        'batch_matrix': '#4CAF50',
        'scan_sequential': '#9C27B0',
        'scan_parallel': '#F44336',
    }
    labels = {
        'sequential': 'Sequential (baseline)',
        'matrix_vec': 'Matrix-Vector',
        'batch_matrix': 'Batch Matrix',
        'scan_sequential': 'Prefix Scan (sequential)',
        'scan_parallel': 'Prefix Scan (parallel)',
    }

    length_results = [r for r in results if r['name'].startswith('length_')]

    # Plot 1: Elapsed time (ms) per string
    for method in methods_main + methods_scan:
        data = [(r['string_length'], r['per_string_us'])
                for r in length_results if r['method'] == method]
        if not data:
            continue
        data.sort()
        x, y = zip(*data)
        ax1.plot(x, y, 'o-', color=colors.get(method, 'gray'),
                 label=labels.get(method, method), linewidth=2, markersize=5)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('String Length')
    ax1.set_ylabel('Time per String (µs)')
    ax1.set_title('Latency vs String Length\n(pattern: (a|b)*abb)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Plot 2: Throughput (MB/s)
    for method in methods_main:
        data = [(r['string_length'], r['throughput_mbs'])
                for r in length_results if r['method'] == method]
        if not data:
            continue
        data.sort()
        x, y = zip(*data)
        ax2.plot(x, y, 'o-', color=colors.get(method, 'gray'),
                 label=labels.get(method, method), linewidth=2, markersize=5)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('String Length')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.set_title('Throughput vs String Length')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    path = os.path.join(outdir, 'throughput_vs_length.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_prefix_scan_scaling(results, outdir='results'):
    """Plot prefix scan time vs sequence length — verifying O(log n) depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    scan_results_seq = [(r['string_length'], r['elapsed_ms'])
                        for r in results if r['method'] == 'scan_seq_raw']
    scan_results_par = [(r['string_length'], r['elapsed_ms'])
                        for r in results if r['method'] == 'scan_par_raw']

    if not scan_results_seq:
        return

    scan_results_seq.sort()
    scan_results_par.sort()

    x_s, y_s = zip(*scan_results_seq)
    x_p, y_p = zip(*scan_results_par)

    ax1.plot(x_s, y_s, 'o-', color='#9C27B0', label='Sequential Scan', linewidth=2)
    ax1.plot(x_p, y_p, 's-', color='#F44336', label='Parallel Scan (Blelloch)', linewidth=2)

    # Reference lines: O(n) and O(n log n)
    x_ref = np.array(x_s)
    y_ref_n = y_s[0] * (x_ref / x_ref[0])
    y_ref_nlogn = y_s[0] * (x_ref / x_ref[0]) * np.log2(x_ref / x_ref[0] + 1)
    ax1.plot(x_ref, y_ref_n, '--', color='gray', alpha=0.5, label='O(n) reference')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length (L)')
    ax1.set_ylabel('Elapsed Time (ms)')
    ax1.set_title('Prefix Scan Scaling: Time vs Length\n(16×16 random permutation matrices)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup of parallel over sequential
    speedups = [s / p for s, p in zip(y_s, y_p)]
    ax2.plot(x_s, speedups, 'o-', color='#4CAF50', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length (L)')
    ax2.set_ylabel('Speedup (Sequential / Parallel)')
    ax2.set_title('Parallel Scan Speedup\n(CPU numpy — no true parallelism)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'prefix_scan_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_dfa_size_scaling(results, outdir='results'):
    """Plot performance vs DFA state count."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    dfa_results = [r for r in results if r['name'].startswith('dfa_scale_')]
    methods = ['sequential', 'matrix_vec', 'scan_parallel']
    colors = {'sequential': '#2196F3', 'matrix_vec': '#FF9800', 'scan_parallel': '#F44336'}
    labels = {'sequential': 'Sequential', 'matrix_vec': 'Matrix-Vector', 'scan_parallel': 'Prefix Scan (par)'}

    for method in methods:
        data = [(r['dfa_states'], r['per_string_us'], r['pattern'])
                for r in dfa_results if r['method'] == method]
        if not data:
            continue
        data.sort()
        x, y, pats = zip(*data)
        ax.plot(x, y, 'o-', color=colors.get(method, 'gray'),
                label=labels.get(method, method), linewidth=2, markersize=6)
        # Label points with pattern names
        if method == 'sequential':
            for xi, yi, p in zip(x, y, pats):
                ax.annotate(p, (xi, yi), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel('DFA States (raw, before padding)')
    ax.set_ylabel('Time per String (µs)')
    ax.set_title('Latency vs DFA Size (string length = 2048)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'dfa_size_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_batch_scaling(results, outdir='results'):
    """Plot batch size vs throughput."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    batch_results = [r for r in results if r['name'].startswith('batch_')]

    for method, color, label in [
        ('sequential', '#2196F3', 'Sequential'),
        ('batch_matrix', '#4CAF50', 'Batch Matrix'),
    ]:
        data = [(r['n_strings'], r['elapsed_ms'], r['throughput_mbs'])
                for r in batch_results if r['method'] == method]
        if not data:
            continue
        data.sort()
        x, t, tp = zip(*data)
        ax1.plot(x, t, 'o-', color=color, label=label, linewidth=2)
        ax2.plot(x, tp, 'o-', color=color, label=label, linewidth=2)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Time (ms)')
    ax1.set_title('Total Time vs Batch Size\n(length=512, (a|b)*abb)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.set_title('Throughput vs Batch Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'batch_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_projected_gpu(outdir='results'):
    """Plot projected GPU tensor-core performance vs CPU."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # A100 int8 tensor core projections
    # 624 TOPS, each 16×16×16 MMA = 8192 ops
    mma_per_sec = 624e12 / 8192
    # Memory bandwidth: 2 TB/s (A100 HBM2e)
    mem_bw = 2e12  # bytes/s

    lengths = [2**i for i in range(6, 25)]
    N = 16

    depth_limited = []     # time based on O(log L) depth of matmuls
    work_limited = []      # time based on total MMA work
    memory_limited = []    # time based on loading transition matrices
    cpu_sequential = []    # estimated CPU sequential time

    for L in lengths:
        log2L = int(np.log2(L))
        # Depth-limited: log2(L) steps, each step is one MMA
        t_depth = log2L / mma_per_sec
        depth_limited.append(t_depth * 1e6)  # µs

        # Work-limited: ~2L matmuls total
        t_work = (2 * L) / mma_per_sec
        work_limited.append(t_work * 1e6)

        # Memory-limited: load L matrices of 256 bytes + store L matrices
        bytes_transferred = L * N * N * 2  # read + write
        t_mem = bytes_transferred / mem_bw
        memory_limited.append(t_mem * 1e6)

        # CPU: ~L * N * N ops at ~10 Gops/s (optimistic single-core)
        t_cpu = L * N * N / 10e9
        cpu_sequential.append(t_cpu * 1e6)

    ax.plot(lengths, depth_limited, '-', color='#F44336', linewidth=2,
            label='GPU depth-limited (O(log L) MMA)')
    ax.plot(lengths, work_limited, '--', color='#FF9800', linewidth=1.5,
            label='GPU work-limited (O(L) MMA)')
    ax.plot(lengths, memory_limited, '-.', color='#9C27B0', linewidth=1.5,
            label='GPU memory-limited (HBM bandwidth)')
    ax.plot(lengths, cpu_sequential, '-', color='#2196F3', linewidth=2,
            label='CPU sequential (estimated)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('String Length (L)')
    ax.set_ylabel('Projected Time (µs)')
    ax.set_title('Projected Performance: A100 Int8 Tensor Cores vs CPU\n'
                 '(16-state DFA, single string, 624 TOPS / 2 TB/s HBM)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'2^{int(np.log2(x))}'))

    # Annotate crossover
    ax.annotate('Practical bottleneck\nis memory bandwidth',
                xy=(2**16, memory_limited[10]), fontsize=8, alpha=0.7,
                xytext=(2**18, memory_limited[10] * 5),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(outdir, 'projected_gpu_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    results = load_results()
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)

    print("Generating charts...")
    plot_throughput_vs_length(results, outdir)
    plot_prefix_scan_scaling(results, outdir)
    plot_dfa_size_scaling(results, outdir)
    plot_batch_scaling(results, outdir)
    plot_projected_gpu(outdir)
    print("Done.")


if __name__ == '__main__':
    main()
