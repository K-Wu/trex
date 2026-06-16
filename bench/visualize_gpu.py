"""
GPU evaluation visualization: crossover analysis, roofline, and throughput charts.

Combines CUDA-internal kernel timings (from dfa_scan executable) with
Python end-to-end benchmarks to produce evaluation figures.
"""

import sys, os, json, re, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


OUTDIR = 'results'


def parse_cuda_benchmark():
    """Run the CUDA executable and parse the benchmark table."""
    result = subprocess.run(
        ['./build/dfa_scan'], capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr

    rows = []
    for line in output.split('\n'):
        m = re.match(
            r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            line
        )
        if m:
            rows.append({
                'length': int(m.group(1)),
                'bk_time_ms': float(m.group(2)),
                'bk_gbs': float(m.group(3)),
                'hs_time_ms': float(m.group(4)),
                'hs_gbs': float(m.group(5)),
            })
    return rows


def load_python_benchmark(path='results/gpu_benchmark_results.json'):
    with open(path) as f:
        return json.load(f)


def plot_kernel_throughput(cuda_data):
    """Plot GPU kernel throughput: Blelloch vs Hillis-Steele."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    lengths = [r['length'] for r in cuda_data]
    bk_gbs = [r['bk_gbs'] for r in cuda_data]
    hs_gbs = [r['hs_gbs'] for r in cuda_data]
    bk_ms = [r['bk_time_ms'] for r in cuda_data]
    hs_ms = [r['hs_time_ms'] for r in cuda_data]

    ax1.plot(lengths, bk_gbs, 'o-', color='#4CAF50', linewidth=2.5,
             markersize=7, label='Blelloch (work-efficient)')
    ax1.plot(lengths, hs_gbs, 's-', color='#2196F3', linewidth=2.5,
             markersize=7, label='Hillis-Steele')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('String Length (L)')
    ax1.set_ylabel('Throughput (GB/s)')
    ax1.set_title('GPU Kernel Throughput\n(CUDA events, persistent memory)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.0f}M'))

    speedup = [bk / hs if hs > 0 else 0 for bk, hs in zip(bk_gbs, hs_gbs)]
    ax2.plot(lengths, speedup, 'D-', color='#FF5722', linewidth=2.5, markersize=7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('String Length (L)')
    ax2.set_ylabel('Blelloch / Hillis-Steele Throughput')
    ax2.set_title('Blelloch Speedup over Hillis-Steele')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.0f}M'))

    crossover_len = None
    for i in range(1, len(speedup)):
        if speedup[i-1] < 1.0 and speedup[i] >= 1.0:
            crossover_len = lengths[i]
            break
    if crossover_len:
        ax2.axvline(x=crossover_len, color='#FF5722', linestyle=':', alpha=0.7)
        ax2.annotate(f'Crossover: L={crossover_len:,}',
                     xy=(crossover_len, 1.0), fontsize=9,
                     xytext=(crossover_len * 2, 1.5),
                     arrowprops=dict(arrowstyle='->', color='#FF5722'))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'gpu_kernel_throughput.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_crossover_analysis(cuda_data, py_data):
    """Crossover analysis: GPU kernel vs CPU sequential."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # GPU kernel throughput (Blelloch) in MB/s
    gpu_lengths = [r['length'] for r in cuda_data]
    gpu_mbs = [r['bk_gbs'] * 1000 for r in cuda_data]

    # CPU throughput from Python benchmark
    cpu_data = [(r['string_length'], r['throughput_mbs'])
                for r in py_data
                if r['suite'] == 'throughput_vs_length'
                and r['method'] == 'cpu_sequential']
    cpu_data.sort()
    cpu_lengths, cpu_mbs = zip(*cpu_data) if cpu_data else ([], [])

    ax1.plot(gpu_lengths, gpu_mbs, 'o-', color='#4CAF50', linewidth=2.5,
             markersize=7, label='GPU Tensor Core (kernel only)')
    if cpu_lengths:
        ax1.plot(cpu_lengths, cpu_mbs, 's-', color='#2196F3', linewidth=2.5,
                 markersize=7, label='CPU Sequential (Python)')

    # End-to-end GPU from Python
    gpu_e2e = [(r['string_length'], r['throughput_mbs'])
               for r in py_data
               if r['suite'] == 'throughput_vs_length'
               and r['method'] == 'gpu_tensor_core']
    if gpu_e2e:
        gpu_e2e.sort()
        e2e_len, e2e_mbs = zip(*gpu_e2e)
        ax1.plot(e2e_len, e2e_mbs, '^--', color='#FF9800', linewidth=2,
                 markersize=6, label='GPU End-to-End (Python ctypes)')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('String Length (L)')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.set_title('CPU vs GPU Throughput\n(crossover analysis)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.0f}M'))

    # Plot 2: GPU speedup over CPU
    if cpu_lengths:
        avg_cpu_mbs = np.mean(cpu_mbs)
        speedup = [g / avg_cpu_mbs for g in gpu_mbs]
        ax2.plot(gpu_lengths, speedup, 'o-', color='#4CAF50', linewidth=2.5,
                 markersize=7, label='GPU kernel / CPU seq')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Parity')

        crossover_idx = None
        for i in range(len(speedup)):
            if speedup[i] >= 1.0:
                crossover_idx = i
                break
        if crossover_idx is not None:
            cl = gpu_lengths[crossover_idx]
            ax2.axvline(x=cl, color='#4CAF50', linestyle=':', alpha=0.7)
            ax2.annotate(f'GPU wins at L={cl:,}',
                         xy=(cl, 1.0), fontsize=10, fontweight='bold',
                         xytext=(cl * 4, speedup[crossover_idx] * 1.5),
                         arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('String Length (L)')
    ax2.set_ylabel('Speedup (GPU kernel / CPU sequential)')
    ax2.set_title('GPU Speedup over CPU\n(kernel-only, excluding transfer)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.0f}M'))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'crossover_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_roofline(cuda_data):
    """Roofline model for tensor-core DFA scan on H200."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # H200 NVL specs
    peak_tops = 1979e12      # int8 tensor TOPS
    peak_bw = 4.8e12         # HBM3e bandwidth (bytes/s)
    TILE = 16
    TILE_ELEMS = TILE * TILE  # 256 bytes per matrix

    # Operational intensity: ops / bytes
    # Each Blelloch step: 1 MMA (N^3 ops for N×N matmul) per matrix
    # Bytes: read 2 matrices (2 * 256 = 512 B), write 1 (256 B) = 768 B
    # Plus log2(L) passes over L matrices
    # Arithmetic intensity = ops / bytes
    ops_per_mma = TILE * TILE * TILE  # 4096 int8 ops
    bytes_per_step = 3 * TILE_ELEMS   # read 2 + write 1

    # Plot roofline
    oi_range = np.logspace(-2, 4, 200)
    roofline = np.minimum(peak_tops, peak_bw * oi_range)
    ax.plot(oi_range, roofline / 1e12, '-', color='black', linewidth=2.5,
            label='H200 NVL Roofline')

    # Mark peak compute and bandwidth
    ridge_point = peak_tops / peak_bw
    ax.axvline(x=ridge_point, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'Ridge point\nOI={ridge_point:.1f}',
                xy=(ridge_point, peak_tops / 1e12), fontsize=8,
                xytext=(ridge_point * 5, peak_tops / 1e12 * 0.7),
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Plot measured points
    for r in cuda_data:
        L = r['length']
        if L < 256:
            continue
        # Measured throughput: GB/s * bytes → ops/s via arithmetic intensity
        bk_time_s = r['bk_time_ms'] / 1000
        log2L = int(np.log2(L)) if L > 0 else 1
        total_ops = 2 * L * ops_per_mma  # Blelloch: ~2L matmuls
        total_bytes = 2 * L * bytes_per_step  # rough
        measured_oi = total_ops / total_bytes
        measured_perf = total_ops / bk_time_s

        ax.plot(measured_oi, measured_perf / 1e12, 'o', color='#4CAF50',
                markersize=8, zorder=5)
        if L in (1024, 65536, 1048576, 16777216):
            label = f'L={L:,}' if L < 1e6 else f'L={L/1e6:.0f}M'
            ax.annotate(label, xy=(measured_oi, measured_perf / 1e12),
                        fontsize=7, xytext=(5, 5),
                        textcoords='offset points')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Operational Intensity (ops/byte)')
    ax.set_ylabel('Performance (TOPS)')
    ax.set_title('Roofline Analysis: Tensor-Core DFA Scan on H200 NVL\n'
                 '(Int8 WMMA, Blelloch prefix scan)')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(0.001, peak_tops / 1e12 * 2)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'roofline_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_scan_algorithm_comparison(cuda_data):
    """Time comparison: Blelloch O(L) work vs Hillis-Steele O(L log L) work."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = [r['length'] for r in cuda_data]
    bk_ms = [r['bk_time_ms'] for r in cuda_data]
    hs_ms = [r['hs_time_ms'] for r in cuda_data]

    ax.plot(lengths, bk_ms, 'o-', color='#4CAF50', linewidth=2.5,
            markersize=7, label='Blelloch (O(L) work, O(log L) depth)')
    ax.plot(lengths, hs_ms, 's-', color='#2196F3', linewidth=2.5,
            markersize=7, label='Hillis-Steele (O(L log L) work, O(log L) depth)')

    # Reference slopes
    L_arr = np.array(lengths, dtype=float)
    # O(L) reference from Blelloch at L=65536
    ref_idx = [i for i, l in enumerate(lengths) if l == 65536]
    if ref_idx:
        idx = ref_idx[0]
        scale_linear = bk_ms[idx] / lengths[idx]
        ax.plot(L_arr, L_arr * scale_linear, '--', color='gray', alpha=0.4,
                label='O(L) reference')
        scale_nlogn = hs_ms[idx] / (lengths[idx] * np.log2(lengths[idx]))
        ax.plot(L_arr[2:], L_arr[2:] * np.log2(L_arr[2:]) * scale_nlogn,
                '-.', color='gray', alpha=0.4, label='O(L log L) reference')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('String Length (L)')
    ax.set_ylabel('Kernel Time (ms)')
    ax.set_title('Scan Algorithm Comparison on H200 NVL\n'
                 '(16×16 Int8 transition matrices, CUDA events)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'{int(x):,}' if x < 1e6 else f'{x/1e6:.0f}M'))

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'scan_algorithm_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_batch_scaling(py_data):
    """Batch scaling: GPU vs CPU."""
    batch_data = [r for r in py_data if r['suite'] == 'batch_scaling']
    if not batch_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method, color, label in [
        ('cpu_sequential', '#2196F3', 'CPU Sequential'),
        ('gpu_tensor_core', '#4CAF50', 'GPU Tensor Core'),
    ]:
        data = [(r['string_length'] * int(r.get('batch_size', r['string_length'])),
                 r['elapsed_ms'], r['throughput_mbs'])
                for r in batch_data if r['method'] == method]
        # Extract batch size from the data differently
        method_data = [r for r in batch_data if r['method'] == method]
        bs = [r['elapsed_ms'] for r in method_data]
        tp = [r['throughput_mbs'] for r in method_data]

        ax1.plot(range(len(bs)), bs, 'o-', color=color, label=label, linewidth=2)
        ax2.plot(range(len(tp)), tp, 'o-', color=color, label=label, linewidth=2)

    batch_sizes = [1, 4, 16, 64, 256, 1024]
    ax1.set_xticks(range(len(batch_sizes)))
    ax1.set_xticklabels(batch_sizes)
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Time (ms)')
    ax1.set_title('Batch Processing Time\n(L=1024, (a|b)*abb)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels(batch_sizes)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.set_title('Batch Throughput')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'gpu_batch_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_summary(cuda_data, py_data):
    """Generate a text summary of the evaluation."""
    lines = []
    lines.append("=" * 70)
    lines.append("TERX Evaluation Summary: Tensor-Core DFA Scan on H200 NVL")
    lines.append("=" * 70)

    lines.append("\n## GPU Kernel Performance (CUDA events, persistent memory)")
    lines.append(f"{'Length':>12} {'Blelloch (ms)':>14} {'BK GB/s':>10} "
                 f"{'H-S (ms)':>14} {'HS GB/s':>10} {'BK/HS':>8}")
    lines.append("-" * 70)
    for r in cuda_data:
        speedup = r['bk_gbs'] / r['hs_gbs'] if r['hs_gbs'] > 0 else 0
        lines.append(f"{r['length']:>12,} {r['bk_time_ms']:>14.4f} "
                     f"{r['bk_gbs']:>10.3f} {r['hs_time_ms']:>14.4f} "
                     f"{r['hs_gbs']:>10.3f} {speedup:>8.2f}x")

    # Peak throughput
    peak_bk = max(r['bk_gbs'] for r in cuda_data)
    peak_hs = max(r['hs_gbs'] for r in cuda_data)
    lines.append(f"\nPeak Blelloch throughput: {peak_bk:.3f} GB/s "
                 f"({peak_bk * 1000:.0f} MB/s)")
    lines.append(f"Peak Hillis-Steele throughput: {peak_hs:.3f} GB/s")
    lines.append(f"Max Blelloch speedup over Hillis-Steele: "
                 f"{peak_bk / peak_hs:.1f}x")

    # Crossover analysis
    cpu_data = [(r['string_length'], r['throughput_mbs'])
                for r in py_data
                if r['suite'] == 'throughput_vs_length'
                and r['method'] == 'cpu_sequential']
    if cpu_data:
        avg_cpu = np.mean([t for _, t in cpu_data])
        lines.append(f"\n## Crossover Analysis")
        lines.append(f"CPU sequential throughput: ~{avg_cpu:.1f} MB/s")
        crossover_found = False
        for r in cuda_data:
            gpu_mbs = r['bk_gbs'] * 1000
            if gpu_mbs > avg_cpu and not crossover_found:
                lines.append(f"GPU kernel surpasses CPU at L={r['length']:,} "
                             f"({gpu_mbs:.0f} vs {avg_cpu:.0f} MB/s)")
                crossover_found = True

    # Roofline
    TILE = 16
    peak_tops = 1979e12
    peak_bw = 4.8e12
    ops_per_mma = TILE ** 3
    bytes_per_step = 3 * TILE * TILE

    lines.append(f"\n## Roofline Analysis")
    lines.append(f"H200 NVL peak int8 tensor: {peak_tops/1e12:.0f} TOPS")
    lines.append(f"H200 NVL peak HBM3e BW: {peak_bw/1e12:.1f} TB/s")
    lines.append(f"Ridge point OI: {peak_tops/peak_bw:.1f} ops/byte")
    lines.append(f"DFA scan OI: {ops_per_mma/bytes_per_step:.2f} ops/byte "
                 f"(memory-bound)")

    best_row = max(cuda_data, key=lambda r: r['bk_gbs'])
    L = best_row['length']
    bk_time_s = best_row['bk_time_ms'] / 1000
    total_ops = 2 * L * ops_per_mma
    achieved_tops = total_ops / bk_time_s
    achievable = peak_bw * (ops_per_mma / bytes_per_step)
    lines.append(f"Achieved performance: {achieved_tops/1e12:.4f} TOPS "
                 f"({achieved_tops/achievable*100:.1f}% of BW-limited roof)")

    lines.append(f"\n## Key Findings")
    lines.append(f"1. Blelloch scan achieves {peak_bk*1000:.0f} MB/s peak "
                 f"kernel throughput")
    lines.append(f"2. Work-efficient Blelloch is {peak_bk/peak_hs:.1f}x "
                 f"faster than Hillis-Steele at large L")
    lines.append(f"3. Kernel is memory-bound (OI={ops_per_mma/bytes_per_step:.2f} "
                 f"< ridge point {peak_tops/peak_bw:.0f})")
    lines.append(f"4. Python bridge overhead (~0.3ms/call) dominates for "
                 f"short strings")
    lines.append(f"5. Batch processing amortizes launch overhead for many "
                 f"short strings")

    text = '\n'.join(lines)
    path = os.path.join(OUTDIR, 'evaluation_summary.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n{text}")
    print(f"\n  Summary saved to {path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Parsing CUDA benchmark data...")
    cuda_data = parse_cuda_benchmark()
    if not cuda_data:
        print("ERROR: No benchmark data from CUDA executable")
        return

    print(f"  Got {len(cuda_data)} data points\n")

    py_data = []
    py_path = 'results/gpu_benchmark_results.json'
    if os.path.exists(py_path):
        py_data = load_python_benchmark(py_path)
        print(f"Loaded {len(py_data)} Python benchmark results\n")

    print("Generating evaluation charts...")
    plot_kernel_throughput(cuda_data)
    plot_scan_algorithm_comparison(cuda_data)
    plot_crossover_analysis(cuda_data, py_data)
    plot_roofline(cuda_data)
    if py_data:
        plot_batch_scaling(py_data)

    generate_summary(cuda_data, py_data)


if __name__ == '__main__':
    main()
