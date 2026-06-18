"""Generate publication-quality figures from k-gram TC benchmark data.

Uses the cleanest data from each measurement approach:
- K-sweep: separate-process runs (no engine re-creation stalls)
- Batch scaling: multi-backend benchmark (consistent for kgram/batched),
  monoid reference from isolated runs at L=512
- TC utilization: from P4 (B=65536, fully warmed)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Clean data ──────────────────────────────────────────────────────────────

# K-sweep (separate process per k, B=16384, L=512, sigma=2, N=16)
K_SWEEP = [
    {'k': 1,  'kern_gcs': 6.24,  'kern_ms': 1.345, 'total_gcs': 5.43},
    {'k': 2,  'kern_gcs': 11.40, 'kern_ms': 0.736, 'total_gcs': 9.80},
    {'k': 4,  'kern_gcs': 20.23, 'kern_ms': 0.415, 'total_gcs': 16.52},
    {'k': 8,  'kern_gcs': 25.92, 'kern_ms': 0.324, 'total_gcs': 20.18},
    {'k': 16, 'kern_gcs': 27.38, 'kern_ms': 0.306, 'total_gcs': 21.43},
]

# Batch scaling at L=512 (kgram k=17, batched, monoid)
# kgram and batched from multi-backend benchmark (consistent);
# monoid from isolated reference runs
BATCH_MONOID = [
    {'B': 256,   'kern_gcs': 5.81},
    {'B': 1024,  'kern_gcs': 21.82},
    {'B': 4096,  'kern_gcs': 60.82},
    {'B': 16384, 'kern_gcs': 83.45},
    {'B': 65536, 'kern_gcs': 66.46},
]
BATCH_KGRAM = [
    {'B': 256,   'kern_gcs': 0.87},
    {'B': 1024,  'kern_gcs': 3.52},
    {'B': 4096,  'kern_gcs': 13.26},
    {'B': 16384, 'kern_gcs': 19.28},
    {'B': 65536, 'kern_gcs': 19.84},
]
BATCH_BATCHED = [
    {'B': 256,   'kern_gcs': 0.21},
    {'B': 1024,  'kern_gcs': 0.79},
    {'B': 4096,  'kern_gcs': 3.41},
    {'B': 16384, 'kern_gcs': 13.08},
    {'B': 65536, 'kern_gcs': 27.21},
]

# Length scaling at B=16384 (from multi-backend benchmark P3, clean range)
LEN_KGRAM = [
    {'L': 32,   'kern_gcs': 8.68},
    {'L': 64,   'kern_gcs': 14.37},
    {'L': 128,  'kern_gcs': 20.35},
    {'L': 256,  'kern_gcs': 22.50},
    {'L': 512,  'kern_gcs': 19.38},
    {'L': 1024, 'kern_gcs': 18.53},
    {'L': 2048, 'kern_gcs': 18.65},
]
LEN_BATCHED = [
    {'L': 32,   'kern_gcs': 9.92},
    {'L': 64,   'kern_gcs': 11.19},
    {'L': 128,  'kern_gcs': 12.21},
    {'L': 256,  'kern_gcs': 12.81},
    {'L': 512,  'kern_gcs': 13.12},
    {'L': 1024, 'kern_gcs': 13.27},
    {'L': 2048, 'kern_gcs': 13.30},
]
LEN_MONOID = [
    {'L': 32,   'kern_gcs': 29.65},
    {'L': 64,   'kern_gcs': 42.69},
    {'L': 128,  'kern_gcs': 72.42},
    {'L': 256,  'kern_gcs': 78.67},
    {'L': 512,  'kern_gcs': 83.45},
    {'L': 1024, 'kern_gcs': 46.97},
    {'L': 2048, 'kern_gcs': 62.42},
]

# TC utilization (P4, B=65536, L=512)
TC_UTIL = [
    {'k': 1,  'tflops': 52.6, 'ai_eff': 16,  'roofline_pct': 2.0,  'measured_pct': 1.33},
    {'k': 2,  'tflops': 50.5, 'ai_eff': 32,  'roofline_pct': 4.0,  'measured_pct': 1.28},
    {'k': 4,  'tflops': 42.6, 'ai_eff': 64,  'roofline_pct': 8.0,  'measured_pct': 1.08},
    {'k': 8,  'tflops': 26.7, 'ai_eff': 128, 'roofline_pct': 15.9, 'measured_pct': 0.67},
    {'k': 16, 'tflops': 10.7, 'ai_eff': 256, 'roofline_pct': 31.8, 'measured_pct': 0.27},
]


# ── Figure 1: K-sweep ───────────────────────────────────────────────────────

def fig_k_sweep():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ks = [r['k'] for r in K_SWEEP]
    kern = [r['kern_gcs'] for r in K_SWEEP]

    ax.bar(range(len(ks)), kern, color='#2196F3', edgecolor='#1565C0',
           linewidth=0.8, zorder=3)

    for i, (k, v) in enumerate(zip(ks, kern)):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    speedups = [kern[i] / kern[0] for i in range(len(kern))]
    for i, s in enumerate(speedups):
        if i > 0:
            ax.text(i, kern[i] * 0.5, f'{s:.1f}x', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('k (characters per WMMA MMA)', fontsize=11)
    ax.set_ylabel('Kernel throughput (Gchar/s)', fontsize=11)
    ax.set_title('K-gram TC: Throughput vs k\n(B=16384, L=512, N=16, |$\\Sigma$|=2)', fontsize=12)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 32)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'kgram_throughput_vs_k.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: Batch scaling ─────────────────────────────────────────────────

def fig_batch_scaling():
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = [
        (BATCH_MONOID, 'Monoid R1 (GPU)', '#4CAF50', 's'),
        (BATCH_KGRAM, 'K-gram TC (k=17)', '#2196F3', 'o'),
        (BATCH_BATCHED, 'TC Batched', '#9C27B0', '^'),
    ]

    for data, label, color, marker in datasets:
        bs = [r['B'] for r in data]
        gcs = [r['kern_gcs'] for r in data]
        ax.plot(bs, gcs, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch size (B)', fontsize=11)
    ax.set_ylabel('Kernel throughput (Gchar/s)', fontsize=11)
    ax.set_title('GPU Backend Throughput vs Batch Size\n(L=512, pattern=(a|b)*abb, N=16)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(0)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'kgram_batch_scaling.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 3: Length scaling ────────────────────────────────────────────────

def fig_length_scaling():
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = [
        (LEN_MONOID, 'Monoid R1 (GPU)', '#4CAF50', 's'),
        (LEN_KGRAM, 'K-gram TC (k=17)', '#2196F3', 'o'),
        (LEN_BATCHED, 'TC Batched', '#9C27B0', '^'),
    ]

    for data, label, color, marker in datasets:
        ls = [r['L'] for r in data]
        gcs = [r['kern_gcs'] for r in data]
        ax.plot(ls, gcs, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('String length (L)', fontsize=11)
    ax.set_ylabel('Kernel throughput (Gchar/s)', fontsize=11)
    ax.set_title('GPU Backend Throughput vs String Length\n(B=16384, pattern=(a|b)*abb, N=16)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(0)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'kgram_length_scaling.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 4: TC utilization + roofline ─────────────────────────────────────

def fig_tc_utilization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ks = [r['k'] for r in TC_UTIL]
    roofline = [r['roofline_pct'] for r in TC_UTIL]
    measured = [r['measured_pct'] for r in TC_UTIL]

    x = np.arange(len(ks))
    w = 0.35

    ax1.bar(x - w/2, roofline, w, label='Roofline estimate',
            color='#90CAF9', edgecolor='#1565C0', zorder=3)
    ax1.bar(x + w/2, measured, w, label='Measured',
            color='#A5D6A7', edgecolor='#2E7D32', zorder=3)

    for i, (r, m) in enumerate(zip(roofline, measured)):
        ax1.text(i - w/2, r + 0.3, f'{r:.1f}%', ha='center', va='bottom', fontsize=8)
        ax1.text(i + w/2, m + 0.3, f'{m:.2f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel('k (characters per MMA)', fontsize=11)
    ax1.set_ylabel('TC utilization (%)', fontsize=11)
    ax1.set_title('Tensor Core Utilization vs k', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Roofline
    ai_range = np.logspace(0, 3.2, 500)
    H200_BW = 4915
    H200_TOPS = 3958
    bw_bound = ai_range * H200_BW / 1e3
    compute_bound = np.full_like(ai_range, H200_TOPS)
    achieved = np.minimum(bw_bound, compute_bound)

    ax2.loglog(ai_range, achieved, 'k-', linewidth=2, label='H200 roofline')
    ax2.fill_between(ai_range, achieved, alpha=0.05, color='gray')

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']
    for i, r in enumerate(TC_UTIL):
        ax2.plot(r['ai_eff'], r['tflops'], 'o', markersize=10, color=colors[i],
                 label=f"k={r['k']} ({r['tflops']:.0f} TOPS)", zorder=5)

    ax2.axvline(805, color='red', linestyle=':', alpha=0.4)
    ax2.text(805, 15, 'Balance\nAI=805', fontsize=8, ha='right', color='red', alpha=0.6)
    ax2.set_xlabel('Arithmetic Intensity (ops/byte)', fontsize=11)
    ax2.set_ylabel('Performance (TOPS)', fontsize=11)
    ax2.set_title('INT8 Roofline (H200 SXM)', fontsize=12)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3, which='both')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'kgram_tc_utilization.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 5: Time breakdown ───────────────────────────────────────────────

def fig_time_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: k-gram kernel time breakdown by k
    ks = [r['k'] for r in K_SWEEP]
    kern_ms = [r['kern_ms'] for r in K_SWEEP]
    ideal_ms = [K_SWEEP[0]['kern_ms'] / r['k'] for r in K_SWEEP]

    x = np.arange(len(ks))
    w = 0.35
    ax1.bar(x - w/2, kern_ms, w, label='Measured', color='#2196F3',
            edgecolor='#1565C0', zorder=3)
    ax1.bar(x + w/2, ideal_ms, w, label='Ideal (linear in 1/k)', color='#BBDEFB',
            edgecolor='#90CAF9', zorder=3)

    for i, (m, d) in enumerate(zip(kern_ms, ideal_ms)):
        eff = d / m * 100 if m > 0 else 0
        ax1.text(i, m + 0.02, f'{eff:.0f}%\neff', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel('k', fontsize=11)
    ax1.set_ylabel('Kernel time (ms)', fontsize=11)
    ax1.set_title('Measured vs Ideal Kernel Time\n(B=16384, L=512)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Right: all-backend comparison at B=16384, L=512
    names = ['Monoid R1', 'K-gram TC\n(k=17)', 'TC Batched']
    kern_vals = [83.45, 19.28, 13.08]
    colors_b = ['#4CAF50', '#2196F3', '#9C27B0']

    bars = ax2.bar(names, kern_vals, color=colors_b, edgecolor='k', linewidth=0.5, zorder=3)

    for bar, val in zip(bars, kern_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f'{val:.1f}\nGc/s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add speedup annotations
    ax2.annotate('', xy=(0, 83.45), xytext=(1, 83.45),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax2.text(0.5, 85, '4.3x', ha='center', fontsize=9, color='gray')

    ax2.set_ylabel('Kernel throughput (Gchar/s)', fontsize=11)
    ax2.set_title('Backend Comparison\n(B=16384, L=512, N=16)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'kgram_time_breakdown.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    print('Generating figures...')
    fig_k_sweep()
    fig_batch_scaling()
    fig_length_scaling()
    fig_tc_utilization()
    fig_time_breakdown()
    print('Done.')
