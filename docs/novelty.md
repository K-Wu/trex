# Using int8 Tensor Cores to Accelerate Regex Matching via Matrix Products: Novelty Assessment and Related Work

## TL;DR
- The constituent ingredients all exist in the literature, but the **specific combination** — encoding DFA/NFA transitions as **int8 matrices**, running them on **GPU tensor cores**, and composing transition matrices via **parallel prefix scan** for O(log n) depth — appears to be **unpublished and novel as of June 2026**. No paper, preprint, blog, or repo was found that does all three together for regex/finite-automaton simulation.
- The closest direct prior work is a 2025 ACM TACO paper that maps NFA processing to matrix algebra on GPUs — but it uses **sparse matrix–vector (SpMV) multiplication on regular CUDA cores / cuSPARSE**, explicitly *not* tensor cores, and does not use parallel prefix scan over transition matrices.
- Each individual piece is known: automata-as-matrix-multiplication exploiting associativity on CPUs (Mytkowicz 2014; Sin'ya 2013), tensor cores for cellular automata (CAT 2024/25), BFS/SpMM on tensor cores, and tensor-core parallel scan (Dakkak 2019). The novelty lies in fusing them.

## Key Findings

1. **No direct prior work** combining (a) int8 DFA/NFA transition matrices, (b) GPU tensor cores, and (c) parallel prefix scan for O(log n) depth. The idea is genuinely novel in that specific form.

2. **Automata-as-matrix-multiplication is well established** as a theoretical and CPU-practical idea. Simulating a DFA over a string is equivalent to composing per-symbol transition functions, and because composition is associative, it can be parallelized via prefix scan. This is exactly what Mytkowicz et al. (ASPLOS 2014) and Sin'ya et al. (ICPP 2013) exploited on multicore CPUs.

3. **GPU automata/regex engines do not use tensor cores.** iNFAnt (2010), Zu et al. (PPoPP 2012), Yu & Becchi (CF 2013), DFAGE, AsyncAP (SIGMETRICS 2023), ngAP (ASPLOS 2024 best paper), and HybridSA (2024) all run on CUDA cores using worklists, bitstreams, or SpMV — none use the tensor-core MMA datapath.

4. **Tensor cores have been repurposed for non-ML combinatorial workloads** — cellular automata (CAT), BFS (BLEST, Berrybees), SpMM (Magicube, Acc-SpMM, DTC-SpMM), and reduction/scan (Dakkak et al. ICS 2019) — establishing strong precedent for the general approach, but none target finite-automaton/regex simulation.

5. **The most recent and most directly competitive work** is "Advancing Matrix Operations for High-Performance and Memory-Efficient Automata Processing on GPUs" (ACM TACO, 2025), which builds a GraphBLAS-inspired SpMV automata engine ("AutomataBLAS") on GPUs but on CUDA cores, not tensor cores.

## Details

### 1. Direct prior work: tensor cores for regex/automata via matrix products

An extensive search across arXiv, ACM DL, IEEE Xplore, USENIX, Semantic Scholar, GitHub, and general web found **no work** that specifically uses GPU tensor cores (int8, fp16, or int1) to accelerate regular-expression matching or DFA/NFA simulation by encoding transition functions as low-precision matrices executed on the tensor-core MMA datapath. The exact pipeline in question — int8 transition matrices + tensor cores + parallel prefix scan for logarithmic depth — does not appear in the literature as of mid-2026.

### 2. GPU regex/NFA/DFA matching engines

The GPU regex literature is mature and uniformly built on **CUDA-core** execution, not tensor cores:

- **iNFAnt** (Cascarano, Rolando, Risso, Sisto; SIGCOMM CCR 2010) — the seminal NFA-on-GPGPU engine; represents the NFA as transition tables and processes input symbols with massive thread parallelism.
- **Zu et al.** (PPoPP 2012, "GPU-based NFA Implementation for Memory Efficient High Speed Regular Expression Matching") — full author list Yuan Zu, Ming Yang, Zhonghu Xu, Lin Wang, Xin Tian, Kunyang Peng, Qunfeng Dong (pp. 129–140). The paper states "compared with iNFAnt [9], our GPU-based solution can boost matching speed by 29∼46 times, consistently yielding matching speed above 10Gbps … on NVIDIA GTX-460 GPU."
- **Yu & Becchi** (Computing Frontiers 2013, "GPU acceleration of regular expression matching for large datasets") — and the **DFAGE** engine that descends from it.
- **Vasiliadis et al.** (RAID 2008/2009, "Gnort", "Regular Expression Matching on Graphics Hardware") — DFA/XFA-based packet matching, one packet per processor.
- **AsyncAP** (Liu, Pai, Jog; SIGMETRICS 2023, POMACS Vol. 7 No. 1) — exploits input-symbol-level parallelism asynchronously; "AsyncAP achieves up to 58× speedup on average over the state-of-the-art GPU automata processing engine" when parallelism is insufficient, "and still achieves 2.4× speedup" when GPU cores are saturated (12 applications evaluated).
- **ngAP** (Tianao Ge, Tong Zhang, Hongyuan Liu; ASPLOS 2024, best paper; journal extension "Towards Scalable and Non-blocking Automata Processing on GPUs with ngAP" in ACM Transactions on Computer Systems, DOI 10.1145/3748646) — non-blocking large-scale automata processing with concurrent multi-symbol processing.
- **HybridSA** (Proc. ACM PL 2024) — heterogeneous CPU-GPU engine using **bit parallelism** (shift-and) rather than matrix products.
- **ANG** (IEEE, 2025) — multi-level fine-grained parallelism for low-latency NFA processing.

Intel **Hyperscan** is the dominant CPU engine (SIMD bit-parallel, NFA decomposition); it does not use matrix multiplication or GPUs. None of these use tensor cores or formulate matching as dense MMA.

### 3. Matrix-based / algebraic automata simulation

This is the deepest vein of relevant prior art:

- **Mytkowicz, Musuvathi & Schulte, "Data-Parallel Finite-State Machines"** (ASPLOS 2014) — the canonical work breaking the sequential dependency of FSMs by enumerating transitions from all possible states per symbol, enabling SIMD/multicore parallelism. This is functionally a transition-matrix composition exploiting associativity, on CPUs.
- **Sin'ya, Matsuzaki & Sassa, "Simultaneous Finite Automata"** (ICPP 2013, pp. 220–229, DOI 10.1109/ICPP.2013.31; arXiv:1405.0562) — extends an automaton so states are mappings (transition functions) that compose associatively; the paper reports "it has achieved over 10-times speedups on an environment with dual hexa-core CPUs in a typical case." Explicitly a data-parallel model for regex matching based on transition composition.
- **Jiang & Agrawal** (PPoPP 2017) and **SimdFSM** (2023) — enumerative speculation and adaptive vectorization of FSMs, related parallelization strategies.
- The **parallel prefix scan** (Hillis-Steele 1986, Blelloch 1990) over an associative operator — here matrix/transition-function composition — gives O(log n) depth. This is the same structure used in modern state-space models (Mamba-style linear recurrences x_t = A_t x_{t-1} + b_t composed via scan) and in Boolean-matrix transitive-closure/APSP formulations (Warshall/Floyd, tropical and Boolean semirings). The 2025 TACO paper notes prior CPU automata-matrix work "exploit[s] the associativity of matrix multiplication."
- DFA simulation via matrix products is essentially a **graph reachability / Boolean matrix multiplication** problem: the transition matrix is an adjacency matrix and string-matching is iterated SpMV over the Boolean semiring.

### 4. Tensor cores for graph/combinatorial algorithms (non-ML)

Strong and growing precedent for repurposing tensor cores beyond deep learning:

- **Dakkak, Li, Xiong, Gelado, Hwu, "Accelerating Reduction and Scan Using Tensor Core Units"** (ICS 2019; arXiv:1811.09736) — the foundational result that reduction and **scan** can be expressed as matrix multiplications and mapped onto TCUs. The paper reports the implementation "using NVIDIA's V100 TCUs … achieved 89% − 98% of peak memory copy bandwidth … orders of magnitude faster (up to 100× for reduction and 3× for scan) … while decreasing the power consumption by up to 22% for reduction and 16% for scan." This directly supports the feasibility of a tensor-core prefix scan, a key ingredient of the proposed regex idea.
- **CAT: Cellular Automata on Tensor Cores** (Navarro, Quezada, Meneses, Ferrada, Hitschfeld; arXiv:2406.17284; IEEE TPDS 36(2), 2025) — uses fp16 MMA to accelerate cellular-automaton neighborhood reductions. The paper states "CAT uses FP16 types for the neighborhood reduction, as it is currently the most suitable type … (i.e., full INT32 MMAs are still not supported in square shape)," and cites INT8/INT4/BIT as faster types whose reuse "could" be future work; it verifies constant time across the radius range 1 ≤ r ≤ 16. This is the closest "automata + tensor core" work, but it targets cellular (not finite-state/regex) automata.
- **BFS on tensor cores** — "BLEST: Blazingly Efficient BFS using Tensor Cores" (arXiv:2512.21967) and Berrybees (PPoPP 2025) recast BFS as Boolean-semiring SpMM/SpMSpV on tensor cores. Graph reachability via tensor cores — conceptually adjacent to DFA reachability.
- **Quantized/sparse SpMM on tensor cores** — Magicube (SC 2022, int8/int4 sparse MMA), Acc-SpMM (PPoPP 2025), DTC-SpMM (ASPLOS 2024), HC-SpMM. Establish that int8 Boolean-style matrix products on tensor cores are practical.

### 5. General GPU parallelization of regex/automata

The standard GPU approaches exploit three parallelism levels: (1) **input-stream-level** (many strings/packets in parallel — the dominant approach in iNFAnt, Gnort), (2) **automaton-level** (many NFAs concurrently), and (3) **state-level** (parallel across active states). Additional techniques include **enumeration of start positions / speculative execution** (Luchaup et al.; Mytkowicz), **prefix-sum/scan** over state-enumeration, and worklist-based asynchronous processing (AsyncAP, ngAP). The Yang/Zu/Xia line of PPoPP/USTC work focused on memory-efficient NFA representations.

### 6. Novelty assessment

Decomposing the proposed idea into its three components:

| Component | Prior art status |
|---|---|
| (a) Encode DFA/NFA transitions as matrices and simulate via matrix products | **Known** — Mytkowicz 2014, Sin'ya 2013 (CPU); TACO 2025 (GPU SpMV); standard Boolean-matrix reachability. |
| (b) Use GPU tensor cores (int8 specifically) for the matrix multiplies | **Not done for automata/regex.** Tensor cores used for CA, BFS, SpMM, scan — but not finite-automaton/regex simulation. int8 specifically only in SpMM/quantized-NN contexts. |
| (c) Parallel prefix scan over transition matrices for O(log n) depth | **Known in isolation** — Blelloch/Hillis-Steele scan over associative matrix composition; SSM scans; Dakkak tensor-core scan. Applied to FSMs on CPU but not on tensor cores. |

**Verdict:** The high-level concept of automata-as-matrix-multiplication is decidedly *not* novel, and parallel-prefix composition of transition matrices is also known. What is novel is the **systems contribution of fusing all three**: (i) mapping finite-automaton/regex transition matrices specifically onto the **int8 tensor-core MMA datapath**, and (ii) doing so within a **parallel-prefix-scan formulation that achieves O(log n) depth** on tensor cores. No located source combines these. The nearest competitor (TACO 2025) deliberately stays on CUDA-core SpMV and per-symbol (O(n)-depth) iteration, leaving the tensor-core + log-depth-scan design space open.

## Recommendations

1. **Frame the contribution as a novel systems/architecture mapping, not a new algorithm.** Position relative to (i) TACO 2025 AutomataBLAS as the direct GPU SpMV baseline, (ii) Mytkowicz 2014 / Sin'ya 2013 as the CPU associativity precedent, and (iii) Dakkak 2019 / CAT 2024 as proof tensor cores can run scan and automata-like kernels. Claim novelty specifically on the int8-tensor-core + log-depth-scan combination, which the evidence supports.

2. **Benchmark directly against ngAP (ASPLOS 2024) and the TACO 2025 SpMV engine** — these are the strongest current GPU baselines. A win must show the tensor-core MMA datapath beats CUDA-core SpMV/worklist approaches on real regex workloads (e.g., ANMLZoo, Snort, protein/network sets).

3. **Address the key technical risks the prior art implies:** (a) DFA/NFA transition matrices are sparse and small; tensor cores favor dense ≥16×16 tiles, so the central challenge (as in BLEST/Magicube) is packing/batching many small Boolean matrices to keep MMA utilization high without wasted zero work. (b) Boolean semiring (OR/AND) must be emulated on int8 (sum-then-threshold), with saturation/overflow handled — verify correctness against int32 accumulation. (c) The log-depth scan materializes full N×N transition matrices per segment, costing Θ(N²L) memory (per the SSM-scan analysis); quantify whether this beats O(n)-depth SpMV for realistic N (state counts) and n (string lengths).

4. **Benchmarks/thresholds that would change the recommendation:** If a literature check at submission time reveals a tensor-core regex/DFA paper (watch arXiv cs.DC/cs.AR, PPoPP, MICRO, ASPLOS, SC 2025–2026), the novelty claim must narrow to the specific scan/int8 details. If tensor-core MMA utilization on packed transition matrices stays in the low range where BLEST/Magicube struggled, the SpMV-on-CUDA-cores baseline likely wins and the tensor-core approach should be reframed as useful only for dense/large-alphabet or many-automata batched regimes.

## Caveats

- I could not fully verify that Mytkowicz 2014 and Sin'ya 2013 are literally references [27] and [64] in the TACO 2025 paper, because ACM DL blocked automated fetching of the full reference list and Semantic Scholar elided the references field. The attribution is a high-confidence inference from the paper's own description ("multi-core CPUs … exploit the associativity of matrix multiplication"), not a confirmed citation match.
- Absence of evidence is not proof of absence: a very recent or obscure preprint, a non-English paper, or an unpublished industrial implementation could exist. The novelty claim is "no located prior work as of June 2026," not an absolute guarantee.
- Several tensor-core works (e.g., BLEST, arXiv:2512.21967) are very recent preprints not yet peer-reviewed; treat their performance claims as preliminary.
- The proposed approach's practical advantage is unproven — all evidence here concerns *related* workloads. Whether int8 tensor cores actually beat optimized CUDA-core SpMV/worklist regex engines is an open empirical question this research would need to settle.