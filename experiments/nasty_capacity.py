#!/usr/bin/env python3
"""
Nasty Capacity Experiment
==========================
Tests the thesis: "Nastier (higher-D) embedding → more information survives
projection to low-D Penrose tiling."

A cut-and-project scheme maps a high-D lattice onto a low-D "physical" space
(keel) while preserving a perpendicular-space residue. If the embedding
dimension is higher, each perpendicular-space dimension carries more
structured information, so reconstruction from partial residue should improve.

Also tests: golden-ratio projection angle vs random irrational angles.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────
EMBED_DIMS = [10, 20, 50, 100, 200, 500]
KEEL_DIM = 5           # physical-space (Penrose-tiling) dimension
N_MEMORIES = 100       # random memories per trial
N_TRIALS = 3           # repetitions for averaging
RESIDUE_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]
PHI = (1 + np.sqrt(5)) / 2  # golden ratio ≈ 1.618


def make_projection_matrix(embed_dim, keel_dim, angle_basis="golden"):
    """
    Build a (embed_dim, keel_dim) projection matrix.
    `angle_basis` controls the irrational rotation angles.
    """
    if angle_basis == "golden":
        # Use powers of golden ratio — classic Penrose/Quasicrystal choice
        angles = np.array([PHI ** k for k in range(keel_dim * (embed_dim // keel_dim + 1))])
        angles = angles[:embed_dim]
    elif angle_basis == "random_irrational":
        # Random irrational-like angles (sqrt of distinct primes)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                  59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                  191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
                  257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                  331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                  401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                  467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557,
                  563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
                  631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
                  709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
                  797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
                  877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953,
                  967, 971, 977, 983, 991, 997][:embed_dim]
        angles = np.sqrt(primes, dtype=float)
    else:
        raise ValueError(f"Unknown angle_basis: {angle_basis}")

    # Build a tall matrix (embed_dim × embed_dim) with angle-modulated entries
    # Use enough columns to span both physical and perpendicular space
    n_cols = embed_dim  # full rank for clean QR decomposition
    G = np.zeros((embed_dim, n_cols))
    for i in range(embed_dim):
        for j in range(n_cols):
            phase = angles[i % len(angles)] * (j + 1)
            G[i, j] = np.sin(phase) + np.cos(phase * PHI)
    
    # Take first keel_dim columns of Q as projection to physical space
    # Remaining columns span perpendicular space
    Q, R = np.linalg.qr(G)
    P_phys = Q[:, :keel_dim]       # embed_dim × keel_dim
    P_perp = Q[:, keel_dim:]       # embed_dim × (embed_dim - keel_dim)
    
    return P_phys, P_perp


def cut_and_project(memory, P_phys, P_perp):
    """
    Cut-and-project: decompose memory into physical + perpendicular components.
    Returns (physical_coords, perpendicular_residue).
    """
    physical = P_phys.T @ memory       # keel_dim
    perp = P_perp.T @ memory           # (embed_dim - keel_dim)
    return physical, perp


def reconstruct(physical, perp_full, P_phys, P_perp, residue_fraction):
    """
    Reconstruct memory from physical coords + partial perpendicular residue.
    residue_fraction controls how much perpendicular info survives.
    """
    n_perp = len(perp_full)
    n_keep = max(1, int(n_perp * residue_fraction))
    
    # Keep top components (by magnitude — most informative)
    # Use deterministic: just truncate (simulates bandwidth limit)
    perp_partial = np.zeros(n_perp)
    perp_partial[:n_keep] = perp_full[:n_keep]
    
    reconstructed = P_phys @ physical + P_perp @ perp_partial
    return reconstructed


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_experiment(angle_basis="golden"):
    """Run full experiment, return results dict."""
    results = {}
    
    for edim in EMBED_DIMS:
        if edim <= KEEL_DIM:
            continue
        
        residue_scores = {f: [] for f in RESIDUE_FRACTIONS}
        
        for trial in range(N_TRIALS):
            rng = np.random.RandomState(42 + trial)
            P_phys, P_perp = make_projection_matrix(edim, KEEL_DIM, angle_basis)
            memories = rng.randn(N_MEMORIES, edim)
            
            for mem in memories:
                phys, perp = cut_and_project(mem, P_phys, P_perp)
                
                for frac in RESIDUE_FRACTIONS:
                    recon = reconstruct(phys, perp, P_phys, P_perp, frac)
                    sim = cosine_sim(mem, recon)
                    residue_scores[frac].append(sim)
        
        results[edim] = {
            frac: (np.mean(scores), np.std(scores))
            for frac, scores in residue_scores.items()
        }
    
    return results


def print_table(results, label=""):
    """Print the summary table."""
    header = f"{'Embed Dim':>10} |"
    for frac in RESIDUE_FRACTIONS:
        header += f" Residue {frac*100:5.0f}% |"
    print(header)
    print("-" * len(header))
    
    for edim in EMBED_DIMS:
        if edim not in results:
            continue
        row = f"{edim:>10} |"
        for frac in RESIDUE_FRACTIONS:
            mean, std = results[edim][frac]
            row += f" {mean:12.4f} |"
        print(row)


def print_comparison(golden_results, random_results):
    """Compare golden vs random irrational."""
    print("\n" + "=" * 80)
    print("COMPARISON: Golden Ratio vs Random Irrational Projection")
    print("=" * 80)
    print(f"{'Dim':>6} | {'Golden mean':>12} | {'Random mean':>12} | {'Δ':>8} | {'Winner':>8}")
    print("-" * 65)
    
    for edim in EMBED_DIMS:
        if edim not in golden_results or edim not in random_results:
            continue
        # Average across all residue fractions for a single number
        g_mean = np.mean([golden_results[edim][f][0] for f in RESIDUE_FRACTIONS])
        r_mean = np.mean([random_results[edim][f][0] for f in RESIDUE_FRACTIONS])
        delta = g_mean - r_mean
        winner = "GOLDEN" if delta > 0.001 else ("RANDOM" if delta < -0.001 else "TIE")
        print(f"{edim:>6} | {g_mean:>12.6f} | {r_mean:>12.6f} | {delta:>+8.4f} | {winner:>8}")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("NASTY CAPACITY EXPERIMENT")
    print("Thesis: Higher-D embedding → more info survives cut-and-project")
    print(f"Keel dim: {KEEL_DIM} | Memories/trial: {N_MEMORIES} | Trials: {N_TRIALS}")
    print("=" * 80)
    
    print("\n─── Golden Ratio Projection ───")
    golden_results = run_experiment("golden")
    print_table(golden_results, "Golden Ratio")
    
    print("\n─── Random Irrational Projection ───")
    random_results = run_experiment("random_irrational")
    print_table(random_results, "Random Irrational")
    
    print_comparison(golden_results, random_results)
    
    # ── Save results ───────────────────────────────────────────────────
    output = []
    output.append("NASTY CAPACITY EXPERIMENT — RESULTS")
    output.append(f"Keel dim: {KEEL_DIM} | Memories/trial: {N_MEMORIES} | Trials: {N_TRIALS}")
    output.append("")
    output.append("=== GOLDEN RATIO PROJECTION ===")
    
    header = f"{'Embed Dim':>10} |"
    for frac in RESIDUE_FRACTIONS:
        header += f" Residue {frac*100:5.0f}% |"
    output.append(header)
    output.append("-" * len(header))
    
    for edim in EMBED_DIMS:
        if edim not in golden_results:
            continue
        row = f"{edim:>10} |"
        for frac in RESIDUE_FRACTIONS:
            mean, std = golden_results[edim][frac]
            row += f" {mean:12.4f} |"
        output.append(row)
    
    output.append("")
    output.append("=== RANDOM IRRATIONAL PROJECTION ===")
    output.append(header)
    output.append("-" * len(header))
    
    for edim in EMBED_DIMS:
        if edim not in random_results:
            continue
        row = f"{edim:>10} |"
        for frac in RESIDUE_FRACTIONS:
            mean, std = random_results[edim][frac]
            row += f" {mean:12.4f} |"
        output.append(row)
    
    output.append("")
    output.append("=== COMPARISON ===")
    output.append(f"{'Dim':>6} | {'Golden mean':>12} | {'Random mean':>12} | {'Delta':>8} | {'Winner':>8}")
    output.append("-" * 65)
    
    for edim in EMBED_DIMS:
        if edim not in golden_results or edim not in random_results:
            continue
        g_mean = np.mean([golden_results[edim][f][0] for f in RESIDUE_FRACTIONS])
        r_mean = np.mean([random_results[edim][f][0] for f in RESIDUE_FRACTIONS])
        delta = g_mean - r_mean
        winner = "GOLDEN" if delta > 0.001 else ("RANDOM" if delta < -0.001 else "TIE")
        output.append(f"{edim:>6} | {g_mean:>12.6f} | {r_mean:>12.6f} | {delta:>+8.4f} | {winner:>8}")
    
    output.append("")
    output.append("=== THESIS CHECK ===")
    # Check if reconstruction improves with embedding dimension
    for frac in RESIDUE_FRACTIONS:
        sims_by_dim = [(edim, golden_results[edim][frac][0]) for edim in EMBED_DIMS if edim in golden_results]
        is_monotonic = all(sims_by_dim[i][1] <= sims_by_dim[i+1][1] for i in range(len(sims_by_dim)-1))
        output.append(f"Residue {frac*100:5.0f}%: monotonic increase with dim? {is_monotonic}")
        if not is_monotonic:
            output.append(f"  Values: {[(e, f'{s:.4f}') for e, s in sims_by_dim]}")
    
    result_text = "\n".join(output)
    
    out_path = Path(__file__).parent / "nasty_capacity_results.txt"
    out_path.write_text(result_text)
    print(f"\nResults saved to {out_path}")
    
    # Also print thesis check
    print("\n=== THESIS CHECK ===")
    for frac in RESIDUE_FRACTIONS:
        sims_by_dim = [(edim, golden_results[edim][frac][0]) for edim in EMBED_DIMS if edim in golden_results]
        is_monotonic = all(sims_by_dim[i][1] <= sims_by_dim[i+1][1] for i in range(len(sims_by_dim)-1))
        trend = "↑ IMPROVES" if sims_by_dim[-1][1] > sims_by_dim[0][1] else "↓ DEGRADES"
        print(f"Residue {frac*100:5.0f}%: {trend} | Monotonic: {is_monotonic}")
        print(f"  Low-D ({sims_by_dim[0][0]}): {sims_by_dim[0][1]:.4f} → High-D ({sims_by_dim[-1][0]}): {sims_by_dim[-1][1]:.4f}")
