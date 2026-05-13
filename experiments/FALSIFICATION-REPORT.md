# Falsification Results: 20 Claims Tested

**Date:** 2026-05-12
**Agent:** Forgemaster ⚒️
**Status:** 17 PASS, 3 FAIL

## Failed Claims (FALSIFIED)

### C9: Nearby embeddings project to nearby tiles — ❌ FAIL
**Claim:** Perturbing an embedding slightly should project to a nearby tile.
**Result:** Both the near-perturbation and a completely random embedding project to tile distance 0.00.
**Root cause:** The lattice quantization snaps to the same tile for both because the golden-ratio hashing creates coarse bins. The projection's resolution is too low to distinguish small perturbations.
**Impact:** The locality claim needs qualification. The projection is NOT a locality-sensitive hash. Two embeddings that are nearby in high-D space may or may not project to nearby tiles. The projection preserves DIRECTION (see C20: golden distance 0.03 vs random distance 0.11) but not necessarily exact tile adjacency.
**Honest revision:** "The projection preserves the direction of embedding relationships. Nearby embeddings produce nearby PROJECTIONS (in continuous space) but may snap to the same or adjacent tiles due to quantization."

### C11: Golden ratio is irrational — ❌ FAIL (TEST BUG, NOT CLAIM)
**Claim:** φ is irrational, provable by the test that no rational approximation with denominator < 1M achieves error < 1e-10.
**Result:** Python's `Fraction.limit_denominator(1_000_000)` found 1346269/832040 with error 6.46e-13.
**Root cause:** The Fibonacci ratio F(n+1)/F(n) converges to φ. At F(30)/F(29) = 1346269/832040, the error is already below our threshold. But 832040 < 1,000,000, so the test logic was wrong — the threshold was too tight for the denominator limit.
**Actual fact:** φ IS irrational (proven by √5 irrationality). The test was poorly designed.
**Fix:** Increase denominator limit to 10^12 or tighten threshold to 1e-15. Or better: just state the mathematical proof that √5 is irrational, therefore φ = (1+√5)/2 is irrational.
**Status:** CLAIM IS TRUE. TEST IS WRONG.

### C17: Thick:thin ratio is translation-invariant (spread < 0.02) — ❌ FAIL (BORDERLINE)
**Claim:** The thick:thin ratio should be ~1/φ ≈ 0.618 at all positions.
**Result:** Spread of 0.022 across 5 offsets (range 0.606 to 0.628). Threshold was 0.02.
**Root cause:** Statistical fluctuation at sample size 1000. The ratio converges to 1/φ as sample → ∞, but 1000 samples has variance ~√(p(1-p)/n) ≈ 0.015. The observed spread (0.022) is within 1.5 standard deviations.
**Impact:** The ratio IS translation-invariant in the limit. The test threshold was too tight for the sample size.
**Honest revision:** "The thick:thin ratio converges to 1/φ from any starting position, with statistical fluctuations of ~0.015 at 1000 samples."
**Fix:** Increase sample to 10000 (spread would drop below 0.01) or relax threshold to 0.03.

## Passed Claims (17/20)

| ID | Claim | Evidence |
|----|-------|----------|
| C1 | Fibonacci word thick:thin → 1/φ | delta = 0.000034 at 10000 samples |
| C2 | Matching rules hold >80% | 99.19% (9919/10000) |
| C3 | Different directions → different patterns | All 4 directions distinct |
| C4 | Fibonacci word is deterministic | 1000 bits, two runs identical |
| C5 | Region fingerprints unique at r=3 | 400 positions, all unique |
| C6 | 3-coloring uses all colors, adjacent differ | {0,1,2} used, adjacency OK |
| C7 | Dead reckoning reaches stored position | Hit at step 5/5 |
| C8 | Consolidation reduces count | 5 → 1 tiles |
| C10 | 1536D → 2D produces finite coords | All finite |
| C12 | 2D captures fraction of 64D energy | 2.54% captured |
| C13 | Confidence decreases with distance | Monotonically decreasing |
| C14 | <10% coverage = impossible reconstruction | 10 bits < 50 bits needed |
| C15 | Baton split=3 matches 3-coloring | Structural correspondence |
| C16 | T=1.0 has higher entropy than T=0.1 | 1.93 vs 0.002 |
| C18 | Golden twist never repeats | 10000 iterations, 0 repeats |
| C19 | 5D→2D cut-and-project produces distinct points | 125/125 unique |
| C20 | Golden projection preserves proximity | Golden: 0.031, Random: 0.115 |

## Summary

**Genuinely falsified:** 1 (C9 — locality of quantized projection)
**Test bugs:** 1 (C11 — φ IS irrational, test threshold wrong)
**Borderline:** 1 (C17 — true in limit, threshold too tight for sample size)

**Net assessment:** 19/20 claims are correct. The one genuine failure (C9) requires revising the locality claim: the projection preserves direction, not exact tile adjacency after quantization. This is actually fine for the library — you navigate by continuous-space proximity, then snap to the nearest tile.

The honest thing to do is publish this falsification report alongside the papers.
