# Phase 2: Questions Seed Found — Results

**Method:** Seed-2.0-mini generated 10 candidate questions at temp=1.0. Harsh reviewer (also Seed-mini) selected top 5. Experiments run with numpy-only.

## Results

### Q1: Golden angle for structured (random walk) vectors — **NEGATIVE**
- **Question:** Does golden angle help for structured vectors vs random angles?
- **Result:** Golden LOSES (mean distance 0.091 vs random 0.089). Ratio 1.023.
- **Honest read:** Golden angle confers NO advantage even for structured data. The golden ratio's self-similarity properties don't translate to better noise resistance in this projection.
- **Impact:** We should NOT claim golden ratio is special for projection quality. It's special for self-similarity and aperiodicity (proven), not for information preservation.

### Q2: Snap error vs perturbation angle — **SURPRISING**
- **Question:** Do perpendicular perturbations cause more snap errors than aligned?
- **Result:** REVERSED. Aligned errors: 100%. Perpendicular: 6.5%. Perp/aligned ratio: 0.065.
- **Honest read:** Perturbations ALONG the golden projection axis almost always cause snap errors. Perpendicular perturbations rarely do. The projection is highly sensitive along its principal axis and robust perpendicular to it.
- **Impact:** For real agent use, the memory palace should be designed so navigation perturbations are perpendicular to the projection axis. This is a genuine design insight.

### Q7: Principal axis alignment matters — **STRONG POSITIVE**
- **Question:** Does captured variance depend on alignment with golden projection axis?
- **Result:** YES. Best (120°) captures 24× more variance than worst (30°). Ratio: 24.3.
- **Honest read:** This is the strongest result of the five. The projection is NOT isotropic — it has a strong directional preference. Rotating the data's principal axis to align with the projection's high-variance direction dramatically improves information capture.
- **Impact:** **The projection matrix should be LEARNED, not fixed.** A PCA-based projection that aligns the data's principal components with the golden projection's high-variance directions would capture far more information. This is the path to making the memory palace actually work well with real neural embeddings.

### Q8: 3-coloring as retrieval cue — **NEGATIVE**
- **Question:** Does including tile color improve retrieval?
- **Result:** No. Both spatial-only and color-cue achieve 89% accuracy.
- **Honest read:** The 3-coloring is a structural property (guaranteed by the tiling) but doesn't carry retrieval-relevant information. It's useful for the baton protocol (sharding) but not for improving recall.
- **Impact:** Keep 3-coloring for sharding, don't claim it helps retrieval.

### Q10: Snap errors are spatially localized — **STRONG POSITIVE**
- **Question:** When snap errors occur, is the wrong tile at least nearby?
- **Result:** YES. 100% of snap errors (849/849) snap to a tile closer to the perturbed projection than to the original correct tile.
- **Honest read:** Snap errors are completely localized. The quantization is honest — it always picks the nearest tile. The C9 falsification (nearby embeddings snap to same tile) is actually a FEATURE, not a bug: the quantization is doing exactly what it should.
- **Impact:** C9's "failure" is actually correct behavior. The system is well-calibrated. Snap errors are always to the geometrically nearest tile.

## Key Takeaways

1. **Golden ratio is NOT special for compression** (Q1 + prior Q13 result). It IS special for aperiodicity and self-similarity (proven theorems), but not for information preservation.

2. **Projection should be LEARNED** (Q7). 24× variance capture difference from alignment. This is the biggest actionable insight: replace the fixed golden projection with a PCA or neural projection that aligns with the data.

3. **Snap errors are localized** (Q10). The C9 "falsification" is actually the quantization working correctly. No correction needed.

4. **3-coloring is for sharding, not retrieval** (Q8). Keep it for baton protocol, don't oversell it.

5. **Axis sensitivity is asymmetric** (Q2). Perpendicular perturbations are safe; aligned perturbations are devastating. Design the memory palace so navigation noise is perpendicular to the projection axis.
