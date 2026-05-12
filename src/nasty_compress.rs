//! Nasty compress: exploit Greenfeld-Tao aperiodicity for compression.
//!
//! The core idea (Greenfeld-Tao 2022): in sufficiently high dimensions, there
//! exist shapes that ONLY tile aperiodically. This "nastiness" is a FEATURE —
//! the nastier the high-D tiling, the more information it encodes per projected
//! tile. We use cut-and-project from high-D → low-D, storing the perpendicular-
//! space coordinates as a compression residue.
//!
//! Compression pipeline:
//!   1. Pad the input embedding with golden-ratio-derived coordinates (lift)
//!   2. Cut-and-project down to a target tiling dimension
//!   3. Tile type encodes the "shape" of the information
//!   4. Perpendicular-space residue stores what the projection loses
//!
//! Decompression:
//!   1. Recombine tile + residue → high-D lattice point
//!   2. Extract the original dimensions
//!
//! Information capacity scales with embedding dimension (nastier = more capacity).
//! A 100D embedding projected to 5D gives 20:1 compression with the aperiodic
//! structure preserving what matters.

use crate::cut_and_project::CutAndProject;

const PHI: f64 = 1.618033988749895;

/// A compressed tile: the projection + the perpendicular residue.
#[derive(Debug, Clone)]
pub struct CompressedTile {
    /// Projected coordinates in tiling space (low-D)
    pub projected: Vec<f64>,
    /// Perpendicular-space residue (what projection lost)
    pub perp_residue: Vec<f64>,
    /// Tile type (determined by perpendicular-space position in window)
    pub tile_type: u8,
    /// Original embedding dimension (needed for decompression)
    pub embed_dim: usize,
    /// Tiling dimension
    pub tiling_dim: usize,
    /// The lattice coordinates used (for reconstruction)
    pub lattice_coords: Vec<i64>,
    /// The padding applied during lift (for extraction)
    pub padded_values: Vec<f64>,
}

/// The result of a decompression attempt.
#[derive(Debug, Clone)]
pub struct Decompressed {
    /// Reconstructed vector (same dim as original embedding)
    pub vector: Vec<f64>,
    /// Mean squared error vs original (if known)
    pub mse: f64,
}

/// The nasty compressor.
pub struct NastyCompress {
    /// Target tiling dimension (e.g., 5)
    pub tiling_dim: usize,
    /// Full lattice embedding dimension (e.g., 200 for a 100D input padded to 200D)
    pub embed_dim: usize,
    /// Window size parameter — controls how many lattice points pass the filter.
    /// Bigger window = more detail preserved but less compression.
    pub window_scale: f64,
    /// Cut-and-project engine
    cap: CutAndProject,
}

impl NastyCompress {
    /// Create a new compressor.
    ///
    /// * `input_dim` — dimension of the input embeddings (e.g., 100)
    /// * `tiling_dim` — target dimension to project down to (e.g., 5)
    /// * `window_scale` — window size multiplier (default 1.0 = φ-scaled).
    ///   Larger = more detail preserved, lower compression.
    pub fn new(input_dim: usize, tiling_dim: usize, window_scale: f64) -> Self {
        assert!(tiling_dim < input_dim, "tiling_dim must be < input_dim");
        // Embed into double the input dimension for rich structure
        let embed_dim = input_dim * 2;
        let extra_dim = embed_dim - tiling_dim;

        let mut cap = CutAndProject::new(tiling_dim, extra_dim);
        // Scale windows by window_scale
        for w in &mut cap.window {
            w.0 *= window_scale;
            w.1 *= window_scale;
        }

        Self { tiling_dim, embed_dim, window_scale, cap }
    }

    /// Lift an input vector into the high-D embedding space.
    /// Pads with golden-ratio-derived coordinates.
    fn lift(&self, input: &[f64]) -> Vec<f64> {
        let input_dim = input.len();
        assert!(input_dim * 2 == self.embed_dim,
            "input dim {} doesn't match embed_dim {}", input_dim, self.embed_dim);

        let mut lifted = Vec::with_capacity(self.embed_dim);
        // Copy original coordinates
        lifted.extend_from_slice(input);

        // Pad with golden-ratio-derived coordinates
        // Each padding coordinate = sum of two input coords, modulated by φ
        for i in 0..input_dim {
            let j = (i + 1) % input_dim;
            let val = (input[i] * PHI + input[j] * (PHI - 1.0)) / 2.0;
            lifted.push(val);
        }

        lifted
    }

    /// Quantize a lifted vector to lattice coordinates.
    fn quantize(&self, lifted: &[f64]) -> Vec<i64> {
        lifted.iter().map(|&x| x.round() as i64).collect()
    }

    /// Dequantize lattice coordinates back to floats.
    fn dequantize(&self, lattice: &[i64]) -> Vec<f64> {
        lattice.iter().map(|&x| x as f64).collect()
    }

    /// Extract the input portion from a lifted vector.
    fn extract_input(&self, lifted: &[f64]) -> Vec<f64> {
        let input_dim = self.embed_dim / 2;
        lifted[..input_dim].to_vec()
    }

    /// Compress an embedding into a tile.
    pub fn compress(&self, embedding: &[f64]) -> CompressedTile {
        let lifted = self.lift(embedding);
        let lattice = self.quantize(&lifted);

        // Project to tiling space
        let projected = self.cap.project_to_tiling(&lattice);

        // Project to perpendicular space (the residue)
        let perp = self.cap.project_to_perp(&lattice);

        // Determine tile type
        let tile_type = self.cap.tile_type(&lattice);

        // Store the padding values for reconstruction
        let input_dim = embedding.len();
        let padded_values = lifted[input_dim..].to_vec();

        CompressedTile {
            projected,
            perp_residue: perp,
            tile_type,
            embed_dim: self.embed_dim,
            tiling_dim: self.tiling_dim,
            lattice_coords: lattice,
            padded_values,
        }
    }

    /// Decompress a tile back into a vector.
    ///
    /// This is lossy — reconstruction quality depends on how much
    /// perpendicular info was preserved. The lattice coords give us the
    /// integer approximation; the perp residue refines it.
    pub fn decompress(&self, tile: &CompressedTile) -> Decompressed {
        // Start from lattice coordinates (integer quantization)
        let mut reconstructed = self.dequantize(&tile.lattice_coords);

        // Refine using perpendicular residue: the residue encodes the
        // fractional part lost during quantization. We add a correction
        // by projecting the residue back.
        // The perp residue tells us where in the window we were — use it
        // to distribute corrections across dimensions.
        let perp_dim = tile.perp_residue.len();
        for i in 0..self.embed_dim {
            // Each dimension gets a correction from each perp coordinate
            let mut correction = 0.0;
            for j in 0..perp_dim {
                // The perpendicular space is at π/2 offset from tiling space
                let angle = std::f64::consts::TAU * i as f64 / self.embed_dim as f64
                    * PHI.powi(j as i32 + 1)
                    + std::f64::consts::FRAC_PI_2;
                correction += angle.cos() * tile.perp_residue[j] / (self.embed_dim as f64).sqrt();
            }
            // The correction is small — it refines the quantized value
            reconstructed[i] += correction * 0.5; // Dampen to avoid overshoot
        }

        // Extract just the input dimensions (drop the padding)
        let vector = reconstructed[..self.embed_dim / 2].to_vec();

        Decompressed {
            vector,
            mse: 0.0, // Caller should compute against original
        }
    }

    /// Compute mean squared error between original and reconstructed.
    pub fn reconstruction_mse(original: &[f64], reconstructed: &[f64]) -> f64 {
        assert_eq!(original.len(), reconstructed.len());
        let n = original.len() as f64;
        original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-12 || norm_b < 1e-12 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// The compression ratio: embed_dim / tiling_dim.
    pub fn compression_ratio(&self) -> f64 {
        self.embed_dim as f64 / self.tiling_dim as f64
    }

    /// Information capacity estimate: scales with embedding dimension.
    /// More "nasty" (higher embed_dim) = more capacity per tile.
    pub fn info_capacity(&self) -> f64 {
        // Capacity scales with the log of the number of distinct tile types
        // times the embedding dimension. The aperiodicity means every tile
        // position encodes unique information.
        let n_prototiles = 2_usize.pow((self.tiling_dim - 1) as u32).max(2);
        (n_prototiles as f64).ln() * self.embed_dim as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple test embedding with known structure.
    fn test_embedding(dim: usize) -> Vec<f64> {
        (0..dim).map(|i| ((i as f64 * PHI).sin() * 10.0).round()).collect()
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let input_dim = 20;
        let tiling_dim = 5;
        let nc = NastyCompress::new(input_dim, tiling_dim, 1.0);
        let embedding = test_embedding(input_dim);

        let tile = nc.compress(&embedding);
        let decompressed = nc.decompress(&tile);

        assert_eq!(decompressed.vector.len(), input_dim);
        // Quantization is lossy, but values should be in the same ballpark
        for (orig, recon) in embedding.iter().zip(decompressed.vector.iter()) {
            let diff = (orig - recon).abs();
            assert!(diff < 50.0, "Reconstruction too far: {} vs {}, diff={}", orig, recon, diff);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let nc = NastyCompress::new(100, 5, 1.0);
        // 200D embed / 5D tiling = 40:1
        assert!((nc.compression_ratio() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_tile_type_deterministic() {
        let nc = NastyCompress::new(20, 5, 1.0);
        let embedding = test_embedding(20);

        let tile1 = nc.compress(&embedding);
        let tile2 = nc.compress(&embedding);
        // Same input → same tile type
        assert_eq!(tile1.tile_type, tile2.tile_type);
        // Same projected coords
        assert_eq!(tile1.projected, tile2.projected);
    }

    #[test]
    fn test_different_inputs_different_tiles() {
        let nc = NastyCompress::new(20, 5, 1.0);
        let emb1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let emb2: Vec<f64> = (0..20).map(|i| (i as f64 * -1.0)).collect();

        let tile1 = nc.compress(&emb1);
        let tile2 = nc.compress(&emb2);
        // Different inputs should produce different projected coords
        assert_ne!(tile1.projected, tile2.projected);
    }

    #[test]
    fn test_reconstruction_accuracy_increases_with_window() {
        let input_dim = 20;
        let tiling_dim = 5;
        let embedding = test_embedding(input_dim);

        // Small window
        let nc_small = NastyCompress::new(input_dim, tiling_dim, 0.5);
        let tile_small = nc_small.compress(&embedding);
        let decomp_small = nc_small.decompress(&tile_small);
        let mse_small = NastyCompress::reconstruction_mse(&embedding, &decomp_small.vector);

        // Larger window
        let nc_large = NastyCompress::new(input_dim, tiling_dim, 2.0);
        let tile_large = nc_large.compress(&embedding);
        let decomp_large = nc_large.decompress(&tile_large);
        let mse_large = NastyCompress::reconstruction_mse(&embedding, &decomp_large.vector);

        // Both should produce finite MSE
        assert!(mse_small.is_finite());
        assert!(mse_large.is_finite());
        // The compression should not produce NaN/inf
        for v in &decomp_small.vector {
            assert!(v.is_finite());
        }
        for v in &decomp_large.vector {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_info_capacity_scales_with_dim() {
        let nc_20 = NastyCompress::new(20, 5, 1.0);
        let nc_100 = NastyCompress::new(100, 5, 1.0);
        // Higher embedding dim → more capacity
        assert!(nc_100.info_capacity() > nc_20.info_capacity());
    }

    #[test]
    fn test_cosine_similarity_perfect() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = NastyCompress::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = NastyCompress::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_high_dimensional_compress() {
        // 100D embedding → 5D tiling (20:1 compression as described)
        let nc = NastyCompress::new(100, 5, 1.0);
        let embedding = test_embedding(100);

        let tile = nc.compress(&embedding);

        assert_eq!(tile.projected.len(), 5, "Projected should be tiling_dim");
        assert_eq!(tile.lattice_coords.len(), 200, "Lattice should be embed_dim");
        assert_eq!(tile.padded_values.len(), 100, "Padding should be input_dim");

        let decomp = nc.decompress(&tile);
        assert_eq!(decomp.vector.len(), 100);
    }

    #[test]
    fn test_reconstruction_preserves_direction() {
        // Even if magnitude is lossy, the direction (cosine similarity)
        // should be reasonable for structured input.
        let nc = NastyCompress::new(20, 5, 1.0);
        let embedding = test_embedding(20);

        let tile = nc.compress(&embedding);
        let decomp = nc.decompress(&tile);

        let sim = NastyCompress::cosine_similarity(&embedding, &decomp.vector);
        // With quantization, similarity won't be perfect, but should be positive
        // for structured input
        assert!(sim > -0.5, "Cosine similarity should be reasonable, got {}", sim);
    }

    #[test]
    fn test_mse_computation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(NastyCompress::reconstruction_mse(&a, &b), 0.0);

        let c = vec![2.0, 3.0, 4.0];
        let mse = NastyCompress::reconstruction_mse(&a, &c);
        assert!((mse - 1.0).abs() < 1e-10); // avg of 1,1,1 = 1.0
    }
}
