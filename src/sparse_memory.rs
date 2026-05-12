//! Safe Rust wrapper for the sparse_memory Fortran module.

/// Configuration for creating a new sparse memory layer.
#[derive(Debug, Clone)]
pub struct SparseMemoryConfig {
    pub n_rows: usize,
    pub n_cols: usize,
    pub hidden_dim: usize,
    pub n_ranks: usize,
    pub top_k: usize,
}

impl Default for SparseMemoryConfig {
    fn default() -> Self {
        Self {
            n_rows: 64,
            n_cols: 64,
            hidden_dim: 128,
            n_ranks: 16,
            top_k: 8,
        }
    }
}

/// A sparse memory layer with UltraMem-inspired architecture.
///
/// This is a pure Rust implementation that mirrors the Fortran sparse_memory
/// module. The Fortran FFI is available for high-performance computation;
/// this Rust implementation provides a safe, portable alternative.
#[derive(Debug)]
pub struct SparseMemoryLayer {
    config: SparseMemoryConfig,
    values: Vec<Vec<Vec<f64>>>,     // (n_rows, n_cols, hidden_dim)
    row_keys: Vec<Vec<f64>>,        // (n_ranks, n_rows)
    col_keys: Vec<Vec<f64>>,        // (n_ranks, n_cols)
    tucker_core: Vec<Vec<f64>>,     // (n_ranks, n_ranks)
}

impl SparseMemoryLayer {
    /// Create a new sparse memory layer with random initialization.
    pub fn new(config: SparseMemoryConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (config.n_ranks + config.hidden_dim) as f64).sqrt();

        let values = (0..config.n_rows)
            .map(|_| (0..config.n_cols)
                .map(|_| (0..config.hidden_dim)
                    .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
                    .collect())
                .collect())
            .collect();

        let row_keys = (0..config.n_ranks)
            .map(|_| (0..config.n_rows)
                .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
                .collect())
            .collect();

        let col_keys = (0..config.n_ranks)
            .map(|_| (0..config.n_cols)
                .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
                .collect())
            .collect();

        let tucker_core = (0..config.n_ranks)
            .map(|_| (0..config.n_ranks)
                .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
                .collect())
            .collect();

        Self { config, values, row_keys, col_keys, tucker_core }
    }

    /// Query the sparse memory layer.
    ///
    /// Returns a weighted sum of the top-k value vectors,
    /// weighted by their Tucker-decomposed scores.
    pub fn query(&self, query_vec: &[f64]) -> Vec<f64> {
        assert_eq!(query_vec.len(), self.config.hidden_dim);

        // Project query to rank space via row keys
        let q_rank: Vec<f64> = (0..self.config.n_ranks)
            .map(|r| self.row_keys[r].iter()
                .zip(query_vec.iter())
                .map(|(k, q)| k * q)
                .sum())
            .collect();

        // Apply Tucker core
        let c_rank: Vec<f64> = (0..self.config.n_ranks)
            .map(|r| (0..self.config.n_ranks)
                .map(|k| self.tucker_core[r][k] * q_rank[k])
                .sum())
            .collect();

        // Compute column scores
        let col_scores: Vec<f64> = (0..self.config.n_cols)
            .map(|c| self.col_keys.iter().zip(c_rank.iter())
                .map(|(keys, cr)| keys[c] * cr)
                .sum())
            .collect();

        // Compute all scores and find top-k
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for r in 0..self.config.n_rows {
            let row_score: f64 = self.row_keys.iter()
                .zip(query_vec.iter())
                .map(|(keys, q)| keys[r] * q)
                .sum();
            for c in 0..self.config.n_cols {
                let score = row_score * col_scores[c];
                let idx = r * self.config.n_cols + c;
                scored.push((idx, score));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);

        // Weighted sum
        let total_score: f64 = scored.iter().map(|(_, s)| s.abs()).sum().max(1e-10);
        let mut output = vec![0.0; self.config.hidden_dim];
        for (idx, score) in &scored {
            let r = idx / self.config.n_cols;
            let c = idx % self.config.n_cols;
            let w = score.abs() / total_score;
            for d in 0..self.config.hidden_dim {
                output[d] += w * self.values[r][c][d];
            }
        }

        output
    }

    /// Get the configuration.
    pub fn config(&self) -> &SparseMemoryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_layer() {
        let config = SparseMemoryConfig {
            n_rows: 4, n_cols: 4, hidden_dim: 8,
            n_ranks: 2, top_k: 2,
        };
        let layer = SparseMemoryLayer::new(config);
        assert_eq!(layer.config().n_rows, 4);
    }

    #[test]
    fn test_query() {
        let config = SparseMemoryConfig {
            n_rows: 4, n_cols: 4, hidden_dim: 8,
            n_ranks: 2, top_k: 2,
        };
        let layer = SparseMemoryLayer::new(config);
        let query = vec![1.0, 0.5, -0.3, 0.7, -0.2, 0.4, 0.1, -0.6];
        let output = layer.query(&query);
        assert_eq!(output.len(), 8);
        // Output should be non-zero (unless extremely unlucky with random init)
        let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Query output should be non-zero");
    }
}
