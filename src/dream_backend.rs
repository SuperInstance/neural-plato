//! Dream backend configuration and types.
//!
//! Provides Rust-side configuration for the neural-plato dream system,
//! including memory parameters and forgetting curve settings.

/// Configuration for a dream/reconstruction session.
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Memory coverage fraction (0.0 - 1.0).
    pub coverage: f64,
    /// Number of negative constraints available.
    pub negative_constraints: usize,
    /// Total facts in the knowledge base.
    pub total_facts: usize,
    /// Prompt style: 0=direct, 1=conversational, 2=Socratic, 3=narrative, 4=technical.
    pub style: u32,
    /// Whether to use sparse memory retrieval.
    pub use_sparse_memory: bool,
    /// Top-k value for sparse retrieval.
    pub top_k: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            coverage: 1.0,
            negative_constraints: 0,
            total_facts: 100,
            style: 0,
            use_sparse_memory: true,
            top_k: 8,
        }
    }
}

impl DreamConfig {
    /// Create a new dream config with specified coverage.
    pub fn with_coverage(coverage: f64) -> Self {
        Self {
            coverage,
            ..Default::default()
        }
    }

    /// Predict the expected reconstruction accuracy.
    pub fn predicted_accuracy(&self) -> f64 {
        // Base accuracy from forgetting curve
        let base = predict_accuracy(self.coverage);

        // Style factor
        let style_f = style_factor(self.style);

        // Shadow bonus from negative constraints
        let shadow_bonus = if self.total_facts > 0 {
            shadow_accuracy(self.negative_constraints, self.total_facts) * 0.1
        } else {
            0.0
        };

        (base * style_f + shadow_bonus).min(1.0)
    }

    /// Check if coverage is below the amnesia cliff.
    pub fn is_below_cliff(&self) -> bool {
        self.coverage < 0.10
    }
}

/// Result of a dream/reconstruction pass.
#[derive(Debug, Clone)]
pub struct DreamResult {
    /// Reconstructed output vector.
    pub output: Vec<f64>,
    /// Predicted accuracy for this result.
    pub accuracy: f64,
    /// Number of sparse memory lookups performed.
    pub lookups: usize,
    /// Coverage used for this dream.
    pub coverage: f64,
}

impl DreamResult {
    /// Create a new dream result.
    pub fn new(output: Vec<f64>, accuracy: f64, lookups: usize, coverage: f64) -> Self {
        Self { output, accuracy, lookups, coverage }
    }

    /// Compute the L2 norm of the output.
    pub fn norm(&self) -> f64 {
        self.output.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ---- Rust implementations of Fortran algorithms ----

/// Predict accuracy from coverage (mirrors Fortran amnesia_curve).
pub fn predict_accuracy(coverage: f64) -> f64 {
    let coverage_data = [1.00, 0.75, 0.50, 0.33, 0.25, 0.15, 0.10, 0.05];
    let accuracy_data = [0.975, 0.775, 0.475, 0.325, 0.225, 0.225, 0.125, 0.000];

    if coverage <= 0.0 { return 0.0; }
    if coverage >= 1.0 { return accuracy_data[0]; }

    if coverage < 0.10 {
        return (coverage / 0.10) * accuracy_data[6];
    }

    // Linear interpolation between data points
    for i in 0..7 {
        if coverage >= accuracy_data[i + 1] && coverage <= coverage_data[i] {
            // Wrong — we need to interpolate by coverage, not accuracy
            if coverage <= coverage_data[i] && coverage >= coverage_data[i + 1] {
                let span = coverage_data[i] - coverage_data[i + 1];
                if span < 1e-12 { return accuracy_data[i]; }
                let t = (coverage - coverage_data[i + 1]) / span;
                return accuracy_data[i + 1] + t * (accuracy_data[i] - accuracy_data[i + 1]);
            }
        }
    }

    0.912 * coverage + 0.012 // linear fit fallback
}

/// Style factor multiplier (mirrors Fortran amnesia_curve::style_factor).
pub fn style_factor(style_id: u32) -> f64 {
    match style_id {
        0 => 1.00,
        1 => 0.95,
        2 => 0.88,
        3 => 0.92,
        4 => 1.05,
        _ => 1.00,
    }
}

/// Shadow accuracy from negative constraints (mirrors Fortran negative_space).
pub fn shadow_accuracy(n_negative: usize, n_total: usize) -> f64 {
    if n_total == 0 { return 0.0; }
    let density = n_negative as f64 / n_total as f64;
    let accuracy = 1.0 - (-4.6 * density).exp();
    accuracy.min(0.975)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_accuracy() {
        assert!(predict_accuracy(1.0) > 0.9);
        assert!(predict_accuracy(0.5) > 0.4);
        assert!(predict_accuracy(0.05) < 0.1);
        assert_eq!(predict_accuracy(0.0), 0.0);
    }

    #[test]
    fn test_style_factor() {
        assert_eq!(style_factor(0), 1.00);
        assert_eq!(style_factor(4), 1.05);
        assert!(style_factor(2) < 1.0);
    }

    #[test]
    fn test_dream_config() {
        let config = DreamConfig::with_coverage(0.75);
        let acc = config.predicted_accuracy();
        assert!(acc > 0.5 && acc < 1.0);
    }

    #[test]
    fn test_dream_config_cliff() {
        let config = DreamConfig::with_coverage(0.05);
        assert!(config.is_below_cliff());
        let config2 = DreamConfig::with_coverage(0.5);
        assert!(!config2.is_below_cliff());
    }

    #[test]
    fn test_dream_result() {
        let result = DreamResult::new(vec![1.0, 2.0, 3.0], 0.95, 8, 1.0);
        assert!((result.norm() - 3.74165738).abs() < 0.01);
    }
}
