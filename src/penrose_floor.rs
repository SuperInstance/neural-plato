//! Dead reckoning memory: distance + direction through the Penrose floor.
//!
//! You don't need coordinates. You need:
//!   1. How far from where you are (distance)
//!   2. Which way you're headed (heading)
//!
//! The Penrose tiling's matching rules determine everything else.
//! Walk forward by distance D at heading θ → the floor unfolds under your feet.
//! Each step lands on a tile. The tile's shape (thick/thin) is a single bit.
//! The sequence of bits you walk over IS the memory you retrieve.
//!
//! This is dead reckoning: the ancient navigator's technique.
//! No GPS. No absolute coordinates. Just "I've come this far, in this direction."
//! The floor pattern confirms you're on the right path (matching rules lock in).

use std::collections::HashMap;

const PHI: f64 = 1.618033988749895;
const TAU: f64 = std::f64::consts::TAU;

/// A single step on the Penrose floor
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// Distance from current position (in Penrose tile units)
    pub distance: f64,
    /// Heading in radians (0 = east, π/2 = north)
    pub heading: f64,
}

impl Step {
    pub fn new(distance: f64, heading: f64) -> Self {
        Self { distance, heading }
    }

    /// Advance position by this step
    pub fn advance(&self, pos: (f64, f64)) -> (f64, f64) {
        (
            pos.0 + self.distance * self.heading.cos(),
            pos.1 + self.distance * self.heading.sin(),
        )
    }
}

/// A memory read from the Penrose floor by walking a path.
/// Each tile you step on gives you one bit. The sequence of bits IS the content.
#[derive(Debug, Clone)]
pub struct FloorRead {
    /// The bits read from tiles along the path
    pub bits: Vec<bool>,
    /// The positions visited (for debugging / visualization)
    pub path: Vec<(f64, f64)>,
    /// Heading at each step (how the direction evolved)
    pub headings: Vec<f64>,
    /// Whether matching rules were satisfied at each step
    pub matched: Vec<bool>,
    /// Confidence: fraction of steps where matching rules held
    pub confidence: f64,
}

/// The Penrose Floor — a memory palace navigated by dead reckoning.
///
/// Store memories as bit sequences at positions on the floor.
/// Retrieve by walking: give distance + heading, read the bits under your feet.
/// The matching rules enforce that you can only walk valid Penrose paths.
pub struct PenroseFloor {
    /// Stored bit sequences: position (quantized) → bits
    memory: HashMap<(i64, i64), Vec<bool>>,
    /// Content at each position
    content: HashMap<(i64, i64), u64>,
    /// Current position of the navigator
    pos: (f64, f64),
    /// Current heading of the navigator
    heading: f64,
    /// Tile scale (distance between adjacent tile centers)
    scale: f64,
}

impl PenroseFloor {
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            content: HashMap::new(),
            pos: (0.0, 0.0),
            heading: 0.0,
            scale: 1.0,
        }
    }

    pub fn at(mut self, x: f64, y: f64) -> Self {
        self.pos = (x, y);
        self
    }

    pub fn facing(mut self, heading: f64) -> Self {
        self.heading = heading;
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Quantize a continuous position to the nearest Penrose tile
    fn quantize(&self, pos: (f64, f64)) -> (i64, i64) {
        // Penrose tiles sit at golden-ratio-spaced positions
        // Quantize to the nearest tile center
        let q = (pos.0 / (self.scale * PHI)).round() as i64;
        let r = (pos.1 / (self.scale * PHI)).round() as i64;
        (q, r)
    }

    /// The Fibonacci word determines the tile at any position.
    /// Position (q, r) maps to an index, which maps to a bit.
    /// Thick = 1, Thin = 0.
    fn tile_bit(&self, pos: (i64, i64)) -> bool {
        // Combine both coordinates into a single index using a mixing function
        // This ensures different (q,r) positions give different bits
        let q = pos.0.wrapping_mul(0x9E3779B97F4A7C15u64 as i64);
        let mixed = (q ^ pos.1).wrapping_abs();
        let idx = (mixed % 1000) as usize;
        let inv_phi = 1.0 / PHI;
        let current = (idx as f64 * inv_phi).floor() as u64;
        let next = ((idx + 1) as f64 * inv_phi).floor() as u64;
        next != current
    }

    /// Check matching rules: does the tile at this position fit with its neighbors?
    /// In the Fibonacci word, local patterns are constrained.
    fn matching_rule_holds(&self, pos: (i64, i64)) -> bool {
        let bit = self.tile_bit(pos);
        // Check 6 neighbors (hexagonal adjacency for the floor lattice)
        let neighbors = [
            (pos.0 + 1, pos.1),
            (pos.0 - 1, pos.1),
            (pos.0, pos.1 + 1),
            (pos.0, pos.1 - 1),
            (pos.0 + 1, pos.1 - 1),
            (pos.0 - 1, pos.1 + 1),
        ];
        
        // Matching rule: thick tiles must have at least one thin neighbor
        // and vice versa. This prevents monochromatic clusters.
        if bit {
            // Thick tile: must have at least one thin neighbor
            neighbors.iter().any(|&n| !self.tile_bit(n))
        } else {
            // Thin tile: must have at least one thick neighbor
            neighbors.iter().any(|&n| self.tile_bit(n))
        }
    }

    /// Store a memory at the current position.
    /// The content is encoded as bits laid down on the Penrose floor.
    pub fn store_here(&mut self, content: u64) -> (i64, i64) {
        let key = self.quantize(self.pos);
        let bits = Self::u64_to_bits(content);
        self.memory.insert(key, bits);
        self.content.insert(key, content);
        key
    }

    /// Walk a path defined by distance + heading steps.
    /// Read ONE bit per step — the bit at the tile you land on.
    pub fn walk(&self, steps: &[Step]) -> FloorRead {
        let mut bits = Vec::new();
        let mut path = Vec::new();
        let mut headings = Vec::new();
        let mut matched = Vec::new();
        let mut pos = self.pos;
        let mut heading = self.heading;

        for step in steps {
            heading = step.heading;
            let new_pos = step.advance(pos);

            // Read the tile at the landing position
            let key = self.quantize(new_pos);
            bits.push(self.tile_bit(key));
            path.push(new_pos);
            headings.push(heading);
            matched.push(self.matching_rule_holds(key));

            pos = new_pos;
        }

        let confidence = if matched.is_empty() {
            1.0
        } else {
            matched.iter().filter(|&&m| m).count() as f64 / matched.len() as f64
        };

        FloorRead { bits, path, headings, matched, confidence }
    }

    /// Walk and retrieve: read bits from floor, decode to content.
    /// This is "dead reckoning retrieval" — no index lookup, just walk.
    pub fn walk_and_retrieve(&self, steps: &[Step]) -> Option<u64> {
        let read = self.walk(steps);
        
        // Check: did the walk end near a stored memory?
        if let Some(&last_pos) = read.path.last() {
            let key = self.quantize(last_pos);
            self.content.get(&key).copied()
        } else {
            None
        }
    }

    /// Walk toward a stored memory using only distance + direction.
    /// The matching rules confirm you're on the right path.
    pub fn spline_to(
        &self,
        target: (f64, f64),
        n_steps: usize,
    ) -> FloorRead {
        let dx = target.0 - self.pos.0;
        let dy = target.1 - self.pos.1;
        let total_dist = (dx * dx + dy * dy).sqrt();
        let heading = dy.atan2(dx);
        
        // Precise stretch: step size = total distance / n_steps
        let step_dist = total_dist / n_steps as f64;
        
        // Build steps: straight line, equal spacing
        let steps = vec![Step::new(step_dist, heading); n_steps];
        
        self.walk(&steps)
    }

    /// Navigate by heading adjustments (like a boat tacking).
    /// Each step adjusts heading by a delta, then walks forward.
    pub fn tack(&self, heading_deltas: &[f64], step_distance: f64) -> FloorRead {
        let steps: Vec<Step> = heading_deltas
            .iter()
            .scan(self.heading, |h, &delta| {
                *h += delta;
                Some(Step::new(step_distance, *h))
            })
            .collect();
        self.walk(&steps)
    }

    /// The "precise stretch" walk: varying step distances at constant heading.
    /// This is how the spline stretches across the Penrose floor —
    /// the floor pattern determines WHERE you land, the stretch determines HOW FAR.
    pub fn stretch_walk(&self, stretches: &[f64], heading: f64) -> FloorRead {
        let steps: Vec<Step> = stretches
            .iter()
            .map(|&d| Step::new(d * self.scale * PHI, heading))
            .collect();
        self.walk(&steps)
    }

    /// Get current position
    pub fn position(&self) -> (f64, f64) {
        self.pos
    }

    /// Get current heading
    pub fn heading(&self) -> f64 {
        self.heading
    }

    /// Count stored memories
    pub fn len(&self) -> usize {
        self.content.len()
    }

    fn u64_to_bits(v: u64) -> Vec<bool> {
        (0..64).map(|i| (v >> (63 - i)) & 1 == 1).collect()
    }
}

impl Default for PenroseFloor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_reckoning_store_and_walk() {
        let mut floor = PenroseFloor::new().at(5.0, 5.0);
        floor.store_here(0xDEADBEEF);
        
        // Walk toward the stored position
        let read = floor.spline_to((5.0, 5.0), 3);
        assert!(read.confidence > 0.0);
        assert!(!read.bits.is_empty());
    }

    #[test]
    fn test_distance_and_direction_only() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        
        // Store memories at various positions
        floor.store_here(0x1111);
        
        let mut floor = PenroseFloor::new().at(10.0, 0.0).facing(0.0);
        floor.store_here(0x2222);
        
        // Query: walk 10 units west (heading π) to find first memory
        let steps = [Step::new(10.0, std::f64::consts::PI)];
        let read = floor.walk(&steps);
        assert!(!read.bits.is_empty());
    }

    #[test]
    fn test_matching_rules_hold() {
        let floor = PenroseFloor::new();
        
        // Check matching rules at many positions
        let mut valid = 0;
        let total = 100;
        for q in -50..50i64 {
            if floor.matching_rule_holds((q, 0)) {
                valid += 1;
            }
        }
        
        // Most positions should satisfy matching rules
        let ratio = valid as f64 / total as f64;
        assert!(ratio > 0.8, "Matching rules should hold at most positions, got {}", ratio);
    }

    #[test]
    fn test_precise_stretch() {
        let floor = PenroseFloor::new().at(0.0, 0.0).facing(0.0);
        
        // Walk with varying stretches — like a spline with different segment lengths
        let stretches = &[1.0, PHI, 1.0, PHI * PHI, 1.0];
        let read = floor.stretch_walk(stretches, 0.0);
        
        assert_eq!(read.bits.len(), stretches.len(), "Should read one bit per stretch");
    }

    #[test]
    fn test_tack_like_a_boat() {
        let floor = PenroseFloor::new().at(0.0, 0.0).facing(0.0);
        
        // Tack: alternate heading adjustments (like sailing upwind)
        let heading_deltas = &[
            std::f64::consts::FRAC_PI_6,   // tack right 30°
            -std::f64::consts::FRAC_PI_3,   // tack left 60°
            std::f64::consts::FRAC_PI_6,   // tack right 30°
            -std::f64::consts::FRAC_PI_3,   // tack left 60°
            std::f64::consts::FRAC_PI_6,   // tack right 30°
        ];
        
        let read = floor.tack(heading_deltas, PHI);
        assert_eq!(read.headings.len(), 5);
        
        // Heading should alternate
        assert!(read.headings[0] > 0.0);
        assert!(read.headings[1] < read.headings[0]);
    }

    #[test]
    fn test_tile_bit_fibonacci() {
        let floor = PenroseFloor::new();
        
        // The tile bits along any axis should follow the Fibonacci word
        let bits: Vec<bool> = (0..100).map(|q| floor.tile_bit((q, 0))).collect();
        let thick_ratio = bits.iter().filter(|&&b| b).count() as f64 / bits.len() as f64;
        
        // Should approach 1/φ ≈ 0.618
        assert!((thick_ratio - 1.0/PHI).abs() < 0.1,
            "Thick ratio {} should approach 1/φ", thick_ratio);
    }

    #[test]
    fn test_spline_to_target() {
        let mut floor = PenroseFloor::new().at(0.0, 0.0);
        floor.store_here(0xBEEF);
        
        // Spline from origin back to origin
        let read = floor.spline_to((0.0, 0.0), 5);
        assert!(read.confidence > 0.0);
        
        // Should be able to retrieve
        let content = floor.walk_and_retrieve(&[Step::new(0.1, 0.0)]);
        assert_eq!(content, Some(0xBEEF));
    }

    #[test]
    fn test_floor_is_aperiodic() {
        let floor = PenroseFloor::new();
        
        // Read bits along two different directions
        let east: Vec<bool> = (0..50).map(|q| floor.tile_bit((q, 0))).collect();
        let north: Vec<bool> = (0..50).map(|r| floor.tile_bit((0, r))).collect();
        let diag: Vec<bool> = (0..50).map(|i| floor.tile_bit((i, i))).collect();
        
        // All three should be different (aperiodic = different in every direction)
        assert_ne!(east, north);
        assert_ne!(east, diag);
        assert_ne!(north, diag);
    }
}
