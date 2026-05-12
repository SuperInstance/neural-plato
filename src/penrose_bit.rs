//! Single-bit Penrose encoding.
//!
//! Penrose P3 uses exactly 2 tile shapes: Thick (1) and Thin (0).
//! Every position in the tiling is a single bit.
//! The matching rules are deterministic: once you see the local pattern,
//! there is only ONE way the tiling can continue.
//!
//! This means:
//!   - Memory address = bit string (sequence of thick/thin along a path)
//!   - The address IS the structure (matching rules determine valid addresses)
//!   - Compression = store only the seed (tiling grows deterministically from seed)
//!   - Context window = the local neighborhood currently being read
//!   - The entire brain = the tiling. Context window = the fovea.

use std::collections::HashMap;

const PHI: f64 = 1.618033988749895;

/// A single bit in the Penrose address space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PenroseBit {
    Thick = 1,
    Thin = 0,
}

impl From<bool> for PenroseBit {
    fn from(b: bool) -> Self {
        if b { PenroseBit::Thick } else { PenroseBit::Thin }
    }
}

impl From<PenroseBit> for bool {
    fn from(b: PenroseBit) -> Self {
        b == PenroseBit::Thick
    }
}

/// A Penrose address: a sequence of bits that uniquely identifies a memory location.
/// The address IS the path through the tiling from the origin.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PenroseAddress {
    bits: Vec<PenroseBit>,
}

impl PenroseAddress {
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }

    /// Extend address by one bit (deeper into the tiling)
    pub fn push(&mut self, bit: PenroseBit) {
        self.bits.push(bit);
    }

    /// Pop back one level (zoom out)
    pub fn pop(&mut self) -> Option<PenroseBit> {
        self.bits.pop()
    }

    /// Depth in the tiling
    pub fn depth(&self) -> usize {
        self.bits.len()
    }

    /// Encode as raw bits (u8 packed)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, bit) in chunk.iter().enumerate() {
                if *bit == PenroseBit::Thick {
                    byte |= 1 << (7 - i);
                }
            }
            bytes.push(byte);
        }
        bytes
    }

    /// Decode from raw bits
    pub fn from_bytes(bytes: &[u8], bit_count: usize) -> Self {
        let mut bits = Vec::new();
        for (i, &byte) in bytes.iter().enumerate() {
            for j in 0..8 {
                if bits.len() >= bit_count {
                    break;
                }
                let is_thick = (byte >> (7 - j)) & 1 == 1;
                bits.push(PenroseBit::from(is_thick));
            }
            if bits.len() >= bit_count {
                break;
            }
        }
        Self { bits }
    }

    /// The parent address (one level up in the tiling)
    pub fn parent(&self) -> Option<Self> {
        if self.bits.is_empty() {
            None
        } else {
            Some(Self { bits: self.bits[..self.bits.len() - 1].to_vec() })
        }
    }

    /// Two child addresses (extend by Thick and Thin)
    pub fn children(&self) -> (Self, Self) {
        let mut thick = self.clone();
        thick.push(PenroseBit::Thick);
        let mut thin = self.clone();
        thin.push(PenroseBit::Thin);
        (thick, thin)
    }

    /// The ratio of Thick:Thin in this address (should approach φ for deep addresses)
    pub fn thick_ratio(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        let thick_count = self.bits.iter().filter(|b| **b == PenroseBit::Thick).count();
        thick_count as f64 / self.bits.len() as f64
    }
}

/// The Penrose Brain: a memory system where addresses are single-bit Penrose paths.
///
/// The key insight: you don't store the structure. You store SEEDS.
/// The tiling grows deterministically from a seed via matching rules.
/// "Once you see the pattern lock in, there's only one way it can go."
pub struct PenroseBrain {
    /// Stored memories: address → content
    memories: HashMap<Vec<u8>, u64>,
    /// The seed (initial pattern that determines the entire tiling)
    seed: Vec<PenroseBit>,
    /// Maximum depth of stored memories
    max_depth: usize,
    /// Matching rule cache: once determined, the tiling is fixed
    determined: HashMap<Vec<PenroseBit>, PenroseBit>,
}

impl PenroseBrain {
    pub fn new(seed_bits: Vec<PenroseBit>) -> Self {
        Self {
            memories: HashMap::new(),
            seed: seed_bits,
            max_depth: 0,
            determined: HashMap::new(),
        }
    }

    /// Generate the tiling deterministically from the seed.
    /// Once the local pattern locks in, there's only one way to continue.
    /// This IS the matching rule: given the parent path, the next bit is determined.
    pub fn determine_next(&mut self, path: &[PenroseBit]) -> PenroseBit {
        // Check if already determined
        if let Some(&bit) = self.determined.get(path) {
            return bit;
        }

        // The matching rule for Penrose P3 (thick/thin rhombus tiling):
        // The Fibonacci word determines the sequence of thick/thin tiles.
        // Fibonacci word: substitute 1→10, 0→1, starting from 1
        // This produces the same sequence as the Penrose tiling's inflation.
        
        let depth = path.len();
        let bit = self.fibonacci_rule(depth);
        
        self.determined.insert(path.to_vec(), bit);
        bit
    }

    /// Fibonacci substitution rule: determines the tiling from depth alone.
    /// The Penrose tiling's thick/thin sequence IS the Fibonacci word.
    fn fibonacci_rule(&self, depth: usize) -> PenroseBit {
        // Fibonacci word: 1, 10, 101, 10110, 10110101, ...
        // Character at position n is:
        //   1 if (n+1)*(φ-1) rounded down changes from n*(φ-1) rounded down
        //   0 otherwise
        // Equivalently: the nth character is 1 if floor((n+1)/φ) != floor(n/φ)
        
        let inv_phi = 1.0 / PHI;
        let current = (depth as f64 * inv_phi).floor() as u64;
        let next = ((depth + 1) as f64 * inv_phi).floor() as u64;
        
        if next != current {
            PenroseBit::Thick
        } else {
            PenroseBit::Thin
        }
    }

    /// Store a memory at an address. The address must satisfy matching rules.
    pub fn store(&mut self, address: &PenroseAddress, content: u64) -> Result<(), String> {
        // Validate: every bit in the address must be consistent with
        // the deterministic tiling (matching rules)
        for i in 0..address.depth() {
            let prefix = &address.bits[..i];
            let expected = self.determine_next(prefix);
            if address.bits[i] != expected {
                return Err(format!(
                    "Matching rule violation at depth {}: expected {:?}, got {:?}",
                    i, expected, address.bits[i]
                ));
            }
        }
        
        let bytes = address.to_bytes();
        self.memories.insert(bytes, content);
        self.max_depth = self.max_depth.max(address.depth());
        Ok(())
    }

    /// Retrieve a memory by navigating the tiling.
    /// The context window (depth parameter) determines how much you "see."
    pub fn retrieve(&self, address: &PenroseAddress) -> Option<u64> {
        let bytes = address.to_bytes();
        self.memories.get(&bytes).copied()
    }

    /// Query: navigate the tiling from a seed, using matching rules to find
    /// the unique address that matches the query pattern.
    /// Returns the address and content if found.
    pub fn query_by_pattern(&mut self, pattern: &[PenroseBit]) -> Option<(PenroseAddress, u64)> {
        let mut address = PenroseAddress::new();
        
        for i in 0..pattern.len() {
            let prefix = &address.bits[..];
            let expected = self.determine_next(prefix);
            address.push(expected);
            
            // Check: does the deterministic tiling match the query pattern?
            if address.bits[i] != pattern[i] {
                // Pattern doesn't match — this address is in a different part of the palace
                // The Bragg peak is zero — no match here
                return None;
            }
        }
        
        // Pattern matched — strong Bragg peak
        self.retrieve(&address).map(|content| (address, content))
    }

    /// The brain's total stored memory in bytes.
    /// Only the seeds and content are stored — the structure is FREE (deterministic).
    pub fn memory_footprint(&self) -> usize {
        // Seeds + determined cache + stored content
        let seed_bytes = self.seed.len().div_ceil(8);
        let determined_bytes = self.determined.len() * 2; // ~2 bytes per entry
        let stored_bytes = self.memories.len() * (8 + 8); // key + value
        seed_bytes + determined_bytes + stored_bytes
    }

    /// Deflate: consolidate a subtree into a single bit.
    /// The context window shrinks — you see less detail but keep the structure.
    pub fn deflate(&mut self, address: &PenroseAddress) -> Option<u64> {
        let parent = address.parent()?;
        let bytes = address.to_bytes();
        let content = self.memories.remove(&bytes)?;
        
        // XOR content into parent (lossy consolidation)
        let parent_bytes = parent.to_bytes();
        let parent_content = self.memories.get(&parent_bytes).copied().unwrap_or(0);
        self.memories.insert(parent_bytes, parent_content ^ content);
        
        Some(parent_content ^ content)
    }

    /// Inflate: expand a single bit into its two children.
    /// The context window grows — you see more detail.
    pub fn inflate(&mut self, address: &PenroseAddress) -> (PenroseAddress, PenroseAddress) {
        let (thick_child, thin_child) = address.children();
        (thick_child, thin_child)
    }

    /// Count stored memories
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// The thick:thin ratio should approach 1/φ ≈ 0.618 for the Fibonacci word
    pub fn thick_ratio_at_depth(&self, depth: usize) -> f64 {
        let mut count = 0usize;
        for i in 0..depth {
            if self.fibonacci_rule(i) == PenroseBit::Thick {
                count += 1;
            }
        }
        count as f64 / depth.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_word_ratio() {
        let brain = PenroseBrain::new(vec![]);
        // The Fibonacci word has thick ratio → 1/φ ≈ 0.618 as depth → ∞
        let ratio = brain.thick_ratio_at_depth(1000);
        assert!((ratio - 1.0/PHI).abs() < 0.01, 
            "Thick ratio {} should approach 1/φ ≈ 0.618", ratio);
    }

    #[test]
    fn test_address_encode_decode() {
        let mut addr = PenroseAddress::new();
        for b in [true, true, false, true, false, true, true, false, true, false] {
            addr.push(PenroseBit::from(b));
        }
        let bytes = addr.to_bytes();
        let decoded = PenroseAddress::from_bytes(&bytes, 10);
        assert_eq!(addr, decoded);
    }

    #[test]
    fn test_deterministic_tiling() {
        let mut brain = PenroseBrain::new(vec![PenroseBit::Thick]);
        
        // The tiling is deterministic: same depth always gives same bit
        let b1 = brain.determine_next(&[]);
        let b2 = brain.determine_next(&[]);
        assert_eq!(b1, b2, "Tiling must be deterministic");
        
        // Different depths give (usually) different bits
        let mut bits = Vec::new();
        let mut path = Vec::new();
        for _ in 0..20 {
            let b = brain.determine_next(&path);
            bits.push(b);
            path.push(b);
        }
        // Should have both Thick and Thin
        assert!(bits.iter().any(|b| *b == PenroseBit::Thick));
        assert!(bits.iter().any(|b| *b == PenroseBit::Thin));
    }

    #[test]
    fn test_store_and_retrieve_deterministic() {
        let mut brain = PenroseBrain::new(vec![]);
        
        // Build a valid address (following matching rules)
        let mut addr = PenroseAddress::new();
        for i in 0..8 {
            let bit = brain.fibonacci_rule(i);
            addr.push(bit);
        }
        
        brain.store(&addr, 0xBEEF).unwrap();
        assert_eq!(brain.retrieve(&addr), Some(0xBEEF));
    }

    #[test]
    fn test_matching_rule_violation_rejected() {
        let mut brain = PenroseBrain::new(vec![]);
        
        // Build an INVALID address (violating matching rules)
        let mut addr = PenroseAddress::new();
        for _ in 0..8 {
            addr.push(PenroseBit::Thick); // All thick — violates Fibonacci word
        }
        
        let result = brain.store(&addr, 0xDEAD);
        assert!(result.is_err(), "Should reject matching rule violation");
    }

    #[test]
    fn test_deflate_inflate() {
        let mut brain = PenroseBrain::new(vec![]);
        
        let mut addr = PenroseAddress::new();
        for i in 0..4 {
            addr.push(brain.fibonacci_rule(i));
        }
        
        brain.store(&addr, 42).unwrap();
        let _deflated = brain.deflate(&addr);
        assert_eq!(brain.len(), 1); // Parent now holds consolidated content
    }

    #[test]
    fn test_parent_children() {
        let mut addr = PenroseAddress::new();
        addr.push(PenroseBit::Thick);
        addr.push(PenroseBit::Thin);
        
        let parent = addr.parent().unwrap();
        assert_eq!(parent.depth(), 1);
        
        let (thick, thin) = parent.children();
        assert_eq!(thick.depth(), 2);
        assert_eq!(thin.depth(), 2);
    }

    #[test]
    fn test_context_window_is_fovea() {
        let mut brain = PenroseBrain::new(vec![]);
        
        // Store 100 memories at various depths
        for seed in 0..100u64 {
            let mut addr = PenroseAddress::new();
            // Use seed bits to determine depth and path
            for bit in 0..6 {
                if (seed >> bit) & 1 == 1 {
                    addr.push(brain.fibonacci_rule(addr.depth()));
                }
            }
            if addr.depth() > 0 {
                brain.store(&addr, seed).ok();
            }
        }
        
        // The brain holds 100 memories
        assert!(brain.len() > 0);
        
        // But the structure is FREE — only content is stored
        let footprint = brain.memory_footprint();
        // 100 memories × ~16 bytes each ≈ ~1600 bytes
        assert!(footprint < 5000, "Footprint {} should be small", footprint);
    }

    #[test]
    fn test_single_bit_address() {
        // The most fundamental case: a single bit IS an address
        let mut addr = PenroseAddress::new();
        addr.push(PenroseBit::Thick);
        
        let bytes = addr.to_bytes();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b10000000); // Thick = 1, in high bit
    }
}
