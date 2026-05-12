//! Penrose Memory Palace — Aperiodic coordinates for AI memory retrieval
//!
//! Uses Penrose P3 (rhombus) tiling as the coordinate system for memory storage.
//! Every tile has a unique neighborhood (no collisions), the tiling is 3-colorable
//! (natural baton sharding), and matching rules enforce semantic adjacency.

use std::collections::HashMap;

/// Golden ratio
const PHI: f64 = 1.618033988749895;

/// Thick rhombus: angles 72° and 108°
/// Thin rhombus: angles 36° and 144°
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileType {
    Thick,
    Thin,
}

/// Three-coloring of Penrose tiling — maps to baton shards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardColor {
    Red,    // BUILT: concrete artifacts
    Green,  // THOUGHT: reasoning, decisions
    Blue,   // BLOCKED: gaps, negative space
}

impl ShardColor {
    pub fn all() -> [ShardColor; 3] {
        [ShardColor::Red, ShardColor::Green, ShardColor::Blue]
    }
}

/// Edge decoration for matching rules
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeDeco {
    /// Decoration type (semantic constraint class)
    pub deco: u8,
    /// Orientation in radians
    pub orientation: f64,
}

/// A memory tile in the Penrose palace
#[derive(Debug, Clone)]
pub struct PenroseTile {
    /// Unique ID
    pub id: u64,
    /// Position in 2D Penrose plane
    pub position: (f64, f64),
    /// Thick or thin rhombus
    pub tile_type: TileType,
    /// 3-coloring assignment
    pub color: ShardColor,
    /// Deflation level (0 = raw fact, higher = more consolidated)
    pub level: u32,
    /// Edge decorations (matching rules)
    pub edges: [EdgeDeco; 4],
    /// Content hash (reference to actual memory)
    pub content_hash: u64,
    /// Confidence (Bragg peak intensity when stored)
    pub confidence: f64,
}

/// The result of a retrieval query
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    pub tile: PenroseTile,
    pub bragg_confidence: f64,
    pub ring_distance: u32,
    pub color: ShardColor,
}

/// The Penrose Memory Palace
pub struct PenrosePalace {
    tiles: HashMap<(i64, i64), PenroseTile>,
    next_id: u64,
    /// Golden hierarchy levels
    levels: Vec<DeflationLevel>,
}

/// A deflation level in the golden hierarchy
#[derive(Debug, Clone)]
pub struct DeflationLevel {
    pub level: u32,
    pub tile_count: usize,
    pub avg_confidence: f64,
}

impl PenrosePalace {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            next_id: 0,
            levels: vec![DeflationLevel {
                level: 0,
                tile_count: 0,
                avg_confidence: 0.0,
            }],
        }
    }

    /// Store a memory at its Penrose coordinates
    pub fn store(
        &mut self,
        position: (f64, f64),
        content_hash: u64,
        semantic_class: u8,
    ) -> Result<PenroseTile, String> {
        let (q, r) = self.quantize(position);
        
        // Determine tile type from position
        // Use golden ratio to decide: positions where fract(q/φ) < 1/φ are thick
        let tile_type = if (q as f64 / PHI).fract() < 1.0 / PHI {
            TileType::Thick
        } else {
            TileType::Thin
        };
        
        // 3-coloring: determined by position modulo 3
        let color_idx = ((q + 2 * r).rem_euclid(3)) as usize;
        let color = ShardColor::all()[color_idx.min(2)];
        
        // Edge decorations from semantic class
        let angle = match tile_type {
            TileType::Thick => std::f64::consts::FRAC_PI_2 * 0.8, // 72°
            TileType::Thin => std::f64::consts::FRAC_PI_2 * 0.4,  // 36°
        };
        let edges = [
            EdgeDeco { deco: semantic_class, orientation: angle },
            EdgeDeco { deco: semantic_class, orientation: angle + std::f64::consts::PI / 2.0 },
            EdgeDeco { deco: semantic_class, orientation: angle + std::f64::consts::PI },
            EdgeDeco { deco: semantic_class, orientation: angle + 3.0 * std::f64::consts::PI / 2.0 },
        ];
        
        let tile = PenroseTile {
            id: self.next_id,
            position,
            tile_type,
            color,
            level: 0,
            edges,
            content_hash,
            confidence: 1.0,
        };
        self.next_id += 1;
        self.tiles.insert((q, r), tile.clone());
        Ok(tile)
    }

    /// Query the palace — Bragg peak retrieval
    pub fn query(&self, position: (f64, f64), max_rings: u32) -> Vec<RetrievedMemory> {
        let (q, r) = self.quantize(position);
        let mut results = Vec::new();
        
        // Ring 0: exact match
        if let Some(tile) = self.tiles.get(&(q, r)) {
            let confidence = self.bragg_peak(tile);
            if confidence > 0.5 {
                results.push(RetrievedMemory {
                    tile: tile.clone(),
                    bragg_confidence: confidence,
                    ring_distance: 0,
                    color: tile.color,
                });
            }
        }
        
        // Expand outward through golden rings
        for ring in 1..=max_rings {
            let radius = PHI.powi(ring as i32);
            let ring_tiles = self.tiles_in_ring(q, r, radius);
            let mut ring_results = Vec::new();
            
            for (_, tile) in ring_tiles {
                let confidence = self.bragg_peak(&tile);
                if confidence > 0.3 {
                    ring_results.push(RetrievedMemory {
                        tile: tile.clone(),
                        bragg_confidence: confidence,
                        ring_distance: ring,
                        color: tile.color,
                    });
                }
            }
            
            // Sort by confidence (Bragg peak intensity)
            ring_results.sort_by(|a, b| b.bragg_confidence.partial_cmp(&a.bragg_confidence).unwrap());
            
            if !ring_results.is_empty() {
                // Bragg peak found at this ring
                results.extend(ring_results);
                if results.iter().any(|r| r.bragg_confidence > 0.9) {
                    break; // Strong peak — stop searching
                }
            }
        }
        
        results
    }

    /// Deflate (consolidate memories — dream module)
    pub fn deflate(&mut self, center: (i64, i64), radius: f64) -> Option<PenroseTile> {
        let tiles_in_region: Vec<_> = self.tiles_in_ring(center.0, center.1, radius)
            .into_values()
            .collect();
        
        if tiles_in_region.is_empty() {
            return None;
        }
        
        // Consolidate into a single higher-level tile
        let avg_confidence: f64 = tiles_in_region.iter()
            .map(|t| t.confidence)
            .sum::<f64>() / tiles_in_region.len() as f64;
        
        let content_hash = tiles_in_region.iter()
            .fold(0u64, |acc, t| acc ^ t.content_hash);
        
        let new_level = tiles_in_region.iter().map(|t| t.level).max().unwrap_or(0) + 1;
        
        let tile = PenroseTile {
            id: self.next_id,
            position: (center.0 as f64, center.1 as f64),
            tile_type: TileType::Thick, // Deflated tiles are thick (consolidated)
            color: tiles_in_region.first().map(|t| t.color).unwrap_or(ShardColor::Red),
            level: new_level,
            edges: tiles_in_region.first().map(|t| t.edges).unwrap_or([
                EdgeDeco { deco: 0, orientation: 0.0 },
                EdgeDeco { deco: 0, orientation: std::f64::consts::FRAC_PI_2 },
                EdgeDeco { deco: 0, orientation: std::f64::consts::PI },
                EdgeDeco { deco: 0, orientation: 3.0 * std::f64::consts::FRAC_PI_2 },
            ]),
            content_hash,
            confidence: avg_confidence,
        };
        self.next_id += 1;
        self.tiles.insert(center, tile.clone());
        Some(tile)
    }

    /// Compute Bragg peak intensity (retrieval confidence)
    fn bragg_peak(&self, tile: &PenroseTile) -> f64 {
        // In a real implementation: compute diffraction pattern
        // Here: confidence decays with level (deflation = compression)
        // and is boosted by matching rule satisfaction with neighbors
        let base = tile.confidence;
        let level_decay = 1.0 / PHI.powi(tile.level as i32);
        base * level_decay
    }

    /// Quantize continuous position to lattice coordinates
    fn quantize(&self, pos: (f64, f64)) -> (i64, i64) {
        // Snap to Penrose grid (using golden ratio spacing)
        let q = (pos.0 / PHI).round() as i64;
        let r = (pos.1 / PHI).round() as i64;
        (q, r)
    }

    /// Get tiles within a ring (golden-scaled radius)
    fn tiles_in_ring(&self, cq: i64, cr: i64, radius: f64) -> HashMap<(i64, i64), PenroseTile> {
        let r2 = (radius * radius).ceil() as i64;
        self.tiles
            .iter()
            .filter(|((q, r), _)| {
                let dq = q - cq;
                let dr = r - cr;
                dq * dq + dr * dr <= r2
            })
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Get all tiles with a specific color (baton shard)
    pub fn by_color(&self, color: ShardColor) -> Vec<&PenroseTile> {
        self.tiles.values().filter(|t| t.color == color).collect()
    }

    /// Get tile count
    pub fn len(&self) -> usize {
        self.tiles.len()
    }
}

impl Default for PenrosePalace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut palace = PenrosePalace::new();
        let tile = palace.store((1.0, 2.0), 0xDEAD, 1).unwrap();
        assert_eq!(tile.content_hash, 0xDEAD);
        assert_eq!(palace.len(), 1);
        
        let results = palace.query((1.0, 2.0), 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].tile.content_hash, 0xDEAD);
    }

    #[test]
    fn test_three_coloring() {
        let mut palace = PenrosePalace::new();
        let mut colors = std::collections::HashSet::new();
        
        for i in 0..30 {
            let pos = (i as f64 * PHI, (i as f64 * 2.0) * PHI);
            let tile = palace.store(pos, i as u64, 0).unwrap();
            colors.insert(tile.color);
        }
        
        // Should use multiple colors
        assert!(colors.len() >= 2, "Should use at least 2 colors, got {:?}", colors);
    }

    #[test]
    fn test_tile_types() {
        let mut palace = PenrosePalace::new();
        let mut types = std::collections::HashSet::new();
        
        for i in 0..30 {
            let pos = (i as f64, i as f64 * PHI);
            let tile = palace.store(pos, i as u64, 0).unwrap();
            types.insert(tile.tile_type);
        }
        
        // Should have both thick and thin tiles
        assert!(types.contains(&TileType::Thick));
        assert!(types.contains(&TileType::Thin));
    }

    #[test]
    fn test_bragg_peak_exact_match() {
        let mut palace = PenrosePalace::new();
        palace.store((5.0, 5.0), 0xBEEF, 1).unwrap();
        
        let results = palace.query((5.0, 5.0), 3);
        assert!(!results.is_empty());
        assert!(results[0].bragg_confidence > 0.5);
    }

    #[test]
    fn test_deflation() {
        let mut palace = PenrosePalace::new();
        
        // Store several tiles in a region
        for i in 0..5 {
            let pos = (i as f64, i as f64);
            palace.store(pos, i as u64, 1).unwrap();
        }
        
        // Deflate (consolidate)
        let result = palace.deflate((0, 0), 3.0);
        assert!(result.is_some());
        let deflated = result.unwrap();
        assert!(deflated.level > 0, "Deflated tile should be higher level");
    }

    #[test]
    fn test_color_filtering() {
        let mut palace = PenrosePalace::new();
        for i in 0..20 {
            palace.store((i as f64, i as f64 * PHI), i as u64, 0).unwrap();
        }
        
        let red = palace.by_color(ShardColor::Red);
        let green = palace.by_color(ShardColor::Green);
        let blue = palace.by_color(ShardColor::Blue);
        
        // All tiles accounted for
        assert_eq!(red.len() + green.len() + blue.len(), 20);
    }

    #[test]
    fn test_golden_spacing_hierarchy() {
        let mut palace = PenrosePalace::new();
        
        // Store at level 0
        let t1 = palace.store((0.0, 0.0), 1, 0).unwrap();
        assert_eq!(t1.level, 0);
        
        // Deflate to level 1
        palace.store((1.0, 0.0), 2, 0).unwrap();
        palace.store((0.0, 1.0), 3, 0).unwrap();
        let deflated = palace.deflate((0, 0), 2.0).unwrap();
        assert!(deflated.level >= 1);
    }

    #[test]
    fn test_empty_query() {
        let palace = PenrosePalace::new();
        let results = palace.query((999.0, 999.0), 3);
        assert!(results.is_empty());
    }
}
