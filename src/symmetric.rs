//! Symmetric mode handling for sparrow.
//!
//! In symmetric mode, items are placed only in the left half of the container,
//! and their mirror positions on the right half are automatically considered
//! for collision detection.

use jagua_rs::geometry::{DTransformation, Transformation};
use jagua_rs::geometry::primitives::Rect;
use std::f32::consts::PI;

/// Holds configuration for symmetric packing mode
#[derive(Debug, Clone, Copy)]
pub struct SymmetricConfig {
    /// The x-coordinate of the symmetry axis (usually strip_width / 2)
    pub axis_x: f32,
    /// Whether symmetric mode is enabled
    pub enabled: bool,
}

impl SymmetricConfig {
    pub fn new(strip_width: f32, enabled: bool) -> Self {
        Self {
            axis_x: strip_width / 2.0,
            enabled,
        }
    }

    pub fn disabled() -> Self {
        Self {
            axis_x: 0.0,
            enabled: false,
        }
    }

    /// Update the axis position when strip width changes
    pub fn update_axis(&mut self, new_strip_width: f32) {
        self.axis_x = new_strip_width / 2.0;
    }
}

/// Compute the mirror transformation of a given transformation around the symmetry axis.
///
/// For a point at (x, y) with rotation r, its mirror around axis_x is:
/// - x' = 2 * axis_x - x
/// - y' = y (unchanged)
/// - r' = PI - r (mirror the rotation)
pub fn mirror_transformation(dt: DTransformation, axis_x: f32) -> DTransformation {
    let (x, y) = dt.translation();
    let r = dt.rotation();

    // Mirror x coordinate
    let mirror_x = 2.0 * axis_x - x;

    // Mirror rotation: flip around vertical axis
    // If original rotation is r, mirror is PI - r (or equivalently -r with a flip)
    let mirror_r = PI - r;

    DTransformation::new(mirror_r, (mirror_x, y))
}

/// Get the valid sampling bounding box for symmetric mode.
/// In symmetric mode, we only sample from the left half of the container.
pub fn get_symmetric_sample_bbox(container_bbox: Rect, axis_x: f32) -> Option<Rect> {
    Rect::try_new(
        container_bbox.x_min,
        container_bbox.y_min,
        axis_x,  // Only sample up to the axis
        container_bbox.y_max,
    ).ok()
}

/// Check if a transformation is in the valid region for symmetric mode (left half).
pub fn is_in_valid_region(dt: DTransformation, axis_x: f32) -> bool {
    dt.translation().0 <= axis_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_transformation() {
        let dt = DTransformation::new(0.0, (1.0, 2.0));
        let axis_x = 5.0;

        let mirrored = mirror_transformation(dt, axis_x);

        // x should be mirrored: 2 * 5.0 - 1.0 = 9.0
        assert!((mirrored.translation().0 - 9.0).abs() < 1e-6);
        // y should be unchanged
        assert!((mirrored.translation().1 - 2.0).abs() < 1e-6);
        // rotation should be PI - 0 = PI
        assert!((mirrored.rotation() - PI).abs() < 1e-6);
    }
}
