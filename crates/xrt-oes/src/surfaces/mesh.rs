//! Surface from 3D mesh model.
//!
//! Represents a surface loaded from vertex/face data (e.g., STL file).
//! Uses simple nearest-vertex lookup for surface height and normal.

use crate::surface::Surface;

/// A surface defined by a triangulated mesh.
///
/// Each vertex has a position (x, y, z) and the surface normal at each
/// query point is interpolated from the nearest triangle.
#[derive(Debug, Clone)]
pub struct MeshSurface {
    /// Vertex positions: [(x, y, z), ...]
    pub vertices: Vec<[f64; 3]>,
    /// Triangle indices: [(v0, v1, v2), ...]
    pub faces: Vec<[usize; 3]>,
    /// Precomputed face normals
    face_normals: Vec<[f64; 3]>,
    /// Precomputed face centroids (for nearest lookup)
    face_centroids: Vec<[f64; 2]>,
}

impl MeshSurface {
    /// Create a mesh surface from vertices and face indices.
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        let face_normals: Vec<[f64; 3]> = faces.iter().map(|&[i0, i1, i2]| {
            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            if norm > 1e-30 {
                [nx / norm, ny / norm, nz / norm]
            } else {
                [0.0, 0.0, 1.0]
            }
        }).collect();

        let face_centroids: Vec<[f64; 2]> = faces.iter().map(|&[i0, i1, i2]| {
            let cx = (vertices[i0][0] + vertices[i1][0] + vertices[i2][0]) / 3.0;
            let cy = (vertices[i0][1] + vertices[i1][1] + vertices[i2][1]) / 3.0;
            [cx, cy]
        }).collect();

        Self { vertices, faces, face_normals, face_centroids }
    }

    /// Find the nearest face to (x, y) in the xy-plane.
    fn nearest_face(&self, x: f64, y: f64) -> usize {
        let mut best = 0;
        let mut best_d2 = f64::INFINITY;
        for (i, c) in self.face_centroids.iter().enumerate() {
            let dx = x - c[0];
            let dy = y - c[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        best
    }
}

impl Surface for MeshSurface {
    fn local_z(&self, x: f64, y: f64) -> f64 {
        let fi = self.nearest_face(x, y);
        let [i0, i1, i2] = self.faces[fi];
        // Average z of the nearest face vertices
        (self.vertices[i0][2] + self.vertices[i1][2] + self.vertices[i2][2]) / 3.0
    }

    fn local_n(&self, x: f64, y: f64) -> [f64; 3] {
        let fi = self.nearest_face(x, y);
        self.face_normals[fi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_flat_quad() {
        // Two triangles forming a flat quad at z=0
        let verts = vec![
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 10.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let m = MeshSurface::new(verts, faces);
        assert!(m.local_z(0.0, 0.0).abs() < 1e-12);
        let n = m.local_n(0.0, 0.0);
        assert!((n[2].abs() - 1.0).abs() < 1e-10, "flat mesh normal should be ±z");
    }

    #[test]
    fn mesh_tilted() {
        // Triangle tilted in y
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 1.0], // tilted up
        ];
        let faces = vec![[0, 1, 2]];
        let m = MeshSurface::new(verts, faces);
        let n = m.local_n(3.0, 3.0);
        // Normal should have a ny component (tilted)
        assert!(n[1].abs() > 0.01, "tilted mesh should have ny component");
    }
}
