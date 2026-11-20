//! ParametricOpticalElement: OE for parametric surfaces (elliptical, parabolical, hyperbolic).
//!
//! Similar to OpticalElement but uses ParametricSurface trait instead of Surface trait.
//! Intersection uses find_intersection_parametric, and coordinates are converted
//! between Cartesian and parametric forms.

use rayon::prelude::*;

use crate::core::beam::{Beam, RayState};
use crate::core::transforms::{RotationParams, rotate_beam};
use crate::math::rootfind::RootFindConfig;

use crate::oes::aperture::Aperture;
use crate::oes::bracketing::bracket_ray;
use crate::oes::deflection;
use crate::oes::intersection::find_intersection_parametric;
use crate::oes::oe::OeParams;
use crate::oes::reflect::{DeflectionMode, RayResult};
use crate::oes::surface::ParametricSurface;

/// An optical element with a parametric surface.
pub struct ParametricOpticalElement<P: ParametricSurface> {
    pub surface: P,
    pub params: OeParams,
}

impl<P: ParametricSurface> ParametricOpticalElement<P> {
    pub fn new(surface: P, params: OeParams) -> Self {
        Self { surface, params }
    }

    /// Reflect a beam off this parametric optical element.
    pub fn reflect(&self, beam: &mut Beam) -> Vec<RayResult> {
        let good: Vec<usize> = (0..beam.nrays()).filter(|&i| beam.state[i] > 0).collect();

        if good.is_empty() {
            return vec![];
        }

        // Transform to local coordinates
        let rotation = RotationParams::default_sequence(
            -self.params.pitch,
            -self.params.roll,
            -self.params.yaw,
        );

        for &i in &good {
            beam.x[i] -= self.params.center[0];
            beam.y[i] -= self.params.center[1];
            beam.z[i] -= self.params.center[2];
        }
        rotate_beam(beam, Some(&good), &rotation, false, false);

        // Core reflection with parametric intersection
        let phys_x = Some(self.params.aperture.phys_x);
        let phys_y = Some(self.params.aperture.phys_y);

        let results = self.reflect_parametric_local(
            beam,
            &good,
            &self.params.aperture,
            self.params.invert_normal,
            self.params.mode,
            phys_x,
            phys_y,
            &self.params.config,
        );

        // Apply results
        crate::oes::reflect::apply_results(beam, &good, &results);

        // Store parametric coordinates
        let parametric = beam.ensure_parametric();
        for (&i, result) in good.iter().zip(results.iter()) {
            if result.state == RayState::Good {
                // For parametric surfaces, intersection returns (s, phi, r) in (x, y, z)
                // of the RayResult after parametric conversion
                let (s, phi, r) = self.surface.xyz_to_param(result.x, result.y, result.z);
                parametric.s[i] = s;
                parametric.phi[i] = phi;
                parametric.r[i] = r;
            }
        }

        // Rotate back to global (inverse rotation — reversed sequence)
        let inv_rotation =
            RotationParams::inverse_sequence(self.params.pitch, self.params.roll, self.params.yaw);

        let good_after: Vec<usize> = good
            .iter()
            .zip(results.iter())
            .filter(|(_, r)| r.state == RayState::Good || r.state == RayState::Over)
            .map(|(&i, _)| i)
            .collect();

        rotate_beam(beam, Some(&good_after), &inv_rotation, false, false);

        for &i in &good_after {
            beam.x[i] += self.params.center[0];
            beam.y[i] += self.params.center[1];
            beam.z[i] += self.params.center[2];
        }

        results
    }

    /// Core parametric reflection pipeline.
    fn reflect_parametric_local(
        &self,
        beam: &Beam,
        good: &[usize],
        aperture: &Aperture,
        invert_normal: i32,
        mode: DeflectionMode,
        phys_x: Option<[f64; 2]>,
        phys_y: Option<[f64; 2]>,
        config: &RootFindConfig,
    ) -> Vec<RayResult> {
        good.par_iter()
            .map(|&i| {
                let x0 = beam.x[i];
                let y0 = beam.y[i];
                let z0 = beam.z[i];
                let a = beam.a[i];
                let b = beam.b[i];
                let c = beam.c[i];

                let (t_min, t_max) = bracket_ray(x0, y0, z0, a, b, c, phys_x, phys_y);

                let isect = find_intersection_parametric(
                    &self.surface,
                    t_min,
                    t_max,
                    x0,
                    y0,
                    z0,
                    a,
                    b,
                    c,
                    invert_normal,
                    config,
                );

                if !isect.converged {
                    return RayResult {
                        state: RayState::Out,
                        x: x0,
                        y: y0,
                        z: z0,
                        a,
                        b,
                        c,
                        path_delta: 0.0,
                        nx: 0.0,
                        ny: 0.0,
                        nz: 1.0,
                        beam_in_dot_normal: 0.0,
                    };
                }

                // For parametric surfaces, isect returns (s, phi, r) in (x, y, z)
                // Convert back to Cartesian for the actual intersection point
                let (ix, iy, iz) = self.surface.param_to_xyz(isect.x, isect.y, isect.z);

                // Aperture check in Cartesian
                let state = aperture.classify_ray(ix, iy);
                if state != RayState::Good {
                    return RayResult {
                        state,
                        x: ix,
                        y: iy,
                        z: iz,
                        a,
                        b,
                        c,
                        path_delta: isect.t,
                        nx: 0.0,
                        ny: 0.0,
                        nz: 1.0,
                        beam_in_dot_normal: 0.0,
                    };
                }

                // Surface normal in parametric coordinates, then expressed in Cartesian
                let [nx, ny, nz] = self.surface.local_n(isect.x, isect.y);

                let bidn = a * nx + b * ny + c * nz;

                let (a_out, b_out, c_out) = match mode {
                    DeflectionMode::Reflect => {
                        deflection::reflect_specular(a, b, c, nx, ny, nz, bidn)
                    }
                    DeflectionMode::PassThrough => (a, b, c),
                    _ => (a, b, c),
                };

                RayResult {
                    state: RayState::Good,
                    x: ix,
                    y: iy,
                    z: iz,
                    a: a_out,
                    b: b_out,
                    c: c_out,
                    path_delta: isect.t,
                    nx,
                    ny,
                    nz,
                    beam_in_dot_normal: bidn,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oes::beamline::OeParamsBuilder;
    use crate::oes::surfaces::elliptical::EllipticalSurface;

    #[test]
    fn elliptical_oe_reflect() {
        let surface = EllipticalSurface::from_pq(10_000.0, 5_000.0, 0.003);
        let oe = ParametricOpticalElement::new(
            surface,
            OeParamsBuilder::new()
                .center(0.0, 0.0, 0.0)
                .pitch(0.003) // match grazing angle
                .build(),
        );

        let mut beam = Beam::new(5);
        for i in 0..5 {
            beam.y[i] = -1000.0;
            beam.a[i] = 0.0;
            beam.b[i] = 1.0;
            beam.c[i] = 0.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }

        let results = oe.reflect(&mut beam);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn parametric_oe_stores_coords() {
        let surface = EllipticalSurface::new(100.0, 50.0, 0.0);
        let oe = ParametricOpticalElement::new(surface, OeParamsBuilder::new().pitch(0.01).build());

        let mut beam = Beam::new(3);
        for i in 0..3 {
            beam.y[i] = -100.0;
            beam.b[i] = 1.0;
            beam.state[i] = RayState::Good as i32;
            beam.e[i] = 10000.0;
        }

        let _results = oe.reflect(&mut beam);
        // After reflection, parametric coords should be allocated
        let parametric = beam.parametric().expect("reflect allocates them");
        assert_eq!(parametric.s.len(), beam.nrays());
        assert_eq!(parametric.phi.len(), beam.nrays());
        assert_eq!(parametric.r.len(), beam.nrays());
    }

    const P: f64 = 10_000.0;
    const Q: f64 = 300.0;
    const THETA: f64 = 0.003;

    /// n×n rays leaving (x0, 0, z0) within ±`half_angle` in x' and z'.
    fn fan(n: usize, half_angle: f64, x0: f64, z0: f64) -> Beam {
        let mut beam = Beam::new(n * n);
        let step = |k: usize| half_angle * (2.0 * k as f64 / (n - 1) as f64 - 1.0);
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                let (xp, zp) = (step(i), step(j));
                let norm = (1.0 + xp * xp + zp * zp).sqrt();
                beam.x[k] = x0;
                beam.z[k] = z0;
                beam.a[k] = xp / norm;
                beam.b[k] = 1.0 / norm;
                beam.c[k] = zp / norm;
                beam.state[k] = RayState::Good as i32;
            }
        }
        beam
    }

    /// Collimated n×n grid within ±`half` in x and z, going along +y.
    fn grid(n: usize, half: f64) -> Beam {
        let mut beam = Beam::new(n * n);
        let step = |k: usize| half * (2.0 * k as f64 / (n - 1) as f64 - 1.0);
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                beam.x[k] = step(i);
                beam.z[k] = step(j);
                beam.state[k] = RayState::Good as i32;
            }
        }
        beam
    }

    /// A 200 mm long mirror at y = P. The length matters: past the conic's
    /// ends r(s) continues as |·|, so an unbounded search finds no sign change.
    fn mirror_at_p<S: ParametricSurface>(surface: S) -> ParametricOpticalElement<S> {
        let mut params = OeParamsBuilder::new()
            .center(0.0, P, 0.0)
            .pitch(THETA)
            .build();
        params.aperture.phys_y = [-100.0, 100.0];
        ParametricOpticalElement::new(surface, params)
    }

    /// Largest distance from (x, z) of the good rays' lines at plane y = `y`.
    /// Lines run both ways, so a virtual focus behind the mirror counts too.
    fn max_miss_at_plane(beam: &Beam, y: f64, x: f64, z: f64) -> f64 {
        let mut miss = 0.0_f64;
        for i in 0..beam.nrays() {
            assert_eq!(beam.state[i], RayState::Good as i32, "ray {i} lost");
            let t = (y - beam.y[i]) / beam.b[i];
            let d = (beam.x[i] + t * beam.a[i] - x).hypot(beam.z[i] + t * beam.c[i] - z);
            assert!(d.is_finite(), "ray {i} has a non-finite path");
            miss = miss.max(d);
        }
        miss
    }

    #[test]
    fn ellipse_images_source_onto_second_focus() {
        let mut beam = fan(11, 1e-5, 0.0, 0.0);
        mirror_at_p(EllipticalSurface::from_pq(P, Q, THETA)).reflect(&mut beam);
        let (sin2, cos2) = (2.0 * THETA).sin_cos();
        let miss = max_miss_at_plane(&beam, P + Q * cos2, 0.0, Q * sin2);
        assert!(miss < 1e-6, "focus miss {miss} mm");
    }

    #[test]
    fn hyperbola_images_source_onto_virtual_focus() {
        use crate::oes::surfaces::hyperbolic::HyperbolicSurface;
        let mut beam = fan(11, 1e-5, 0.0, 0.0);
        mirror_at_p(HyperbolicSurface::from_pq(P, Q, THETA)).reflect(&mut beam);
        let (sin2, cos2) = (2.0 * THETA).sin_cos();
        let miss = max_miss_at_plane(&beam, P - Q * cos2, 0.0, -Q * sin2);
        assert!(miss < 1e-6, "virtual focus miss {miss} mm");
    }

    #[test]
    fn paraboloid_from_p_collimates_point_source() {
        use crate::oes::surfaces::parabolical::ParabolicalSurface;
        let mut beam = fan(11, 1e-5, 0.0, 0.0);
        mirror_at_p(ParabolicalSurface::from_p(P, THETA)).reflect(&mut beam);
        let sin2 = (2.0 * THETA).sin();
        for i in 0..beam.nrays() {
            assert_eq!(beam.state[i], RayState::Good as i32, "ray {i} lost");
            assert!(beam.a[i].abs() < 1e-10, "ray {i}: a = {}", beam.a[i]);
            assert!(
                (beam.c[i] - sin2).abs() < 1e-10,
                "ray {i}: c = {}",
                beam.c[i]
            );
        }
    }

    #[test]
    fn paraboloid_from_q_focuses_parallel_beam() {
        use crate::oes::surfaces::parabolical::ParabolicalSurface;
        let mut beam = grid(11, 0.05);
        mirror_at_p(ParabolicalSurface::from_q(Q, THETA)).reflect(&mut beam);
        let (sin2, cos2) = (2.0 * THETA).sin_cos();
        let miss = max_miss_at_plane(&beam, P + Q * cos2, 0.0, Q * sin2);
        assert!(miss < 1e-6, "focus miss {miss} mm");
    }
}
