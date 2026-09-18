// Kirchhoff diffraction integral — WGSL compute shader
//
// Phase reduction for f32 precision:
// Instead of computing sin(k*path) where k*path can be ~5e10,
// we compute sin(k*delta_path) where delta_path = path - ref_path.
//
// To avoid catastrophic cancellation in `path - ref_path` (both ~1000mm),
// we compute delta_path via the algebraic identity:
//   path² - ref² = (path-ref)(path+ref)
//   delta_path = (path² - ref²) / (path + ref_path)
// where path² - ref² is computed from expanded terms involving
// small ray-centroid offsets.

struct Ray {
    x: f32, y: f32, z: f32,
    nx: f32, ny: f32, nz: f32,
    nl: f32,
    energy: f32,
    es_re: f32, es_im: f32,
    ep_re: f32, ep_im: f32,
};

struct Pixel {
    x: f32, y: f32, z: f32,
    _pad: f32,
};

struct PixelResult {
    es_re: f32, es_im: f32,
    ep_re: f32, ep_im: f32,
};

struct Params {
    n_rays: u32,
    n_pixels: u32,
    chbar_inv_1e7: f32,
    _pad: u32,
    // Ray centroid for phase reduction
    cx: f32,
    cy: f32,
    cz: f32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> pixels: array<Pixel>;
@group(0) @binding(2) var<storage, read_write> results: array<PixelResult>;
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

fn cmul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    if (pixel_idx >= params.n_pixels) {
        return;
    }

    let pixel = pixels[pixel_idx];

    // Precompute ref_path = distance(pixel, centroid)
    let ref_dx = pixel.x - params.cx;
    let ref_dy = pixel.y - params.cy;
    let ref_dz = pixel.z - params.cz;
    let ref_path = sqrt(ref_dx * ref_dx + ref_dy * ref_dy + ref_dz * ref_dz);

    var es_re: f32 = 0.0;
    var es_im: f32 = 0.0;
    var ep_re: f32 = 0.0;
    var ep_im: f32 = 0.0;

    for (var j: u32 = 0u; j < params.n_rays; j = j + 1u) {
        let ray = rays[j];

        let dx = pixel.x - ray.x;
        let dy = pixel.y - ray.y;
        let dz = pixel.z - ray.z;
        let path_sq = dx * dx + dy * dy + dz * dz;
        let path = sqrt(path_sq);

        if (path < 1e-20) {
            continue;
        }

        let inv_path = 1.0 / path;

        let ns = (ray.nx * dx + ray.ny * dy + ray.nz * dz) * inv_path;

        let k = ray.energy * params.chbar_inv_1e7;

        // Compute delta_path = path - ref_path using algebraic identity
        // to avoid catastrophic cancellation in f32.
        //
        // path² - ref² = Σ[(px-rx)² - (px-cx)²]
        //              = Σ[(cx-rx)(2*px - cx - rx)]
        //              = Σ[δ * (2*p - c - r)]
        // where δ = cx - rx (small, ~0.1mm)
        let ox = params.cx - ray.x;  // centroid - ray (small offset)
        let oy = params.cy - ray.y;
        let oz = params.cz - ray.z;
        let diff_sq = ox * (2.0 * pixel.x - params.cx - ray.x)
                    + oy * (2.0 * pixel.y - params.cy - ray.y)
                    + oz * (2.0 * pixel.z - params.cz - ray.z);
        // delta_path = diff_sq / (path + ref_path)
        let delta_path = diff_sq / (path + ref_path);

        let delta_phase = k * delta_path;
        let cos_phase = cos(delta_phase);
        let sin_phase = sin(delta_phase);

        let obliquity = ray.nl + ns;
        let amp = k / (4.0 * PI) * obliquity * inv_path;

        // i * amp * exp(i*delta_phase) = amp * (-sin + i*cos)
        let u_re = -amp * sin_phase;
        let u_im = amp * cos_phase;

        let es_contrib = cmul(ray.es_re, ray.es_im, u_re, u_im);
        es_re += es_contrib.x;
        es_im += es_contrib.y;

        let ep_contrib = cmul(ray.ep_re, ray.ep_im, u_re, u_im);
        ep_re += ep_contrib.x;
        ep_im += ep_contrib.y;
    }

    results[pixel_idx].es_re = es_re;
    results[pixel_idx].es_im = es_im;
    results[pixel_idx].ep_re = ep_re;
    results[pixel_idx].ep_im = ep_im;
}
