// Kirchhoff diffraction integral — WGSL compute shader
//
// Each workgroup processes one pixel, summing over all rays.
// Uses complex arithmetic for field amplitudes.

struct Ray {
    // Position on OE [mm]
    x: f32,
    y: f32,
    z: f32,
    // Surface normal
    nx: f32,
    ny: f32,
    nz: f32,
    // Obliquity factor
    nl: f32,
    // Energy [eV]
    energy: f32,
    // Field amplitudes (real, imag)
    es_re: f32,
    es_im: f32,
    ep_re: f32,
    ep_im: f32,
};

struct Pixel {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
};

struct PixelResult {
    es_re: f32,
    es_im: f32,
    ep_re: f32,
    ep_im: f32,
};

struct Params {
    n_rays: u32,
    n_pixels: u32,
    chbar_inv_1e7: f32, // 1.0 / (CHBAR) * 1e7
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> pixels: array<Pixel>;
@group(0) @binding(2) var<storage, read_write> results: array<PixelResult>;
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
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
    var es_re: f32 = 0.0;
    var es_im: f32 = 0.0;
    var ep_re: f32 = 0.0;
    var ep_im: f32 = 0.0;

    for (var j: u32 = 0u; j < params.n_rays; j = j + 1u) {
        let ray = rays[j];

        let dx = pixel.x - ray.x;
        let dy = pixel.y - ray.y;
        let dz = pixel.z - ray.z;
        let path = sqrt(dx * dx + dy * dy + dz * dz);

        if (path < 1e-20) {
            continue;
        }

        let inv_path = 1.0 / path;

        // ns = dot(normal, (pixel - ray)) / path
        let ns = (ray.nx * dx + ray.ny * dy + ray.nz * dz) * inv_path;

        // k = E / CHBAR * 1e7 [mm⁻¹]
        let k = ray.energy * params.chbar_inv_1e7;

        // Phase
        let phase = k * path;
        let cos_phase = cos(phase);
        let sin_phase = sin(phase);

        // U = i * k/(4π) * (nl + ns) * exp(ikr) / path
        let obliquity = ray.nl + ns;
        let amp = k / (4.0 * PI) * obliquity * inv_path;

        // i * amp * exp(ikr) = amp * (-sin + i*cos)
        let u_re = -amp * sin_phase;
        let u_im = amp * cos_phase;

        // Es_result += Es * U
        let es_contrib = cmul(ray.es_re, ray.es_im, u_re, u_im);
        es_re += es_contrib.x;
        es_im += es_contrib.y;

        // Ep_result += Ep * U
        let ep_contrib = cmul(ray.ep_re, ray.ep_im, u_re, u_im);
        ep_re += ep_contrib.x;
        ep_im += ep_contrib.y;
    }

    results[pixel_idx].es_re = es_re;
    results[pixel_idx].es_im = es_im;
    results[pixel_idx].ep_re = ep_re;
    results[pixel_idx].ep_im = ep_im;
}
