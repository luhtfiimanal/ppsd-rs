//! Seismic PPSD (Probabilistic Power Spectral Density) computation.
//!
//! A pure-Rust implementation of the McNamara & Buland (2004) PPSD algorithm,
//! matching [ObsPy's PPSD](https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.html)
//! output within 0.5 dB tolerance. Uses [`stationxml_rs::Response`] directly
//! for instrument response evaluation — no intermediate types needed.
//!
//! # Algorithm
//!
//! The PPSD pipeline for each time segment:
//!
//! 1. **Welch's method** — overlap-averaged periodogram with cosine taper and linear detrend
//! 2. **Instrument response removal** — evaluates PAZ, FIR, and Coefficients stages from
//!    [`stationxml_rs::Response`], then divides PSD by |H(f)|²
//! 3. **Velocity→acceleration correction** — multiplies by ω² = (2πf)²
//! 4. **dB conversion** — 10 × log₁₀(PSD)
//! 5. **Period binning** — averages dB values into octave-spaced period bins
//!
//! # Quick Start
//!
//! ```no_run
//! use ppsd_rs::process_segment;
//!
//! // Load inventory (FDSN StationXML or SC3ML — auto-detected)
//! let inv = stationxml_rs::read_from_file("IA.JAGI.xml").unwrap();
//! let channel = &inv.networks[0].stations[0].channels[0];
//! let response = channel.response.as_ref().unwrap();
//! let sample_rate = channel.sample_rate;
//!
//! // Compute PSD for one segment
//! # let samples = vec![0.0_f64; 72000];
//! # let nfft = 4096;
//! # let nlap = 2048;
//! # let psd_periods = vec![1.0];
//! # let bin_left = vec![0.5];
//! # let bin_right = vec![1.5];
//! let result = process_segment(
//!     &samples, sample_rate, nfft, nlap,
//!     response, &psd_periods, &bin_left, &bin_right,
//! ).unwrap();
//! ```
//!
//! # Response Evaluation
//!
//! [`eval_response`] walks all stages in a [`stationxml_rs::Response`] and computes
//! the combined |H(f)|²:
//!
//! | Stage type | Type | Method |
//! |------------|------|--------|
//! | Sensor | [`stationxml_rs::PolesZeros`] | Laplace transfer function × gain |
//! | ADC | [`stationxml_rs::Coefficients`] (Digital) | DTFT of numerators × gain |
//! | FIR filter | [`stationxml_rs::FIR`] | DTFT with DC normalization × gain |
//! | Gain-only | (none of above) | gain² |
//!
//! FIR coefficients are DC-normalized to match evalresp behavior.
//! [`stationxml_rs::Symmetry`] expansion is handled automatically
//! (`None` = all coefficients, `Even` = mirror, `Odd` = mirror with negation).

#![warn(missing_docs)]

use num_complex::Complex;
use realfft::RealFftPlanner;
use stationxml_rs::{
    CfTransferFunction, FIR, PzTransferFunction, Response, ResponseStage, Symmetry,
};
use std::f64::consts::PI;

// ─── Error ─────────────────────────────────────────────────────────

/// Errors that can occur during PSD computation.
#[derive(Debug, thiserror::Error)]
pub enum PpsdError {
    /// Response has no stages to evaluate.
    #[error("response has no stages")]
    EmptyResponse,

    /// Unsupported transfer function type encountered.
    #[error("unsupported transfer function type in stage {stage}: {detail}")]
    UnsupportedStage {
        /// Stage number (1-based).
        stage: u32,
        /// Description of what is unsupported.
        detail: String,
    },

    /// A digital stage has no decimation info (needed for DTFT sample rate).
    #[error("stage {0} has FIR/Coefficients but no decimation info")]
    MissingDecimation(u32),

    /// A digital stage has a non-positive or NaN sample rate.
    ///
    /// Guards against silent NaN propagation through the DTFT inner loop, which
    /// would otherwise occur when `input_sample_rate` is `0.0`, negative, or NaN
    /// (`f / 0.0 = ±Inf`, `cos/sin(±Inf) = NaN`).
    #[error("stage {0} has invalid sample rate (must be finite and > 0)")]
    InvalidSampleRate(u32),
}

/// Result type for ppsd operations.
pub type Result<T> = std::result::Result<T, PpsdError>;

// ─── Public types ──────────────────────────────────────────────────

// ─── Step 1: Cosine taper ──────────────────────────────────────────

/// Cosine taper (Tukey window) matching ObsPy's `cosine_taper(npts, p)`.
///
/// Produces a window of length `npts` where a fraction `p` of the total
/// length is tapered with a raised cosine (half on each side).
///
/// # Arguments
///
/// * `npts` — Number of points in the window.
/// * `p` — Taper fraction, 0.0 (rectangular) to 1.0 (full Hann). ObsPy default is 0.05.
///
/// # Examples
///
/// ```
/// let taper = ppsd_rs::cosine_taper(100, 0.1);
/// assert_eq!(taper.len(), 100);
/// // Edges are tapered, center is 1.0
/// assert!((taper[50] - 1.0).abs() < 1e-12);
/// ```
pub fn cosine_taper(npts: usize, p: f64) -> Vec<f64> {
    let mut taper = vec![0.0; npts];

    let frac = if p == 0.0 || p == 1.0 {
        (npts as f64 * p / 2.0) as usize
    } else {
        (npts as f64 * p / 2.0 + 0.5) as usize
    };

    let idx1 = 0usize;
    let mut idx2 = frac.saturating_sub(1);
    let mut idx3 = npts.saturating_sub(frac);
    let idx4 = npts - 1;

    if idx1 == idx2 {
        idx2 += 1;
    }
    if idx3 == idx4 {
        idx3 -= 1;
    }

    // Rising ramp
    let ramp_len = (idx2 - idx1) as f64;
    for (i, val) in taper.iter_mut().enumerate().take(idx2 + 1).skip(idx1) {
        *val = 0.5 * (1.0 - (PI * (i as f64 - idx1 as f64) / ramp_len).cos());
    }
    // Flat top
    for val in taper.iter_mut().take(idx3).skip(idx2 + 1) {
        *val = 1.0;
    }
    // Falling ramp
    let ramp_len = (idx4 - idx3) as f64;
    for (i, val) in taper.iter_mut().enumerate().take(idx4 + 1).skip(idx3) {
        *val = 0.5 * (1.0 + (PI * (idx3 as f64 - i as f64) / ramp_len).cos());
    }

    taper
}

// ─── Step 2: FFT and power spectrum ────────────────────────────────

/// Compute one-sided power spectrum via real FFT.
///
/// Returns `(freqs, power)` where `freqs` are in Hz and `power` is scaled
/// by `1/fs` (power spectral density). Non-DC, non-Nyquist bins are doubled
/// for the single-sided spectrum.
///
/// # Arguments
///
/// * `samples` — Input signal (time domain).
/// * `sample_rate` — Sampling rate in Hz.
///
/// # Returns
///
/// Tuple of `(frequencies_hz, power_spectral_density)`, each of length `N/2 + 1`.
pub fn fft_power(samples: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut input = samples.to_vec();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).unwrap();

    let n_freqs = spectrum.len(); // n/2 + 1
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut power = Vec::with_capacity(n_freqs);

    for (i, c) in spectrum.iter().enumerate() {
        freqs.push(i as f64 * sample_rate / n as f64);
        let mut p = (c.re * c.re + c.im * c.im) / (sample_rate * n as f64);
        // Double non-DC, non-Nyquist bins for single-sided spectrum
        if i > 0 && i < n_freqs - 1 {
            p *= 2.0;
        }
        power.push(p);
    }

    (freqs, power)
}

// ─── Step 3: Instrument response evaluation ────────────────────────

/// Evaluate a single PAZ stage |H(f)|² at given frequencies.
fn eval_paz(
    pz: &stationxml_rs::PolesZeros,
    stage_gain: f64,
    stage_number: u32,
    freqs: &[f64],
) -> Result<Vec<f64>> {
    if pz.pz_transfer_function_type == PzTransferFunction::DigitalZTransform {
        return Err(PpsdError::UnsupportedStage {
            stage: stage_number,
            detail: "DigitalZTransform PAZ not supported (requires sample rate)".into(),
        });
    }

    Ok(freqs
        .iter()
        .map(|&f| {
            let s = match pz.pz_transfer_function_type {
                PzTransferFunction::LaplaceRadians => Complex::new(0.0, 2.0 * PI * f),
                PzTransferFunction::LaplaceHertz => Complex::new(0.0, f),
                PzTransferFunction::DigitalZTransform => unreachable!(),
            };

            let mut h = Complex::new(1.0, 0.0);
            for z in &pz.zeros {
                h *= s - Complex::new(z.real, z.imaginary);
            }
            for p in &pz.poles {
                h /= s - Complex::new(p.real, p.imaginary);
            }
            h *= pz.normalization_factor;
            h *= stage_gain;
            h.norm_sqr() // |H(f)|²
        })
        .collect())
}

/// Compute DC-normalized DTFT |H(f)|² × gain² for a set of coefficients.
///
/// Shared by FIR and digital Coefficients stage evaluation.
fn dtft_power_normalized(
    coeffs: &[f64],
    stage_gain: f64,
    sample_rate: f64,
    freqs: &[f64],
) -> Vec<f64> {
    // DC-normalize: fold norm² into gain² so the inner loop uses raw coefficients.
    let coeff_sum: f64 = coeffs.iter().sum();
    let norm_sq = if coeff_sum.abs() > 1e-30 {
        1.0 / (coeff_sum * coeff_sum)
    } else {
        1.0
    };
    let scale = stage_gain * stage_gain * norm_sq;

    freqs
        .iter()
        .map(|&f| {
            let phase_step = -2.0 * PI * f / sample_rate;
            let mut re = 0.0;
            let mut im = 0.0;
            for (k, &coeff) in coeffs.iter().enumerate() {
                let phase = phase_step * k as f64;
                re += coeff * phase.cos();
                im += coeff * phase.sin();
            }
            (re * re + im * im) * scale
        })
        .collect()
}

/// Evaluate a FIR stage |H(f)|² via DTFT at given frequencies.
///
/// Coefficients are DC-normalized (matching evalresp behavior).
/// Symmetry expansion: Even mirrors coefficients, Odd mirrors with negation.
fn eval_fir(fir: &FIR, stage_gain: f64, sample_rate: f64, freqs: &[f64]) -> Vec<f64> {
    match fir.symmetry {
        Symmetry::None => {
            dtft_power_normalized(&fir.numerator_coefficients, stage_gain, sample_rate, freqs)
        }
        Symmetry::Even => {
            let mut full = fir.numerator_coefficients.clone();
            full.extend(fir.numerator_coefficients.iter().rev());
            dtft_power_normalized(&full, stage_gain, sample_rate, freqs)
        }
        Symmetry::Odd => {
            let mut full = fir.numerator_coefficients.clone();
            full.extend(fir.numerator_coefficients.iter().rev().map(|c| -c));
            dtft_power_normalized(&full, stage_gain, sample_rate, freqs)
        }
    }
}

/// Get sample rate for a digital stage from its Decimation info.
fn stage_sample_rate(stage: &ResponseStage) -> Option<f64> {
    stage.decimation.as_ref().map(|d| d.input_sample_rate)
}

/// Evaluate full instrument response |H(f)|² at given frequencies.
///
/// Walks all stages in [`stationxml_rs::Response::stages`], evaluating PAZ, FIR,
/// and Coefficients stages. Returns the product of all stage |H(f)|² values.
///
/// FIR coefficients are DC-normalized (divided by their sum) to match evalresp
/// behavior. Symmetry expansion is handled for `Even` and `Odd` FIR filters.
///
/// # Arguments
///
/// * `response` — Instrument response from a [`stationxml_rs::Channel`].
/// * `freqs` — Frequencies (Hz) at which to evaluate the response.
///
/// # Returns
///
/// Vector of |H(f)|² values, same length as `freqs`.
///
/// # Errors
///
/// Returns [`PpsdError`] if:
/// - Response has no stages ([`PpsdError::EmptyResponse`])
/// - A digital stage lacks decimation info ([`PpsdError::MissingDecimation`])
/// - An unsupported transfer function type is encountered ([`PpsdError::UnsupportedStage`])
///
/// # Examples
///
/// ```no_run
/// let inv = stationxml_rs::read_from_file("station.xml").unwrap();
/// let response = inv.networks[0].stations[0].channels[0]
///     .response.as_ref().unwrap();
/// let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
/// let h_sq = ppsd_rs::eval_response(response, &freqs).unwrap();
/// assert_eq!(h_sq.len(), freqs.len());
/// ```
pub fn eval_response(response: &Response, freqs: &[f64]) -> Result<Vec<f64>> {
    if response.stages.is_empty() {
        return Err(PpsdError::EmptyResponse);
    }

    let mut result = vec![1.0_f64; freqs.len()];

    for stage in &response.stages {
        let gain = stage.stage_gain.as_ref().map(|g| g.value).unwrap_or(1.0);

        let stage_resp = if let Some(pz) = &stage.poles_zeros {
            eval_paz(pz, gain, stage.number, freqs)?
        } else if let Some(fir) = &stage.fir {
            let fs = stage_sample_rate(stage).ok_or(PpsdError::MissingDecimation(stage.number))?;
            if !fs.is_finite() || fs <= 0.0 {
                return Err(PpsdError::InvalidSampleRate(stage.number));
            }
            eval_fir(fir, gain, fs, freqs)
        } else if let Some(coeffs) = &stage.coefficients {
            match coeffs.cf_transfer_function_type {
                CfTransferFunction::Digital => {
                    let fs = stage_sample_rate(stage)
                        .ok_or(PpsdError::MissingDecimation(stage.number))?;
                    if !fs.is_finite() || fs <= 0.0 {
                        return Err(PpsdError::InvalidSampleRate(stage.number));
                    }
                    dtft_power_normalized(&coeffs.numerators, gain, fs, freqs)
                }
                CfTransferFunction::AnalogRadians | CfTransferFunction::AnalogHertz => {
                    return Err(PpsdError::UnsupportedStage {
                        stage: stage.number,
                        detail: "analog Coefficients stage not supported".into(),
                    });
                }
            }
        } else {
            // Gain-only stage: multiply all by gain² directly, no allocation needed
            let gain_sq = gain * gain;
            for r in result.iter_mut() {
                *r *= gain_sq;
            }
            continue;
        };

        for (r, h) in result.iter_mut().zip(stage_resp.iter()) {
            *r *= h;
        }
    }

    Ok(result)
}

// ─── Step 4: Konno-Ohmachi smoothing ───────────────────────────────

/// Konno-Ohmachi log-frequency spectral smoothing.
///
/// Smooths a spectrum using the Konno & Ohmachi (1998) windowing function,
/// which is constant-width in log-frequency space. The smoothing kernel is
/// `W(f, fc) = [sin(b × log₁₀(f/fc)) / (b × log₁₀(f/fc))]⁴`.
///
/// # Arguments
///
/// * `freqs` — Frequency values (Hz). Entries ≤ 0 are passed through unsmoothed.
/// * `spectrum` — Spectral values at each frequency.
/// * `bandwidth` — Smoothing bandwidth parameter (typically 40.0). Higher = less smoothing.
pub fn konno_ohmachi_smooth(freqs: &[f64], spectrum: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = freqs.len();
    let mut smoothed = vec![0.0; n];

    // Precompute log10 of all frequencies to avoid N² log10 calls.
    let log_freqs: Vec<f64> = freqs.iter().map(|&f| f.log10()).collect();

    for i in 0..n {
        if freqs[i] <= 0.0 {
            smoothed[i] = spectrum[i];
            continue;
        }

        let log_fc = log_freqs[i];
        let mut wsum = 0.0;
        let mut vsum = 0.0;
        for j in 0..n {
            if freqs[j] <= 0.0 {
                continue;
            }
            let log_ratio = log_freqs[j] - log_fc;
            let w = if log_ratio.abs() < 1e-10 {
                1.0
            } else {
                let arg = bandwidth * log_ratio;
                let sinc = arg.sin() / arg;
                sinc.powi(4)
            };
            wsum += w;
            vsum += w * spectrum[j];
        }
        if wsum > 0.0 {
            smoothed[i] = vsum / wsum;
        }
    }

    smoothed
}

// ─── Step 5: Welch's method ────────────────────────────────────────

/// Compute PSD using Welch's method matching ObsPy/matplotlib's `mlab.psd`.
///
/// Applies a cosine taper (`p=0.2`), linear detrend, overlap-averaged FFT,
/// and `scale_by_freq=True` normalization. Returns one-sided PSD.
///
/// # Arguments
///
/// * `data` — Input time-series samples.
/// * `nfft` — FFT length (number of samples per sub-segment).
/// * `sample_rate` — Sampling rate in Hz.
/// * `noverlap` — Number of overlapping samples between consecutive sub-segments.
///
/// # Returns
///
/// Tuple of `(frequencies_hz, power_spectral_density)`, each of length `nfft/2 + 1`.
pub fn welch_psd(
    data: &[f64],
    nfft: usize,
    sample_rate: f64,
    noverlap: usize,
) -> (Vec<f64>, Vec<f64>) {
    let taper = cosine_taper(nfft, 0.2);
    let taper_norm_sq: f64 = taper.iter().map(|w| w * w).sum();

    let step = nfft - noverlap;
    let n_segments = if data.len() >= nfft {
        (data.len() - nfft) / step + 1
    } else {
        0
    };

    let n_freqs = nfft / 2 + 1;
    let mut power_avg = vec![0.0; n_freqs];

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut spectrum = fft.make_output_vec();
    let mut buf = vec![0.0_f64; nfft];

    for seg_idx in 0..n_segments {
        let offset = seg_idx * step;
        buf.copy_from_slice(&data[offset..offset + nfft]);
        detrend_linear(&mut buf);

        for (s, w) in buf.iter_mut().zip(taper.iter()) {
            *s *= w;
        }

        fft.process(&mut buf, &mut spectrum).unwrap();

        for (i, c) in spectrum.iter().enumerate() {
            power_avg[i] += c.re * c.re + c.im * c.im;
        }
    }

    if n_segments == 0 {
        let freqs = (0..n_freqs)
            .map(|i| i as f64 * sample_rate / nfft as f64)
            .collect();
        return (freqs, power_avg);
    }

    let scale = sample_rate * taper_norm_sq * n_segments as f64;
    for (i, p) in power_avg.iter_mut().enumerate() {
        *p /= scale;
        if i > 0 && i < n_freqs - 1 {
            *p *= 2.0;
        }
    }

    let freqs = (0..n_freqs)
        .map(|i| i as f64 * sample_rate / nfft as f64)
        .collect();

    (freqs, power_avg)
}

/// Linear detrend (least-squares line removal), matching `mlab.detrend_linear`.
fn detrend_linear(data: &mut [f64]) {
    let n = data.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        let mean = sy / n;
        for d in data.iter_mut() {
            *d -= mean;
        }
        return;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    for (i, d) in data.iter_mut().enumerate() {
        *d -= intercept + slope * i as f64;
    }
}

// ─── Step 6: Period binning ────────────────────────────────────────

/// Bin PSD values into period bins by averaging (ObsPy style).
///
/// For each bin defined by `[bin_left[i], bin_right[i]]`, averages all `spec_db`
/// values whose corresponding `psd_periods` fall within the bin. Returns `None`
/// for bins with no matching periods.
///
/// # Arguments
///
/// * `psd_periods` — Period values (seconds) from the Welch FFT, in descending order.
/// * `spec_db` — PSD values in dB at each period.
/// * `bin_left` — Left edges of the period bins (seconds).
/// * `bin_right` — Right edges of the period bins (seconds).
pub fn period_bin_average(
    psd_periods: &[f64],
    spec_db: &[f64],
    bin_left: &[f64],
    bin_right: &[f64],
) -> Vec<Option<f64>> {
    bin_left
        .iter()
        .zip(bin_right.iter())
        .map(|(&left, &right)| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for (j, &period) in psd_periods.iter().enumerate() {
                if period >= left && period <= right {
                    sum += spec_db[j];
                    count += 1;
                }
            }
            if count > 0 {
                Some(sum / count as f64)
            } else {
                None
            }
        })
        .collect()
}

/// Compute period binning edges matching ObsPy `PPSD._setup_period_binning`.
///
/// Generates octave-spaced bins covering the range of `psd_periods`.
///
/// # Arguments
///
/// * `psd_periods` — Period values (seconds) from the Welch FFT.
/// * `period_step_octaves` — Step size between bin centers in octaves (ObsPy default: 0.04).
/// * `period_smoothing_width_octaves` — Width of each bin in octaves (ObsPy default: 0.3).
///
/// # Returns
///
/// Tuple of `(left_edges, right_edges, centers)` — all in seconds.
pub fn setup_period_binning(
    psd_periods: &[f64],
    period_step_octaves: f64,
    period_smoothing_width_octaves: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let step_factor = 2.0_f64.powf(period_step_octaves);
    let smooth_factor = 2.0_f64.powf(period_smoothing_width_octaves);

    let period_min = psd_periods.iter().cloned().reduce(f64::min).unwrap();
    let period_max = psd_periods.iter().cloned().reduce(f64::max).unwrap();

    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    let mut centers = Vec::new();

    let mut per_left = period_min / smooth_factor.sqrt();
    loop {
        let per_right = per_left * smooth_factor;
        let per_center = (per_left * per_right).sqrt();

        if per_center > period_max {
            break;
        }

        if per_right > period_min && per_left < period_max {
            lefts.push(per_left);
            rights.push(per_right);
            centers.push(per_center);
        }

        per_left *= step_factor;
    }

    (lefts, rights, centers)
}

// ─── Step 7: Full PSD pipeline ─────────────────────────────────────

/// Process one PPSD segment, replicating ObsPy's `PPSD.__process` exactly.
///
/// Performs the full pipeline:
/// 1. **Welch PSD** — `mlab.psd` with cosine taper and linear detrend
/// 2. **Reorder** — skip DC bin, reverse to period order
/// 3. **Response removal** — divide by |H(f)|² and multiply by ω² (velocity→acceleration)
/// 4. **dB conversion** — 10 × log₁₀(PSD)
/// 5. **Period bin averaging** — octave-spaced bins
///
/// Takes [`stationxml_rs::Response`] directly — no intermediate types needed.
///
/// # Arguments
///
/// * `segment` — Time-domain samples for one PPSD segment.
/// * `sample_rate` — Sampling rate in Hz.
/// * `nfft` — FFT length for Welch's method.
/// * `nlap` — Overlap samples for Welch's method.
/// * `response` — Instrument response from [`stationxml_rs::Channel`].
/// * `psd_periods` — Period values (seconds) from the Welch FFT.
/// * `bin_left` — Left edges of period bins (seconds).
/// * `bin_right` — Right edges of period bins (seconds).
///
/// # Returns
///
/// Vector of `Option<f64>` — dB values per period bin. `None` for empty bins.
///
/// # Errors
///
/// Returns [`PpsdError`] if response evaluation fails (see [`eval_response`]).
#[allow(clippy::too_many_arguments)]
pub fn process_segment(
    segment: &[f64],
    sample_rate: f64,
    nfft: usize,
    nlap: usize,
    response: &Response,
    psd_periods: &[f64],
    bin_left: &[f64],
    bin_right: &[f64],
) -> Result<Vec<Option<f64>>> {
    // 1. Welch PSD
    let (freqs, power) = welch_psd(segment, nfft, sample_rate, nlap);

    // 2. Skip DC bin (index 0), reverse to period order
    let spec: Vec<f64> = power[1..].iter().rev().copied().collect();

    // 3. Instrument response removal + 4. dB conversion (fused)
    let resp_power = eval_response(response, &freqs[1..])?;
    let resp_power_rev: Vec<f64> = resp_power.iter().rev().copied().collect();

    let dtiny = 1e-20_f64;
    let n_spec = spec.len();
    let spec_db: Vec<f64> = (0..n_spec)
        .map(|i| {
            let f = freqs[n_spec - i]; // reversed freq (skip DC)
            let w = 2.0 * PI * f;
            let val = w * w * spec[i] / resp_power_rev[i];
            10.0 * (if val < dtiny { dtiny } else { val }).log10()
        })
        .collect();

    // 5. Period bin averaging
    Ok(period_bin_average(
        psd_periods,
        &spec_db,
        bin_left,
        bin_right,
    ))
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use stationxml_rs::{Coefficients, Decimation, PoleZero, PolesZeros, StageGain};
    use std::path::PathBuf;

    fn test_vectors_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("pyscripts")
            .join("test_vectors")
    }

    /// Helper: extract a `Vec<f64>` from a JSON array field.
    fn json_f64_vec(val: &serde_json::Value, key: &str) -> Vec<f64> {
        val[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }

    /// Helper: parse `Vec<Vec<Option<f64>>>` from JSON (for expected dB values).
    fn json_option_f64_matrix(val: &serde_json::Value, key: &str) -> Vec<Vec<Option<f64>>> {
        val[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|seg| seg.as_array().unwrap().iter().map(|v| v.as_f64()).collect())
            .collect()
    }

    /// Helper: compare PSD result against expected dB values with tolerance.
    fn assert_psd_match(
        result: &[Option<f64>],
        expected: &[Option<f64>],
        seg_idx: usize,
        label: &str,
    ) {
        assert_eq!(
            result.len(),
            expected.len(),
            "segment {seg_idx}: bin count mismatch"
        );

        let mut max_err = 0.0_f64;
        let mut n_compared = 0usize;
        for (bin, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            match (got, exp) {
                (Some(g), Some(e)) => {
                    let err = (g - e).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    assert!(
                        err < 0.5,
                        "segment {seg_idx} bin {bin}: got {g:.4} dB, expected {e:.4} dB, err={err:.4} dB"
                    );
                    n_compared += 1;
                }
                (None, None) => {}
                (Some(g), None) => {
                    panic!("segment {seg_idx} bin {bin}: got {g:.4} dB, expected None");
                }
                (None, Some(e)) => {
                    panic!("segment {seg_idx} bin {bin}: got None, expected {e:.4} dB");
                }
            }
        }
        eprintln!("{label} segment {seg_idx}: compared {n_compared} bins, max_err={max_err:.6} dB");
    }

    /// Helper: build a PAZ-only Response from test vector JSON.
    fn response_from_json(data: &serde_json::Value) -> Response {
        let zeros_re = json_f64_vec(data, "zeros_real");
        let zeros_im = json_f64_vec(data, "zeros_imag");
        let poles_re = json_f64_vec(data, "poles_real");
        let poles_im = json_f64_vec(data, "poles_imag");

        Response {
            instrument_sensitivity: None,
            stages: vec![ResponseStage {
                number: 1,
                stage_gain: Some(StageGain {
                    value: data["stage_gain"].as_f64().unwrap(),
                    frequency: data["normalization_frequency"].as_f64().unwrap(),
                }),
                poles_zeros: Some(PolesZeros {
                    input_units: stationxml_rs::Units {
                        name: "M/S".into(),
                        description: None,
                    },
                    output_units: stationxml_rs::Units {
                        name: "V".into(),
                        description: None,
                    },
                    pz_transfer_function_type: PzTransferFunction::LaplaceRadians,
                    normalization_factor: data["normalization_factor"].as_f64().unwrap(),
                    normalization_frequency: data["normalization_frequency"].as_f64().unwrap(),
                    zeros: zeros_re
                        .iter()
                        .zip(zeros_im.iter())
                        .enumerate()
                        .map(|(i, (&r, &im))| PoleZero {
                            number: i as u32,
                            real: r,
                            imaginary: im,
                        })
                        .collect(),
                    poles: poles_re
                        .iter()
                        .zip(poles_im.iter())
                        .enumerate()
                        .map(|(i, (&r, &im))| PoleZero {
                            number: i as u32,
                            real: r,
                            imaginary: im,
                        })
                        .collect(),
                }),
                coefficients: None,
                fir: None,
                decimation: None,
            }],
        }
    }

    /// Helper: build FIR stages from test vector JSON.
    fn fir_stages_from_json(data: &serde_json::Value) -> Vec<ResponseStage> {
        data.as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let coefficients = json_f64_vec(s, "coefficients");
                let symmetry = match s["symmetry"].as_str().unwrap() {
                    "asymmetric" => Symmetry::None,
                    "even" => Symmetry::Even,
                    "odd" => Symmetry::Odd,
                    other => panic!("unknown symmetry: {other}"),
                };
                let stage_gain = s["stage_gain"].as_f64().unwrap();
                let sample_rate = s["sample_rate"].as_f64().unwrap();

                ResponseStage {
                    number: (i + 2) as u32,
                    stage_gain: Some(StageGain {
                        value: stage_gain,
                        frequency: 1.0,
                    }),
                    poles_zeros: None,
                    coefficients: None,
                    fir: Some(FIR {
                        input_units: stationxml_rs::Units {
                            name: "COUNTS".into(),
                            description: None,
                        },
                        output_units: stationxml_rs::Units {
                            name: "COUNTS".into(),
                            description: None,
                        },
                        symmetry,
                        numerator_coefficients: coefficients,
                    }),
                    decimation: Some(Decimation {
                        input_sample_rate: sample_rate,
                        factor: 1,
                        offset: 0,
                        delay: 0.0,
                        correction: 0.0,
                    }),
                }
            })
            .collect()
    }

    // -- Cosine taper --

    #[test]
    fn test_cosine_taper() {
        let path = test_vectors_dir().join("cosine_taper.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let n = data["n"].as_u64().unwrap() as usize;
        let p = data["taper_percentage"].as_f64().unwrap();
        let expected = json_f64_vec(&data, "taper_weights");

        let result = cosine_taper(n, p);

        assert_eq!(result.len(), expected.len());
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-12,
                "taper mismatch at index {i}: got {r}, expected {e}"
            );
        }
    }

    // -- FFT power spectrum --

    #[test]
    fn test_fft_power() {
        let path = test_vectors_dir().join("fft_power.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let sample_rate = data["sample_rate"].as_f64().unwrap();
        let signal_tapered = json_f64_vec(&data, "signal_tapered");
        let expected_freqs = json_f64_vec(&data, "freqs");
        let expected_power = json_f64_vec(&data, "power");

        let (freqs, power) = fft_power(&signal_tapered, sample_rate);

        assert_eq!(freqs.len(), expected_freqs.len());
        for (i, (r, e)) in freqs.iter().zip(expected_freqs.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-10,
                "freq mismatch at bin {i}: got {r}, expected {e}"
            );
        }

        assert_eq!(power.len(), expected_power.len());
        for (i, (r, e)) in power.iter().zip(expected_power.iter()).enumerate() {
            let tol = e.abs() * 1e-6 + 1e-20;
            assert!(
                (r - e).abs() < tol,
                "power mismatch at bin {i}: got {r:.6e}, expected {e:.6e}"
            );
        }
    }

    // -- Instrument response (PAZ only) --

    #[test]
    fn test_instrument_response() {
        let path = test_vectors_dir().join("instrument_response.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let response = response_from_json(&data);

        let freqs = json_f64_vec(&data, "freqs");
        let expected_power = json_f64_vec(&data, "response_power");

        let result = eval_response(&response, &freqs).unwrap();

        assert_eq!(result.len(), expected_power.len());
        for (i, (r, e)) in result.iter().zip(expected_power.iter()).enumerate() {
            let rel_err = if *e != 0.0 {
                ((r - e) / e).abs()
            } else {
                r.abs()
            };
            assert!(
                rel_err < 1e-6,
                "response power mismatch at bin {i} (freq={:.4}): got {r:.6e}, expected {e:.6e}, rel_err={rel_err:.2e}",
                freqs[i]
            );
        }
    }

    // -- FIR response --

    #[test]
    fn test_fir_response() {
        let path = test_vectors_dir().join("fir_response.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let coefficients = json_f64_vec(&data, "coefficients");
        let sample_rate = data["sample_rate"].as_f64().unwrap();
        let stage_gain = data["stage_gain"].as_f64().unwrap();

        let fir = FIR {
            input_units: stationxml_rs::Units {
                name: "COUNTS".into(),
                description: None,
            },
            output_units: stationxml_rs::Units {
                name: "COUNTS".into(),
                description: None,
            },
            symmetry: Symmetry::None,
            numerator_coefficients: coefficients,
        };

        let freqs = json_f64_vec(&data, "freqs");
        let expected_power = json_f64_vec(&data, "response_power");

        let result = eval_fir(&fir, stage_gain, sample_rate, &freqs);

        assert_eq!(result.len(), expected_power.len());
        for (i, (r, e)) in result.iter().zip(expected_power.iter()).enumerate() {
            let rel_err = if *e > 1e-20 {
                ((r - e) / e).abs()
            } else {
                (r - e).abs()
            };
            assert!(
                rel_err < 1e-6,
                "FIR power mismatch at bin {i} (freq={:.4}): got {r:.6e}, expected {e:.6e}, rel_err={rel_err:.2e}",
                freqs[i]
            );
        }
    }

    // -- Full PSD (PAZ only) --

    #[test]
    fn test_full_psd() {
        let path = test_vectors_dir().join("full_psd.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let sample_rate = data["sample_rate"].as_f64().unwrap();
        let seg_len = data["seg_len"].as_u64().unwrap() as usize;
        let nfft = data["nfft"].as_u64().unwrap() as usize;
        let nlap = data["nlap"].as_u64().unwrap() as usize;
        let overlap = data["overlap"].as_f64().unwrap();
        let n_segments = data["n_segments"].as_u64().unwrap() as usize;

        let input_data = json_f64_vec(&data, "input_data");
        let psd_periods = json_f64_vec(&data, "psd_periods");
        let bin_left = json_f64_vec(&data, "period_bin_left");
        let bin_right = json_f64_vec(&data, "period_bin_right");

        let response = response_from_json(&data["response"]);

        let expected_db = json_option_f64_matrix(&data, "psd_values_db");
        let step = (seg_len as f64 * (1.0 - overlap)) as usize;

        for (seg_idx, expected_seg) in expected_db.iter().enumerate().take(n_segments) {
            let start = seg_idx * step;
            let segment = &input_data[start..start + seg_len];

            let result = process_segment(
                segment,
                sample_rate,
                nfft,
                nlap,
                &response,
                &psd_periods,
                &bin_left,
                &bin_right,
            )
            .unwrap();

            assert_psd_match(&result, expected_seg, seg_idx, "PAZ-only");
        }
    }

    // -- Konno-Ohmachi smoothing --

    #[test]
    fn test_konno_ohmachi() {
        let path = test_vectors_dir().join("konno_ohmachi.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let freqs = json_f64_vec(&data, "freqs");
        let spectrum = json_f64_vec(&data, "spectrum");
        let bandwidth = data["bandwidth"].as_f64().unwrap();
        let expected = json_f64_vec(&data, "smoothed");

        let result = konno_ohmachi_smooth(&freqs, &spectrum, bandwidth);

        assert_eq!(result.len(), expected.len());
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            let tol = e.abs() * 1e-4 + 1e-20;
            assert!(
                (r - e).abs() < tol,
                "smoothing mismatch at bin {i} (freq={:.4}): got {r:.6e}, expected {e:.6e}",
                freqs[i]
            );
        }
    }

    // -- Sample-rate validation guards --

    fn make_units() -> stationxml_rs::Units {
        stationxml_rs::Units {
            name: "COUNTS".into(),
            description: None,
        }
    }

    fn fir_stage_with_sample_rate(sample_rate: f64) -> ResponseStage {
        ResponseStage {
            number: 1,
            stage_gain: Some(StageGain {
                value: 1.0,
                frequency: 1.0,
            }),
            poles_zeros: None,
            coefficients: None,
            fir: Some(FIR {
                input_units: make_units(),
                output_units: make_units(),
                symmetry: Symmetry::None,
                numerator_coefficients: vec![0.25, 0.5, 0.25],
            }),
            decimation: Some(Decimation {
                input_sample_rate: sample_rate,
                factor: 1,
                offset: 0,
                delay: 0.0,
                correction: 0.0,
            }),
        }
    }

    fn digital_coeffs_stage_with_sample_rate(sample_rate: f64) -> ResponseStage {
        ResponseStage {
            number: 1,
            stage_gain: Some(StageGain {
                value: 1.0,
                frequency: 1.0,
            }),
            poles_zeros: None,
            fir: None,
            coefficients: Some(Coefficients {
                input_units: make_units(),
                output_units: make_units(),
                cf_transfer_function_type: CfTransferFunction::Digital,
                numerators: vec![0.25, 0.5, 0.25],
                denominators: vec![],
            }),
            decimation: Some(Decimation {
                input_sample_rate: sample_rate,
                factor: 1,
                offset: 0,
                delay: 0.0,
                correction: 0.0,
            }),
        }
    }

    #[test]
    fn test_eval_response_rejects_zero_sample_rate_fir() {
        let response = Response {
            instrument_sensitivity: None,
            stages: vec![fir_stage_with_sample_rate(0.0)],
        };
        let freqs = vec![0.1, 0.5, 1.0];
        let err = eval_response(&response, &freqs).unwrap_err();
        assert!(
            matches!(err, PpsdError::InvalidSampleRate(1)),
            "expected InvalidSampleRate(1), got {err:?}"
        );
    }

    #[test]
    fn test_eval_response_rejects_negative_sample_rate_fir() {
        let response = Response {
            instrument_sensitivity: None,
            stages: vec![fir_stage_with_sample_rate(-100.0)],
        };
        let freqs = vec![0.1, 0.5, 1.0];
        let err = eval_response(&response, &freqs).unwrap_err();
        assert!(matches!(err, PpsdError::InvalidSampleRate(1)));
    }

    #[test]
    fn test_eval_response_rejects_nan_sample_rate_fir() {
        let response = Response {
            instrument_sensitivity: None,
            stages: vec![fir_stage_with_sample_rate(f64::NAN)],
        };
        let freqs = vec![0.1, 0.5, 1.0];
        let err = eval_response(&response, &freqs).unwrap_err();
        assert!(matches!(err, PpsdError::InvalidSampleRate(1)));
    }

    #[test]
    fn test_eval_response_rejects_zero_sample_rate_coefficients() {
        let response = Response {
            instrument_sensitivity: None,
            stages: vec![digital_coeffs_stage_with_sample_rate(0.0)],
        };
        let freqs = vec![0.1, 0.5, 1.0];
        let err = eval_response(&response, &freqs).unwrap_err();
        assert!(
            matches!(err, PpsdError::InvalidSampleRate(1)),
            "expected InvalidSampleRate(1), got {err:?}"
        );
    }

    // -- Full PSD with FIR --

    #[test]
    fn test_full_psd_with_fir() {
        let path = test_vectors_dir().join("full_psd_with_fir.json");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        let sample_rate = data["sample_rate"].as_f64().unwrap();
        let seg_len = data["seg_len"].as_u64().unwrap() as usize;
        let nfft = data["nfft"].as_u64().unwrap() as usize;
        let nlap = data["nlap"].as_u64().unwrap() as usize;
        let overlap = data["overlap"].as_f64().unwrap();
        let n_segments = data["n_segments"].as_u64().unwrap() as usize;

        let input_data = json_f64_vec(&data, "input_data");
        let psd_periods = json_f64_vec(&data, "psd_periods");
        let bin_left = json_f64_vec(&data, "period_bin_left");
        let bin_right = json_f64_vec(&data, "period_bin_right");

        // Build Response: PAZ stage + FIR stages
        let mut response = response_from_json(&data["response"]);
        let fir_stages = fir_stages_from_json(&data["fir_stages"]);
        response.stages.extend(fir_stages);

        let expected_db = json_option_f64_matrix(&data, "psd_values_db");
        assert!(n_segments > 0, "expected at least 1 segment in test vector");
        let step = (seg_len as f64 * (1.0 - overlap)) as usize;

        for (seg_idx, expected_seg) in expected_db.iter().enumerate().take(n_segments) {
            let start = seg_idx * step;
            let segment = &input_data[start..start + seg_len];

            let result = process_segment(
                segment,
                sample_rate,
                nfft,
                nlap,
                &response,
                &psd_periods,
                &bin_left,
                &bin_right,
            )
            .unwrap();

            assert_psd_match(&result, expected_seg, seg_idx, "PAZ+FIR");
        }
    }
}
