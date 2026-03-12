# ppsd-rs

Seismic PPSD (Probabilistic Power Spectral Density) computation in Rust, matching ObsPy output. Zero `unsafe`, zero C dependencies.

[![Crates.io](https://img.shields.io/crates/v/ppsd-rs.svg)](https://crates.io/crates/ppsd-rs)
[![docs.rs](https://docs.rs/ppsd-rs/badge.svg)](https://docs.rs/ppsd-rs)
[![CI](https://github.com/luhtfiimanal/ppsd-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/luhtfiimanal/ppsd-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)

## What is PPSD?

[PPSD](https://doi.org/10.1785/012003001) (McNamara & Buland, 2004) is the standard method for characterizing ambient seismic noise at a station. It computes power spectral density from continuous waveforms, removes the instrument response, and accumulates statistics over time. PPSD is used by IRIS, FDSN, BMKG, and most seismic networks for station quality monitoring.

This crate implements the exact same algorithm as [ObsPy's PPSD](https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.html), validated against ObsPy output within 0.5 dB tolerance. It uses [`stationxml-rs`](https://crates.io/crates/stationxml-rs) types directly for instrument response evaluation -- no intermediate types, no conversion boilerplate.

## Features

- **Full PPSD pipeline** matching ObsPy's `PPSD.__process` step-by-step
- **Welch's method** (mlab.psd compatible) with cosine taper and linear detrend
- **Instrument response evaluation** from `stationxml_rs::Response` -- walks PAZ, FIR, and Coefficients stages automatically
- **FIR filter support** with DTFT evaluation, DC normalization (matching evalresp), and symmetry expansion (None/Even/Odd)
- **Konno-Ohmachi smoothing** with configurable bandwidth
- **Period binning** with octave-based averaging (ObsPy style)
- **Direct `stationxml-rs` integration** -- takes `&Response` from any FDSN StationXML or SC3ML inventory
- **Zero unsafe** -- pure Rust math, no FFI
- **Zero C dependencies** -- compiles anywhere `rustc` runs

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ppsd-rs = "0.1"
stationxml-rs = "0.2"
```

### Compute PSD for one segment

```rust
use ppsd_rs::{process_segment, setup_period_binning, welch_psd};

// Load inventory (auto-detects FDSN StationXML or SC3ML)
let inv = stationxml_rs::read_from_file("IA.JAGI.xml").unwrap();
let channel = &inv.networks[0].stations[0].channels[0];
let response = channel.response.as_ref().unwrap();
let sample_rate = channel.sample_rate;

// ObsPy-compatible parameters
let ppsd_length = 3600.0;  // 1-hour segments
let nfft = (ppsd_length * sample_rate) as usize;
let nlap = nfft / 2;  // 50% overlap

// Period binning (matching ObsPy defaults)
let freqs: Vec<f64> = (1..=nfft / 2)
    .map(|i| i as f64 * sample_rate / nfft as f64)
    .collect();
let psd_periods: Vec<f64> = freqs.iter().rev().map(|&f| 1.0 / f).collect();
let (bin_left, bin_right, _centers) =
    setup_period_binning(&psd_periods, 0.04, 0.3);

// Process one segment (3600s of data)
let samples: Vec<f64> = load_your_data(); // your data source
let result = process_segment(
    &samples, sample_rate, nfft, nlap,
    response, &psd_periods, &bin_left, &bin_right,
).unwrap();

// result is Vec<Option<f64>> -- dB values per period bin
for (i, db) in result.iter().enumerate() {
    if let Some(val) = db {
        println!("bin {i}: {val:.1} dB");
    }
}
```

### Evaluate instrument response only

```rust
use ppsd_rs::eval_response;

let inv = stationxml_rs::read_from_file("station.xml").unwrap();
let response = inv.networks[0].stations[0].channels[0]
    .response.as_ref().unwrap();

let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
let h_squared = eval_response(response, &freqs).unwrap();
// h_squared[i] = |H(f_i)|² including all stages (PAZ + FIR + ADC)
```

## API Overview

```rust
use ppsd_rs::{
    // Full pipeline
    process_segment,           // one PPSD segment → dB values
    // Response evaluation
    eval_response,             // stationxml_rs::Response → |H(f)|²
    // Individual steps
    welch_psd,                 // Welch's method (mlab.psd)
    cosine_taper,              // Tukey window
    fft_power,                 // one-sided power spectrum
    konno_ohmachi_smooth,      // spectral smoothing
    period_bin_average,        // octave binning
    setup_period_binning,      // bin edge computation
    // Types
    PsdConfig, PsdResult,      // configuration and output
    PpsdError, Result,         // error handling
};
```

| Function | Description |
|----------|-------------|
| `process_segment()` | Full PPSD pipeline: Welch → response removal → dB → period binning |
| `eval_response()` | Evaluate `stationxml_rs::Response` at frequencies, returns \|H(f)\|² |
| `welch_psd()` | PSD via Welch's method (cosine taper, linear detrend, overlap-average) |
| `cosine_taper()` | Tukey window matching ObsPy's `cosine_taper(npts, p)` |
| `fft_power()` | One-sided power spectrum via real FFT |
| `konno_ohmachi_smooth()` | Konno-Ohmachi log-frequency smoothing |
| `period_bin_average()` | Average dB values into octave-spaced period bins |
| `setup_period_binning()` | Compute bin edges matching ObsPy `PPSD._setup_period_binning` |

## Response Evaluation

`eval_response()` walks all stages in `stationxml_rs::Response` and computes the combined |H(f)|²:

| Stage type | stationxml-rs type | Evaluation method |
|------------|-------------------|-------------------|
| Sensor (PAZ) | `PolesZeros` | Laplace H(s) = A₀ × ∏(s-zᵢ)/∏(s-pⱼ) × gain |
| ADC / Digital | `Coefficients` (Digital) | DTFT of numerators × gain |
| FIR filter | `FIR` | DTFT with DC normalization × gain |
| Gain-only | (none of above) | gain² |

FIR coefficients are DC-normalized to match evalresp behavior. Symmetry expansion handles `None` (all coefficients), `Even` (mirror), and `Odd` (mirror with negation).

## Architecture

```
src/
  lib.rs           -- single-file library: types, response eval, PSD math, tests
```

### Design Decisions

- **Direct `stationxml-rs` coupling**: takes `&Response` directly -- no intermediate `InstrumentResponse` type. Both crates share the same author; updates propagate naturally.
- **Single file**: the crate is focused enough that one `lib.rs` keeps everything discoverable without module navigation overhead.
- **`Result<T>` over panic**: response evaluation can fail (empty response, missing decimation, unsupported stage) -- errors are explicit, not hidden behind defaults.
- **DC normalization**: FIR coefficients are normalized to unit DC gain, matching evalresp's internal behavior. This is critical for correct dB output.
- **`&[f64]` over stream**: user provides pre-sliced sample arrays -- simple, composable, no hidden I/O or buffering.

### TDD with ObsPy

Test vectors are generated by Python/ObsPy scripts, ensuring the Rust output matches ObsPy PPSD within 0.5 dB tolerance across all 7 pipeline stages:

```bash
cd pyscripts && uv run python -m pyscripts
cargo test
```

## Development

```bash
cargo build                     # build
cargo test                      # all tests (requires test vectors)
cargo clippy -- -D warnings     # lint (strict)
cargo fmt -- --check            # format check
cargo doc --no-deps --open      # browse docs locally
```

## References

- [McNamara & Buland (2004)](https://doi.org/10.1785/012003001) -- original PPSD method paper
- [ObsPy PPSD](https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.html) -- Python reference implementation
- [FDSN StationXML](https://www.fdsn.org/xml/station/) -- instrument response format specification
- [evalresp](https://ds.iris.edu/ds/nodes/dmc/software/downloads/evalresp/) -- IRIS response evaluation (FIR normalization behavior)

## Sister Projects

- [stationxml-rs](https://crates.io/crates/stationxml-rs) -- FDSN StationXML and SC3ML reader/writer (provides `Response` types used by this crate)
- [miniseed-rs](https://crates.io/crates/miniseed-rs) -- miniSEED v2/v3 decoder and encoder
- [seedlink-rs](https://crates.io/crates/seedlink-rs) -- SeedLink protocol client/server

## License

Apache-2.0
