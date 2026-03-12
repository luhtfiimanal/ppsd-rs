# CLAUDE.md — ppsd-rs

Seismic PPSD (Probabilistic Power Spectral Density) computation in Rust, matching ObsPy output. Uses `stationxml-rs` types directly for instrument response evaluation. Apache 2.0.

## CRITICAL

- **Diskusi dulu sebelum implementasi** — investigasi, jelaskan, diskusikan, baru code
- **Jangan push tanpa persetujuan user**
- **stdout workaround**: `script -q -c "cargo test" /dev/null` (Claude Code bug)
- **Zero unsafe** — pure Rust math, no FFI

## Scope

PSD math library — takes `&stationxml_rs::Response` + `&[f64]` samples, outputs dB values matching ObsPy PPSD within 0.5 dB tolerance. No I/O, no async, no networking.

**Key dependency**: `stationxml-rs` for `Response`, `ResponseStage`, `FIR`, `PolesZeros`, `Symmetry` types. Direct coupling by design — both crates share the same author.

## API

```rust
use ppsd_rs::{process_segment, eval_response};

// Consumer provides stationxml_rs::Response (parsed from FDSN or SC3ML)
let inv = stationxml_rs::read_from_file("station.xml")?;
let response = inv.networks[0].stations[0].channels[0].response.as_ref().unwrap();

// Full PSD pipeline for one segment
let result = process_segment(&samples, sample_rate, nfft, nlap, response, ...)?;

// Or just evaluate response transfer function
let h_squared = eval_response(response, &freqs)?;
```

## Module Structure

Single file: `src/lib.rs` — all PSD math + response evaluation.

Public functions:
- `process_segment()` — full PPSD pipeline per segment
- `eval_response()` — PAZ + FIR + Coefficients |H(f)|² evaluation
- `welch_psd()` — Welch's method (mlab.psd compatible)
- `cosine_taper()` — Tukey window
- `fft_power()` — one-sided power spectrum
- `konno_ohmachi_smooth()` — spectral smoothing
- `period_bin_average()` — octave binning
- `setup_period_binning()` — bin edge computation

## Commands

```bash
cargo build                          # build
cargo test                           # test all
cargo clippy -- -D warnings          # lint (strict)
cargo fmt -- --check                 # format check

# pyscripts (TDD vector generation)
cd pyscripts && uv sync
cd pyscripts && uv run python -m pyscripts
cd pyscripts && uv run ruff check src
cd pyscripts && uv run basedpyright src
```

## TDD Strategy

Python/ObsPy generates test vectors → Rust tests assert against them.

1. `cd pyscripts && uv run python -m pyscripts`
2. Write Rust test loading `test_vectors/*.json` — RED
3. Implement Rust code — GREEN
4. Validate: output matches ObsPy within < 0.5 dB tolerance

Test vectors saved as JSON in `pyscripts/test_vectors/` (gitignored, regenerate locally).

## Response Evaluation

Walks `Response.stages` sequentially, evaluating each stage type:

| Stage | stationxml-rs type | Method |
|-------|-------------------|--------|
| Sensor (PAZ) | `PolesZeros` | Laplace H(s) × stage_gain |
| ADC (digital) | `Coefficients` | DTFT × stage_gain |
| FIR filter | `FIR` | DTFT (DC-normalized) × stage_gain |
| Gain-only | (none of above) | stage_gain² |

FIR coefficients are DC-normalized to match evalresp behavior. Symmetry expansion: `None` = all coefficients, `Even` = mirror, `Odd` = mirror with negation.

## Code Quality

- `cargo fmt` + `cargo clippy -- -D warnings`
- `thiserror` for error types
- No `unsafe` anywhere
- pyscripts: `basedpyright` strict + `ruff`
