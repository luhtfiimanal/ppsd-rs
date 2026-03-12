"""Generate PSD test vectors using ObsPy/NumPy as oracle for Rust TDD.

Outputs JSON files in test_vectors/ that Rust tests load to validate each step:
1. cosine_taper - taper function on known input
2. fft_power - FFT and power spectrum of a sine wave
3. instrument_response - transfer function from known poles/zeros
4. konno_ohmachi - smoothing on synthetic spectrum
5. full_psd - complete pipeline matching ObsPy PPSD output
6. fir_response - DTFT of real Q330 FIR coefficients
7. full_psd_with_fir - complete pipeline with PAZ + FIR stages
"""

# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from obspy import Stream, Trace, UTCDateTime  # type: ignore[import-untyped]
from obspy.core.inventory import (  # type: ignore[import-untyped]
    Channel,
    InstrumentSensitivity,
    Inventory,
    Network,
    PolesZerosResponseStage,
    Response,
    Site,
    Station,
)
from obspy.signal.invsim import cosine_taper as obspy_cosine_taper  # type: ignore[import-untyped]
from obspy.signal.spectral_estimation import PPSD  # type: ignore[import-untyped]

VECTORS_DIR = Path(__file__).resolve().parent.parent.parent / "test_vectors"


def _to_list(a: NDArray[Any]) -> list[float]:
    return [float(x) for x in a]


def _to_list_nan(a: NDArray[Any]) -> list[float | None]:
    """Convert array to list, NaN becomes null."""
    return [None if np.isnan(x) else float(x) for x in a]


def generate_cosine_taper() -> None:
    """Test vector for cosine taper function (5% each side)."""
    n = 1000
    taper_percentage = 0.05
    taper = obspy_cosine_taper(n, p=taper_percentage)

    # Also test on a signal
    signal = np.ones(n, dtype=np.float64)
    tapered = signal * taper

    out = {
        "n": n,
        "taper_percentage": taper_percentage,
        "taper_weights": _to_list(taper),
        "input_signal": _to_list(signal),
        "tapered_signal": _to_list(tapered),
    }
    path = VECTORS_DIR / "cosine_taper.json"
    path.write_text(json.dumps(out))
    print(f"  wrote {path.name} ({n} samples)")


def generate_fft_power() -> None:
    """Test vector: FFT power spectrum of known sine wave.

    A 10 Hz sine at 100 Hz sample rate should produce a clear peak at bin 10 Hz.
    """
    sample_rate = 100.0
    duration = 10.0  # seconds
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    freq_hz = 10.0
    amplitude = 1.0

    signal = amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    # Demean
    signal_demeaned = signal - np.mean(signal)

    # Taper
    taper = obspy_cosine_taper(n, p=0.05)
    signal_tapered = signal_demeaned * taper

    # Real FFT
    fft_result = np.fft.rfft(signal_tapered)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Power spectrum: |FFT|^2 / (fs * N) - single-sided
    power = np.abs(fft_result) ** 2 / (sample_rate * n)
    # Double non-DC, non-Nyquist bins for single-sided spectrum
    power[1:-1] *= 2.0

    out = {
        "sample_rate": sample_rate,
        "n": n,
        "frequency_hz": freq_hz,
        "amplitude": amplitude,
        "signal": _to_list(signal),
        "signal_demeaned": _to_list(signal_demeaned),
        "signal_tapered": _to_list(signal_tapered),
        "fft_real": _to_list(np.real(fft_result)),
        "fft_imag": _to_list(np.imag(fft_result)),
        "freqs": _to_list(freqs),
        "power": _to_list(power),
    }
    path = VECTORS_DIR / "fft_power.json"
    path.write_text(json.dumps(out))
    print(f"  wrote {path.name} ({n} samples, peak at {freq_hz} Hz)")


def generate_instrument_response() -> None:
    """Test vector: instrument response (transfer function) from known poles/zeros.

    Uses a simplified STS-2 seismometer response (velocity sensor).
    """
    # Simplified STS-2 poles and zeros (velocity, rad/s)
    zeros = [complex(0, 0), complex(0, 0)]  # two zeros at origin
    poles = [
        complex(-0.037004, 0.037016),
        complex(-0.037004, -0.037016),
        complex(-251.33, 0),
        complex(-131.04, -467.29),
        complex(-131.04, 467.29),
    ]
    normalization_factor = 60077000.0
    normalization_frequency = 1.0  # Hz
    stage_gain = 1500.0  # PAZ stage gain (V / (m/s))

    # Evaluate at test frequencies
    freqs = np.logspace(-2, 2, 200)  # 0.01 to 100 Hz
    omega = 2.0 * np.pi * freqs
    s = 1j * omega

    # Build transfer function: H(s) = stage_gain * norm * prod(s - z) / prod(s - p)
    # This matches ObsPy's get_evalresp_response for a single PAZ stage.
    h = np.ones(len(freqs), dtype=complex)
    for z in zeros:
        h *= s - z
    for p in poles:
        h /= s - p
    h *= normalization_factor
    h *= stage_gain

    out = {
        "zeros_real": [z.real for z in zeros],
        "zeros_imag": [z.imag for z in zeros],
        "poles_real": [p.real for p in poles],
        "poles_imag": [p.imag for p in poles],
        "normalization_factor": normalization_factor,
        "normalization_frequency": normalization_frequency,
        "stage_gain": stage_gain,
        "freqs": _to_list(freqs),
        "response_amplitude": _to_list(np.abs(h)),
        "response_phase_rad": _to_list(np.angle(h)),
        "response_power": _to_list(np.abs(h) ** 2),
    }
    path = VECTORS_DIR / "instrument_response.json"
    path.write_text(json.dumps(out))
    print(f"  wrote {path.name} ({len(freqs)} freq points)")


def generate_konno_ohmachi() -> None:
    """Test vector: Konno-Ohmachi smoothing on synthetic log-spaced spectrum."""
    # Create a spectrum with a sharp peak
    freqs = np.logspace(-2, 2, 500)
    spectrum = np.ones(len(freqs)) * 1e-10  # background
    # Add a peak near 1 Hz
    peak_idx = np.argmin(np.abs(freqs - 1.0))
    spectrum[peak_idx - 2 : peak_idx + 3] = 1e-5

    bandwidth = 40.0  # Konno-Ohmachi bandwidth parameter

    # Konno-Ohmachi smoothing kernel
    smoothed = np.zeros_like(spectrum)
    for i, fc in enumerate(freqs):
        if fc == 0:
            smoothed[i] = spectrum[i]
            continue
        ratio = freqs / fc
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log10(ratio)
            arg = bandwidth * log_ratio
            weights = np.where(
                np.abs(log_ratio) < 1e-10,
                1.0,
                np.where(
                    np.abs(arg) > 1e-10,
                    (np.sin(arg) / arg) ** 4,
                    1.0,
                ),
            )
        weights = np.nan_to_num(weights, nan=0.0)
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * spectrum) / np.sum(weights)

    out = {
        "freqs": _to_list(freqs),
        "spectrum": _to_list(spectrum),
        "bandwidth": bandwidth,
        "smoothed": _to_list(smoothed),
    }
    path = VECTORS_DIR / "konno_ohmachi.json"
    path.write_text(json.dumps(out))
    print(f"  wrote {path.name} ({len(freqs)} freq points, bw={bandwidth})")


# Real Q330 FIR filter coefficients (67 taps, asymmetric).
# This is the Q330_FIR_4 decimation filter used for 20 sps BHZ channels.
Q330_FIR_4_COEFFICIENTS: list[float] = [
    -3.65942e-17,
    3.67488e-07,
    4.14999e-06,
    1.65202e-05,
    4.10887e-05,
    7.80802e-05,
    1.22657e-04,
    1.63051e-04,
    1.79489e-04,
    1.51996e-04,
    5.84562e-05,
    -1.13033e-04,
    -3.66693e-04,
    -6.86001e-04,
    -1.02817e-03,
    -1.31777e-03,
    -1.44480e-03,
    -1.27228e-03,
    -6.45988e-04,
    5.76735e-04,
    2.44003e-03,
    4.83216e-03,
    7.49853e-03,
    1.00528e-02,
    1.19403e-02,
    1.25556e-02,
    1.12681e-02,
    7.51693e-03,
    7.67085e-04,
    -8.67975e-03,
    -2.06294e-02,
    -3.36755e-02,
    -4.59824e-02,
    -5.54189e-02,
    -5.97484e-02,
    -5.66998e-02,
    -4.46371e-02,
    -2.26048e-02,
    9.66286e-03,
    5.18525e-02,
    1.02285e-01,
    1.58499e-01,
    2.16566e-01,
    2.72509e-01,
    3.22032e-01,
    3.61372e-01,
    3.87690e-01,
    3.99147e-01,
    3.95093e-01,
    3.76055e-01,
    3.43711e-01,
    3.00674e-01,
    2.50178e-01,
    1.95647e-01,
    1.40496e-01,
    8.79676e-02,
    4.18818e-02,
    5.38673e-03,
    -1.89921e-02,
    -3.08859e-02,
    -3.12775e-02,
    -2.24152e-02,
    -8.46462e-03,
    5.20780e-03,
    1.39956e-02,
    1.55425e-02,
    9.90015e-03,
]


def generate_fir_response() -> None:
    """Test vector: DTFT of real Q330_FIR_4 coefficients.

    Evaluates H(f) = Σ a[k] * exp(-j*2π*f*k/SR) at log-spaced frequencies.
    Oracle: direct numpy computation.
    """
    coeffs = np.array(Q330_FIR_4_COEFFICIENTS)
    sample_rate = 20.0  # Hz (20 sps channel)
    freqs = np.logspace(-2, np.log10(sample_rate / 2), 200)  # 0.01 to 10 Hz (Nyquist)

    # Normalize to unit DC gain (matches evalresp behavior)
    coeffs = coeffs / np.sum(coeffs)

    # DTFT: H(f) = Σ a[k] * exp(-j*2π*f*k/SR)
    n = len(coeffs)
    k = np.arange(n)
    h = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        h[i] = np.sum(coeffs * np.exp(-1j * 2 * np.pi * f * k / sample_rate))

    response_power = np.abs(h) ** 2

    out = {
        "coefficients": [float(c) for c in coeffs],
        "symmetry": "asymmetric",
        "sample_rate": sample_rate,
        "stage_gain": 1.0,
        "freqs": _to_list(freqs),
        "response_power": _to_list(response_power),
    }
    path = VECTORS_DIR / "fir_response.json"
    path.write_text(json.dumps(out))
    print(f"  wrote {path.name} ({len(freqs)} freq points, {n} coefficients)")


def _make_synthetic_inventory_with_fir(sample_rate: float) -> Inventory:
    """Create ObsPy Inventory with STS-2-like PAZ + Q330 FIR + ADC gain for testing."""
    from obspy.core.inventory import (
        CoefficientsTypeResponseStage,
        FIRResponseStage,
    )

    paz_stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=1500.0,
        stage_gain_frequency=1.0,
        input_units="M/S",
        output_units="V",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[complex(0, 0), complex(0, 0)],
        poles=[
            complex(-0.037004, 0.037016),
            complex(-0.037004, -0.037016),
            complex(-251.33, 0),
            complex(-131.04, -467.29),
            complex(-131.04, 467.29),
        ],
        normalization_factor=60077000.0,
    )

    from obspy.core.util.obspy_types import FloatWithUncertaintiesAndUnit

    # ADC stage (digitizer gain, modeled as trivial COEFFICIENTS stage)
    adc_stage = CoefficientsTypeResponseStage(
        stage_sequence_number=2,
        stage_gain=419430.0,
        stage_gain_frequency=1.0,
        input_units="V",
        output_units="COUNTS",
        cf_transfer_function_type="DIGITAL",
        numerator=[FloatWithUncertaintiesAndUnit(1.0)],
        denominator=[],
        decimation_input_sample_rate=sample_rate,
        decimation_factor=1,
        decimation_offset=0,
        decimation_delay=0.0,
        decimation_correction=0.0,
    )

    # Q330 FIR stage (67 coefficients, asymmetric, unity gain)
    fir_stage = FIRResponseStage(
        stage_sequence_number=3,
        stage_gain=1.0,
        stage_gain_frequency=1.0,
        input_units="COUNTS",
        output_units="COUNTS",
        symmetry="NONE",  # asymmetric
        coefficients=Q330_FIR_4_COEFFICIENTS,
        decimation_input_sample_rate=sample_rate,
        decimation_factor=1,
        decimation_offset=0,
        decimation_delay=0.0,
        decimation_correction=0.0,
    )

    # Overall sensitivity = PAZ gain * ADC gain * FIR gain = 1500 * 419430 * 1.0
    overall_sensitivity = 1500.0 * 419430.0 * 1.0

    channel = Channel(
        code="BHZ",
        location_code="",
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        depth=0.0,
        sample_rate=sample_rate,
        response=Response(
            instrument_sensitivity=InstrumentSensitivity(
                value=overall_sensitivity,
                frequency=1.0,
                input_units="M/S",
                output_units="COUNTS",
            ),
            response_stages=[paz_stage, adc_stage, fir_stage],
        ),
    )

    station = Station(
        code="TEST",
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        site=Site(name="Test Station"),
        channels=[channel],
    )

    network = Network(code="XX", stations=[station])
    return Inventory(networks=[network], source="test")


def generate_full_psd_with_fir() -> None:
    """Full PSD pipeline with PAZ + FIR, matching ObsPy PPSD output.

    Same noise as generate_full_psd (seed=42), but inventory includes
    an ADC gain stage and a Q330 FIR filter stage in addition to PAZ.
    ObsPy evaluates ALL stages via evalresp internally.
    """
    sample_rate = 20.0
    duration = 7200.0
    n = int(sample_rate * duration)
    ppsd_length = 3600.0
    overlap = 0.5

    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=0, scale=1000.0, size=n).astype(np.float64)

    tr = Trace(data=data.copy())
    tr.stats.network = "XX"
    tr.stats.station = "TEST"
    tr.stats.location = ""
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = sample_rate
    tr.stats.starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)

    inv = _make_synthetic_inventory_with_fir(sample_rate)

    ppsd = PPSD(
        tr.stats,
        metadata=inv,
        ppsd_length=ppsd_length,
        overlap=overlap,
        period_step_octaves=0.04,
        period_smoothing_width_octaves=0.3,
    )
    ppsd.add(Stream([tr]))

    nfft = ppsd.nfft
    nlap = ppsd.nlap
    seg_len = ppsd.len

    period_bin_left = ppsd._period_binning[0]
    period_bin_right = ppsd._period_binning[4]
    period_bin_center = ppsd._period_binning[2]

    psd_periods: NDArray[Any] = np.asarray(ppsd.psd_periods)

    out = {
        "sample_rate": sample_rate,
        "duration_secs": duration,
        "n_samples": n,
        "seed": 42,
        "noise_scale": 1000.0,
        "ppsd_length": ppsd_length,
        "overlap": overlap,
        "nfft": nfft,
        "nlap": nlap,
        "seg_len": seg_len,
        "period_step_octaves": 0.04,
        "period_smoothing_width_octaves": 0.3,
        "period_bin_left": _to_list(period_bin_left),
        "period_bin_right": _to_list(period_bin_right),
        "period_bin_center": _to_list(period_bin_center),
        "n_period_bins": len(period_bin_center),
        "psd_periods": _to_list(psd_periods),
        "times_processed": [str(t) for t in ppsd.times_processed],
        "n_segments": len(ppsd.times_processed),
        "psd_values_db": [_to_list_nan(v) for v in ppsd.psd_values],
        "input_data": _to_list(data),
        # PAZ response (same as full_psd)
        "response": {
            "zeros_real": [0.0, 0.0],
            "zeros_imag": [0.0, 0.0],
            "poles_real": [-0.037004, -0.037004, -251.33, -131.04, -131.04],
            "poles_imag": [0.037016, -0.037016, 0.0, -467.29, 467.29],
            "normalization_factor": 60077000.0,
            "normalization_frequency": 1.0,
            "stage_gain": 1500.0,
            "input_units": "M/S",
        },
        # FIR stages for Rust to construct
        "fir_stages": [
            {
                "coefficients": [1.0],
                "symmetry": "asymmetric",
                "stage_gain": 419430.0,
                "sample_rate": sample_rate,
            },
            {
                "coefficients": [float(c) for c in Q330_FIR_4_COEFFICIENTS],
                "symmetry": "asymmetric",
                "stage_gain": 1.0,
                "sample_rate": sample_rate,
            },
        ],
    }

    path = VECTORS_DIR / "full_psd_with_fir.json"
    path.write_text(json.dumps(out))
    size_mb = path.stat().st_size / 1024 / 1024
    print(
        f"  wrote {path.name} ({size_mb:.1f} MB, "
        f"{len(ppsd.times_processed)} segments, {len(period_bin_center)} freq bins)"
    )


def _make_synthetic_inventory(sample_rate: float) -> Inventory:
    """Create minimal ObsPy Inventory with STS-2-like response for testing."""
    paz_stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=1500.0,
        stage_gain_frequency=1.0,
        input_units="M/S",
        output_units="V",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[complex(0, 0), complex(0, 0)],
        poles=[
            complex(-0.037004, 0.037016),
            complex(-0.037004, -0.037016),
            complex(-251.33, 0),
            complex(-131.04, -467.29),
            complex(-131.04, 467.29),
        ],
        normalization_factor=60077000.0,
    )

    channel = Channel(
        code="BHZ",
        location_code="",
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        depth=0.0,
        sample_rate=sample_rate,
        response=Response(
            instrument_sensitivity=InstrumentSensitivity(
                value=629145000.0,
                frequency=1.0,
                input_units="M/S",
                output_units="COUNTS",
            ),
            response_stages=[paz_stage],
        ),
    )

    station = Station(
        code="TEST",
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        site=Site(name="Test Station"),
        channels=[channel],
    )

    network = Network(code="XX", stations=[station])
    return Inventory(networks=[network], source="test")


def generate_full_psd() -> None:
    """Full PSD pipeline matching ObsPy PPSD output.

    This replicates the EXACT ObsPy algorithm:
    1. mlab.psd (Welch's method) with nfft, nlap, cosine_taper(0.2), detrend_linear
    2. Skip DC bin, reverse to period order
    3. Response removal with w^2 factor (velocity→acceleration)
    4. Period binning (octave averaging, NOT Konno-Ohmachi)
    5. dB conversion
    """
    sample_rate = 20.0
    duration = 7200.0
    n = int(sample_rate * duration)
    ppsd_length = 3600.0
    overlap = 0.5

    # Deterministic
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=0, scale=1000.0, size=n).astype(np.float64)

    # Create trace
    tr = Trace(data=data.copy())
    tr.stats.network = "XX"
    tr.stats.station = "TEST"
    tr.stats.location = ""
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = sample_rate
    tr.stats.starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)

    inv = _make_synthetic_inventory(sample_rate)

    # Compute PPSD
    ppsd = PPSD(
        tr.stats,
        metadata=inv,
        ppsd_length=ppsd_length,
        overlap=overlap,
        period_step_octaves=0.04,
        period_smoothing_width_octaves=0.3,
    )
    ppsd.add(Stream([tr]))

    # Export PPSD internals for Rust to replicate
    nfft = ppsd.nfft
    nlap = ppsd.nlap
    seg_len = ppsd.len  # samples per segment

    # Period binning edges
    period_bin_left = ppsd._period_binning[0]
    period_bin_right = ppsd._period_binning[4]
    period_bin_center = ppsd._period_binning[2]

    # The raw PSD periods (from Welch FFT)
    psd_periods: NDArray[Any] = np.asarray(ppsd.psd_periods)

    out = {
        "sample_rate": sample_rate,
        "duration_secs": duration,
        "n_samples": n,
        "seed": 42,
        "noise_scale": 1000.0,
        # ObsPy internal params
        "ppsd_length": ppsd_length,
        "overlap": overlap,
        "nfft": nfft,
        "nlap": nlap,
        "seg_len": seg_len,
        "period_step_octaves": 0.04,
        "period_smoothing_width_octaves": 0.3,
        # Binning
        "period_bin_left": _to_list(period_bin_left),
        "period_bin_right": _to_list(period_bin_right),
        "period_bin_center": _to_list(period_bin_center),
        "n_period_bins": len(period_bin_center),
        "psd_periods": _to_list(psd_periods),
        # ObsPy output
        "times_processed": [str(t) for t in ppsd.times_processed],
        "n_segments": len(ppsd.times_processed),
        "psd_values_db": [_to_list_nan(v) for v in ppsd.psd_values],
        # Input data (for Rust to slice segments from)
        "input_data": _to_list(data),
        # Instrument response
        "response": {
            "zeros_real": [0.0, 0.0],
            "zeros_imag": [0.0, 0.0],
            "poles_real": [-0.037004, -0.037004, -251.33, -131.04, -131.04],
            "poles_imag": [0.037016, -0.037016, 0.0, -467.29, 467.29],
            "normalization_factor": 60077000.0,
            "normalization_frequency": 1.0,
            "stage_gain": 1500.0,
            "input_units": "M/S",
        },
    }

    path = VECTORS_DIR / "full_psd.json"
    path.write_text(json.dumps(out))
    size_mb = path.stat().st_size / 1024 / 1024
    print(
        f"  wrote {path.name} ({size_mb:.1f} MB, "
        f"{len(ppsd.times_processed)} segments, {len(period_bin_center)} freq bins)"
    )


def main() -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating PSD test vectors...")

    generate_cosine_taper()
    generate_fft_power()
    generate_instrument_response()
    generate_konno_ohmachi()
    generate_full_psd()
    generate_fir_response()
    generate_full_psd_with_fir()

    print("Done!")


if __name__ == "__main__":
    main()
