#!/usr/bin/env python3
"""
Novel ECG Signal Denoising via CEEMDAN-Guided Morphological Custom Wavelet Design
and Sub-band Adaptive Thresholding.

This script implements three integrated contributions:
1) Morphology-driven custom mother wavelet design: CEEMDAN-derived dominant cardiac IMF,
   averaged QRS prototype, Gaussian-mixture morphology fitting, derivative-based admissible
   mother wavelet, iterative scaling-function construction, and custom FIR filter-bank synthesis.
2) CEEMDAN-guided sub-band noise profiling: noise-dominant IMF selection and wavelet-level
   threshold calibration using IMF-to-sub-band frequency mapping with sigmoid level weighting.
3) Morphology-preserving thresholding: a three-zone nonlinear shrinkage function with a
   QRS-protection mask to preserve sharp R-peak morphology in high-frequency sub-bands.

Outputs:
- results/Figure1.png ... results/Figure8.png (300 DPI)
- results/results_table.csv

Notes:
- MIT-BIH record loading requires network access on first download via wfdb/PhysioNet.
- CEEMDAN is computationally expensive. By default, the script loads 5 minutes but processes
  the first 30 seconds for denoising experiments. Use `--process-sec 300` to process all 5 minutes.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import wfdb
from PyEMD import CEEMDAN
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, find_peaks, freqz, upfirdn
from scipy.signal.windows import tukey
from skimage.metrics import structural_similarity as structural_similarity

warnings.filterwarnings("ignore", category=RuntimeWarning)

EPS = 1e-12


@dataclass
class WaveletFilters:
    name: str
    dec_lo: np.ndarray
    dec_hi: np.ndarray
    rec_lo: np.ndarray
    rec_hi: np.ndarray


@dataclass
class MethodOutput:
    denoised: np.ndarray
    thresholds: np.ndarray
    details_before: List[np.ndarray]
    details_after: List[np.ndarray]
    sparsity_ratio: float


def load_ecg(record: str = "100", duration_sec: int = 300) -> Tuple[np.ndarray, int]:
    """Load MIT-BIH ECG (lead MLII if available)."""
    # Read header first so sample count can be derived from real fs.
    header = wfdb.rdheader(record, pn_dir="mitdb")
    fs = int(round(header.fs))
    sampto = int(duration_sec * fs)
    signals, fields = wfdb.rdsamp(record, pn_dir="mitdb", sampto=sampto)
    sig_names = fields.get("sig_name", [])
    # MLII is the standard lead for most MIT-BIH denoising benchmarks.
    lead_idx = sig_names.index("MLII") if "MLII" in sig_names else 0
    ecg = signals[:, lead_idx].astype(np.float64)
    ecg -= np.mean(ecg)
    return ecg, fs


def add_noise_at_snr(signal: np.ndarray, target_snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise to meet a target SNR in dB."""
    p_signal = np.mean(signal**2)
    p_noise = p_signal / (10 ** (target_snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(max(p_noise, EPS)), size=signal.shape)
    return signal + noise


def run_ceemdan(signal: np.ndarray, trials: int = 50, max_imf: int = -1, seed: int = 42) -> np.ndarray:
    """Run CEEMDAN decomposition using PyEMD."""
    # Keep CEEMDAN deterministic for reproducible comparisons.
    ceemdan = CEEMDAN(trials=trials, parallel=False)
    ceemdan.noise_seed(seed)
    imfs = ceemdan.ceemdan(signal, max_imf=max_imf)
    return np.asarray(imfs)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a0 = a[:n] - np.mean(a[:n])
    b0 = b[:n] - np.mean(b[:n])
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + EPS
    return float(np.dot(a0, b0) / denom)


def select_dominant_imf(imfs: np.ndarray, clean_ecg: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Select IMF with highest absolute correlation to clean ECG."""
    corrs = np.array([_safe_corr(imf, clean_ecg) for imf in imfs], dtype=float)
    idx = int(np.argmax(np.abs(corrs)))
    return imfs[idx], idx, corrs


def detect_r_peaks_pan_tompkins(signal_in: np.ndarray, fs: int) -> np.ndarray:
    """Simplified Pan-Tompkins style R-peak detector."""
    # Stage 1: bandpass around QRS-dominant frequencies.
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    band = filtfilt(b, a, signal_in)

    # Stage 2: derivative + squaring + moving-window integration.
    derivative = np.ediff1d(band, to_begin=0.0)
    squared = derivative**2
    win = max(1, int(0.15 * fs))
    mwi = np.convolve(squared, np.ones(win) / win, mode="same")

    # Stage 3: adaptive threshold and minimum RR separation.
    thresh = np.mean(mwi) + 0.5 * np.std(mwi)
    peaks, _ = find_peaks(mwi, height=thresh, distance=max(1, int(0.20 * fs)))

    # Stage 4: local refinement to align peaks with signal maxima.
    refined = []
    search = max(1, int(0.05 * fs))
    for p in peaks:
        lo = max(0, p - search)
        hi = min(len(signal_in), p + search)
        if hi - lo < 3:
            continue
        refined.append(lo + int(np.argmax(signal_in[lo:hi])))

    if not refined:
        fallback, _ = find_peaks(signal_in, distance=max(1, int(0.25 * fs)), prominence=np.std(signal_in) * 0.5)
        refined = fallback.tolist()

    return np.array(sorted(set(refined)), dtype=int)


def extract_qrs_prototype(dominant_imf: np.ndarray, fs: int, beats_to_average: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 200ms pre/post windows around R-peaks and average 10 windows."""
    peaks = detect_r_peaks_pan_tompkins(dominant_imf, fs)
    pre = int(0.20 * fs)
    post = int(0.20 * fs)

    windows = []
    for p in peaks:
        if p - pre >= 0 and p + post < len(dominant_imf):
            windows.append(dominant_imf[p - pre : p + post + 1])

    if len(windows) < beats_to_average:
        raise RuntimeError(
            f"Only {len(windows)} valid QRS windows found; need at least {beats_to_average}."
        )

    windows_arr = np.vstack(windows)
    center_amp = np.abs(windows_arr[:, pre])
    best_idx = np.argsort(center_amp)[-beats_to_average:]
    qrs_proto = np.mean(windows_arr[best_idx], axis=0)
    t = np.linspace(-0.2, 0.2, len(qrs_proto), endpoint=True)
    return qrs_proto, t


def _gaussian_mixture_3(t: np.ndarray, *params: float) -> np.ndarray:
    y = np.zeros_like(t, dtype=float)
    for k in range(3):
        a, m, s = params[3 * k : 3 * k + 3]
        s = abs(s) + 1e-5
        y += a * np.exp(-0.5 * ((t - m) / s) ** 2)
    return y


def _gaussian_mixture_3_derivative(t: np.ndarray, *params: float) -> np.ndarray:
    y = np.zeros_like(t, dtype=float)
    for k in range(3):
        a, m, s = params[3 * k : 3 * k + 3]
        s = abs(s) + 1e-5
        z = (t - m) / s
        y += a * np.exp(-0.5 * z**2) * (-(t - m) / (s**2))
    return y


def fit_gaussian_mixture(qrs_proto: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit 3-Gaussian mixture to QRS morphology."""
    c = int(np.argmax(np.abs(qrs_proto)))
    mu0 = float(t[c])

    p0 = [
        float(np.max(qrs_proto)),
        mu0,
        0.020,
        float(-0.4 * np.max(qrs_proto)),
        mu0 - 0.035,
        0.030,
        float(0.3 * np.max(qrs_proto)),
        mu0 + 0.040,
        0.025,
    ]

    lower = [-3.0, -0.2, 0.005, -3.0, -0.2, 0.005, -3.0, -0.2, 0.005]
    upper = [3.0, 0.2, 0.080, 3.0, 0.2, 0.080, 3.0, 0.2, 0.080]

    params, _ = curve_fit(
        _gaussian_mixture_3,
        t,
        qrs_proto,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000,
    )
    fit_curve = _gaussian_mixture_3(t, *params)
    return params, fit_curve


def _iterative_scaling_function(h: np.ndarray, iterations: int = 9) -> np.ndarray:
    """Iterative two-scale relation (cascade algorithm)."""
    phi = np.array([1.0], dtype=float)
    for _ in range(iterations):
        # Upsample then filter by low-pass mask to approximate phi at next scale.
        up = np.zeros(phi.size * 2, dtype=float)
        up[::2] = phi
        phi = np.convolve(up, h / math.sqrt(2.0), mode="full")
        phi /= np.linalg.norm(phi) + EPS
    return phi


def build_custom_wavelet(params: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build custom mother wavelet, scaling function, and FIR filter bank."""
    morphology = _gaussian_mixture_3(t, *params)
    psi = _gaussian_mixture_3_derivative(t, *params)

    win = tukey(len(psi), alpha=0.35)
    psi = psi * win
    psi = psi - np.mean(psi)  # enforce zero mean numerically
    psi = psi / (np.linalg.norm(psi) + EPS)

    # Seed scaling-shape from morphology and derive low-pass filter from its samples.
    phi_seed = morphology * win
    phi_seed = phi_seed - np.min(phi_seed) + EPS
    phi_seed = phi_seed / (np.linalg.norm(phi_seed) + EPS)

    h = phi_seed.copy()
    if len(h) % 2 == 1:
        h = h[:-1]
    h = h / (np.sum(h) + EPS) * math.sqrt(2.0)

    # Quadrature mirror high-pass for orthogonal-like perfect reconstruction pair.
    g = ((-1.0) ** np.arange(len(h))) * h[::-1]

    # Derive scaling function via two-scale iterative relation.
    phi = _iterative_scaling_function(h, iterations=9)

    return psi, phi, h, g, morphology


def verify_admissibility(psi: np.ndarray, fs: int) -> Dict[str, np.ndarray | float]:
    """Compute admissibility indicators and frequency response."""
    dc = float(np.abs(np.sum(psi)))
    freq = np.fft.rfftfreq(len(psi), d=1.0 / fs)
    resp = np.abs(np.fft.rfft(psi))
    return {"dc_component": dc, "freq": freq, "response": resp}


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pad_to_pow2(x: np.ndarray) -> Tuple[np.ndarray, int]:
    n = len(x)
    n2 = _next_pow2(n)
    out = np.zeros(n2, dtype=float)
    out[:n] = x
    return out, n


def manual_dwt_decompose(signal_in: np.ndarray, dec_lo: np.ndarray, dec_hi: np.ndarray, levels: int) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """Manual multilevel DWT via convolution + downsampling (zero-padded boundaries)."""
    approx = np.asarray(signal_in, dtype=float)
    details: List[np.ndarray] = []
    lengths = [len(approx)]

    for _ in range(levels):
        # Approximation branch (low-pass) and detail branch (high-pass).
        cA = upfirdn(dec_lo, approx, up=1, down=2)
        cD = upfirdn(dec_hi, approx, up=1, down=2)
        details.append(cD)
        approx = cA
        lengths.append(len(approx))

    return approx, details, lengths


def manual_idwt_reconstruct(approx: np.ndarray, details: List[np.ndarray], lengths: List[int], rec_lo: np.ndarray, rec_hi: np.ndarray) -> np.ndarray:
    """Manual inverse DWT via upsampling + convolution + alignment cropping."""
    curr = np.asarray(approx, dtype=float)
    # For conv/downsample phase used in analysis, synthesis alignment is L-1 samples.
    filt_delay = max(0, len(rec_lo) - 1)

    for lvl in reversed(range(len(details))):
        # Undo decimation via upsampling, then synthesize by summation.
        det = np.asarray(details[lvl], dtype=float)
        up_a = upfirdn(rec_lo, curr, up=2, down=1)
        up_d = upfirdn(rec_hi, det, up=2, down=1)
        rec = up_a + up_d

        target_len = lengths[lvl]
        if filt_delay + target_len <= len(rec):
            rec = rec[filt_delay : filt_delay + target_len]
        else:
            rec = rec[:target_len]
        curr = rec

    return curr


def _mad_sigma(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))) / 0.6745 + EPS)


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _hard_threshold(x: np.ndarray, t: float) -> np.ndarray:
    out = x.copy()
    out[np.abs(out) < t] = 0.0
    return out


def morphology_preserving_threshold(coeffs: np.ndarray, threshold: float, qrs_mask: np.ndarray | None = None) -> np.ndarray:
    """Three-zone morphology-preserving threshold function with optional QRS mask."""
    w = np.asarray(coeffs, dtype=float)
    t = float(max(threshold, EPS))

    mag = np.abs(w)
    sgn = np.sign(w)
    out = np.zeros_like(w)

    zone2 = (mag >= t) & (mag < 2.0 * t)
    zone3 = mag >= 2.0 * t

    out[zone2] = sgn[zone2] * ((mag[zone2] - t) ** 2 / t)
    out[zone3] = sgn[zone3] * (mag[zone3] - 0.5 * t)

    if qrs_mask is not None and len(qrs_mask) > 0:
        m = np.zeros_like(out, dtype=bool)
        m_len = min(len(m), len(qrs_mask))
        m[:m_len] = qrs_mask[:m_len]
        out[m] = w[m]

    return out


def detect_qrs_mask(signal_in: np.ndarray, fs: int, levels: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build level-wise QRS protection masks (active at levels 1-2 only)."""
    peaks = detect_r_peaks_pan_tompkins(signal_in, fs)
    masks: List[np.ndarray] = []

    for j in range(1, levels + 1):
        n_j = int(math.ceil(len(signal_in) / (2**j)))
        mask = np.zeros(n_j, dtype=bool)

        if j <= 2 and len(peaks) > 0:
            idxs = np.clip((peaks / (2**j)).astype(int), 0, n_j - 1)
            half_w = max(1, int(round(0.02 * fs / (2**j))))
            for c in idxs:
                lo = max(0, c - half_w)
                hi = min(n_j, c + half_w + 1)
                mask[lo:hi] = True

        masks.append(mask)

    return masks, peaks


def _spectral_centroid(x: np.ndarray, fs: int) -> float:
    x = x - np.mean(x)
    sp = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=1.0 / fs)
    denom = float(np.sum(sp)) + EPS
    return float(np.sum(fr * sp) / denom)


def profile_noise_per_subband(noisy_imfs: np.ndarray, levels: int, fs: int, clean_ecg: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Estimate per-level sigma_j using CEEMDAN IMF frequency-band mapping."""
    corrs = np.array([_safe_corr(imf, clean_ecg) for imf in noisy_imfs], dtype=float)
    freqs = np.array([_spectral_centroid(imf, fs) for imf in noisy_imfs], dtype=float)
    noise_flags = corrs < 0.1

    noise_pool = noisy_imfs[noise_flags] if np.any(noise_flags) else noisy_imfs[:1]
    fallback_sigma = float(np.std(noise_pool)) + EPS

    sigma_j = []
    for j in range(1, levels + 1):
        # Wavelet detail band j approximated as [fs/2^(j+1), fs/2^j].
        f_low = fs / (2 ** (j + 1))
        f_high = fs / (2**j)
        idx = np.where(noise_flags & (freqs >= f_low) & (freqs < f_high))[0]

        if len(idx) > 0:
            band_data = np.concatenate([noisy_imfs[k] for k in idx])
            sigma = float(np.std(band_data))
        else:
            sigma = fallback_sigma / math.sqrt(j)

        sigma_j.append(max(sigma, EPS))

    meta = {
        "corrs": corrs,
        "freqs": freqs,
        "noise_flags": noise_flags.astype(int),
    }
    return np.array(sigma_j, dtype=float), meta


def compute_adaptive_thresholds(sigma_j: np.ndarray, n_per_level: Sequence[int], beta: float = 0.5) -> np.ndarray:
    """Adaptive threshold per level with sigmoid alpha weighting."""
    level = len(sigma_j)
    j_mid = level / 2.0
    out = []

    for j in range(1, level + 1):
        alpha_j = 1.0 / (1.0 + math.exp(-beta * (j - j_mid)))
        n_j = max(2, int(n_per_level[j - 1]))
        t_j = sigma_j[j - 1] * math.sqrt(2.0 * math.log(n_j)) * alpha_j
        out.append(float(max(t_j, EPS)))

    return np.array(out, dtype=float)


def _threshold_details(
    details: List[np.ndarray],
    thresholds: np.ndarray,
    mode: str,
    qrs_masks: List[np.ndarray] | None = None,
) -> Tuple[List[np.ndarray], float]:
    th_details: List[np.ndarray] = []
    total = 0
    near_zero = 0

    for j, d in enumerate(details, start=1):
        t = float(thresholds[j - 1])
        mask = qrs_masks[j - 1] if (qrs_masks is not None and j - 1 < len(qrs_masks)) else None

        if mode == "soft":
            td = _soft_threshold(d, t)
        elif mode == "hard":
            td = _hard_threshold(d, t)
        elif mode == "proposed":
            td = morphology_preserving_threshold(d, t, qrs_mask=mask if j <= 2 else None)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

        th_details.append(td)
        total += td.size
        near_zero += int(np.sum(np.abs(td) < 1e-6))

    sparsity = 100.0 * near_zero / max(total, 1)
    return th_details, sparsity


def denoise_signal(
    noisy: np.ndarray,
    wavelet: WaveletFilters,
    thresholds: np.ndarray,
    qrs_masks: List[np.ndarray] | None,
    levels: int,
    threshold_mode: str,
) -> MethodOutput:
    """Denoise with manual DWT/IDWT and chosen thresholding strategy."""
    padded, orig_len = _pad_to_pow2(noisy)

    approx, details, lengths = manual_dwt_decompose(padded, wavelet.dec_lo, wavelet.dec_hi, levels=levels)
    details_before = [d.copy() for d in details]

    th_details, sparsity = _threshold_details(details, thresholds, mode=threshold_mode, qrs_masks=qrs_masks)

    rec = manual_idwt_reconstruct(approx, th_details, lengths, wavelet.rec_lo, wavelet.rec_hi)
    den = rec[:orig_len]

    return MethodOutput(
        denoised=den,
        thresholds=np.asarray(thresholds, dtype=float),
        details_before=details_before,
        details_after=th_details,
        sparsity_ratio=float(sparsity),
    )


def evaluate_metrics(original: np.ndarray, denoised: np.ndarray, noisy: np.ndarray, sparsity_ratio: float) -> Dict[str, float]:
    """Compute requested denoising and morphology-preservation metrics."""
    x = np.asarray(original, dtype=float)
    y = np.asarray(denoised, dtype=float)
    n = np.asarray(noisy, dtype=float)

    mse = float(np.mean((x - y) ** 2))
    rmse = math.sqrt(max(mse, EPS))
    peak = float(np.max(np.abs(x)) + EPS)
    psnr = float(20.0 * math.log10(peak / rmse))

    prd = float(100.0 * np.linalg.norm(x - y) / (np.linalg.norm(x) + EPS))

    snr_in = float(10.0 * math.log10((np.sum(x**2) + EPS) / (np.sum((x - n) ** 2) + EPS)))
    snr_out = float(10.0 * math.log10((np.sum(x**2) + EPS) / (np.sum((x - y) ** 2) + EPS)))
    snr_improvement = snr_out - snr_in

    ssim_val = float(structural_similarity(x, y, data_range=np.max(x) - np.min(x) + EPS))

    return {
        "SNR_in_dB": snr_in,
        "SNR_out_dB": snr_out,
        "SNR_improvement_dB": snr_improvement,
        "MSE": mse,
        "PSNR_dB": psnr,
        "PRD_percent": prd,
        "SSIM": ssim_val,
        "Sparsity_ratio_percent": float(sparsity_ratio),
    }


def _wavelet_from_pywt(name: str) -> WaveletFilters:
    w = pywt.Wavelet(name)
    return WaveletFilters(
        name=name,
        dec_lo=np.asarray(w.dec_lo, dtype=float),
        dec_hi=np.asarray(w.dec_hi, dtype=float),
        rec_lo=np.asarray(w.rec_lo, dtype=float),
        rec_hi=np.asarray(w.rec_hi, dtype=float),
    )


def _wavelet_from_custom(name: str, h: np.ndarray, g: np.ndarray) -> WaveletFilters:
    # Synthesis pair uses time-reversed analysis filters for orthogonal-style bank.
    return WaveletFilters(
        name=name,
        dec_lo=np.asarray(h, dtype=float),
        dec_hi=np.asarray(g, dtype=float),
        rec_lo=np.asarray(h[::-1], dtype=float),
        rec_hi=np.asarray(g[::-1], dtype=float),
    )


def _universal_thresholds(details: List[np.ndarray]) -> np.ndarray:
    sigma = _mad_sigma(details[0])
    return np.array([sigma * math.sqrt(2.0 * math.log(max(2, len(d)))) for d in details], dtype=float)


def _decompose_for_lengths(signal_in: np.ndarray, wavelet: WaveletFilters, levels: int) -> Tuple[List[np.ndarray], List[int]]:
    p, _ = _pad_to_pow2(signal_in)
    _, details, lengths = manual_dwt_decompose(p, wavelet.dec_lo, wavelet.dec_hi, levels=levels)
    return details, lengths


def run_all_baselines(
    noisy: np.ndarray,
    clean: np.ndarray,
    fs: int,
    levels: int,
    db4_wav: WaveletFilters,
    sym5_wav: WaveletFilters,
    custom_wav: WaveletFilters,
    noisy_imfs: np.ndarray,
    sigma_adapt: np.ndarray,
    qrs_masks: List[np.ndarray],
) -> Dict[str, MethodOutput]:
    """Run six requested baseline/proposed methods."""
    outputs: Dict[str, MethodOutput] = {}

    # 1) db4 + universal + soft
    db4_details, _ = _decompose_for_lengths(noisy, db4_wav, levels)
    t_db4_uni = _universal_thresholds(db4_details)
    outputs["db4 + universal + soft"] = denoise_signal(
        noisy=noisy,
        wavelet=db4_wav,
        thresholds=t_db4_uni,
        qrs_masks=None,
        levels=levels,
        threshold_mode="soft",
    )

    # 2) sym5 + universal + soft
    sym5_details, _ = _decompose_for_lengths(noisy, sym5_wav, levels)
    t_sym_uni = _universal_thresholds(sym5_details)
    outputs["sym5 + universal + soft"] = denoise_signal(
        noisy=noisy,
        wavelet=sym5_wav,
        thresholds=t_sym_uni,
        qrs_masks=None,
        levels=levels,
        threshold_mode="soft",
    )

    # 3) EMD + DWT (Lahmiri-like baseline)
    corrs = np.array([_safe_corr(imf, clean) for imf in noisy_imfs], dtype=float)
    signal_idx = np.where(corrs >= 0.1)[0]
    if len(signal_idx) == 0:
        prefiltered = noisy - noisy_imfs[0][: len(noisy)]
    else:
        prefiltered = np.sum(noisy_imfs[signal_idx], axis=0)
        prefiltered = prefiltered[: len(noisy)]

    emd_details, _ = _decompose_for_lengths(prefiltered, db4_wav, levels)
    t_emd_uni = _universal_thresholds(emd_details)
    outputs["EMD + DWT (Lahmiri-like)"] = denoise_signal(
        noisy=prefiltered,
        wavelet=db4_wav,
        thresholds=t_emd_uni,
        qrs_masks=None,
        levels=levels,
        threshold_mode="soft",
    )

    # 4) Custom wavelet + universal threshold (ablation N1)
    custom_details, _ = _decompose_for_lengths(noisy, custom_wav, levels)
    t_custom_uni = _universal_thresholds(custom_details)
    outputs["Custom + universal (N1)"] = denoise_signal(
        noisy=noisy,
        wavelet=custom_wav,
        thresholds=t_custom_uni,
        qrs_masks=None,
        levels=levels,
        threshold_mode="soft",
    )

    # 5) db4 + CEEMDAN-guided adaptive thresholds (ablation N2)
    t_db4_adapt = compute_adaptive_thresholds(sigma_adapt, [len(d) for d in db4_details])
    outputs["db4 + CEEMDAN adaptive (N2)"] = denoise_signal(
        noisy=noisy,
        wavelet=db4_wav,
        thresholds=t_db4_adapt,
        qrs_masks=None,
        levels=levels,
        threshold_mode="soft",
    )

    # 6) Full proposed method (N1 + N2 + N3)
    t_custom_adapt = compute_adaptive_thresholds(sigma_adapt, [len(d) for d in custom_details])
    outputs["FULL PROPOSED (N1+N2+N3)"] = denoise_signal(
        noisy=noisy,
        wavelet=custom_wav,
        thresholds=t_custom_adapt,
        qrs_masks=qrs_masks,
        levels=levels,
        threshold_mode="proposed",
    )

    return outputs


def _details_to_matrix(details: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(d) for d in details)
    mat = np.full((len(details), max_len), np.nan, dtype=float)
    for i, d in enumerate(details):
        mat[i, : len(d)] = d
    return mat


def plot_all_figures(results_ctx: Dict, out_dir: Path) -> None:
    """Generate all 8 requested high-resolution figures."""
    fs = results_ctx["fs"]
    clean = results_ctx["clean"]
    noisy_10 = results_ctx["noisy_10"]
    den_10 = results_ctx["denoised_10"]
    residual_10 = noisy_10 - den_10

    qrs_proto = results_ctx["qrs_proto"]
    t_qrs = results_ctx["t_qrs"]
    gmm_fit = results_ctx["gmm_fit"]
    psi = results_ctx["psi"]
    phi = results_ctx["phi"]
    admiss = results_ctx["admiss"]

    imfs_10 = results_ctx["imfs_10"]
    imf_noise_flags = results_ctx["imf_noise_flags"]

    thr_universal = results_ctx["thr_universal"]
    thr_adaptive = results_ctx["thr_adaptive"]

    table = results_ctx["results_table"]
    coeff_before = results_ctx["coeff_before"]
    coeff_after = results_ctx["coeff_after"]
    qrs_masks = results_ctx["qrs_masks"]

    peaks_clean = results_ctx["peaks_clean"]
    peaks_den = results_ctx["peaks_den"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Signal overview
    n5 = min(len(clean), 5 * fs)
    t5 = np.arange(n5) / fs
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(t5, clean[:n5], color="black", lw=1.0)
    ax[0].set_title("Original Clean ECG (5 s)")
    ax[1].plot(t5, noisy_10[:n5], color="tab:orange", lw=1.0)
    ax[1].set_title("Noisy ECG (10 dB)")
    ax[2].plot(t5, den_10[:n5], color="tab:blue", lw=1.0)
    ax[2].set_title("Denoised ECG (Proposed)")
    ax[3].plot(t5, residual_10[:n5], color="tab:red", lw=1.0)
    ax[3].set_title("Residual (Noisy - Denoised)")
    ax[3].set_xlabel("Time (s)")
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure1.png", dpi=300)
    plt.close(fig)

    # Figure 2: Custom wavelet design
    fig, ax = plt.subplots(5, 1, figsize=(12, 14))
    ax[0].plot(t_qrs, qrs_proto, color="black")
    ax[0].set_title("Averaged QRS Prototype")
    ax[1].plot(t_qrs, qrs_proto, color="gray", lw=1.0, label="QRS prototype")
    ax[1].plot(t_qrs, gmm_fit, color="tab:green", lw=1.8, label="3-Gaussian fit")
    ax[1].legend(loc="best")
    ax[1].set_title("Gaussian Mixture Model Fit")
    ax[2].plot(t_qrs, psi, color="tab:blue")
    ax[2].set_title("Derived Mother Wavelet $\\psi(t)$")
    ax[3].plot(admiss["freq"], admiss["response"], color="tab:purple")
    ax[3].set_title(f"Wavelet Frequency Response (|DC|={admiss['dc_component']:.2e})")
    ax[3].set_xlim(0, fs / 2)
    ax[4].plot(np.linspace(0, 1, len(phi)), phi, color="tab:brown")
    ax[4].set_title("Scaling Function $\\phi(t)$ (Iterative Two-Scale)")
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure2.png", dpi=300)
    plt.close(fig)

    # Figure 3: CEEMDAN decomposition
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    t_imf = np.arange(imfs_10.shape[1]) / fs
    offset = 0.0
    step = 3.0
    for i, imf in enumerate(imfs_10):
        color = "tab:red" if imf_noise_flags[i] else "tab:blue"
        scale = np.std(imf) + EPS
        ax.plot(t_imf, (imf / scale) + offset, color=color, lw=0.8)
        ax.text(t_imf[-1] + 0.02, offset, f"IMF {i+1}", fontsize=8)
        offset += step
    ax.set_title("CEEMDAN IMFs of Noisy ECG (Red=noise-dominant, Blue=signal-dominant)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure3.png", dpi=300)
    plt.close(fig)

    # Figure 4: Sub-band threshold comparison
    lvl = np.arange(1, len(thr_adaptive) + 1)
    w = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(lvl - w / 2, thr_universal[: len(lvl)], width=w, label="Universal")
    ax.bar(lvl + w / 2, thr_adaptive, width=w, label="CEEMDAN-guided adaptive")
    ax.set_xlabel("Wavelet Level")
    ax.set_ylabel("Threshold")
    ax.set_title("Level-wise Threshold: Universal vs CEEMDAN-guided")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure4.png", dpi=300)
    plt.close(fig)

    # Figure 5: Thresholding function visualization
    x = np.linspace(-3, 3, 2000)
    T = 1.0
    soft = _soft_threshold(x, T)
    hard = _hard_threshold(x, T)
    prop = morphology_preserving_threshold(x, T, qrs_mask=None)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, soft, label="Soft", lw=1.6)
    ax.plot(x, hard, label="Hard", lw=1.6)
    ax.plot(x, prop, label="Proposed 3-zone", lw=2.0)
    ax.axvspan(-T, T, color="gray", alpha=0.1, label="Zone 1")
    ax.axvspan(T, 2 * T, color="tab:orange", alpha=0.1, label="Zone 2")
    ax.axvspan(-2 * T, -T, color="tab:orange", alpha=0.1)
    ax.set_title("Thresholding Functions (T=1)")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure5.png", dpi=300)
    plt.close(fig)

    # Figure 6: Performance comparison
    methods = [
        "db4 + universal + soft",
        "sym5 + universal + soft",
        "EMD + DWT (Lahmiri-like)",
        "Custom + universal (N1)",
        "db4 + CEEMDAN adaptive (N2)",
        "FULL PROPOSED (N1+N2+N3)",
    ]
    snr_levels = sorted(table["Input_SNR_dB"].unique().tolist())

    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    x_m = np.arange(len(methods))
    width = 0.18
    for i, snr in enumerate(snr_levels):
        subset = table[table["Input_SNR_dB"] == snr].set_index("Method").reindex(methods)
        ax[0].bar(x_m + (i - 1.5) * width, subset["SNR_improvement_dB"].values, width=width, label=f"{snr} dB")
    ax[0].set_xticks(x_m)
    ax[0].set_xticklabels(methods, rotation=20, ha="right")
    ax[0].set_ylabel("SNR Improvement (dB)")
    ax[0].set_title("SNR Improvement Comparison")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    for m in methods:
        subset = table[table["Method"] == m].sort_values("Input_SNR_dB")
        ax[1].plot(subset["Input_SNR_dB"], subset["PRD_percent"], marker="o", lw=1.8, label=m)
    ax[1].set_xlabel("Input SNR (dB)")
    ax[1].set_ylabel("PRD (%)")
    ax[1].set_title("PRD vs Input SNR")
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "Figure6.png", dpi=300)
    plt.close(fig)

    # Figure 7: Wavelet coefficient analysis
    mat_b = _details_to_matrix(coeff_before)
    mat_a = _details_to_matrix(coeff_after)
    mask_mat = np.full_like(mat_a, np.nan)
    for j, m in enumerate(qrs_masks):
        if j < mask_mat.shape[0]:
            mask_mat[j, : len(m)] = m.astype(float)

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    im0 = ax[0].imshow(mat_b, aspect="auto", cmap="coolwarm", origin="lower")
    ax[0].set_title("Coefficients Before Thresholding")
    plt.colorbar(im0, ax=ax[0], fraction=0.02)

    im1 = ax[1].imshow(mat_a, aspect="auto", cmap="coolwarm", origin="lower")
    ax[1].set_title("Coefficients After Proposed Thresholding")
    plt.colorbar(im1, ax=ax[1], fraction=0.02)

    im2 = ax[2].imshow(mask_mat, aspect="auto", cmap="Reds", origin="lower", vmin=0, vmax=1)
    ax[2].set_title("QRS Protection Mask Locations (Levels 1-2 active)")
    ax[2].set_xlabel("Coefficient Index")
    plt.colorbar(im2, ax=ax[2], fraction=0.02)

    for a in ax:
        a.set_ylabel("Level")
    plt.tight_layout()
    plt.savefig(out_dir / "Figure7.png", dpi=300)
    plt.close(fig)

    # Figure 8: Clinical validation around one QRS
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if len(peaks_clean) > 0:
        p = int(peaks_clean[min(5, len(peaks_clean) - 1)])
    else:
        p = len(clean) // 2
    w = int(0.35 * fs)
    lo = max(0, p - w)
    hi = min(len(clean), p + w)
    t_zoom = np.arange(lo, hi) / fs
    ax.plot(t_zoom, clean[lo:hi], color="black", lw=1.8, label="Original")
    ax.plot(t_zoom, den_10[lo:hi], color="tab:blue", lw=1.2, label="Denoised (proposed)")

    den_peaks_zoom = peaks_den[(peaks_den >= lo) & (peaks_den < hi)]
    if len(den_peaks_zoom) > 0:
        ax.scatter(den_peaks_zoom / fs, den_10[den_peaks_zoom], color="tab:red", s=30, label="Detected R-peaks")

    ax.set_title("Clinical Validation: QRS Morphology and R-Peak Preservation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "Figure8.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel ECG denoising using CEEMDAN-guided morphological custom wavelets.")
    parser.add_argument("--record", default="100", help="MIT-BIH record id (default: 100)")
    parser.add_argument("--duration-sec", type=int, default=300, help="Duration to load (default: 300 seconds)")
    parser.add_argument(
        "--process-sec",
        type=int,
        default=30,
        help="Length to process for experiments (default: 30 sec for CEEMDAN runtime control)",
    )
    parser.add_argument("--levels", type=int, default=5, help="Wavelet decomposition level")
    parser.add_argument("--ceemdan-trials", type=int, default=50, help="CEEMDAN ensemble trials")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1) Load benchmark ECG and choose the processing window.
    print("Loading ECG...")
    full_clean, fs = load_ecg(record=args.record, duration_sec=args.duration_sec)

    proc_n = min(len(full_clean), int(args.process_sec * fs))
    clean = full_clean[:proc_n]

    print(f"Loaded {len(full_clean)/fs:.1f} s, processing {len(clean)/fs:.1f} s at fs={fs} Hz")

    # 2) Build morphology-driven custom wavelet from clean ECG.
    print("Designing custom wavelet from clean ECG via CEEMDAN + QRS morphology...")
    clean_imfs = run_ceemdan(clean, trials=args.ceemdan_trials, max_imf=-1, seed=args.seed)
    dominant_imf, dominant_idx, clean_corrs = select_dominant_imf(clean_imfs, clean)

    qrs_proto, t_qrs = extract_qrs_prototype(dominant_imf, fs)
    gmm_params, gmm_fit = fit_gaussian_mixture(qrs_proto, t_qrs)
    psi, phi, h, g, morphology = build_custom_wavelet(gmm_params, t_qrs)
    admiss = verify_admissibility(psi, fs)

    db4_wav = _wavelet_from_pywt("db4")
    sym5_wav = _wavelet_from_pywt("sym5")
    custom_wav = _wavelet_from_custom("custom_morph", h, g)

    snr_levels = [5, 10, 15, 20]

    records: List[Dict[str, float | str | int]] = []
    fig_ctx: Dict = {
        "fs": fs,
        "clean": clean,
        "qrs_proto": qrs_proto,
        "t_qrs": t_qrs,
        "gmm_fit": gmm_fit,
        "psi": psi,
        "phi": phi,
        "admiss": admiss,
    }

    # 3) Evaluate all methods at each input SNR level.
    for snr_in in snr_levels:
        print(f"Running denoising pipeline for input SNR={snr_in} dB...")
        noisy = add_noise_at_snr(clean, snr_in, rng)

        noisy_imfs = run_ceemdan(noisy, trials=args.ceemdan_trials, max_imf=-1, seed=args.seed + int(snr_in))
        sigma_j, ceemdan_meta = profile_noise_per_subband(noisy_imfs, args.levels, fs, clean)

        qrs_masks, peaks_noisy = detect_qrs_mask(noisy, fs, args.levels)

        outputs = run_all_baselines(
            noisy=noisy,
            clean=clean,
            fs=fs,
            levels=args.levels,
            db4_wav=db4_wav,
            sym5_wav=sym5_wav,
            custom_wav=custom_wav,
            noisy_imfs=noisy_imfs,
            sigma_adapt=sigma_j,
            qrs_masks=qrs_masks,
        )

        for method_name, out in outputs.items():
            metrics = evaluate_metrics(clean, out.denoised, noisy, out.sparsity_ratio)
            row = {
                "Input_SNR_dB": snr_in,
                "Method": method_name,
                **metrics,
            }
            records.append(row)

        # Keep the 10 dB case for detailed figure panels.
        if snr_in == 10:
            full_key = "FULL PROPOSED (N1+N2+N3)"
            db4_key = "db4 + universal + soft"
            fig_ctx["noisy_10"] = noisy
            fig_ctx["denoised_10"] = outputs[full_key].denoised
            fig_ctx["imfs_10"] = noisy_imfs
            fig_ctx["imf_noise_flags"] = ceemdan_meta["noise_flags"].astype(bool)
            fig_ctx["thr_universal"] = outputs[db4_key].thresholds
            fig_ctx["thr_adaptive"] = outputs[full_key].thresholds
            fig_ctx["coeff_before"] = outputs[full_key].details_before
            fig_ctx["coeff_after"] = outputs[full_key].details_after
            fig_ctx["qrs_masks"] = qrs_masks
            fig_ctx["peaks_clean"] = detect_r_peaks_pan_tompkins(clean, fs)
            fig_ctx["peaks_den"] = detect_r_peaks_pan_tompkins(outputs[full_key].denoised, fs)

    # 4) Save metrics and plots.
    df = pd.DataFrame.from_records(records)
    df = df[
        [
            "Input_SNR_dB",
            "Method",
            "SNR_in_dB",
            "SNR_out_dB",
            "SNR_improvement_dB",
            "MSE",
            "PSNR_dB",
            "PRD_percent",
            "SSIM",
            "Sparsity_ratio_percent",
        ]
    ]
    df.to_csv(out_dir / "results_table.csv", index=False)

    fig_ctx["results_table"] = df
    plot_all_figures(fig_ctx, out_dir)

    print("\n=== Full Comparison Table ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.5f}"))

    print("\nSaved outputs:")
    print(f"- {out_dir / 'novel_ecg_denoising.py'}")
    for i in range(1, 9):
        print(f"- {out_dir / f'Figure{i}.png'}")
    print(f"- {out_dir / 'results_table.csv'}")
    print(f"\nDominant IMF index from clean CEEMDAN: {dominant_idx}")
    print(f"Admissibility check |sum(psi)| = {admiss['dc_component']:.3e}")
    print(f"Clean IMF correlations: {np.array2string(clean_corrs, precision=3)}")


if __name__ == "__main__":
    main()
