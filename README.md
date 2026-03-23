# Novel ECG Signal Denoising

Implementation of a CEEMDAN-guided ECG denoising framework with:

1. Morphology-driven custom wavelet design from CEEMDAN-selected dominant IMF and Gaussian-mixture QRS modeling.
2. CEEMDAN-guided per-subband noise profiling with adaptive level thresholds.
3. Morphology-preserving three-zone thresholding with QRS protection mask.

## Files

- `novel_ecg_denoising.py` - full pipeline script
- `results_table.csv` - generated metrics table
- `Figure1.png` ... `Figure8.png` - generated figures

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default run (loads 5 minutes, processes first 30 seconds for tractable CEEMDAN runtime):

```bash
python novel_ecg_denoising.py
```

Process full 5 minutes:

```bash
python novel_ecg_denoising.py --process-sec 300
```

## Output

- Console: full comparison table for all 6 methods x 4 input SNRs.
- Disk: `results_table.csv` and `Figure1.png` through `Figure8.png`.
