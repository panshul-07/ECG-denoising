# ECG Denoising Demo Notes (VS Code + Presentation)

## 1) Run in VS Code (one-click)
1. Open folder: `/Users/panshulaj/Documents/dsp proj`
2. Open **Terminal -> Run Task**
3. Run: `Setup: Create venv + install deps` (first time only)
4. Run: `Run: ECG quick` (fast demo, ~10s signal)
5. Outputs are generated in `results/`:
   - `results/results_table.csv`
   - `results/Figure1.png` ... `results/Figure8.png`

Alternative:
- Press `F5` and choose `ECG Denoising: Quick (10s, 10 trials)`.

## 2) Dataset (what to say)
- Dataset: **MIT-BIH Arrhythmia Database** (PhysioNet), **Record 100**.
- Lead used: **MLII**.
- Script loads up to 5 minutes (`--duration-sec 300`) and processes a shorter segment for speed (`--process-sec 10` or `30`).
- Synthetic Gaussian noise is added at input SNR = **5, 10, 15, 20 dB** for controlled benchmarking.

## 3) How the method works (simple explanation)
1. CEEMDAN decomposes clean ECG into IMFs.
2. Dominant IMF is selected by highest correlation with clean ECG.
3. QRS beats are detected, aligned, averaged -> QRS prototype.
4. 3-Gaussian model is fit to prototype.
5. Derivative of fitted model becomes custom mother wavelet.
6. For noisy ECG, CEEMDAN estimates noise per sub-band.
7. Adaptive threshold per wavelet level is computed.
8. 3-zone thresholding + QRS mask protects R-peaks.
9. Signal is reconstructed and compared with baselines.

## 4) Comparison with previous works (what is different)
Previous works:
- Fixed wavelets (`db4`, `sym5`) not tied to ECG morphology.
- Global noise estimate (single threshold strategy).
- Standard hard/soft thresholding may distort sharp QRS details.

This project:
- Morphology-driven custom wavelet from CEEMDAN + QRS Gaussian model.
- CEEMDAN-guided **per-sub-band** noise profiling.
- New 3-zone thresholding with QRS protection mask.

## 5) What current results show (important)
- After fixing inverse-DWT alignment, classical baselines now show expected positive SNR gains in many settings.
- `db4 + CEEMDAN adaptive (N2)` performs best overall in current run.
- Custom-wavelet variants (`Custom + universal`, `FULL PROPOSED`) still underperform due unresolved custom filter-bank reconstruction quality.

Use this line in presentation:
- "Our ablation shows CEEMDAN-guided adaptive thresholding is effective. The remaining work is to enforce strict perfect-reconstruction constraints for the morphology-derived filter bank to unlock full-method gains."

## 6) Quick speaking script (2 minutes)
- "We used MIT-BIH record 100 (MLII) and injected controlled Gaussian noise at 5–20 dB."
- "We compared six methods: two classical wavelet baselines, one EMD+DWT baseline, two ablations, and the full proposed method."
- "Our key novelty is morphology-aware design: QRS-derived custom wavelet + CEEMDAN sub-band noise modeling + 3-zone thresholding with QRS protection."
- "The current strongest practical result is CEEMDAN-guided adaptive thresholding with db4; this validates the sub-band adaptive idea."
- "The custom-wavelet branch needs further PR-constrained filter-bank refinement, which is our next step."

## 7) Presentation file
- Local PPT: `results/ECG_Denoising_Presentation_24BEC0177.pptx`
- Open with:
```bash
open "/Users/panshulaj/Documents/dsp proj/results/ECG_Denoising_Presentation_24BEC0177.pptx"
```
