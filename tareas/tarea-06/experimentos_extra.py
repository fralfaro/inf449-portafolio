"""Experimentos extra Tarea 6: robustez al ruido (SNR gaussiano y Poisson), L=4 y 16.
Mismo pipeline, semillas y parametros que homework6.ipynb. No modifica notebook ni informe.
Uso: python experimentos_extra.py   (salida: results/extra_*.csv, results/extra_summary.md)
"""
import sys, time
import numpy as np
from experimentos_extra_lib import *

def measure(O, masks, prop, noise, rng):
    Y = np.abs(prop(masks * O[None])) ** 2
    kind, val = noise
    if kind == "gauss":                      # identico al notebook
        sigma = np.sqrt(np.mean(Y**2) / 10 ** (val / 10))
        Y = Y + sigma * rng.standard_normal(Y.shape)
    elif kind == "poisson":                  # val = fotones en el pico de intensidad
        peak = Y.max()
        Y = rng.poisson(Y * (val / peak)).astype(float) * (peak / val)
    return Y

KEYS = ["psnr_amp", "ssim_amp", "psnr_ph", "ssim_ph"]
CONFIGS = [("gauss", 20), ("gauss", 30), ("gauss", 40), ("gauss", 50), ("poisson", 1e3), ("poisson", 1e4)]
rows = ["| Ruido | L | PSNR amp [dB] | SSIM amp | PSNR fase [dB] | SSIM fase |", "|---|---|---|---|---|---|"]
for noise in CONFIGS:
    tag = f"{noise[0]}_{int(noise[1])}"
    for L in L_LIST:
        res = []
        t0 = time.perf_counter()
        for k in range(1, N_IMAGES + 1):
            amp, ph, O = make_object(dataset, k)
            Y = measure(O, masks_by_L[L], prop_fwd, noise, np.random.default_rng(10_000 * L + k))
            Oh = hio_reconstruct(Y, masks_by_L[L], prop_fwd, prop_bwd, n_iter=N_ITER, beta=BETA,
                                 hio_block=HIO_BLOCK, er_block=ER_BLOCK)
            m, _, _ = evaluate(Oh, amp, ph)
            res.append([k, 25 - k] + [m[q] for q in KEYS])
        res = np.array(res)
        np.savetxt(RES_DIR / f"extra_{tag}_L{L}.csv", res, delimiter=",", fmt="%.6f",
                   header="k_fase,k_amplitud," + ",".join(KEYS), comments="")
        mu, sd = res[:, 2:].mean(0), res[:, 2:].std(0)
        rows.append(f"| {noise[0]} {noise[1]:g} | {L} | {mu[0]:.2f} ± {sd[0]:.2f} | {mu[1]:.4f} ± {sd[1]:.4f} | "
                    f"{mu[2]:.2f} ± {sd[2]:.2f} | {mu[3]:.4f} ± {sd[3]:.4f} |")
        print(rows[-1], f"({time.perf_counter()-t0:.0f}s)", flush=True)
        (RES_DIR / "extra_summary.md").write_text("\n".join(rows) + "\n", encoding="utf8")
