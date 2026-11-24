"""Experimentos extra Tarea 6: robustez al ruido (SNR gaussiano y ruido de Poisson).
Reutiliza el pipeline de homework6.ipynb (mismas semillas, máscaras y parámetros).
No modifica el notebook ni el informe. Salida: results/extra_*.csv y results/extra_summary.md
"""
import time, sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SEED=4
DATA_DIR=Path("dataset"); RES_DIR=Path("results"); RES_DIR.mkdir(exist_ok=True)
M=384; N_IMAGES=24; L_LIST=[4,16]
WVL=643e-6; DELTA1=0.02; DELTAN=0.02; Z=np.linspace(0,160,5)
BETA=0.9; N_ITER=100; HIO_BLOCK,ER_BLOCK=10,10


def load_dataset(data_dir, n_images=24, size=384):
    # Carga k.gif (k=1..n), normaliza a [0,1] (mat2gray) y recorta el centro size x size.
    images = []
    for k in range(1, n_images + 1):
        img = np.asarray(Image.open(data_dir / f"{k}.gif").convert("L"), dtype=np.float64)
        img = (img - img.min()) / (img.max() - img.min())          # mat2gray
        d1, d2 = img.shape
        r0, c0 = (d1 - size) // 2, (d2 - size) // 2
        images.append(img[r0:r0 + size, c0:c0 + size])
    return np.stack(images)


dataset = load_dataset(DATA_DIR, N_IMAGES, M)


def ft2(g, dx):
    # DFT 2D centrada (Schmidt): fftshift(fft2(fftshift(g))) * dx^2
    return sfft.fftshift(sfft.fft2(sfft.fftshift(g, axes=(-2, -1))), axes=(-2, -1)) * dx**2

def ift2(G, df):
    # IDFT 2D centrada (Schmidt): ifftshift(ifft2(ifftshift(G))) * (N df)^2
    N = G.shape[-1]
    return sfft.ifftshift(sfft.ifft2(sfft.ifftshift(G, axes=(-2, -1))), axes=(-2, -1)) * (N * df)**2

def ang_spec_pro(Uin, wvl, d1, d2, Dz):
    # Port de ang_spec_pro.m: propagación por espectro angular en un paso.
    N = Uin.shape[-1]
    k = 2 * np.pi / wvl
    n = np.arange(-N // 2, N // 2)
    x1, y1 = np.meshgrid(n * d1, n * d1); r1sq = x1**2 + y1**2
    df1 = 1 / (N * d1)
    fX, fY = np.meshgrid(n * df1, n * df1); fsq = fX**2 + fY**2
    m = d2 / d1
    x2, y2 = np.meshgrid(n * d2, n * d2); r2sq = x2**2 + y2**2
    Q1 = np.exp(-1j * k / 2 * (1 - m) / Dz * r1sq)
    Q2 = np.exp(1j * np.pi**2 * 2 * Dz / m / k * fsq)
    Q3 = np.exp(-1j * k / 2 * (m - 1) / (m * Dz) * r2sq)
    return Q3 * ift2(Q2 * ft2(Q1 * Uin / m, d1), df1)

def ang_spec_multi_prop(Uin, wvl, delta1, deltan, z):
    # Port literal de ang_spec_multi_prop.m (split-step, sin absorbente: sg = 1).
    z = np.asarray(z, dtype=float)
    N = Uin.shape[-1]
    n = np.arange(-N // 2, N // 2)
    nx, ny = np.meshgrid(n, n)
    k = 2 * np.pi / wvl
    delta_z = np.diff(z)
    alpha = z / z[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:-1]
    r1sq = (nx * delta[0])**2 + (ny * delta[0])**2
    U = Uin * np.exp(-1j * k / 2 * (1 - m[0]) / delta_z[0] * r1sq)
    for idx in range(len(z) - 1):
        deltaf = 1 / (N * delta[idx])
        fsq = (nx * deltaf)**2 + (ny * deltaf)**2
        Q2 = np.exp(1j * np.pi**2 * 2 * delta_z[idx] / m[idx] / k * fsq)
        U = ift2(Q2 * ft2(U / m[idx], delta[idx]), deltaf)
    rnsq = (nx * delta[-1])**2 + (ny * delta[-1])**2
    Q3 = np.exp(-1j * k / 2 * (m[-1] - 1) / (m[-1] * delta_z[-1]) * rnsq)
    return Q3 * U



class NearFieldPropagator:
    # P[Δz] por espectro angular multi-paso (Schmidt), precalculado. Acepta (..., N, N).
    def __init__(self, N, wvl, delta1, deltan, z):
        z = np.asarray(z, dtype=float)
        k = 2 * np.pi / wvl
        n = np.arange(-N // 2, N // 2)
        nx, ny = np.meshgrid(n, n)
        nsq = (nx**2 + ny**2).astype(float)
        dz = np.diff(z)
        alpha = z / z[-1]
        delta = (1 - alpha) * delta1 + alpha * deltan
        m = delta[1:] / delta[:-1]
        H = np.ones((N, N), dtype=complex)
        scale = 1.0
        for i in range(len(dz)):
            df = 1 / (N * delta[i])
            H *= np.exp(1j * np.pi**2 * 2 * dz[i] / m[i] / k * nsq * df**2)
            scale /= m[i]
        self.Q1 = np.exp(-1j * k / 2 * (1 - m[0]) / dz[0] * nsq * delta[0]**2)
        self.Q3 = np.exp(-1j * k / 2 * (m[-1] - 1) / (m[-1] * dz[-1]) * nsq * delta[-1]**2)
        self.trivial_quadratic = np.allclose(m, 1.0)      # Q1 = Q3 = 1 si delta1 == deltan
        self.H = sfft.ifftshift(H) * scale                # H en orden "no centrado" de la FFT

    def __call__(self, U):
        if self.trivial_quadratic:
            return sfft.ifft2(self.H * sfft.fft2(U, workers=-1), workers=-1)
        U = sfft.ifftshift(self.Q1 * U, axes=(-2, -1))
        U = sfft.ifft2(self.H * sfft.fft2(U, workers=-1), workers=-1)
        return self.Q3 * sfft.fftshift(U, axes=(-2, -1))



prop_fwd = NearFieldPropagator(M, WVL, DELTA1, DELTAN, Z)
prop_bwd = NearFieldPropagator(M, WVL, DELTA1, DELTAN, -Z)


def complementary_masks(masks):
    # Port de complementary_masks.m: T(:,:,i) = masks(:,:,c); T(:,:,i+1) = ~T(:,:,i).
    L = masks.shape[0]
    assert L % 2 == 0, "L debe ser par para máscaras complementarias por pares"
    T = np.empty_like(masks)
    T[0::2] = masks[: L // 2]
    T[1::2] = 1.0 - T[0::2]
    return T

def select_masks(L, M, rng, complementary=True):
    # Port de selectMasks.m para codes == 1 (máscaras aleatorias).
    masks = (rng.random((L, M, M)) > 0.5).astype(np.float64)
    return complementary_masks(masks) if complementary else masks

masks_by_L = {L: select_masks(L, M, np.random.default_rng(SEED + L)) for L in L_LIST}


def make_object(dataset, k):
    # k en 1..24: fase = imagen k, amplitud = imagen 25-k (como en main.m).
    amplitude = dataset[(25 - k) - 1]
    phase = dataset[k - 1]
    return amplitude, phase, amplitude * np.exp(1j * phase)



def hio_schedule(n_iter, hio_block=10, er_block=10):
    # True = iteración HIO, False = iteración ER. Ciclos [hio_block x HIO, er_block x ER].
    if er_block == 0:
        return np.ones(n_iter, dtype=bool)
    cycle = np.r_[np.ones(hio_block, bool), np.zeros(er_block, bool)]
    return np.resize(cycle, n_iter)

def hio_reconstruct(Y_list, masks, prop_fwd, prop_bwd, n_iter=100, beta=0.9, hio_block=10, er_block=10,
                    O_init=None, callback=None, return_history=False):
    # Hybrid Input-Output para phase retrieval con L aperturas codificadas.
    # Y_list: (L, M, M) mediciones; masks: (L, M, M) máscaras binarias.
    # n_iter iteraciones totales, en ciclos de hio_block HIO + er_block ER (er_block=0 -> HIO puro).
    # callback(it, O_hat) opcional. Retorna O_hat (M, M) complejo y, si return_history, E_k.
    Y = np.asarray(Y_list, dtype=float)
    A = np.asarray(masks, dtype=float)
    sqrtY = np.sqrt(np.maximum(Y, 0.0))                    # magnitud medida (ruido -> Y < 0)
    weight = A.sum(axis=0)                                 # Σ_l A_l  (= L/2 si complementarias)
    if np.any(weight == 0):
        raise ValueError("Hay píxeles no iluminados por ninguna máscara.")

    if O_init is None:                                     # fieldvec = ones(M,N), con energía correcta
        O_init = np.full(A.shape[1:], np.sqrt(Y.sum() / weight.sum()), dtype=complex)
    g = A * O_init[None]                                   # ondas de salida ψ_l = A_l ⊙ O
    schedule = hio_schedule(n_iter, hio_block, er_block)
    history = np.empty(n_iter)

    for it in range(n_iter):
        # --- (1) Restricción de medición P_M, para las L máscaras -------------------
        G = prop_fwd(g)                                    # P[Δz]{ψ_l}
        G_new = sqrtY * G / np.maximum(np.abs(G), 1e-30)   # √Y_l · exp(i∠G_l)
        g_prime = prop_bwd(G_new)                          # P[-Δz]{·}  -> g'_k

        # --- (2) Restricción de objeto: proyección P_S (soporte + consistencia) ------
        O_hat = (A * g_prime).sum(axis=0) / weight         # objeto consenso
        PS_gp = A * O_hat[None]                            # P_S(g'_k)   (región válida)
        violation = g_prime - PS_gp                        # (I - P_S) g'_k
        history[it] = np.linalg.norm(violation) / np.linalg.norm(g_prime)

        if schedule[it]:                                   # HIO
            PS_g = A * ((A * g).sum(axis=0) / weight)[None]
            g = PS_gp + (g - PS_g) - beta * violation      # P_S g' + (I-P_S)(g - β g')
        else:                                              # ER: g = P_S(g')
            g = PS_gp
        if callback is not None:
            callback(it, O_hat)

    return (O_hat, history) if return_history else O_hat



def psnr_eq(I, K, max_i=1.0):
    # Ecs. (5)-(6).
    mse = np.mean((I.astype(float) - K.astype(float)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(max_i**2 / mse)

def ssim_eq(x, y, data_range=1.0, K1=0.01, K2=0.03, sigma=1.5, truncate=3.5):
    # Ec. (7) con ventana gaussiana (estadísticos poblacionales), promedio del mapa SSIM.
    x = x.astype(float); y = y.astype(float)
    w = lambda img: gaussian_filter(img, sigma=sigma, truncate=truncate, mode="reflect")
    mu_x, mu_y = w(x), w(y)
    var_x = w(x * x) - mu_x**2
    var_y = w(y * y) - mu_y**2
    cov_xy = w(x * y) - mu_x * mu_y
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    S = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))
    pad = int(truncate * sigma + 0.5)                      # borde de la ventana (11x11)
    return S[pad:-pad, pad:-pad].mean()

def ssim_sk(x, y):
    return structural_similarity(x, y, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)

def align_global_phase(O_hat, O_true):
    # Elimina la ambigüedad de fase global: O_hat * exp(-i θ*), θ* = ∠<O_true, O_hat>.
    return O_hat * np.exp(-1j * np.angle(np.vdot(O_true, O_hat)))

def evaluate(O_hat, amplitude, phase):
    O_al = align_global_phase(O_hat, amplitude * np.exp(1j * phase))
    amp_hat = np.clip(np.abs(O_al), 0, 1)
    ph_hat = np.clip(np.angle(O_al), 0, 1)
    return {
        "psnr_amp": peak_signal_noise_ratio(amplitude, amp_hat, data_range=1.0),
        "ssim_amp": ssim_sk(amplitude, amp_hat),
        "psnr_ph": peak_signal_noise_ratio(phase, ph_hat, data_range=1.0),
        "ssim_ph": ssim_sk(phase, ph_hat),
    }, amp_hat, ph_hat

