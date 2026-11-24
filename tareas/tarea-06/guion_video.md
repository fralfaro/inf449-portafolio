# Guion del video – Homework 6 (Phase Retrieval), ~6–8 min

Objetivo: conectar el informe (`informe.pdf`) con el código (`homework6.ipynb`).

| Min | Qué mostrar | Qué decir |
|---|---|---|
| 0:00 | Portada del informe | Tema: recuperar amplitud y fase de un objeto a partir de intensidades con aperturas codificadas complementarias (L=4 y L=16), usando HIO. |
| 0:30 | Informe §2.1 y notebook §2.2 (`phase_swap.png`) | Por qué importa la fase: al intercambiar fases de Fourier, la estructura sigue a la fase. |
| 1:15 | Notebook §3 (propagador) y celda de verificaciones | Port del MATLAB; el propagador rápido coincide con el literal a 1e-14 y es unitario. N=384 por el criterio λΔz/δ². |
| 2:15 | Notebook §4–5, figuras `masks_L4.png` y `forward_example.png` | Máscaras complementarias por pares (A₂ = ¬A₁); Yℓ = \|P{Aℓ⊙O}\|² + Ω con SNR de 40 dB; fase = imagen k, amplitud = imagen 25−k. |
| 3:15 | Notebook §6 (`hio_reconstruct`) y §8 (`beta_ablation.png`) | HIO con consenso entre máscaras. **Aclarar:** HIO puro diverge con ruido, por eso el resultado principal usa bloques 10 HIO + 10 ER (β=0.9); el HIO puro se muestra en la ablación. |
| 4:15 | Notebook §7 | PSNR y SSIM según Ecs. (5)–(7); coinciden con skimage. |
| 4:45 | Notebook §9–11 y tabla resumen; `metrics_per_image.png`, `metrics_boxplot.png` | L=4: 44.10 dB / 0.987 (amp), 25.57 dB / 0.823 (fase). L=16: 51.39 dB / 0.997 y 29.10 dB / 0.904. Mejora en las 24 imágenes. |
| 5:45 | `convergence.png`, `visual_examples.png`, `phase_error_vs_amplitude.png`, celda de estadísticas | L=16 converge en 15 iteraciones frente a 25. El error de fase se concentra donde la amplitud es casi nula (Spearman ≈ −0.9). |
| 6:45 | Conclusiones del informe | Calidad vs. costo: 4× más mediciones y ≈3.6× más cómputo por iteración. Limitaciones: fase en [0,1], ruido gaussiano, solo máscaras aleatorias. |
| 7:30 | Carpeta del proyecto | Archivos entregados: notebook, informe, CSV y figuras. |
