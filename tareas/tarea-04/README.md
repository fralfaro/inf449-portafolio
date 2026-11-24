# Tarea 4 — Elementos difractivos y deconvolución

La solución principal está en [`homework4.py`](homework4.py) y el notebook de entrega en [`homework4.ipynb`](homework4.ipynb). Se usa primero el dataset local `McMaster/`; la descarga desde Kaggle es opcional y solo se intenta con `--descargar`.

Desde la raíz del repositorio:

```bash
python tareas/tarea-04/homework4.py --device auto --workers 2
```

Para una prueba corta:

```bash
python tareas/tarea-04/homework4.py --limit 2 --size 128 --iterations 10 --sin-barrido --sin-doe
```

La configuración principal del informe es `size=256`, `sigma=0`, `tau=0.01`, `iterations=25` y `tau_bayer=0.001`. Con `--device auto`, PyTorch usa CUDA cuando está disponible para las FFT propias de PSF y Wiener; SciPy, scikit-image y las métricas siguen en CPU. Usa `--device cpu` para forzar NumPy/SciPy.

El experimento completo guarda las figuras en `Fig/`, las métricas por imagen/método en `resultados/metricas_mcmaster.csv` y las tablas LaTeX. `--solo-tablas` regenera resúmenes y figuras de métricas sin volver a simular. El CSV histórico de barrido contiene promedios sin desglose por imagen y se omite con un aviso; activa el experimento completo para generar el esquema actual.
