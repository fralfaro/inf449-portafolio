# Tarea 4 — Elementos difractivos y deconvolución

La solución principal está en [`homework4.py`](homework4.py) y el notebook de entrega en [`homework4.ipynb`](homework4.ipynb). El script descarga el dataset McMaster desde el enlace de Kaggle indicado en la guía si no encuentra las 18 imágenes localmente.

Desde la raíz del repositorio:

```bash
python3 tareas/tarea-04/homework4.py
```

Para una prueba corta:

```bash
python3 tareas/tarea-04/homework4.py --limit 2 --size 128 --iterations 10
```

El experimento completo guarda las figuras en `Fig/` y las métricas por imagen/método en `resultados/metricas_mcmaster.csv`. La configuración usada en el informe es `size=256`, `iterations=25` y `balance=0.01`.
