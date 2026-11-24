# Tarea 1: ejecución local

## Requisitos

Python 3.10 o superior. Las 24 imágenes Kodak ya están en `tareas/tarea-01/PhotoCD_PCD0992/`; no es necesario descargarlas.

## Windows (PowerShell)

Ejecuta estos comandos desde la raíz del repositorio:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r tareas/tarea-01/requirements.txt
python -m ipykernel install --user --name inf449-tarea1 --display-name "Python (INF449 Tarea 1)"
python -m notebook tareas/tarea-01/homework1.ipynb
```

Si PowerShell bloquea la activación del entorno, usa directamente el ejecutable de ese entorno:

```powershell
.venv\Scripts\python.exe -m pip install -r tareas/tarea-01/requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name inf449-tarea1 --display-name "Python (INF449 Tarea 1)"
.venv\Scripts\python.exe -m notebook tareas/tarea-01/homework1.ipynb
```

En Jupyter, selecciona el kernel **Python (INF449 Tarea 1)** y luego **Run > Run All Cells**. Las tablas CSV se guardan en `tareas/tarea-01/resultados/`. La galería con las 24 imágenes está desactivada por defecto para que la ejecución sea más rápida; cambia `MOSTRAR_TODAS_LAS_IMAGENES` a `True` en la última celda para mostrarla.

## macOS o Linux

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tareas/tarea-01/requirements.txt
python -m ipykernel install --user --name inf449-tarea1 --display-name "Python (INF449 Tarea 1)"
python -m notebook tareas/tarea-01/homework1.ipynb
```

## Datos

El notebook usa `tareas/tarea-01/PhotoCD_PCD0992/` y verifica que encuentre exactamente los archivos `01.png` a `24.png`. Si falta alguno, revisa esa carpeta antes de ejecutar el notebook.

