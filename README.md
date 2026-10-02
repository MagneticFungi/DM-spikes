# DM-spikes

Librería de perfiles de densidad de materia oscura. El flujo numérico compone
densidad inicial, potencial, inversión de Eddington, conservación de acción,
integral de densidad final y saturación por aniquilación. También incluye las
alternativas analíticas de ley de potencia y esfera pseudo-isoterma.

Para calcular un perfil a la vez repartiendo sus radios entre procesos, ver
[ejecución de perfiles](docs/parallel_profiles.md). Las dos rutas de ejecución
son serial y radial; los perfiles analíticos usan la ruta serial.

```python
from dm_spikes.profile_jobs import run_profile, run_profile_radial, iter_profiles_radial
```

```bash
python -m pip install -e .
python -m dm_spikes.profile_cli profiles.json --output serial.json
python -m dm_spikes.profile_cli profiles.json --workers 8 --output-dir resultados
python -m dm_spikes.profile_cli profiles.json --id halo-nfw --workers 8 --output nfw.json
python -B -m pytest -q
```

`profiles.json` es un manifiesto del usuario; la guía describe su formato.
Los archivos de salida deben ser nuevos. Con `--workers`, cada
worker construye y reutiliza sus callables y cachés; `--output-dir` guarda el
perfil terminado antes de iniciar el siguiente. El pool usa `spawn` y exige un
número explícito de workers. Un planificador externo puede seleccionar un ID
y una salida propia. No se han cambiado las cuadraturas ni el mapeo numérico.
`execution.py` permanece fuera de este flujo.

Las pruebas y mediciones actuales se describen en
[validación por radios](docs/radial_validation.md). El
[informe por perfiles](docs/parallel_validation.md) documenta una etapa anterior
cuyo backend ya no forma parte de la librería.
