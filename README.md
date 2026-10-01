# DM-spikes

Librería de perfiles de densidad de materia oscura. El flujo numérico compone
densidad inicial, potencial, inversión de Eddington, conservación de acción,
integral de densidad final y saturación por aniquilación. También incluye las
alternativas analíticas de ley de potencia y esfera pseudo-isoterma.

Para barridos independientes, ver [ejecución de perfiles](docs/parallel_profiles.md)
y [validación y mediciones](docs/parallel_validation.md).

```python
from dm_spikes.profile_jobs import run_profile, run_profiles
```

```bash
python -m pip install -e .
python -m dm_spikes.profile_cli examples/profile_jobs.json --output resultados.json
python -m dm_spikes.profile_cli examples/profile_jobs.json --workers 2 --output paralelo.json
python -m dm_spikes.profile_cli examples/profile_jobs.json --id power-law-analytic --output uno.json
python -B -m pytest -q
```

Los archivos de salida deben ser nuevos. El pool local requiere un número
explícito de workers y distribuye perfiles completos, con `spawn`. Para un
planificador externo, cada tarea puede seleccionar un ID y escribir su propia
salida. Los ejemplos numéricos son deliberadamente pequeños, pero pueden tardar
mucho más que los analíticos. `execution.py` es un script antiguo fuera de este
flujo.
