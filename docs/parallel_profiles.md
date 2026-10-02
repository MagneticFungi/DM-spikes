# Ejecución de perfiles y paralelización por radios

La modalidad radial calcula **un perfil a la vez y reparte sus radios** entre
procesos. Cada worker construye una vez la densidad, el potencial, la DF y el
perfil final. Reutiliza sus callables y cachés al recibir nuevos radios. Solo se
transmiten configuraciones serializables y bloques de radios, nunca cierres ni
cachés construidas. Las APIs escalares y la ruta serial se conservan.
`execution.py` queda fuera de este flujo.

## Diagnóstico y decisiones

- El potencial inicial mantiene dos LRU (4096 masas y 2048 integrales exteriores)
  y anclas ordenadas de masa. Cada worker conserva su propia copia. Aumentar
  procesos duplica memoria y parte del trabajo que una ejecución serial reutiliza.
- El coste dominante está en las cuadraturas anidadas de densidad final. Cada
  muestra requiere el mapeo de energía, raíces, integrales de acción y Eddington.
- Se reparte una evaluación completa de `rho_prime(r)` por tarea de forma
  predeterminada. Las cuadraturas adaptativas dentro de cada radio siguen siendo
  escalares, con sus mismas tolerancias, sumas y control de error.
- Los workers reciben la configuración una vez y construyen los callables al
  atender su primera tarea. Una construcción fallida se propaga con ID y causa.
  No se permite crear pools anidados. Se usa siempre `spawn`, incluso con un worker.
- Las historias de caché pueden diferir de serial; la comparación numérica usa
  tolerancias. No se introducen tablas, interpolaciones ni aproximaciones al mapeo.
  Unidades, captura, energía, fronteras, acción y saturación siguen las APIs actuales.
- Las alternativas analíticas ya aceptan arrays. El modo radial exige perfiles
  numéricos; para cálculos analíticos baratos conviene la ruta serial vectorizada.

## Uso desde Python

Instalar con `python -m pip install -e .`. No se agregan dependencias de ejecución.
El usuario proporciona `profiles.json`, una lista de configuraciones como la
mostrada más abajo. Guardar este ejemplo en un módulo ejecutable con protección
`__main__`; en notebooks, usar la CLI para los pools.

```python
import json
from dm_spikes.profile_jobs import run_profile, run_profiles, run_profile_radial

if __name__ == "__main__":
    with open("profiles.json", encoding="utf-8") as stream:
        jobs = json.load(stream)
    serial = run_profiles(jobs)
    single = run_profile(jobs[0])
    # Perfiles secuenciales y radios distribuidos dentro de cada perfil.
    radial = run_profiles(jobs, backend="radial", max_workers=8, block_size=1)
    single_radial = run_profile_radial(jobs[0], max_workers=8)
```

El pool radial requiere workers explícitos; `serial` sigue siendo el backend por
omisión. La biblioteca no usa Slurm, afinidad asumida ni CPU detectadas como
permiso para ocuparlas. Cada perfil radial tiene un pool nuevo: al terminar se
cierra, se entrega el resultado y solo entonces puede empezar el siguiente.

Para consumir o guardar cada perfil terminado, usar `iter_profiles_radial`:

```python
import json
from pathlib import Path
from dm_spikes.profile_jobs import iter_profiles_radial

if __name__ == "__main__":
    jobs = json.loads(Path("profiles.json").read_text(encoding="utf-8"))
    directory = Path("resultados")
    directory.mkdir(exist_ok=True)
    for index, result in enumerate(iter_profiles_radial(jobs, max_workers=8)):
        with (directory / f"perfil_{index:03d}.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, allow_nan=False)
        # El siguiente perfil solo empieza al avanzar el iterador.
```

El iterador valida todo el lote antes del primer pool y cierra el pool actual
antes de entregar su resultado. Interrumpir la iteración entre perfiles no deja
workers pendientes. `run_profiles(..., backend="radial")` recoge todo en una lista;
para conservar lo completado si falla un perfil posterior, guardar usando el
iterador o la CLI con `--output-dir`. Las APIs Python no escriben archivos.

`block_size=1` equilibra dinámicamente radios con costes distintos. Bloques mayores
agrupan entradas consecutivas de la lista sin cambiar su evaluación. Cada worker
conserva su caché entre bloques: no se reconstruye por radio. Se mantienen como
máximo `2 * workers` tareas pendientes de resolución. El máximo efectivo del pool
es `min(max_workers, ceil(numero_de_radios / block_size))`. Para 600 radios y
128 workers, bloques de 8 permiten como máximo 75 procesos; bloques de 1 permiten
128. Más procesos no garantizan mayor velocidad. En Windows, `ProcessPoolExecutor`
admite hasta 61 procesos por pool; la validación local usa uno y dos.

Los resultados conservan el orden de los perfiles y radios, incluso desordenados
o repetidos. Incluyen `id`, copia de `config`, `radii`, `rho_prime`, `rho_spike`
si se pide saturación y metadatos `execution`. En modo radial estos últimos
incluyen PID del padre, PIDs de workers que devolvieron resultados, workers
solicitados y máximo efectivo, bloque, hilos internos y segundos completos,
incluido arranque/cierre del pool. En serial, el tiempo es el de cálculo dentro
del proceso.

## Configuración numérica

Un JSON puede contener un objeto o una lista de estos objetos. Este ejemplo
ilustra el formato; sus parámetros no constituyen un ajuste del halo.

```json
{
  "id": "halo-nfw",
  "M_bh": 4000000.0,
  "radii": [0.001, 0.01, 0.1],
  "density": {
    "factory": "dm_spikes.density_models:nfw",
    "parameters": {"rho_s": 0.02, "r_s": 1500.0}
  },
  "boundary": "finite_escape",
  "density_derivatives": true,
  "potential_derivatives": true
}
```

`engine` omitido significa `numerical`. `M_bh` y una lista no vacía de radios
positivos son obligatorios. `G` y `clight` usan exactamente los defaults de
`constants.py`: pc, km/s y Msun, densidad en Msun/pc³.

| Factoría | Parámetros | Densidad |
| --- | --- | --- |
| `dm_spikes.density_models:power_law` | `rho0`, `r0`, `gamma` | `rho0*(r/r0)^(-gamma)` |
| `dm_spikes.density_models:nfw` | `rho_s`, `r_s` | `rho_s/[x*(1+x)^2]`, `x=r/r_s` |
| `dm_spikes.density_models:hernquist` | `total_mass`, `scale_radius` | `M*a/[2*pi*r*(r+a)^3]` |

Estas factorías entregan la densidad inicial y sus derivadas exactas; el potencial,
la DF y el mapeo siguen siendo numéricos. Elegir frontera explícita: NFW/Hernquist
usan `finite_escape`, potencial cero en infinito. Una ley pura con
`0 < gamma <= 2` usa `confining` y `r_ref` positivo, potencial cero allí; para
`2 < gamma < 3` la frontera convergente es `finite_escape`. La energía pasada
al mapeo y a la DF no se desplaza ni renormaliza automáticamente.

- `density_derivatives` y `potential_derivatives`, falsas por omisión, pasan
  `.prime` y `.second` de los callables a Eddington cuando se activan.
- `eddington`: `escape_slope`, `central_potential`, `tail_survey_log_radius`,
  con el significado y defaults de `make_eddington_df`.
- `final`: `epsabs`, `epsrel`, `limit`, `map_kwargs`, reenviados sin alteración
  a `make_final_profile`; las opciones de acción se anidan en `map_kwargs`.
- `annihilation`: `m`, `observable_sigma_v`, opcionalmente `bh_age`, aplicados
  mediante `rho_spike`. `m/(observable_sigma_v*bh_age)` debe tener las mismas
  unidades de densidad; no se convierten implícitamente GeV, cm³/s o años.

Para una densidad arbitraria, definir una factoría en un módulo importable,
instalado o accesible por `PYTHONPATH`, y usar `"factory": "mi_modulo:mi_factoria"`.
Recibe los argumentos de `parameters` y devuelve un callable escalar; si se piden
sus derivadas, debe adjuntar `.prime` y `.second`. Los cierres se construyen en los
hijos. Las factorías son código Python de confianza, determinista y sin escrituras
compartidas. No se aceptan lambdas, callables, arrays NumPy, objetos, NaN ni infinito
en la configuración; convertir los radios con `.tolist()`. Los IDs deben ser únicos.

La configuración JSON y su estructura se validan antes del pool. Errores al importar,
construir o evaluar se propagan como `ProfileExecutionError`, con ID y causa. Los
fallos durante una evaluación radial incluyen índice y radio. Un proceso terminado
abruptamente identifica el perfil incompleto; no se inventa el radio responsable.
Los fallos no se convierten en ceros. El cero físico por captura se conserva.

Las alternativas `engine="analytic_power_law"` y `"analytic_isothermal"` mantienen
sus parámetros y su vectorización actual; siguen disponibles en serial.
No hay sustitución automática de un perfil numérico por
una fórmula analítica según su nombre.

## CLI y planificador externo

```bash
python -m dm_spikes.profile_cli profiles.json --output serial.json
python -m dm_spikes.profile_cli profiles.json --workers 8 --block-size 1 --output-dir resultados
python -m dm_spikes.profile_cli profiles.json --id halo-nfw --output nfw-serial.json
python -m dm_spikes.profile_cli profiles.json --id halo-nfw --workers 8 --output nfw-radial.json
```

Un job array externo mapea su índice a un ID y ejecuta uno de los comandos con
`--id`, con archivo o directorio propio. La CLI no interpreta variables del
planificador. La ruta radial funciona en un nodo; varios nodos requieren tareas
externas. Los workers deben respetar la asignación de cada tarea.

`--workers` activa el modo radial; el anterior `--parallel-axis radii` sigue
aceptándose como alias opcional. `--output-dir` requiere el modo radial y crea
el directorio. Produce archivos
como `000_halo-nfw.json`, con índice y ID saneado; el ID original y los parámetros
se conservan dentro. Escribe cada resultado desde el padre **antes de comenzar
el siguiente perfil**. Ante un fallo, los perfiles guardados permanecen y los
siguientes no se calculan. No hay guardado parcial por radio ni reanudación
automática. Para repetir un perfil, seleccionar su ID y una salida nueva.

Para `--output`, la carpeta debe existir y el archivo se escribe al finalizar
la ejecución solicitada. Sin opción de salida se devuelve JSON por stdout.
Con `--output-dir`, stdout informa cada perfil guardado. Los errores salen por
stderr con código distinto de cero. Las salidas usan creación exclusiva: no se
sobrescriben archivos existentes, incluso si otra tarea los crea después de la
comprobación previa. Un corte durante la escritura puede dejar JSON parcial;
consumirlo solo tras una salida exitosa y validar el JSON.

Ante un error se cancelan las tareas pendientes que aún puedan cancelarse y se
espera a las evaluaciones ya en curso. Esto puede tardar si una integral es lenta;
no hay timeout interno ni terminación forzada de cálculos científicos.

## Recursos y despliegue

Cada pool establece `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS` al valor de
`inner_threads` (CLI `--inner-threads`, por omisión 1) y `OMP_MAX_ACTIVE_LEVELS=1`
**antes de spawn**, y restaura el entorno del padre incluso ante errores. Así
limita bibliotecas que respetan esas variables sin añadir dependencias. No puede
reconfigurar bibliotecas ya cargadas en el padre ni garantizar controles de
bibliotecas personalizadas. Para serial, fijar el entorno antes de iniciar Python:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
python -m dm_spikes.profile_cli profiles.json --id halo-nfw --output nfw.json
```

En PowerShell se usa `$env:OMP_NUM_THREADS='1'`, etc. Las variables son globales:
las llamadas a pools de esta biblioteca se serializan con un lock. Evitar que
otros hilos cambien ese entorno o lancen procesos durante una llamada. No se
cambia globalmente el método de arranque de `multiprocessing`.

Para ajustar el despliegue hacen falta CPU y RAM asignadas por tarea, tiempo
máximo, sistema operativo, Python/NumPy/SciPy y BLAS, políticas de afinidad e
hilos, almacenamiento/cuotas y la malla y tolerancias de producción. Con `P`
procesos y `T` hilos, empezar con `P*T` dentro de la asignación y memoria para
cada intérprete, SciPy y cachés. Disponer de 128 CPU lógicas no demuestra que
128 workers sea la opción más rápida: medir con la carga real antes de ampliarla.

## Pruebas y benchmark

La suite rápida usa procesos reales; verifica construcción numérica y captura
para ley de potencia, NFW y Hernquist, además de aislamiento, cachés y fallos.

```bash
python -B -m pytest -q
```

La prueba lenta integra la densidad completa en dos radios fuera de captura y
uno dentro. Compara serial con dos workers radiales, verifica orden, identidad
y saturación, y guarda tiempos completos, parámetros, versiones y resultados.
En PowerShell:

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'; $env:VECLIB_MAXIMUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
$env:OMP_MAX_ACTIVE_LEVELS='1'; $env:DM_SPIKES_RUN_SLOW='1'
$env:DM_SPIKES_BENCHMARK_OUTPUT='benchmark-radial.json'
python -B -m pytest tests/test_radial_numerical.py -q -s
```

En POSIX usar `export` para las mismas variables. El archivo de salida debe ser
nuevo. La prueba usa `epsrel=0.05` explícito y compara rutas con `rtol=2e-6`, sin
mocks ni tablas. No certifica un error físico global de `2e-6`; este incluye
inversión y mapeo. Una sola repetición no prueba escalado universal. Ver
[radial_validation.md](radial_validation.md) para resultados y límites actuales;
[parallel_validation.md](parallel_validation.md) es el informe histórico del
backend por perfiles, ya retirado.
