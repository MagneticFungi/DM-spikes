# Barridos de perfiles

La unidad de ejecución es **un perfil con todos sus radios**. `profile_jobs`
construye la densidad, el potencial, la DF y el perfil final en el proceso que
los evalúa. Solo viajan configuraciones y resultados JSON. Las APIs escalares
existentes se conservan; `execution.py` queda fuera de esta implementación.

## Diagnóstico y alcance

- `initial_profile` conserva dos LRU (4096 masas y 2048 integrales exteriores)
  y dos listas ordenadas de hasta 2048 anclas de masa. Se reutilizan entre
  radios del mismo perfil; no se comparten entre procesos ni entre perfiles.
- Eddington integra de nuevo para cada energía. La integral final anida
  cuadraturas de energía y momento angular; cada muestra llama al mapeo,
  que a su vez busca raíces e integra acciones radiales. Ese coste domina.
- Los cierres devueltos por las factorías y las cachés del potencial no tienen
  una serialización estándar portable. Se transmite una referencia importable
  a una factoría de densidad y sus parámetros, nunca la función construida.
- No se paralelizan radios ni cuadraturas. Eso perdería reutilización de caché,
  añadiría comunicaciones en integrandos escalares, podría modificar la suma
  y el control de error y permitiría sobreasignación por pools anidados.
- Las dos alternativas analíticas existentes ya aceptan arrays de radios.
  Se llaman una vez por perfil. El backend serial es el predeterminado;
  activar procesos es una decisión explícita, particularmente para ellas.
- No hay interpolación, tablas de DF, aproximaciones nuevas al mapeo ni cambios
  de unidades, captura, acción o cero de energía. Las derivadas opcionales son
  las entradas que ya admite Eddington y se activan explícitamente.

## Python: serial, pool local y perfil individual

Instalar el proyecto en el entorno elegido con `python -m pip install -e .`.
No se agregan dependencias. Este ejemplo debe guardarse en un archivo Python:

```python
import json
from dm_spikes.profile_jobs import run_profile, run_profiles

if __name__ == "__main__":
    with open("examples/profile_jobs.json", encoding="utf-8") as f:
        jobs = json.load(f)
    serial = run_profiles(jobs)
    parallel = run_profiles(jobs, backend="process", max_workers=2, inner_threads=1)
    single = run_profile(jobs[0])
```

Sustituir `2` por los procesos permitidos por la asignación. El backend
`process` exige `max_workers` explícito, limita el pool al número de perfiles
y utiliza siempre `spawn`, incluso con un worker. No usa el estado del padre,
Slurm, afinidad asumida ni un número de núcleos detectado como autorización.
No se ofrecen `fork` ni pools anidados. Si solo hay un perfil, su conjunto de
radios se calcula en un proceso; no se divide automáticamente para ocupar CPUs.

Los resultados mantienen el orden de entrada y el de los radios, incluyendo
radios repetidos. Cada resultado contiene `id`, una copia de `config`, `radii`,
`rho_prime`, `rho_spike` cuando se pide saturación, y `execution` (PID y segundos
del cálculo, excluyendo arranque/importaciones/transferencia). Los IDs de un
barrido deben ser únicos. Los resultados numéricos son equivalentes; PID y
tiempos naturalmente difieren. Las funciones Python no escriben archivos.

Usar el pool desde scripts protegidos con `if __name__ == "__main__"`.
En un notebook, usar la CLI o un script importable para el pool; la ruta serial
sí puede llamarse directamente. Las factorías personalizadas deben estar en un
módulo instalado o en `PYTHONPATH` accesible en todos los procesos/nodos.

## Configuración numérica

`examples/numerical_profiles.json` contiene Hernquist, NFW y ley de potencia.
Es un conjunto de demostración con `G=1`, `clight=10` y tolerancia final `1e-3`:
esas elecciones explícitas facilitan estudiar captura, **no son parámetros
astrofísicos recomendados**. Cada cálculo exterior puede tardar minutos.

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

`engine` omitido significa `numerical`. `M_bh` y una lista 1D no vacía de radios
positivos son obligatorios. `G` y `clight` predeterminados son exactamente los
de `constants.py`: pc, km/s y Msun. La densidad se expresa en Msun/pc³.

Las factorías incluidas devuelven las siguientes densidades y sus derivadas
radiales exactas; no devuelven potenciales ni DF analíticos:

| Factoría | Parámetros | Densidad |
| --- | --- | --- |
| `dm_spikes.density_models:power_law` | `rho0`, `r0`, `gamma` | `rho0*(r/r0)^(-gamma)` |
| `dm_spikes.density_models:nfw` | `rho_s`, `r_s` | `rho_s/[x*(1+x)^2]`, `x=r/r_s` |
| `dm_spikes.density_models:hernquist` | `total_mass`, `scale_radius` | `M*a/[2*pi*r*(r+a)^3]` |

Elegir la frontera explícitamente. NFW/Hernquist admiten `finite_escape`
con potencial cero en infinito. Una ley pura con `0 < gamma <= 2` usa
`confining` y un `r_ref` positivo, con potencial cero allí; para `2 < gamma < 3`
la frontera convergente es `finite_escape`. No se infiere otra normalización
ni un desplazamiento de energía para la DF.

Opciones numéricas:

- `density_derivatives: true` entrega `.prime` y `.second` de la densidad a
  Eddington; `potential_derivatives: true` entrega las del potencial numérico.
  Ambas son falsas por defecto, conservando la diferenciación de la API escalar.
- `eddington`: `escape_slope`, `central_potential`, `tail_survey_log_radius`,
  con exactamente el significado y defaults de `make_eddington_df`.
- `final`: `epsabs`, `epsrel`, `limit`, `map_kwargs`, reenviados sin modificación
  a `make_final_profile`. Los ajustes de acción se anidan dentro de `map_kwargs`.
- `annihilation`: `m`, `observable_sigma_v` y opcionalmente `bh_age`. Se llama
  a `rho_spike` sobre el resultado; `m/(observable_sigma_v*bh_age)` debe tener
  las mismas unidades de densidad. No se realiza una conversión implícita de
  GeV, cm³/s o años. Los valores del ejemplo analítico solo ilustran el contrato.

Para una densidad arbitraria, definir por ejemplo `mi_modelo.py`:

```python
def crear_densidad(*, amplitud, escala):
    def density(r):
        return amplitud / (r / escala) / (1 + r / escala)**3
    return density
```

Configurar `"factory": "mi_modelo:crear_densidad"`. El cierre se construye
dentro de cada worker. Si se solicitan derivadas, la factoría también debe
adjuntar los atributos callables `.prime` y `.second`. Las factorías son código
Python de confianza y deberían ser deterministas, sin escrituras compartidas.
No pasar lambdas, funciones, arrays NumPy, objetos, NaN o infinito en el JSON;
usar `.tolist()` para radios NumPy. Se valida todo el lote antes de crear el pool.
Los errores de importación, dominio y cálculo se atribuyen al ID y propagan
como `ProfileExecutionError`, con la excepción original como causa; nunca se
rellenan fallos con ceros. Cero por captura sigue siendo un resultado físico.

## Alternativas analíticas

Seleccionar `engine: "analytic_power_law"` con `parameters` de `cusp_profile`
(`gamma`, opcionalmente `rho0`, `r0`, `rtol`, `max_terms`), o
`engine: "analytic_isothermal"` con `rho0` y `sigma_v`. `M_bh`, `G`, `clight`
son comunes; `R_S` se calcula con `schwarzschild_radius` y se pasa a la API
actual. No hay elección automática de una alternativa física según el nombre
del perfil. Para barridos analíticos baratos, usar arrays y ejecución serial.

## CLI y planificador externo

Desde el mismo entorno y directorio:

```bash
python -m dm_spikes.profile_cli examples/profile_jobs.json --output serial.json
python -m dm_spikes.profile_cli examples/profile_jobs.json --workers 2 --output pool.json
python -m dm_spikes.profile_cli examples/numerical_profiles.json --id nfw --output nfw.json
```

Un job array externo solo necesita mapear su índice a un ID del manifiesto y
ejecutar el tercer comando. Cada tarea debe usar un archivo de salida distinto
(por ejemplo `resultados/ID.json` con una tabla de nombres controlada por el
script). La carpeta debe existir. La CLI nunca deduce el nombre del archivo del
ID ni interpreta variables del planificador. `--id` ejecuta solo ese perfil en
el proceso actual y es incompatible con `--workers`.

Sin `--output`, se devuelve JSON por stdout. Los errores van a stderr y causan
un código distinto de cero. La salida solo se abre después de completar el
cálculo, con creación exclusiva: dos tareas no pueden sobrescribir el mismo
archivo. Un corte de energía durante la escritura podría dejar un JSON parcial;
consumirlo solo tras salida exitosa y validar JSON. No hay reanudación automática.
Si un pool se rompe se informan los IDs pendientes; no siempre puede determinarse
qué tarea causó una terminación abrupta del proceso. Ante un error ordinario se
cancelan tareas pendientes y se espera a las que ya estaban ejecutándose.

## Recursos y despliegue

Cada pool establece `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS` y
`OMP_MAX_ACTIVE_LEVELS=1` **antes de spawn**, y restaura el entorno del padre
al terminar, incluso si falla. Esto limita las bibliotecas que respetan dichas
variables sin añadir `threadpoolctl`. No reconfigura los runtimes numéricos ya
cargados en el padre. Bibliotecas personalizadas pueden necesitar otros controles.

Para serial y tareas `--id`, establecer límites antes de iniciar Python. En un
shell POSIX del servidor, por ejemplo:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
python -m dm_spikes.profile_cli jobs.json --id halo-001 --output halo-001.json
```

En PowerShell se usa `$env:OMP_NUM_THREADS='1'`, etc., antes del comando.
Las variables de entorno son globales al proceso: las llamadas a pools de esta
librería se serializan mediante un lock. Evitar que otros hilos del programa
lancen procesos o cambien ese entorno durante la llamada. No existe pool global
ni cambio global del método de arranque de `multiprocessing`.

Para ajustar el despliegue faltará conocer: CPUs y RAM **asignadas por tarea**,
límite de tiempo, número de perfiles, radios y tolerancias de producción,
sistema operativo, versión de Python/NumPy/SciPy y backend BLAS, políticas de
afinidad/hilos del centro, módulos/entorno disponibles y rutas/permisos/cuotas de
almacenamiento. Con `P` procesos y `T` hilos internos, empezar con `P*T` dentro
de las CPUs asignadas, dejando memoria para intérpretes, SciPy y cachés por
worker. El pool es de un nodo; los nodos se distribuyen con tareas externas.

## Verificación y mediciones

```bash
python -B -m pytest -q
python -B benchmarks/benchmark_profiles.py --repeats 3 --workers 2 --output bench.json
python -B benchmarks/benchmark_profiles.py --manifest examples/numerical_profiles.json --repeats 1 --workers 2 --output bench-numerical.json
```

La comparación del flujo numérico directo fuera de captura es una prueba lenta
opcional. Para ejecutarla en PowerShell:

```powershell
$env:DM_SPIKES_RUN_SLOW='1'
python -B -m pytest tests/test_profile_jobs.py::test_direct_numerical_profiles_outside_capture_match_spawn -q -s
```

En POSIX, prefijar el comando con `DM_SPIKES_RUN_SLOW=1`. Usa dos pendientes de
ley de potencia, radios interiores/exteriores y `epsrel=0.05` explícito para la
cuadratura final; compara serial y spawn con `rtol=2e-12`, sin mocks. Esa
equivalencia comprueba transporte y aislamiento, no certifica un error físico
global de `2e-12`: el error final incluye la inversión y el mapeo.

El benchmark compara cada salida, registra configuración, versiones, tiempos
completos (incluyendo arranque/cierre del pool) y pico RSS histórico del padre.
Ese pico no es memoria incremental ni suma del pool; los workers se excluyen.
En cargas pequeñas, la importación de SciPy y la comunicación pueden dominar.
No hay promesa de aceleración universal. Ver `parallel_validation.md` para
las mediciones realizadas en esta máquina y los límites científicos observados.
