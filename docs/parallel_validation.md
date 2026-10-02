# Validación de la ejecución por perfiles — 1 de octubre de 2026

> Informe histórico de la implementación por perfiles completos. Las pruebas,
> ejemplos y benchmarks citados aquí fueron eliminados en una limpieza posterior
> del repositorio y el backend `process` fue retirado después; sus cifras no
> describen la suite actual. La extensión por
> radios y su verificación están en [radial_validation.md](radial_validation.md).

## Estado inicial y conservación del trabajo

Antes de editar: **96 pruebas pasadas en 16,72 s**. El árbol ya contenía
modificaciones, eliminaciones y módulos nuevos del usuario. Se añadieron archivos;
los SHA-256 de los 48 archivos preexistentes inventariados permanecieron iguales,
incluyendo `execution.py`, los módulos científicos y las pruebas originales.
No se restauraron eliminaciones ni se hicieron commits.

La arquitectura usa un perfil con todos sus radios por tarea. Cada proceso
construye sus propios callables y cachés a partir de JSON. `spawn` es explícito,
hay validación antes del pool, resultados en orden de entrada, errores con ID,
límites de hilos antes de importar bibliotecas en el hijo y escritura exclusiva
desde el coordinador. No se modificaron cuadraturas ni convenciones físicas.

## Archivos añadidos

- `src/dm_spikes/profile_jobs.py`: configuración, ejecución individual/serial,
  pool, aislamiento del entorno y errores identificados.
- `src/dm_spikes/profile_cli.py`: JSON, selección por ID y salida exclusiva.
- `src/dm_spikes/density_models.py`: factorías importables de las densidades
  iniciales de ley de potencia, NFW y Hernquist y sus derivadas exactas.
- `tests/test_profile_jobs.py` y `tests/profile_job_fixtures.py`: pruebas con
  procesos reales, sin sustituir la física en la prueba numérica exterior.
- `examples/profile_jobs.json` y `examples/numerical_profiles.json`: ejemplos
  analíticos y numéricos; estos últimos usan constantes de demostración.
- `benchmarks/benchmark_profiles.py` y los tres JSON de evidencia en esa carpeta.
- `README.md`, `docs/parallel_profiles.md` y este informe.

Las funciones nuevas se importan desde `dm_spikes.profile_jobs`; no se editó
el `__init__.py` que ya tenía cambios del usuario.

## Pruebas ejecutadas

La suite rápida final pasó con **116 passed, 1 skipped en 58,12 s**. La prueba
omitida es intencionalmente larga y se ejecutó separadamente con
`DM_SPIKES_RUN_SLOW=1`: **1 passed en 894,31 s**. En total se comprobaron las
96 pruebas originales y 21 nuevas, aunque no todas en una única invocación.

Las nuevas pruebas cubren:

- Serial frente a procesos con uno y dos workers, alternativas analíticas,
  saturación, radios repetidos/desordenados y correspondencia con las APIs
  escalares; tolerancia relativa `2e-12` en esta comparación determinista.
- PID distinto al padre y concurrencia real mediante una barrera de archivos
  entre dos workers. Los cierres se construyen dentro del hijo.
- Estado mutable del módulo del padre que no aparece bajo `spawn`, variables
  de hilos en los hijos y restauración del entorno incluso tras errores.
- Error numérico ordinario, terminación abrupta de un worker, rechazo de pool
  anidado y posibilidad de ejecutar otro pool después de un fallo.
- Rechazo previo al trabajo de callables, arrays NumPy, NaN y claves no JSON;
  IDs duplicados y límites de recursos inválidos.
- Construcción numérica de ley de potencia, NFW y Hernquist en procesos,
  conservación del cero por captura y configuración íntegra en los resultados.
- CLI por ID y pool, códigos de fallo, ausencia de salida cuando falla el cálculo,
  rechazo de sobrescritura y dos tareas independientes que pasan la comprobación
  inicial y compiten por el mismo archivo: solo una puede crearlo.
- Flujo numérico completo, sin mocks, para dos leyes de potencia y radios
  interiores/exteriores de captura, descrito a continuación.

También se comprobó la sintaxis de los módulos nuevos y del benchmark con la
gramática de Python 3.10. La ejecución real fue con Python 3.14 en Windows;
no se ha ejecutado esta suite en Linux ni en Python 3.10.

## Comparación numérica completa

Configuración: `rho0=0.16`, `r0=1`, pendientes `gamma=0.75` y `0.6`, `G=M_bh=1`,
`clight=10`, frontera `confining`, `r_ref=1`, radios `[0.04, 0.081]`, derivadas
de densidad/potencial activadas. El radio de captura es `0.08`.
La cuadratura final usa **`epsrel=0.05` explícito**, manteniendo los demás
defaults del mapeo y Eddington. Esa tolerancia permite una prueba de integración
acotada; no es una recomendación para producción científica.

| Pendiente | Densidad exterior serial | Densidad exterior spawn |
| --- | ---: | ---: |
| 0.75 | 0.00020061823428129367 | 0.00020061823428129367 |
| 0.6 | 0.00016883793210032948 | 0.00016883793210032948 |

Los resultados fueron idénticos en los floats devueltos, incluyendo los ceros
interiores de captura. La fase serial duró **615,913 s**, y el pool de dos procesos
**276,591 s**, incluido su arranque/cierre. Estos tiempos provienen de una sesión
con carga cambiante: durante parte de la fase serial había otros sondeos
numéricos activos; durante la fase paralela ya habían terminado. Por ello **no
se calcula un factor de aceleración controlado** a partir de esta prueba.
La evidencia íntegra está en
[`numerical_equivalence_2026-10-01.json`](../benchmarks/numerical_equivalence_2026-10-01.json).

En otro sondeo de `gamma=0.75`, apretar la cuadratura final a `epsrel=0.01`,
`epsabs=1e-8` produjo `0.00020061598163910043` en 921,891 s. La diferencia
relativa con el caso anterior es `1.12286e-5`; esto comprueba sensibilidad a esas
dos configuraciones, **no** constituye una cota global del error físico.
El sondeo con `epsrel=0.05` utilizó 441 evaluaciones del mapeo/DF y 336,589 s
solo en la cuadratura final. Los tiempos diagnósticos también tuvieron carga
concurrente. Parámetros y resultados están en
[`numerical_probes.json`](../benchmarks/numerical_probes.json).

## Benchmark analítico reproducible

Comando ejecutado, una vez terminados los otros cálculos de esta tarea:

```bash
python -B benchmarks/benchmark_profiles.py --repeats 3 --workers 2 --output benchmarks/analytic_benchmark_2026-10-01.json
```

Condiciones: Windows 11 10.0.26200, AMD Ryzen 5 4500U, seis núcleos lógicos
visibles, aproximadamente 7,36 GiB de RAM visible; Python 3.14.6, NumPy 2.5.0,
SciPy 1.18.0. Escritorio interactivo, sin afinidad ni aislamiento del sistema
operativo. Un hilo interno por biblioteca, fijado antes de importar también en
el padre. Ocho perfiles (cuatro leyes de potencia y cuatro pseudo-isotermos),
256 radios por perfil y tres repeticiones por modalidad. El tiempo medido es el
de la llamada de barrido: incluye arranque/importación de workers y cierre del
pool; excluye arranque/importación del intérprete padre y escritura del informe.

| Modalidad | Tiempos (s) | Mediana (s) | Pico RSS histórico del padre (MiB) |
| --- | --- | ---: | ---: |
| Serial vectorizado | 0.069387, 0.067085, 0.068642 | **0.068642** | 79.7 |
| Spawn, 1 worker | 1.646783, 1.770133, 1.735755 | **1.735755** | 80.3 |
| Spawn, 2 workers | 1.724253, 1.818947, 1.737531 | **1.737531** | 80.3 |

Diferencia relativa máxima entre resultados: **0**. En esta carga pequeña
la vectorización serial basta y el pool añade casi todo el tiempo. No se
extrapola ese resultado a perfiles numéricos costosos ni a un servidor distinto.
El benchmark no divide radios ni crea pools internos.

El pico RSS de esa tabla pertenece solo al padre y acumula su historia; no es
memoria incremental ni memoria total del pool. Además, una instantánea durante
la prueba numérica mostró 97,50 MiB para el padre y 79,65/79,87 MiB para los dos
workers (257,02 MiB sumados, sin shell/launcher). Tampoco es un pico agregado.
No se añadieron dependencias para medir memoria.

Configuraciones, tiempos sin redondear, versiones, PIDs y valores de referencia:
[`analytic_benchmark_2026-10-01.json`](../benchmarks/analytic_benchmark_2026-10-01.json).
Para repetirlo, usar un archivo de salida nuevo.

## Límites científicos y pendientes de despliegue

La capa de ejecución está verificada con procesos reales. La validación física
exhaustiva de todos los perfiles y radios sigue siendo un problema distinto:

- Para NFW y Hernquist se comprobaron individualmente potencial, mapeo y una
  evaluación de DF con la configuración de ejemplo. Las pruebas originales
  también verifican estos componentes contra referencias analíticas. Las
  pruebas nuevas verifican construcción y captura en procesos.
- El sondeo completo exterior de Hernquist con `epsrel=0.001` fue interrumpido
  después de **1325,55 s** (1293,53 s de CPU; pico RSS de 79,73 MiB). No había
  resultado final ni excepción numérica: se registra como **no completado**,
  no como convergencia ni defecto científico. No se completó una integral final
  exterior de NFW en esta sesión. Ambos casos tienen manifiestos reproducibles
  para extender la validación en el servidor.
- Existe una limitación numérica previa: `initial_profile.second` puede rechazar
  una segunda derivada exactamente nula por falta de precisión certificable,
  como en la ley de potencia `gamma=1` con potencial lineal. La suite original
  verifica ese rechazo. Activar `potential_derivatives` conserva ese comportamiento;
  la capa nueva no introduce una fórmula especial ni lo convierte en cero.
- Las comprobaciones de colas de la densidad arbitraria siguen siendo heurísticas
  numéricas, y las tolerancias de Eddington, acción e integral final no constituyen
  por sí solas una cota certificada del error total. No se alteró ese modelo.
- No hay resultados parciales/reanudación automática de un lote fallido, timeout
  automático por perfil ni pool distribuido entre nodos. Las tareas independientes
  por ID permiten aislar, guardar y volver a ejecutar perfiles desde fuera.
- El tiempo de apagado ante un error espera a los trabajos ya en curso; solo
  se cancelan los que todavía pueden cancelarse. Una muerte abrupta identifica
  los perfiles pendientes, sin atribuir falsamente una causa numérica concreta.

Para desplegar falta conocer CPUs/RAM asignadas, tiempo máximo de las tareas,
entorno Python/BLAS, afinidad, políticas del planificador y rutas/cuotas de
almacenamiento. Esos datos ajustan `max_workers`, hilos, tamaño de lotes y límites
externos, sin acoplar la librería al servidor. La guía de uso y los comandos
por ID están en [`parallel_profiles.md`](parallel_profiles.md).
