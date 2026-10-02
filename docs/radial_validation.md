# Validación de la paralelización por radios — 1 de octubre de 2026

> Este informe registra la etapa en que también existía el backend por perfiles
> completos. La API actual conserva solo `serial` y `radial`; las referencias al
> backend `process` y las pruebas de esa ruta son históricas.

## Estado inicial y alcance

El checkout estaba limpio al iniciar esta extensión, pero la limpieza previa
del repositorio había eliminado las pruebas, ejemplos y benchmarks de la etapa
por perfiles. El comando inicial `python -B -m pytest -q` no encontró pruebas.
Las 96 pruebas originales y las 116 del informe histórico no estaban disponibles
para volver a ejecutarlas. No se restauraron archivos eliminados por el usuario.

Se modificaron `src/dm_spikes/profile_jobs.py`, `src/dm_spikes/profile_cli.py`,
`README.md` y la guía `docs/parallel_profiles.md`. Se añadió al informe histórico
`docs/parallel_validation.md` una nota que distingue ambas etapas. Las pruebas
nuevas están en `tests/test_radial_jobs.py`, `tests/radial_test_factories.py` y
`tests/test_radial_numerical.py`; este último contiene la medición reproducible.

Los hashes de los restantes archivos preexistentes coinciden con el inventario
anterior a esta extensión: los módulos científicos, `execution.py`, `__init__.py`,
notebooks y `pyproject.toml` no se modificaron. No hay dependencias nuevas ni
commits. El borrador del script de los cuatro perfiles fue retirado por petición
del usuario; no forma parte de esta entrega.

## Arquitectura y costes

El coste dominante sigue siendo la integral final con sus evaluaciones de
Eddington y mapeo mediante conservación de acción. El potencial mantiene cachés
de masa e integrales; construirlo y reutilizarlo dentro del worker evita repetir
la construcción por radio. No se serializan callables ni cachés.

| Ruta | Unidad repartida | Reutilización | Concurrencia entre perfiles |
| --- | --- | --- | --- |
| Serial | Ninguna | Todos los radios del perfil | No |
| `backend="process"` anterior | Perfil completo | Todos sus radios en el worker | Sí |
| `backend="radial"` nuevo | Radio o bloque del perfil actual | Todos los bloques que recibe cada worker | No |

El modo radial usa `spawn`, una configuración por worker y bloques pequeños de
radios, con envío acotado a dos tareas pendientes por worker. Por omisión cada
tarea evalúa un radio completo. Se conserva el orden original de resultados,
independientemente de cuándo terminen. No se paralelizan nodos de cuadratura ni
se cambian tolerancias, energía, fronteras, captura, DF, mapeo o aniquilación.

Se cierra el pool antes de entregar el perfil terminado. El iterador permite
guardarlo antes de iniciar otro; la CLI implementa ese guardado con creación
exclusiva desde el padre. Ante un fallo conserva lo ya escrito y detiene el lote.
Los errores incluyen ID y, cuando está disponible, índice y valor del radio.

Cada proceso mantiene su propio estado; aumentar workers duplica cachés, memoria
y parte del trabajo previo. `block_size=1` favorece el reparto dinámico de radios
de coste desigual. Bloques mayores son opcionales y pueden reducir los procesos
efectivamente aprovechables. No hay pools anidados ni elección automática de
128 workers. Los hilos internos se limitan antes de importar bibliotecas en el
hijo y el entorno del padre se restaura al terminar, incluso ante fallos.

## Pruebas rápidas

`python -B -m pytest -q`: **25 pasadas y 1 omitida en 52,90 s**. La omitida es la
comparación completa de cuadraturas: ejecutada por separado con
`DM_SPIKES_RUN_SLOW=1`, **1 pasada en 790,37 s**. Son 26 pruebas verificadas en
total, en dos invocaciones. Se comprueba:

- Uno y dos procesos reales, `spawn`, bloques de distintos tamaños y límite de
  workers según el número de bloques; orden de radios repetidos/desordenados.
- Dos workers simultáneos mediante barrera, una construcción por worker en
  múltiples tareas y ausencia del estado mutable del padre en los hijos.
- Entorno de hilos en cada worker, restauración tras éxito y fallo, y rechazo de
  pools anidados. Un pool nuevo funciona después de un error o muerte abrupta.
- Error ordinario con causa e ID, terminación abrupta con perfil incompleto y
  error de evaluación con índice y valor del radio; no se fabrican ceros.
- Rechazo de configuraciones no JSON, NaN, arrays, callables, IDs duplicados y
  recursos inválidos antes de iniciar trabajo. No se mezclan perfiles en un pool.
- Construcción numérica de ley de potencia, NFW y Hernquist y captura en serial,
  pool por perfiles y pool radial. Estas pruebas rápidas no integran sus densidades
  finales exteriores. Las alternativas analíticas conservan la vectorización y
  coinciden entre serial y el pool por perfiles con densidades positivas.
- Cierre del pool antes de entregar cada resultado; CLI por ID tanto serial como
  radial, guardado antes de empezar el siguiente perfil, conservación de salidas
  ante fallo posterior y rechazo de sobrescritura.

Los cinco archivos Python modificados o añadidos se analizaron con la gramática
de Python 3.10. La ejecución real usa Python 3.14 en Windows; no se ha verificado
el runtime de Python 3.10 ni Linux en esta sesión.

## Benchmark de la integral completa

La prueba `tests/test_radial_numerical.py` es también el benchmark reproducible.
La guía de ejecución describe las variables de entorno y el comando para
repetirlo, guardando un JSON nuevo. No requiere un script de producción.

Configuración: ley de potencia con `gamma=0.75`, `rho0=0.16`, `r0=1`,
`G=M_bh=1`, `clight=10`, frontera `confining`, `r_ref=1`, derivadas de densidad y
potencial activadas. Radios `[0.0815, 0.04, 0.081]`, con captura en `r <= 0.08`.
Se usa **`epsrel=0.05` explícito** en la integral final, manteniendo los defaults
del mapeo y Eddington. Son parámetros de prueba, no una configuración astrofísica
ni una recomendación de tolerancia científica.

Condiciones: Windows 11, AMD Ryzen 5 4500U, seis CPU lógicas visibles, Python
3.14.6, NumPy 2.5.0 y SciPy 1.18.0. Una repetición serial y una con dos workers
radiales, bloques de un radio, `spawn` y un hilo interno. Escritorio interactivo
sin afinidad ni aislamiento. Las pruebas rápidas coexistieron brevemente con
partes de la medición: los tiempos son observaciones de esta sesión, no una
medida controlada de escalabilidad. Incluyen arranque/cierre de los workers y
excluyen el arranque del padre y la escritura del informe.

| Ruta | Tiempo completo (s) |
| --- | ---: |
| Serial | 505,344025 |
| Radial, 2 workers | 282,761878 |

La razón de tiempos observada es **1,78717×**. No se extrapola a 128 workers,
otros perfiles, tolerancias o servidor. Los valores devueltos fueron:

| Radio | Densidad serial | Densidad radial |
| --- | ---: | ---: |
| 0.0815 | 0.0005122839583720074 | 0.0005122839583720074 |
| 0.04 | 0.0 | 0.0 |
| 0.081 | 0.00020061823428127763 | 0.00020061823428129367 |

Diferencia relativa máxima: **7,99361 × 10⁻¹⁴**, menor que `rtol=2e-6` usado
para comparar rutas. También coincidieron la saturación por aniquilación y la
identidad/configuración. La integral completa se ejecutó en ambos casos, sin
sustituir mapeo ni DF. La diferencia pequeña es compatible con distintas
historias de caché y no certifica el error físico total.

Configuraciones, resultados, PIDs, versiones y tiempos sin redondear están en
[`radial_benchmark_2026-10-01.json`](radial_benchmark_2026-10-01.json), copiado sin
modificar de la salida de la prueba. El benchmark automatizado no muestrea memoria;
la siguiente observación se tomó externamente mientras ejecutaba el pool.

Una instantánea durante la fase radial registró 101466112 bytes de working set
en el padre y 83025920/83406848 bytes en sus dos workers: **255,49 MiB** sumados.
Es una muestra, no un pico agregado, ni memoria incremental. Se midió mediante
`Get-Process` de Windows, sin dependencias adicionales; no incluye shell/launcher.

## Límites científicos y despliegue

La equivalencia de rutas no certifica la precisión física global. El presupuesto
de error incluye cuadraturas, diferenciación, inversión y mapeo. La validación
completa exterior de esta extensión está acotada a la ley de potencia indicada;
NFW y Hernquist todavía necesitan mediciones de sus integrales finales exteriores
y de la malla real de producción. No se infiere que el cero por captura valide
esas integrales.

Permanece la limitación previa descrita en el informe histórico: la segunda
derivada del potencial puede rechazar una curvatura exactamente nula, como en
`gamma=1` con potencial lineal, al no poder certificar su precisión. No se agregó
una excepción científica ni un cambio silencioso de derivadas para ese caso.

No hay reanudación por radio, timeout interno ni distribución de un pool entre
nodos. Ante errores, los trabajos ya ejecutándose se esperan al cerrar el pool.
`--id` permite aislar y repetir perfiles desde un planificador externo.

No se ha ejecutado con 128 workers ni dentro del servidor de la universidad.
Para elegir concurrencia faltan RAM asignada, afinidad, límites de tiempo,
entorno Python/BLAS, cuotas de almacenamiento y medidas con radios/tolerancias
de producción. Cada worker necesita intérprete, bibliotecas y cachés propios.

## Actualización: rutas serial y radial

Se retiró el backend que repartía perfiles completos. `run_profiles` acepta
`serial` y `radial`; en la CLI, `--workers` selecciona el modo radial.
`--parallel-axis radii` se mantiene como alias de compatibilidad. La suite
rápida del estado actualizado terminó con **27 pruebas pasadas y 1 omitida en
47,08 s**. La prueba lenta de integral completa no se repitió tras este cambio
de planificación; el benchmark anterior sigue identificado arriba con sus
condiciones y límites.
