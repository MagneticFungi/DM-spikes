# DM-spikes

Librería Python para calcular la respuesta adiabática autoconsistente de un halo
esférico al crecimiento de un agujero negro. El motor numérico sigue el ciclo
del apéndice A de [MacMillan y Henriksen](https://arxiv.org/html/astro-ph/0201153v1#A1):

1. Construye el potencial inicial `Phi_i` y la función de distribución
   isotrópica `f_i(E)` mediante `make_initial_potential` y la inversión de
   Eddington. Ambos permanecen fijos.
2. Inicia `Phi_f = Phi_i - G*M_bh/r`.
3. Calcula la acción radial en el potencial final actual y encuentra la energía
   inicial con la misma acción y el mismo momento angular físico.
4. Integra la DF deformada para obtener una nueva densidad.
5. Vuelve a construir el potencial del halo **con `make_initial_potential`** y
   suma una vez el término del agujero negro. Repite hasta que el cambio
   fraccional de densidad sea menor que `rtol` en todos los radios.

El valor predeterminado `rtol=1e-4` implementa una comparación relativa punto
por punto, `max(abs(rho_new/rho_previous - 1)) < rtol`. El artículo no precisa
la normalización de su cambio de densidad de `1e-4`. El ciclo no usa mezcla.
Para `M_bh > 0`, la integral de densidad final comienza en
`L_c = 2 c R_S = 4 G M_bh / c` en todas las iteraciones. La aniquilación se
aplica después, de forma opcional.
Si la captura da densidad cero, el cambio cero a cero cuenta como cero;
cero a positivo cuenta como uno y obliga a continuar la iteración.

## Instalación y unidades

Se requiere Python 3.10 o posterior, NumPy y SciPy:

```sh
python -m pip install -e .
```

Las unidades son pc, km/s, masas solares y `Msun/pc^3` para densidad. El
potencial inicial no contiene al agujero negro. `finite_escape` fija el cero de
potencial en infinito; `confining` requiere `r_ref` y fija allí el cero del
potencial del halo. La DF inicial debe emplear el mismo cero que `Phi_i`.

## API numérica

```python
import numpy as np
from dm_spikes import solve_self_consistent
from dm_spikes.density_models import hernquist

rho_i = hernquist(total_mass=1.0, scale_radius=1.0)
radii = np.geomspace(0.01, 100.0, 129)
result = solve_self_consistent(
    rho_i, radii, M_bh=0.01, G=1.0, boundary="finite_escape",
)
rho_final = result.density
phi_final = result.potential
rho_at_other_radius = result.density_profile(0.5)
```

`solve_self_consistent` acepta opcionalmente `initial_potential` e `initial_df`
si ya se conocen. `result.history` registra el cambio máximo de densidad por
iteración. Si se agota `max_iterations`,
`SelfConsistentConvergenceError.result` conserva el último estado y
`converged=False`. Con `iteration_callback=func`, la función recibe el estado
completo después de cada actualización de Poisson. Con
`return_on_max_iterations=True`, el límite devuelve ese último estado sin
lanzar la excepción.

`result.density` y `result.distribution` corresponden al potencial usado
en esa iteración, disponible en `result.density_potential`.
`result.potential` es la actualización de Poisson de esa densidad, incluido
el agujero negro, y se usa en la siguiente iteración. Los trabajos JSON
guardan ambos como `density_potential_grid` y `potential_grid`.
El campo `potential` del NPZ final conserva la actualización de Poisson.

`find_turning_points`, `radial_action`, `solve_initial_energy`,
`make_final_distribution`, `rho_prime_at_r` y `make_final_profile` trabajan con
el **potencial final suministrado**, que incluye halo y agujero negro. La
integral final admite las fronteras `finite_escape` y `confining`. No queda un
motor numérico final limitado al potencial kepleriano.

Con captura, la integral de densidad a radio `r` usa
`L' in [L_c, r*sqrt(2*(E'-Phi_f(r)))]`. En `finite_escape` integra
`E' in [Phi_f(r)+L_c^2/(2*r^2), 0]`; si el límite inferior es al menos
cero, la densidad es cero. En `confining`, el límite superior es infinito.
En la iteración `k`, esta energía mínima es
`E'_min^(k)(r) = Phi_DM^(k)(r) - G*M_bh/r * (1 - 4*R_S/r)`:
el potencial del halo cambia y `L_c` permanece fijo.
La acción radial y la DF inicial permanecen sin cambio; la restricción actúa
solo como límite de la integral final. `solve_self_consistent` y los trabajos
numéricos fijan `L_c` a partir de la masa del agujero negro, con `clight`
configurable. Si se construye un potencial final propio, `make_final_profile`
y `rho_prime_at_r` aceptan `capture_angular_momentum=L_c` explícitamente.

El potencial de Poisson necesita una densidad en todos los radios positivos.
Entre puntos de la malla, el ciclo interpola suavemente el logaritmo de
`rho_final/rho_inicial` en `log(r)`. Fuera de la malla mantiene constante el
cociente del extremo correspondiente. Si hay valores nulos por captura,
interpola el cociente directamente para admitir ceros sin tomar su logaritmo.
Estas son condiciones de frontera de
la representación finita; un perfil sin perturbación reproduce exactamente
la densidad inicial. Para estimar el error de discretización, se deben
comparar cálculos con dominios y resoluciones radiales distintos. La ruta
directa puede ser costosa porque anida cuadraturas y búsquedas de raíces. Las
integrales angular y, para `finite_escape`, de energía usan reglas de Gauss
sucesivas; aceptan el resultado cuando dos órdenes consecutivos concuerdan
dentro de la tolerancia indicada.

Para cálculos NFW y Hernquist extensos, la configuración opcional `tabulation` acelera
este mismo ciclo. La DF inicial se tabula una vez en energía y se valida en
los puntos medios de cada intervalo; fuera de la tabla se evalúa directamente.
También se tabula una vez la relación inversa `E_i(I_r,L)`. Cada consulta
interpolada se comprueba calculando directamente la acción radial inicial;
si no conserva la acción dentro de la tolerancia, se usa el solucionador
directo. Los potenciales iniciales NFW y Hernquist se evalúan con sus fórmulas
analíticas, incluidas sus derivadas y diferencias respecto del centro. Después
de cada actualización de Poisson, el potencial del halo se interpola en radio
logarítmico; la tabla se sustituye en la siguiente iteración. Las regiones
donde la precisión de los valores del potencial no permite certificar la
interpolación se evalúan directamente. La acción final sigue calculándose
para el potencial vigente, que cambia entre iteraciones.

Para energías iniciales próximas al centro del NFW o Hernquist, el mapa busca y comprueba
la solución en `E - Phi_i(0)`, usando la diferencia analítica del potencial.
Así, la búsqueda de raíces no redondea cada ensayo a la escala mucho mayor
de `Phi_i(0)`. Cerca del escape se conserva la energía absoluta. Al devolver
la energía para evaluar la DF, se restituye su cero original; esa conversión
a un único `float64` puede limitar la precisión si después se vuelve a
calcular una acción casi circular a partir del escalar devuelto.

Con `max_workers=N`, las evaluaciones independientes de la DF inicial y de
la tabla inversa de acciones también se distribuyen entre procesos. La DF
se refina por lotes de muestras nuevas; los valores ya calculados se reutilizan.
La tabla de acciones se ensambla en el orden de su malla, independientemente
del orden en que terminen las tareas. El pool de preparación se cierra antes
de abrir el pool de radios, por lo que ambos no compiten simultáneamente.
El número de tareas disponibles puede ser menor que `N` (el primer lote de
DF tiene 33 muestras). El refinamiento de interpoladores y la actualización
de Poisson siguen en el proceso principal.

`run_profile(..., on_progress=callback)` informa de las etapas y del número
de muestras iniciales completadas. Los scripts `dm_spikes_execution3.py` y
`dm_spikes_execution4.py` imprimen esos mensajes, además de los NPZ guardados
por iteración. Hernquist conserva su potencial inicial analítico durante el
mapa adiabático; el halo actualizado se obtiene numéricamente mediante Poisson.

## CLI y scripts

Un manifiesto JSON numérico puede contener:

```json
{
  "id": "halo-nfw",
  "engine": "numerical",
  "M_bh": 4000000.0,
  "radii": [0.001, 0.01, 0.1],
  "density": {
    "factory": "dm_spikes.density_models:nfw",
    "parameters": {"rho_s": 0.004, "r_s": 20000.0}
  },
  "boundary": "finite_escape",
  "tabulation": {"initial_df_rtol": 0.000001, "initial_action": true,
                  "final_potential": true, "potential_rtol": 0.0000001},
  "solver": {"rtol": 0.0001, "max_iterations": 100}
}
```

```sh
python -m dm_spikes.profile_cli profiles.json --workers <N> --output profile.json
```

`solver.radial_grid` puede especificar una malla más amplia o fina que los
radios de salida; debe contenerlos. `--workers N` indica el número máximo de
procesos CPU que tú asignas al perfil actual; no se deduce ni se fija a partir
del servidor. Si se omite, el cálculo es serial. Los procesos calculan grupos
independientes de radios con el potencial fijo de una iteración. El programa
espera a recibir **todos** los radios, comprueba la convergencia y resuelve
Poisson antes de iniciar la siguiente iteración. En un lote, termina un perfil
antes de comenzar el siguiente. `--block-size` controla cuántos radios recibe
cada tarea (por defecto, 1) y `--inner-threads` controla los hilos de bibliotecas
numéricas por proceso (por defecto, 1). Las mismas opciones están disponibles
en `run_profile(..., max_workers=N, block_size=1, inner_threads=1)` y en ambos
scripts de ejecución. `run_profile_radial` e `iter_profiles_radial` conservan
la interfaz anterior; el iterador entrega cada perfil terminado antes de
iniciar el siguiente. El backend usa CPU; las GPU no participan.

```sh
python dm_spikes_execution.py --workers <N> --block-size 2
python dm_spikes_execution2.py --workers <N> --block-size 2
```

En un servidor con 2 nodos NUMA conviene medir distintos valores de `N` y de
`block-size`; el número de hilos lógicos no implica una aceleración lineal.
Cada worker mantiene su propio potencial y DF iniciales en memoria. La
comunicación entre procesos envía bloques y mallas pequeños y no crea archivos
temporales de perfiles. El resultado registra los procesos utilizados en
`parallel.worker_pids`. `--output-dir` guarda los perfiles de un lote uno por
uno. Sustituye `<N>` por el número de procesos que deseas usar. En scripts
Python propios, llama al backend paralelo desde `if __name__ == "__main__":`
porque los procesos se crean con `spawn`. Los motores `analytic_power_law` y
`analytic_isothermal` siguen disponibles como
referencias analíticas.

`dm_spikes_execution.py` calcula tres leyes de potencia y NFW;
`dm_spikes_execution2.py` calcula NFW hasta 1.20 Mpc. Ambos usan el ciclo
autoconsistente y guardan NPZ en `results/`. Las carpetas `(old)` y los NPZ
existentes son resultados previos; no se recalcularon al cambiar el motor.

`dm_spikes_execution3.py` calcula un NFW con `r_s=20000 pc`,
`rho_s=0.003568 Msun/pc^3` y `M_bh=2.6e6 Msun`, desde `4.001 R_S` hasta
`380 Mpc`. Activa las tablas de DF y acción iniciales y la tabla de potencial
del halo actualizada por iteración. Acepta `--workers N` y guarda un perfil completo por iteración
(`nfw_380mpc_iter_001.npz`, `nfw_380mpc_iter_002.npz`, etc.) en una carpeta
nueva de `results/` o en `--output-dir`. Cada NPZ incluye radios, densidad,
historial, cambio relativo, estado de convergencia y tiempo transcurrido.
Se detiene al converger o al completar como máximo 100 iteraciones. En ambos
casos también escribe `nfw_380mpc.npz` con el último perfil y potencial;
`converged=False` indica que se alcanzó el límite sin converger. Los NPZ de
iteraciones se guardan al terminar todos los radios y la actualización de
Poisson correspondiente.

Para reproducir resultados con la librería anterior al ciclo autoconsistente,
`dm_spikes_execution2_legacy.py` conserva el cálculo NFW previo de 1.20 Mpc.
Debe ejecutarse con esa versión antigua instalada; su salida es `nfw.npz`.

## Pruebas

```sh
python -m pip install -e ".[test]"
python -m pytest tests -m "not slow"
python -m pytest tests -m slow
```
