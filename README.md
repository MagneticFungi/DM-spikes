# DM-spikes

Librería científica para calcular spikes de densidad de materia oscura alrededor
de un agujero negro. El flujo numérico construye el potencial de una densidad
inicial, obtiene su función de distribución isotrópica mediante la inversión de
Eddington, conserva la acción radial y calcula la densidad final integrando el
espacio de fases. La saturación por aniquilación se aplica de forma opcional.
También incluye perfiles analíticos para una ley de potencia y una esfera
pseudoisoterma.

El paquete requiere Python >= 3.10, NumPy y SciPy. Desde la raíz del repositorio,
con el entorno Python deseado activo, se instala en modo editable con:

```sh
python -m pip install -e .
```

Para trabajar con los notebooks y sus gráficas, el extra `notebooks` añade
Matplotlib, IPykernel y JupyterLab:

```sh
python -m pip install -e ".[notebooks]"
python -m jupyterlab
```

La estructura actual separa el paquete, los scripts de ejecución y los datos:

```text
DM-spikes/
├── src/dm_spikes/                  # Librería instalable
├── dm_spikes_execution.py          # Tres leyes de potencia y un NFW
├── dm_spikes_execution2.py         # NFW hasta 1.20 Mpc
├── execution.py                   # Script anterior de Gondolo–Silk
├── df_inicial_ley_potencia.ipynb    # Comparación de la DF inicial
├── perfiles_analiticos.ipynb        # Exploración de perfiles analíticos
├── results/                       # Perfiles actuales en NPZ
├── notebooks (old)/                # Notebooks anteriores
├── results (old)/                  # Resultados locales anteriores y GIF
├── results_servidor (old)/         # Resultados anteriores del servidor
├── README.md
└── pyproject.toml
```

Los módulos de `src/dm_spikes` tienen las siguientes responsabilidades:

| Módulo | Función |
| --- | --- |
| [`__init__.py`](src/dm_spikes/__init__.py) | Reexporta las principales funciones científicas y analíticas. |
| [`constants.py`](src/dm_spikes/constants.py) | Constantes gravitatorias `G` y `C_LIGHT`. |
| [`density_models.py`](src/dm_spikes/density_models.py) | Factorías de densidades iniciales de ley de potencia, NFW y Hernquist, con derivadas radiales. |
| [`initial_profile.py`](src/dm_spikes/initial_profile.py) | Potencial inicial esférico, sus derivadas y cachés de integrales. |
| [`eddington.py`](src/dm_spikes/eddington.py) | Inversión isotrópica de Eddington para construir `f(E)`. |
| [`adiabatic_map.py`](src/dm_spikes/adiabatic_map.py) | Puntos de retorno, acciones radiales y mapeo de la energía inicial por conservación de acción. |
| [`final_profile.py`](src/dm_spikes/final_profile.py) | Distribución final y densidad integrada en un potencial kepleriano con corte por captura. |
| [`annihilations.py`](src/dm_spikes/annihilations.py) | Saturación armónica de una densidad suministrada por aniquilación. |
| [`power_law_profile.py`](src/dm_spikes/power_law_profile.py) | Perfil analítico de ley de potencia, series `J_gamma`, factor `g_gamma`, coeficientes y radios del spike y del núcleo. |
| [`pseudo_isothermal_sphere.py`](src/dm_spikes/pseudo_isothermal_sphere.py) | Perfil analítico pseudoisotermo y radio de empalme. |
| [`profile_jobs.py`](src/dm_spikes/profile_jobs.py) | Configuración JSON y ejecución serial o paralela por radios. |
| [`profile_cli.py`](src/dm_spikes/profile_cli.py) | Interfaz de línea de comandos para manifiestos JSON. |

La convención de unidades es pc, km/s y masas solares; las densidades se expresan
en `Msun/pc^3`. El potencial inicial no incluye al agujero negro. Para usar
`make_initial_potential` y `make_eddington_df`, la frontera `finite_escape` fija
el potencial en cero en infinito; `confining` requiere `r_ref` y fija el potencial
en cero allí. La DF y el mapeo deben compartir el mismo cero de energía.

Los perfiles finales incluyen un corte por captura y se anulan para
`r <= 4 R_S`, con `R_S = 2 G M_bh / c^2`. Para la saturación,
`rho_spike` usa `rho_sat = m / (observable_sigma_v * bh_age)`; los parámetros
deben tener unidades compatibles con la densidad, sin conversión automática.

Las alternativas analíticas aceptan arrays de radios. Por ejemplo:

```python
import numpy as np
from dm_spikes import cusp_profile, isothermal_profile, schwarzschild_radius

M_bh = 4.0e6
R_S = schwarzschild_radius(M_bh)
radii = np.geomspace(4.001 * R_S, 1.0, 100)

rho_cusp = cusp_profile(radii, M_bh, gamma=1.0, R_S=R_S)
rho_isothermal = isothermal_profile(
    radii, M_bh, sigma_v=100.0, rho0=0.0062, R_S=R_S
)
```

La ruta numérica se compone mediante `make_initial_potential`,
`make_eddington_df` y `make_final_profile`. La inversión y el mapeo se evalúan
directamente; calcular muchos radios puede ser costoso.

Para organizar cálculos desde una configuración, `dm_spikes.profile_jobs` ofrece:

- `run_profile(config)`: un perfil en el proceso actual.
- `run_profiles(configs)`: varios perfiles en orden, con ejecución serial por defecto.
- `run_profile_radial(config, max_workers=...)`: un perfil numérico con radios repartidos entre procesos.
- `iter_profiles_radial(configs, max_workers=...)`: entrega un perfil numérico terminado antes de iniciar el siguiente.

Los motores disponibles son `numerical`, `analytic_power_law` y
`analytic_isothermal`. El modo radial exige el motor `numerical`; las
alternativas analíticas usan la ruta serial vectorizada. El backend por perfiles
completos ya no forma parte de la API: `run_profiles` acepta `serial` y `radial`.

La CLI recibe un objeto JSON o una lista de objetos. Guarda, por ejemplo, este
manifiesto numérico en un archivo propio llamado `profiles.json`:

```json
[
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
    "density_derivatives": true,
    "potential_derivatives": true
  }
]
```

`density.factory` identifica una factoría importable como `module:factory` que
devuelve una densidad escalar. Las otras factorías incluidas son `power_law`
(`rho0`, `r0`, `gamma`) y `hernquist` (`total_mass`, `scale_radius`). Los radios
y parámetros del manifiesto son datos JSON; no se pasan funciones ni arrays
NumPy entre procesos. Los IDs de un lote deben ser únicos.

Para motores analíticos, se omiten `density`, `boundary` y las opciones de
derivadas, y se usa `parameters`: `gamma` para `analytic_power_law`, o `rho0` y
`sigma_v` para `analytic_isothermal`. Opcionalmente, `annihilation` recibe `m`,
`observable_sigma_v` y `bh_age`.

Con ese manifiesto, las formas de ejecución son:

```sh
python -m dm_spikes.profile_cli profiles.json --output serial.json
python -m dm_spikes.profile_cli profiles.json --workers 2 --output-dir resultados
python -m dm_spikes.profile_cli profiles.json --id halo-nfw --workers 2 --output nfw.json
```

La instalación también registra el comando equivalente `dm-spikes-profiles`:

```sh
dm-spikes-profiles profiles.json --output serial.json
```

Los archivos de salida deben ser nuevos. `--workers` activa el modo radial,
`--block-size` indica cuántos radios recibe cada tarea y `--inner-threads` limita
los hilos de las bibliotecas numéricas en cada proceso. Cada worker construye
una vez su cadena de cálculo y reutiliza sus funciones y cachés. Se usa `spawn`
y el número de workers debe indicarse explícitamente. Los resultados conservan
el orden de entrada e incluyen configuración, radios, `rho_prime`, información
de ejecución y, si se solicita saturación, `rho_spike`.

`--output-dir` guarda cada perfil terminado antes de iniciar el siguiente. Para
usar el modo radial desde Python, las llamadas deben estar dentro de
`if __name__ == "__main__":` en un script importable. Desde notebooks se puede
invocar la CLI.

Los dos scripts de cálculo actuales construyen potenciales específicos de los
modelos con sus derivadas explícitas y usan la inversión numérica de Eddington,
el mapeo adiabático y la integral final:

```sh
python dm_spikes_execution.py --workers 2
python dm_spikes_execution2.py --workers 2
```

- `dm_spikes_execution.py` calcula leyes de potencia con `gamma = 1.0`, `0.5`
  y `1.5`, y un NFW. Usa 300 radios por perfil, desde `4.001 R_S` hasta el radio
  del spike; el NFW usa el extremo del caso `gamma = 1.0`.
- `dm_spikes_execution2.py` calcula un NFW con 300 radios desde `4.001 R_S`
  hasta `1.20 Mpc`.

Ambos admiten `--workers`, `--block-size`, `--m-bh`, `--nfw-rs`, `--epsrel` y
`--output-dir`. Por defecto usan dos workers, bloques de un radio,
`M_bh = 4e6 Msun`, escala NFW de `20000 pc` y `epsrel = 1e-3` para la integral
final. Crean una carpeta nueva con fecha y hora dentro de `results/`, o usan una
ruta nueva indicada por `--output-dir`. Guardan cada perfil terminado en un NPZ
con radios, densidad, parámetros, unidades y tiempo de cálculo. Los radios se
reparten entre procesos; los perfiles se calculan en orden.

`results/` contiene actualmente `nfw.npz`, `power_law_gamma_1.npz`,
`power_law_gamma_0p5.npz` y `power_law_gamma_1p5.npz`. Se pueden leer con NumPy:

```python
with np.load("results/nfw.npz", allow_pickle=False) as data:
    radii = data["radii"]
    rho_prime = data["rho_prime"]
```

Los notebooks activos son [df_inicial_ley_potencia.ipynb](df_inicial_ley_potencia.ipynb),
que compara la DF analítica de una ley de potencia con la inversión de Eddington,
y [perfiles_analiticos.ipynb](perfiles_analiticos.ipynb), que explora los perfiles
analíticos y sus coeficientes. Las carpetas `(old)` conservan trabajo anterior;
varios de esos notebooks requieren adaptar llamadas a la API actual y rutas de
datos. `execution.py` conserva su flujo propio de Gondolo–Silk.

El árbol actual no contiene una carpeta `tests/` ni una suite de pruebas
automatizadas incluida en el repositorio.
