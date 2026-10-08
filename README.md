# The Geometry of Empowerment

Includes code to reproduce all figures in ``The Geometry of Empowerment'' by Ji et al. 

## Regenerating the figures

```bash
uv sync --locked
uv run python make_all_figures.py                         # all canonical figures
uv run python make_all_figures.py 7 9 10                   # a subset
uv run python make_all_figures.py --out /tmp/geom-figures
uv run python make_all_figures.py 4 --recompute-fig4      # refresh the expensive cache
```

Outputs land in `figures/paper-figures/` as PDF and PNG files with neighboring
`*_plot_data.json` records. 

The six scripts in `plotting_scripts/` compute results and generate the paper
figures. To run these scripts from the repository root:

```bash
uv run python -m plotting_scripts.fig5_adaptation_geometry --out /tmp/geom-figures
```

## Layout

- `utils/` — core math for the figures: KL and mutual information (`information.py`),
  Blahut-Arimoto and random occupancy search, occupancy LPs, adaptation bounds,
  chart geometry, potential policies
- `environments/` — grid MDPs (walled grid, key pickup)
- `plotting/` — figure builders, styling, fonts, and icons. Builders take
  computed data and never import `utils/` or `environments/`
- `tests/` — mathematical checks and figure-entry-point regression tests (`pytest`)
- `plotting_scripts/` — six scripts that compute results and generate the
  paper figures
- `data/` — created locally when caching results

## Licence

The code is MIT (`LICENSE`). Fonts and icons vendored under `plotting/` keep
their own licences and carry the attribution those licences require --- icons by
Font Awesome (Fonticons, Inc.) under CC BY 4.0, fonts under the GUST Font
Licence. See `THIRD_PARTY_NOTICES.md`.