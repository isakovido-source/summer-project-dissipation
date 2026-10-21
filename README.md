# Floquet topological invariants of driven graphene

Numerics for computing Chern numbers, quasienergy spectra, and winding
numbers of graphene driven by a circularly polarized laser (Floquet theory,
Sambe space).

## Project layout

```
summer_project/
├── floquet_graphene/       ← the actual physics code (a Python package)
│   ├── constants.py        ← ALL default parameters live here, and only here
│   ├── lattice.py          ← lattice vectors, Brillouin-zone grids
│   ├── hamiltonians.py     ← builds the Floquet Hamiltonians (3 variants)
│   ├── gaps.py             ← direct band-gap calculations
│   ├── topology.py         ← Chern number (FHS method), winding number
│   ├── plotting.py         ← all matplotlib/plotly figures
│   ├── reporting.py        ← the text summary table
│   └── runners.py          ← "recipes" combining the above for one question
├── scripts/                ← one small file per thing you want to run
├── tests/                  ← pytest checks (Hermiticity, known limits, ...)
├── legacy/                 ← your original 1200-line script, kept for reference
├── results/                ← generated figures land here (not tracked by git)
├── requirements.txt
└── .gitignore
```


