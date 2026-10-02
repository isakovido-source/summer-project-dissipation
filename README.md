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

The rule behind this split: **`floquet_graphene/` answers "how do I compute
X", `scripts/` answers "what do I want to know right now".** If you want a
new plot or a new sweep, you almost always just add a new short file in
`scripts/`, importing functions that already exist in the package.

## How to run something

From the project root (the `summer_project` folder, in a terminal or VS
Code's terminal):

```bash
python scripts/chern_at_omega.py
python scripts/omega_sweep.py
python scripts/make_figures.py
python scripts/full_spectrum_table.py
python scripts/annotated_spectrum.py
python scripts/gap_and_chern_scan.py
```

Each script is short and only sets the specific parameters for that question
(look inside — they're ~15 lines each). Open one, change `omega=0.36` to
whatever you need, save, and run again.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

`tests/test_hamiltonians.py` checks three things that would catch real bugs:
the Hamiltonian is Hermitian, it reduces to the known static-graphene value
at A_0 = 0, and its eigenvalues are always real. Whenever you change
`hamiltonians.py`, running `pytest` takes two seconds and tells you if you
broke something. This is the single habit that will save you the most time
going forward — add a small test every time you fix a bug, so it can never
silently come back.

## What changed relative to your original single-file script

I did **not** rewrite the physics — every formula is untouched, and I
numerically cross-checked several functions (`compute_direct_gap`,
`compute_floquet_chern_adaptive`) against your original script to confirm
they return bit-for-bit identical results. What changed is organization:

- Split into modules as above, so each file is ~100–250 lines with one
  clear job, instead of one 1200-line file.
- Fixed real bugs: `main()` was being called twice at the bottom of the
  file; a stray `""""` was accidentally opening a second docstring.
- Removed dead code: unused `j0`, `j1` imports and the unused `Z_CONST`.
- Renamed the three misspelled constants: `NEARES_NIEGHBOORS_LATTICE` →
  `LATTICE_CONSTANT`, `FERMI_VELOSOTY` → `FERMI_VELOCITY`. Everything else
  keeps its original name.
- Renamed the three Hamiltonian builders to lowercase
  (`create_Floquet_Graphene_Hamiltonian` → `build_floquet_hamiltonian`, etc.)
  to follow Python's `snake_case` convention for functions.
- `get_lattice_vectors` + a new `build_bz_grid` helper in `lattice.py`
  replace three copy-pasted Brillouin-zone grid constructions.
- `run_generate_graphs` and `run_scan_with_all_gaps` no longer decide on
  their own where to save files or whether to plot — the calling script
  does, which is what makes them reusable and testable.

## What's deliberately *not* done yet (your next steps)

These are more invasive, so they're left for you to do incrementally, with
git as your safety net (see below) — and I'm glad to help with any of them
when you're ready:

1. **A `DriveParams` dataclass** bundling `A_0, t, pol, a` into one object,
   instead of passing all four through every function. This removes the
   biggest remaining risk in the code (mismatched default values scattered
   across many function signatures).
2. Splitting the remaining plotting functions into a `compute_...` /
   `plot_...` pair each, the way `run_scan_with_all_gaps` /
   `plot_gap_and_chern_scan` already are (see `runners.py` and
   `plotting.py` for the finished example).
3. Adding physics-based tests for `topology.py` (e.g. the Chern numbers of
   all bands should sum to 0) and `gaps.py`.
4. When you start adding dissipation: a new `dissipation.py` module plus a
   new Lindblad/master-equation solver, without touching the files above.

## Working with me on this going forward

The most useful way to continue is small, verifiable steps:

1. Tell me which of the "next steps" above you want to tackle (or paste a
   new physics feature you want to add).
2. I'll show you the change as a diff/new file content, explain *why* it's
   structured that way, and — where possible — a quick check you (or I,
   with a Python sandbox) can run to confirm nothing broke.
3. You copy the change into your VS Code project, run `pytest` and your
   usual scripts, and commit.

You can paste code directly in the chat, or re-upload a file when you want
me to look at its current state — re-upload it whenever you've made local
edits I haven't seen, since I only know what you've shown me.

## Git: the workflow you actually need day-to-day

You already have `git init` done and GitHub Desktop connected — that's the
hard part. From here, the loop is just:

1. Make a change (e.g., add a new script, or apply one of the "next
   steps").
2. Run your scripts/tests to confirm it still works.
3. In GitHub Desktop: write a one-line summary of what you changed (e.g.
   "Split single script into floquet_graphene package"), click **Commit to
   main**, then **Push origin**.

That's it — you don't need branches, merges, or anything else from the book
to get real value. The only habit worth building now: **commit after every
change that works**, even small ones. That way, if a later change breaks
something, you can always look at (or restore) the last working commit.
One immediate thing to commit: delete the empty `Calculation functions.py`
and `python_basic_practice.py` placeholder files from your repo, or rename
them to valid module names (no spaces) if you plan to use them.
