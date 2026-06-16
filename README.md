# MO-ADAGRAD

This is the MATLAB code for MO-ADAGRAD, the objective-function free algorithm 
for multi-objective optimization presented in:

> M. De Santis, G. Eichfelder, M. Porcelli, *"Objective-Function Free Multi-Objective
> Optimization: Rate of Convergence and Performance of an Adagrad-like algorithm"*, 2026.


Beside the implementation of **MO-Adagrad** (Algorithm 1 in the paper), 
the Armijo line-search method used as a baseline **MO-Descent** (Algorithm 2 in the paper) 
is implemented to reproduce two sets of experiments: the noisy bi-objective runs (Section 4.2,
Tables 2–3) and the multi-task classification runs (Section 4.3, Table 4, Figures 2–3).
 

## Repository layout

```
Codes-submissionFeb2026-bench+mix+MT/
├── MultiAdagrad.m              core solver: MO-Adagrad and MO-Descent, shared by the two
│                                main_*.m drivers below
├── main_mixcutest.m             driver: noisy bi-objective instances built from CUTEst pairs
│                                (Table 2, "CUTEst" rows)
├── main_benchmark.m              driver: noisy bi-objective benchmark instances
│                                (lovison3/4, MOP1, T1, T2 — Table 2 "benchmark" rows, Table 3)
├── MT_Quadrants_Circle.m         multi-task experiment "Quadrants-Circle" (Table 4, Fig. 2)
├── MT_Diagonals_Circle.m         multi-task experiment "Diagonals-Circle" (Table 4, Fig. 3)
├── s2mpjlib.m                    S2MPJ runtime support library (Gratton & Toint), required
│                                by every file in mix_CUTEst_instances/
├── mix_CUTEst_instances/         22 single-objective CUTEst problems, translated to MATLAB
│                                via S2MPJ; each exposes analytic function+gradient values
├── benchmark_instances/          definitions of lovison3, lovison4, MOP1, T1, T2 (problem
│                                setup + symbolic gradient of each component function)
└── README.md                     this file
```

## Requirements

- MATLAB.
- **Optimization Toolbox** — `fmincon` is used everywhere to solve the small QP (Ω(x))
  that gives the common descent direction g^s_k.  
- **Symbolic Math Toolbox** — required only by the `*fun1.m` / `*fun2.m` files inside
  `benchmark_instances/` (lovison3, lovison4, MOP1, T1, T2). They build the gradient with
  `syms`/`gradient` at every call.  
- **Statistics and Machine Learning Toolbox** — only for the `gscatter` calls used to draw
  the test-set scatter plots at the end of `MT_Quadrants_Circle.m` / `MT_Diagonals_Circle.m`.
  If unavailable, comment out the corresponding plotting block; the training loop itself
  doesn't need it.

No installation step is required beyond having the toolboxes above: everything is plain
`.m` files, and `s2mpjlib.m` only needs to be on the MATLAB path.

## Running the experiments

In all three cases, `param.solver` selects which method is run:
- `'MOFFO'` → MO-Adagrad (the proposed method)
- `'LS'` → MO-Descent (Armijo line-search baseline)

Both drivers below run both solvers in the same loop, so a single call produces results
for both methods.

### 1. Noisy CUTEst-derived pairs (`main_mixcutest.m`)
 
This builds 10 bi-objective instances by pairing single-objective CUTEst problems (the
pairs are hard-coded in the `list` cell array at the top of the script — e.g.
`BROWNDEN`/`ALLINITU`, `ZANGWIL2`/`ROSENBR`, etc.) and runs each from 3 starting points
(the start of the first problem, of the second, and their average), at the noise levels
in `valnoise`. Results are appended to `results_mix_cutest.txt` in the current folder,
one line per (solver, pair, starting point, noise level) combination, formatted as in
the header comment of `MultiAdagrad.m`.

The folder `mix_CUTEst_instances/` contains the single-objective problems needed. 
Any S2MPJ-format problem `NAME` exposes
`[pb,pbm] = NAME('setup')` to get the starting point/name, and
`[f,g] = NAME('fx',x)` to evaluate the function and gradient at `x`.

### 2. Noisy benchmark pairs (`main_benchmark.m`)
 
This runs the 5 benchmark problems in `list` (`lovison3_unc`, `lovison4_unc`, `MOP1`,
`T1`, `T2`) over all 4 noise levels in `valnoise` and 10 pre-defined starting
points. Results are written to both `results_MObench.txt` and
`res_num.txt` (a fixed-format numeric table: solver, starting-point index, noise level,
weighted gradient/function evaluations, f1, f2). There is also a `if 0 ... end` block at
the bottom of the loop with plotting code (criterion-space trajectory, decrease of
‖g^s‖, etc.); flip the `0` to `1` to enable it for a single run.

### 3. Multi-task classification (`MT_Quadrants_Circle.m`, `MT_Diagonals_Circle.m`)
 
Each script generates N = 10000 points in [-2,2]², splits them 8000/2000 into
train/test, trains a softmax (4-class) + logistic (binary) model jointly for 500
(`epochs`, *quadrants* script) — the actual number of epochs/iterations is set near
the top of each file — and reports gradient/function evaluations, training time, and
test accuracy, then plots the predicted test labels for both tasks (Figures 2–3 in the
paper). Edit `param.solver` near the top (`'MOFFO'` or `'LS'`) to pick the method; the
script must be re-run once per solver to compare them (it does not loop over both like
the other two drivers).

 
## N.B: computation of the common descent direction through fmincon

- The common descent direction is computed by `comp_dir` (a local function at the
  bottom of `MultiAdagrad.m`, duplicated locally inside the two `MT_*.m` scripts) by
  solving (Ω(x)) with `fmincon`'s SQP algorithm; its exit flag is returned as `gs_flag`
  and treated as successful also when `fmincon` reports a small first-order optimality
  measure even if `exitflag ~= 1`.


## Citation

If you use this code, please cite the paper:

```
@article{desantis2026offo,
  title   = {Objective-Function Free Multi-Objective Optimization: Rate of Convergence
             and Performance of an Adagrad-like algorithm},
  author  = {De Santis, Marianna and Eichfelder, Gabriele and Porcelli, Margherita},
  year    = {2026}
}
```
