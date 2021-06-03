# Fast Pure Exploration via Frank-Wolfe (NeurIPS 2021)
This is the repository for the NeurIPS 2021 paper "Fast Pure Exploration via Frank-Wolfe" by Po-An Wang, Ruo-Chun Tzeng and Alexandre Proutiere.

 * [utilities/envelope.jl](utilities/envelope.jl) contains the key functions for our sampling rules and introduces a generic way for the objective function and its sub-gradient (i.e., `f`, `∇f` in our code) and the generalized log-likelihood ratio (i.e., `alt_min` and `glrt` in our [utilities/peps.jl](utilities/peps.jl)) for the active learning problems under various structures.

## Package Requirement
Julia with version 1.5.4.
 * LinearAlgebra, Distributions, Statistics, Random
 * JuMP, Tulip
 * Distributed, JLD2
 * Plots, StatsPlots, CPUTime, Printf, LaTeXStrings, IterTools


## Experiments
 * Classical Best-Arm Identification
 * Linear Best-Arm Identification, Linear Threshold
 * Lipschitz Best-Arm Identification

## Execution Instructions
Please go to the corresponding folder, e.g., [standard](standard/), [linear](linear/), or [lipschitz](lipschitz/) and then execute the following commands:
 * For Best Arm Identification problem, execute, e.g., `julia -O3 -p8 experiment_bai1.jl` for parallel computing with 8 processes to speeding-up the computation.
 * For Threshold Bandit problem, the command is similar as above, just replace the filename with, e.g., `experiment_threshold.jl`.
 * After completing the experiments, the performance statistics are saved in the `.dat` file. You can visualize the result by e.g., `julia -O3 viz.jl BAI1`.

Please note that except for [linear/experiment_bai.jl](linear/experiment_bai.jl), all other experiments support multiple confidence  `δs` as input.
The reason why [linear/experiment_bai.jl](linear/experiment_bai.jl) cannot support multiple confidence `δs` is because of the stopping rule of XYAdaptive.

## Baseline Tables
|Name                 | Abbrev. | Description                                                                                  |
|:-------------------:|:--------:|:-------------------------------------------------------------------------------------------:|
|FW-based Sampling    | FWS     | Our Frank-Wolfe based Sampling                                                               |
|Track-and-Stop-D     | T-D     | Track and Stop (Garivier and Kaufmann, 2016) with D-Tracking                                 |
|Optimistic TaS-C     | O-C     | Optimistic Track and Stop (Degenne, Koolen and Ménard, 2019) with C-Tracking                 |
|Menard-C             | M-C     | Gradient Ascent algorithm (Ménard, 2019) with C-Tracking                                     |
|DaBomb-C             | D-C     | AdaHedge vs Best-Response (Section 3.1 in Degenne, Koolen and Ménard, 2019) with C-Tracking  |
|ConvexGame-C         | CG-C    | LineGame-C with C-Tracking (Degenne et al. 2020)                                             |
|LinGame-C            | Lk-C    | LineGame with C-Tracking (Degenne et al. 2020)                                               |
|LazyTaS              | LT      | Lazy TaS with modified threshold in A.1 (Jedra and Proutiere, 2020)                          |
|XY-Adaptive          | XY-A    | XY-Adaptive (Soare et al. 2014)                                                              |
