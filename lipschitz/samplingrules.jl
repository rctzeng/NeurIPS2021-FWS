import Distributions: Multinomial;
using LinearAlgebra;
include("../utilities/regret.jl");
include("../utilities/tracking.jl");
include("../utilities/envelope.jl");
###################################################################### Ours ######################################################################
"""
Ours: Frank-Wolfe based Sampling
"""
struct FWSampling
end
long(sr::FWSampling) = "FW-Sampling";
abbrev(sr::FWSampling) = "FWS";
mutable struct FWSamplingState
    x;
    FWSamplingState(K) = new([1.0/K for i=1:K]);
end
function start(sr::FWSampling, N)
    FWSamplingState(length(N));
end
function nextsample(sr::FWSamplingState, pep::LipschitzBestArm, star, ξ, N, S)
    K = length(N); t = sum(N); hw = N./t; hμ = S./N; hi = argmax(hμ);
    r = t^(-9/10)/K;
    z = [0.0 for i=1:K];
    if !hμ_in_lambda(hμ, hi, K) || is_complete_square(floor(Int, t/K))
        z = [1.0/K for i=1:K];
    else
        pseudoL = pseudo_lipschitz(hμ, pep.arms, pep.L);
        f, ∇f, fidx = compute_f_∇f_lipschitz_bai(sr.x, hμ, r, pep.arms, pseudoL);
        if length(fidx) == 1 # best challenger
            challenger_idx = argmax(∇f[fidx[1]]);
            z = [(challenger_idx==j) ? 1 : 0 for j=1:K];
        else # solve LP of the zero-sum matrix game
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-sr.x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
    end
    # update
    setfield!(sr, :x, sr.x*((t-1.0)/t) + z*1.0/t);
    return argmax(sr.x ./ hw);
end

###################################################################### Baselines ######################################################################
"""
Uniform sampling
"""
struct RoundRobin end
long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";
function start(sr::RoundRobin, N)
    return sr;
end
function nextsample(sr::RoundRobin, pep::LipschitzBestArm, istar, ξ, N, S)
    return 1+(sum(N) % length(N));
end

"""
Track and Stop (Garivier and Kaufmann, 2016) implemented by Wouter Koolen
This stands for the strongest baseline without exploring Lipschitz structure.
"""
struct TrackAndStop
    TrackingRule;
end
long(sr::TrackAndStop) = "Track-and-Stop " * abbrev(sr.TrackingRule);
abbrev(sr::TrackAndStop) = "T-" * abbrev(sr.TrackingRule);

struct TrackAndStopState
    t;
    TrackAndStopState(TrackingRule, N) = new(
        ForcedExploration(TrackingRule(N))
    );
end
function start(sr::TrackAndStop, N)
    TrackAndStopState(sr.TrackingRule, N);
end
function nextsample(sr::TrackAndStopState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    # oracle at ξ (the closest feasible bandit model) --> standard structure
    _, w = oracle(pep, ξ);
    # tracking
    return track(sr.t, N, w);
end


"""
Gradient Ascent algorithm (Ménard, 2019) given as inputs our ∇f and pseudo Lipschitz constant.
We create this baseline for comparing Frank-Wolfe vs Mirror Ascent.
This is not in Ménard's paper and also has no asymptotic optimal convergence guarantee.
"""
struct Menard
    TrackingRule;
    scale;
end
long(sr::Menard) = "Menard " * abbrev(sr.TrackingRule);
abbrev(sr::Menard) = "M-" * abbrev(sr.TrackingRule);

function start(sr::Menard, N)
    MenardState(sr.TrackingRule, sr.scale, N);
end
struct MenardState
    h;
    t;
    MenardState(TrackingRule, scale, N) = new(
        FixedShare(length(N), S=scale),
        TrackingRule(N)
    );
end
function nextsample(sr::MenardState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    hμ = S./N; # emp. estimates
    # query the learner
    w = act(sr.h);
    # gradient
    pseudoL = pseudo_lipschitz(hμ, pep.arms, pep.L);
    _, ∇f, fidx = compute_f_∇f_lipschitz_bai(w, hμ, 0, pep.arms, pseudoL);
    # update learner
    incur!(sr.h, -∇f[fidx[1]]);
    # tracking
    return track(sr.t, N, w);
end
