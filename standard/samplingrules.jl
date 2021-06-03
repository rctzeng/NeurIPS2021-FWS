######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
#  * Baselines are all implemented by Wouter Koolen.                                                 #
######################################################################################################
import Distributions: Multinomial
include("../utilities/regret.jl");
include("../utilities/tracking.jl");
include("../utilities/envelope.jl");
###################################################################### Ours ######################################################################
"""
Ours: Frank-Wolfe based Sampling
"""
struct FWSampling end
long(sr::FWSampling) = "FW-Sampling";
abbrev(sr::FWSampling) = "FWS";
mutable struct FWSamplingState
    x;
    FWSamplingState(K) = new([1.0/K for i=1:K]);
end
function start(sr::FWSampling, N)
    FWSamplingState(length(N));
end
function nextsample(sr::FWSamplingState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    r = t^(-9.0/10)/K;
    hw = N./t; # emp. sampled rate
    hμ = S./N; # emp. estimates
    hi = argmax(hμ);
    z = [0.0 for i=1:K];
    if !hμ_in_lambda(hμ, hi, K) || is_complete_square(floor(Int, t/K))
        z = [1.0/K for i=1:K];
    else
        f, ∇f, fidx = compute_f_∇f_standard_bai(sr.x, hμ, ξ, hi, r, K);
        if length(fidx) == 1 # best challenger
            challenger_idx = argmax(∇f[fidx[1]]);
            z = [(challenger_idx==j) ? 1 : 0 for j=1:K];
        else # solve LP of the zero-sum matrix game
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-sr.x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
    end
    setfield!(sr, :x, sr.x*((t-1.0)/t) + z*1.0/t);
    return argmax(sr.x ./ hw);
end

###################################################################### Baselines ######################################################################
"""
Uniform sampling implemented by Wouter Koolen
"""
struct RoundRobin end
long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";
function start(sr::RoundRobin, N)
    return sr;
end
function nextsample(sr::RoundRobin, pep, istar, ξ, N, S)
    return 1+(sum(N) % length(N));
end

"""
Track and Stop (Garivier and Kaufmann, 2016) implemented by Wouter Koolen
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
    # oracle at ξ (the closest feasible bandit model)
    _, w = oracle(pep, ξ);
    # tracking
    return track(sr.t, N, w);
end

"""
Optimistic Track and Stop (Degenne, Koolen and Ménard, 2019) implemented by Wouter Koolen
Wouter's implementation has a bug: `optimistic_oracle` fails infrequently (<5 failures in our 3000 repeated experiments).
Our workaround to this is to switch to `oracle` if `optimistic_oracle` fails.
"""
struct OptimisticTrackAndStop
    TrackingRule;
end
long(sr::OptimisticTrackAndStop) = "Optimistic TaS " * abbrev(sr.TrackingRule);
abbrev(sr::OptimisticTrackAndStop) = "O-" * abbrev(sr.TrackingRule);
struct OptimisticTrackAndStopState
    t;
    OptimisticTrackAndStopState(TrackingRule, N) = new(
        TrackingRule(N)
    );
end
function start(sr::OptimisticTrackAndStop, N)
    OptimisticTrackAndStopState(sr.TrackingRule, N);
end
function nextsample(sr::OptimisticTrackAndStopState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    # Optimistic oracle at ξ (the closest feasible bandit model)
    w = minimum(ξ);
    _, w = optimistic_oracle(pep, ξ, N);
    try
        _, w = optimistic_oracle(pep, ξ, N);
    catch e
        _, w = oracle(pep, ξ);
    end
    # tracking
    return track(sr.t, N, w);
end

"""
Gradient Ascent algorithm (Ménard, 2019) implemented by Wouter Koolen.
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
    # best response λ-player
    _, (_, λs), (_, _) = glrt(pep, w, hμ);
    # gradient
    ∇ = [d(getexpfam(pep, k), hμ[k], λs[k]) for k in eachindex(hμ)];
    # update learner
    incur!(sr.h, -∇);
    # tracking
    return track(sr.t, N, w);
end

"""
AdaHedge vs Best-Response (Section 3.1 in Degenne, Koolen and Ménard, 2019) implemented by Wouter Koolen
"""
struct DaBomb
    TrackingRule;
    M;
end
long(sr::DaBomb) = "DaBomb " * abbrev(sr.TrackingRule);
abbrev(sr::DaBomb) = "D-" * abbrev(sr.TrackingRule);
struct DaBombState
    hs; # one online learner per answer
    t;
    DaBombState(TrackingRule, N, M) = new(
        map(x -> AdaHedge(length(N)), 1:M),
        TrackingRule(N)
    );
end
function start(sr::DaBomb, N)
    DaBombState(sr.TrackingRule, N, sr.M);
end
# optimistic gradients
function optimistic_gradient(pep, hμ, t, N, λs)
    [let dist = getexpfam(pep, k),
     ↑ = dup(dist, hμ[k], log(t)/N[k]),
     ↓ = ddn(dist, hμ[k], log(t)/N[k])
     max(d(dist, ↑, λs[k]), d(dist, ↓, λs[k]), log(t)/N[k])
     end
     for k in eachindex(hμ)];
end
function nextsample(sr::DaBombState, pep, istar, ξ, N, S)
    K = length(N);
    t = sum(N);
    hμ = S./N; # emp. estimates
    # query the learner
    w = act(sr.hs[istar]);
    # best response λ-player
    _, (_, λs), (_, ξs) = glrt(pep, w, hμ);
    ∇ = optimistic_gradient(pep, hμ, t, N, λs);
    incur!(sr.hs[istar], -∇);
    # tracking
    return track(sr.t, N, w);
end
