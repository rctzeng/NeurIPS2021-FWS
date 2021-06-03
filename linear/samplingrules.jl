######################################################################################################
# Copy from https://github.com/xuedong/LinBAI.jl  by Xuedong Shang                                   #
#  * Fixed Xuedong's incorrect implementation of XYAdaptive using the version in Tanner Fiez's repo: #
#    https://github.com/fiezt/Transductive-Linear-Bandit-Code/blob/master/XY_ADAPTIVE.py             #
######################################################################################################
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
    Vxinv;
    FWSamplingState(K, Vxinv) = new([1.0/K for i=1:K], Vxinv);
end
function start(sr::FWSampling, N, Vxinv)
    FWSamplingState(length(N), Vxinv);
end
function nextsample(sr::FWSamplingState, pep::LinearBestArm, N, S, Vinv)
    K = length(N); t = sum(N); hw = N./t; hμ = Vinv*S; hr = [hμ'pep.arms[k] for k=1:K]; hi = argmax(hr);
    r = t^(-9/10)/K; z = [0.0 for i=1:K];
    if !hμ_in_lambda(hr, hi, K) || is_complete_square(floor(Int, t/K))
        z = [1.0/K for i=1:K];
    else
        f, ∇f, fidx = compute_f_∇f_linear_bai(sr.x, hμ, r, pep.arms, sr.Vxinv)
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
    nextVxinv = sherman_morrison(sr.Vxinv*(t-1.0)/t, z[1]*pep.arms[1]/t);
    for k=2:K
        nextVxinv = sherman_morrison(nextVxinv, z[k]*pep.arms[k]/t);
    end
    setfield!(sr, :Vxinv, nextVxinv);
    return hi, argmax(sr.x ./ hw);
end
function nextsample(sr::FWSamplingState, pep::LinearThreshold, N, S, Vinv)
    K = length(N); t = sum(N); hw = N./t; hμ = Vinv*S; hr = [hμ'pep.arms[k] for k=1:K]; hi = argmax(hr);
    r = t^(-9/10)/K; z = [0.0 for i=1:K];
    if !hμ_in_lambda_threshold(hr, K, pep.threshold) || is_complete_square(t)
        z = [1.0/K for i=1:K];
    else
        f, ∇f, fidx = compute_f_∇f_linear_threshold(pep, sr.x, hμ, r, sr.Vxinv);
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
    nextVxinv = sherman_morrison(sr.Vxinv*(t-1.0)/t, z[1]*pep.arms[1]/t);
    for k=2:K
        nextVxinv = sherman_morrison(nextVxinv, z[k]*pep.arms[k]/t);
    end
    setfield!(sr, :Vxinv, nextVxinv);
    return hi, argmax(sr.x ./ hw);
end

###################################################################### Baselines ######################################################################
"""
Our implementation of Lazy T&S (Jedra and Proutiere, 2020)
with heuristic stopping time (see their appendix A.1)
and without their special design for coping with "many-arm" setting
"""
struct LazyTrackAndStop
end
long(sr::LazyTrackAndStop) = "LazyTaS";
abbrev(sr::LazyTrackAndStop) = "LT";
mutable struct LazyTrackAndStopState
    sumw;
    w;
    LazyTrackAndStopState(N) = new(zeros(length(N)), [1.0/length(N) for i=1:length(N)]);
end
function start(sr::LazyTrackAndStop, N)
    LazyTrackAndStopState(N);
end
function check_power2(t)
    exponent = floor(Int, log2(t));
    return (t == 2^exponent);
end
function β(sr::LazyTrackAndStop, A, c1, c2, c3, dim, δ, t) # threshold
    #return c2 * log(sqrt(det((1/(0.1*c1))*A + Matrix{Float64}(I,dim,dim))) / δ); # (Eq 9) in main paper of (Jedra and Proutiere, 2020)
    return c2 * (log(1/δ) + 0.5*log(t) + c3); #The version called modified threshold described in appendix A.1 of (Jedra and Proutiere, 2020)
end
function nextsample(sr::LazyTrackAndStopState, pep, N, S, Vinv, A, A0, i0, c0)
    K = length(N); t = sum(N); hμ = Vinv*S; dim = length(hμ); hi = argmax([hμ'pep.arms[k] for k=1:K]);
    if check_power2(t) # lazy update
        w = copy(sr.w);
        for i=1:1000 # the setting in Lazy T&S (Jedra and Proutiere, 2020)
            Vwinv = pinv(sum([w[k]*pep.arms[k]*(pep.arms[k]') for k=1:K]));
            _, ∇f, fidx = compute_f_∇f_linear_bai(w, hμ, 0, pep.arms, Vwinv); # simplified by our Proposition 2
            w_next = zeros(K);
            w_next[argmax(∇f[fidx[1]])] = 1.0;
            w = w*(i/(i+1)) + w_next/(i+1);
            if norm(w_next) / ((i+1)*norm(w)) < 0.001
                break
            end
        end
        setfield!(sr, :w, w);
    end
    setfield!(sr, :sumw, sr.sumw+sr.w);
     # we simplify the arm tracking rule, without implementing their special design for coping issues arised in "many-arm" setting
    (minimum(eigvals(A)) < c0 * sqrt(t)) ? arm = A0[i0+1] : arm = argmin(N - sr.sumw);

    return hi, arm, (i0+1) % dim; # tracking
end

"""
LineGame-C (Degenne et al. 2020) implemented by Xuedong Shang
"""
struct ConvexGame
    TrackingRule
end
long(sr::ConvexGame) = "ConvexGame-" * abbrev(sr.TrackingRule);
abbrev(sr::ConvexGame) = "CG-" * abbrev(sr.TrackingRule);
struct ConvexGameState
    h  # one online learner
    t  # tracking rule
    ConvexGameState(TrackingRule, P) = new(LinBAIAdaHedge(length(P)), TrackingRule(vec(P)))
end
function start(sr::ConvexGame, N, P)
    ConvexGameState(sr.TrackingRule, P)
end
# optimistic gradients
function optimistic_gradient(pep, hμ, t, P::Matrix, λs, Vinv)
    nb_I = nanswers(pep, hµ)
    K = narms(pep, hµ)
    grads = zeros(size(P))
    for k = 1:K
        arm = pep.arms[k]
        for i = 1:nb_I
            ref_value = (hµ .- λs[i, :])'arm
            confidence_width = log(t)
            deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
            ref_value > 0 ? grads[i, k] = 0.5 * (ref_value + deviation)^2 :
            grads[i, k] = 0.5 * (ref_value - deviation)^2
            grads[i, k] = min(grads[i, k], confidence_width)
        end
    end
    return grads
end
function nextsample(sr::ConvexGameState, pep, star, ξ, N, P, S, Vinv)
    nb_I = size(P)[1]
    K = size(P)[2]
    t = sum(N)
    hμ = Vinv * S # emp. estimates
    # query the learner
    vec_W = act(sr.h)
    W = permutedims(reshape(vec_W, (K, nb_I)), (2, 1))  # W is the I*K matrix of answers and pulls.
    # best response λ-player
    _, λs, (_, ξs) = glrt_cgc(pep, W, hμ)
    ∇ = vec(transpose(optimistic_gradient(pep, hμ, t, P, λs, Vinv)))
    incur!(sr.h, -∇)
    # tracking
    big_index = track(sr.t, vec(transpose(P)), vec_W)
    i = div(big_index - 1, K) + 1
    k = ((big_index - 1) % K) + 1
    return i, k
end

"""
LineGame (Degenne et al. 2020) implemented by Xuedong Shang
"""
struct LearnerK
    TrackingRule
end
long(sr::LearnerK) = "LinGame-" * abbrev(sr.TrackingRule);
abbrev(sr::LearnerK) = "Lk-" * abbrev(sr.TrackingRule);
struct LearnerKState
    hs  # I online learners
    t  # tracking rule
    LearnerKState(TrackingRule, N) = new(
        Dict{Int64,LinBAIAdaHedge}(),  # We could allocate one LinBAIAdaHedge for each answer, but for some problems there are 2^d answers.
        TrackingRule(vec(N)),
    )
end
function start(sr::LearnerK, N, P)
    LearnerKState(sr.TrackingRule, N)
end
# optimistic gradients
function optimistic_gradient(pep, hμ, t, N::Vector, λ, Vinv)
    K = length(pep.arms)
    grads = zeros(length(N))
    for k = 1:K
        arm = pep.arms[k]
        ref_value = (hµ .- λ)'arm
        confidence_width = log(t)
        deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
        ref_value > 0 ? grads[k] = 0.5 * (ref_value + deviation)^2 :
        grads[k] = 0.5 * (ref_value - deviation)^2
        grads[k] = min(grads[k], confidence_width)
    end
    return grads
end
function nextsample(sr::LearnerKState, pep, star, ξ, N, P, S, Vinv)
    t = sum(N)
    hμ = Vinv * S # emp. estimates
    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)
    star = istar(pep, hµ)
    if !haskey(sr.hs, star)  # if we never saw that star, initialize an LinBAIAdaHedge learner
        sr.hs[star] = LinBAIAdaHedge(length(N))
    end
    # query the learner
    w = act(sr.hs[star])
    # best response λ-player
    _, (_, λ), (_, ξs) = glrt_cgc(pep, w, hμ)
    ∇ = optimistic_gradient(pep, hμ, t, N, λ, Vinv)
    incur!(sr.hs[star], -∇)
    # tracking
    k = track(sr.t, vec(N), w)
    return star, k
end

"""
Our implementation of XY-Adaptive (Soare et al. 2014) by merging Xuedong Shang' with Tanner Fiez' codes
"""
struct XYAdaptive end
long(sr::XYAdaptive) = "XY-Adaptive";
abbrev(sr::XYAdaptive) = "XY-A";
mutable struct XYAdaptiveState
    phase;
    XYAdaptiveState() = new(0);
end
function start(sr::XYAdaptive, N)
    XYAdaptiveState();
end
function randmin(vector, rank = 1)
   # returns an integer, not a CartesianIndex
    vector = vec(vector)
    Sorted = sort(vector, rev = false)
    m = Sorted[rank]
    Ind = findall(x -> x == m, vector)
    index = Ind[floor(Int, length(Ind) * rand())+1]
    return index
end
function drop_arms(Xactive, Vinv, μ, phase, δ)
    X = copy(Xactive)
    K = length(Xactive)
    for i = 1:K
        arm = X[i]
        for j = 1:K
            if j == i
                continue
            end
            arm_prime = X[j]
            y = arm_prime - arm
            # After replacing Xuedong's implementation with Fiez' implementation: https://github.com/fiezt/Transductive-Linear-Bandit-Code/blob/master/XY_ADAPTIVE.py#L125,
            # the below ↓ condition is consistent to Eq (11) in (Soare 2014).
            if (y' * Vinv * y * 2 * log(2*π^2*K^2*phase^2/(6*δ)))^0.5 <= y'μ
                filter!(x -> x ≠ arm, Xactive)
                break
            end
        end
    end
    return Xactive
end
function nextsample(sr::XYAdaptiveState, pep, N, S, Vinv, Xactive, α, ρ, ρ_old, t_old, δ)
    t = sum(N)
    hμ = Vinv * S # emp. estimates
    nb_I = nanswers(pep, hµ)
    K = length(Xactive)
    star = istar(pep, hµ)
    Y = build_gaps(Xactive)
    nb_gaps = length(Y)
    k = randmin([maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[j]) * Y[i] for i = 1:nb_gaps]) for j = 1:nb_I])
    if ρ / t < α * ρ_old / t_old
        t_old = t;
        ρ_old = ρ;
        Xcopy = copy(pep.arms);
        Xactive = drop_arms(Xcopy, Vinv, hμ, sr.phase, δ);
        setfield!(sr, :phase, sr.phase+1);
    end
    ρ = maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[k]) * Y[i] for i = 1:nb_gaps])
    return star, k, Xactive, ρ, ρ_old, t_old
end



"""
LinGapE (Xu et al. 2018) implemented by Xuedong Shang
"""
struct LinGapE end
long(sr::LinGapE) = "LinGapE";
abbrev(sr::LinGapE) = "LG";
function start(sr::LinGapE, N)
    return sr
end
function gap(arm1, arm2, μ)
    (arm1 - arm2)'μ
end
function confidence(arm1, arm2, Vinv)
    sqrt(transpose(arm1 - arm2) * Vinv * (arm1 - arm2))
end
function nextsample(sr::LinGapE, pep, star, ξ, N, S, Vinv, β)
    hμ = Vinv * S # emp. estimates
    nb_I = nanswers(pep, hµ)
    K = length(pep.arms)
    star = istar(pep, hµ)
    c_t = sqrt(2 * β)
    ucb, ambiguous = findmax([gap(pep.arms[i], pep.arms[star], hμ) +
                              confidence(pep.arms[i], pep.arms[star], Vinv) * c_t for i = 1:K])
    k = argmin([confidence(
        pep.arms[star],
        pep.arms[ambiguous],
        sherman_morrison(Vinv, pep.arms[i]),
    ) for i = 1:K])
    return star, k, ucb
end



"""
Uniform sampling implemented by Xuedong Shang
"""
struct RoundRobin end
long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "RR";
function start(sr::RoundRobin, N)
    return sr
end
function nextsample(sr::RoundRobin, pep, N, S, Vinv)
    return (1 + (sum(N) % length(N))) * ones(Int64, 2)
end
