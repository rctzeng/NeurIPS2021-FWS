######################################################################################################
# Merged the implementation of Wouter Koolen and Xuedong Shang                                       #
# * https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/                           #
# * https://github.com/xuedong/LinBAI.jl/blob/master/linear/pep_linear.jl                            #
# For the function `glrt` for linear and lipschitz structures, we use functions in                   #
# `utilities/envelope.jl` to implement, which is based on the Envelope Theorem.                      #
######################################################################################################
# A Pure Exploration problem (pep) is parameterised by
# - a domain, \mathcal M, representing the prior knowledge about the structure
# - a query, as embodied by a correct-answer function istar
# To specify a PE problem here, we need to compute the following things:
# - nanswers: number of possible answers
# - istar: correct answer for feasible μ
# - glrt: value and best response (λ and ξ) to (N, ̂μ) or (w, μ)
# - oracle: characteristic time and oracle weights at μ
using IterTools;
include("envelope.jl"); # an elegant way to compute alt_min for Lipschitz structure

"""
Classical.
All functions are implemented by Wouter Koolen.
Note that we can also use Envelope Theorem to simplify `glrt` and to implement `oracle`.
"""
struct BestArm
    expfam; # common exponential family
end
nanswers(pep::BestArm, μ) = length(μ);
istar(pep::BestArm, μ) = argmax(μ);
getexpfam(pep::BestArm, k) = pep.expfam;
function glrt(pep::BestArm, w, μ)
    @assert length(size(μ)) == 1
    ⋆ = argmax(μ); # index of best arm among μ
    val, k, θ = minimum(
        begin
        # transport ⋆ and k to weighted midpoint
        θ = (w[⋆]*μ[⋆]+w[k]*μ[k])/(w[⋆]+w[k]);
        w[⋆]*d(pep.expfam, μ[⋆], θ) + w[k]*d(pep.expfam, μ[k], θ), k, θ
        end
        for k in eachindex(μ)
        if k != ⋆
    );
    λ = copy(μ);
    λ[⋆] = θ;
    λ[k] = θ;
    val, (k, λ), (⋆, μ);
end
# solve for x such that d(μ1, μx) + x*d(μi, μx) == v
# where μx = (μ1+x*μi)/(1+x)
function X(expfam, μ1, μi, v)
    kl1i = d(expfam, μ1, μi); # range of V(x) is [0, kl1i]
    @assert 0 ≤ v ≤ kl1i "0 ≤ $v ≤ $kl1i";
    α = binary_search(
        z -> let μz = (1-z)*μ1+z*μi
        (1-z)*d(expfam, μ1, μz) + z*d(expfam, μi, μz) - (1-z)*v
        end,
        0, 1, ϵ = kl1i*1e-10);
    α/(1-α), (1-α)*μ1+α*μi
end
# oracle problem
function oracle(pep, μs)
    μstar = maximum(μs);
    if all(μs .== μstar) # yes, this happens
        return Inf, ones(length(μs))/length(μs);
    end
    # determine upper range for subsequent binary search
    hi = minimum(
        d(pep.expfam, μstar, μ)
        for μ in μs
        if μ != μstar
    );
    val = binary_search(
        z -> sum(
            let μx = X(pep.expfam, μstar, μ, z)[2];
            d(pep.expfam, μstar, μx) / d(pep.expfam, μ, μx)
            end
            for μ in μs
            if μ != μstar
            ) - 1,
        0, hi);
    ws = [(μ == μstar) ? 1. : X(pep.expfam, μstar, μ, val)[1] for μ in μs];
    Σ = sum(ws);
    Σ/val, ws ./ Σ;
end
# oracle problem for best μ in confidence interval
function optimistic_oracle(pep::BestArm, hμ, N)
    t = sum(N);
    μdn = [ddn(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ)];
    μup = [dup(pep.expfam, hμ[k], log(t)/N[k]) for k in eachindex(hμ)];
    # try to make each arm the best-looking arm so far, then move everybody else down as far as possible
    minimum(oracle(pep, ((k == j) ? μup[k] : μdn[k] for k in eachindex(hμ))) for j in eachindex(hμ) if μup[j] > maximum(μdn));
end

"""
Linear Structure.
Most functions are implemented by Xuedong Shang.
We use Envelope Theorem to simplify `glrt` and to implement `oracle_linear_bai` and `oracle_linear_threshold`.
"""
function sherman_morrison(Vinv, u, v)
    num = (Vinv*u)*transpose(transpose(Vinv)*v)
    denum = 1 + transpose(v)*Vinv*u
    return Vinv .- num / denum
end
function sherman_morrison(Vinv, u)
    Vinv_u = Vinv*u
    num = Vinv_u*transpose(Vinv_u)
    denum = 1 + transpose(u)*Vinv_u
    return Vinv .- num / denum
end
function build_gaps(arms)
    gaps = Vector{Float64}[]
    for pair in subsets(arms, 2)
        gap1 = pair[1] - pair[2]
        push!(gaps, gap1)
        gap2 = pair[2] - pair[1]
        push!(gaps, gap2)
        #@show pair
    end
    return gaps
end

struct LinearBestArm
    expfam; # common exponential family
    arms;  # array of size K of arms in R^d
end
nanswers(pep::LinearBestArm, μ) = length(pep.arms);
narms(pep::LinearBestArm, μ) = length(pep.arms);
istar(pep::LinearBestArm, μ) = argmax([arm'μ for arm in pep.arms]);
armstar(pep::LinearBestArm, μ) = pep.arms[argmax([arm'μ for arm in pep.arms])];
getexpfam(pep::LinearBestArm, k) = pep.expfam;
############################################ Required by CG-C and Lk-C implemented by Xuedong Shang ############################################
function alt_min_cgc(pep::LinearBestArm, w, µ, k) # by Xuedong Shang
    sum_w = sum(w)
    w = w/sum(w)  # avoid dividing by small quantities
    arm = pep.arms[k]
    arm_star = armstar(pep, µ)
    @assert arm != arm_star
    K = length(pep.arms)
    direction = arm .- arm_star
    d = length(direction)
    # Construction of the matrix V = (sum_k w_k a_k a_k^T)^{-1}
    sum_arms_matrix = zeros(d,d)
    for j in 1:K
        sum_arms_matrix .+= w[j].* (pep.arms[j]*transpose(pep.arms[j]))
    end
    Vinv = inv(sum_arms_matrix)
    # Closest point
    η = sum_w * (direction'µ) / ((direction')*Vinv*direction)
    λ = µ .- η/sum_w * Vinv * direction
    # Divergence to that point
    val = .5 * sum_w * (direction'µ)^2 / ((direction')*Vinv*direction)
    return val, λ, k
end
function alt_min_cgc(pep::LinearBestArm, w, µ) # by Xuedong Shang
    minimum(alt_min_cgc(pep, w, µ, i) for i in 1:nanswers(pep, µ) if i != istar(pep, µ))
end
function glrt_cgc(pep::LinearBestArm, w::Vector, μ) # by Xuedong Shang
    @assert length(size(μ)) == 1
    star = istar(pep, µ)
    val, λ, k = alt_min_cgc(pep, w, µ)
    #Return: distance to closest alternative, best arm and closest λ for that alternative, best arm and vector for closest point in model
    val, (k, λ), (star, μ);
end
function glrt_cgc(pep::LinearBestArm, P::Matrix, μ) # by Xuedong Shang
    # P is a matrix of size nb_answers*nb_arms (both of which may be different from length(µ)).
    # In the BAI case, nb_answers = nb_arms != length(µ)
    nb_answers = size(P)[1]
    @assert length(size(μ)) == 1
    val = 0
    λs = zeros(nb_answers, length(µ))  # nb_answers * length(µ0)
    star = istar(pep, µ)
    for i in 1:nb_answers
        if i != star
            λs[i, :] = copy(µ)  # µ belongs to ¬i
        else
            val_i, λ_i, _ = alt_min_cgc(pep, P[i,:], µ)
            #println("i $i ; val_i $val_i ; λ_i $λ_i")
            λs[i, :] = λ_i
            val += val_i
        end
    end
    val, λs, (star, μ);
end
################################################################################################################################################
############################################## Our elegant implementation by Envelope Theorem ##################################################
function glrt(pep::LinearBestArm, N, μ, Vxinv)
    @assert length(size(μ)) == 1
    ⋆ = argmax(μ); # index of best arm among μ
    val, _, _, _ = alt_min_linear_bai(N, μ, pep.arms, Vxinv);
    return val, ⋆, μ;
end
function oracle_linear_bai(pep::LinearBestArm, μ)
    K = length(pep.arms); dim = length(μ); reward = [μ'pep.arms[k] for k=1:K]; μi = argmax(reward);
    @assert all(hμ_in_lambda(reward, μi, K)) "μ'arm has >=2 best arms.";
    x = [1.0/K for k=1:K]; Vxinv = zeros(Float64, dim,dim);
    for k in 1:K
        Vxinv += pep.arms[k]*(pep.arms[k]')/K;
    end
    for t=1:10000
        if is_complete_square(floor(Int, t/K))
            z = [1.0/K for k=1:K];
        else
            f, ∇f, fidx = compute_f_∇f_linear_bai(x, μ, t^(-0.9)/K, pep.arms, Vxinv);
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
        x = t/(t+1.0) * x + z/(t+1.0);
        for k=1:K
            Vxinv = sherman_morrison(Vxinv*t/(t+1.0), z[k]*pep.arms[k]/(t+1.0));
        end
    end
    value, _, _, _ = alt_min_linear_bai(x, μ, pep.arms, Vxinv);
    return 1/value, x;
end
#########################################################################################################################################
struct LinearThreshold
    expfam; # common exponential family
    arms;  # array of size K of arms in R^d
    threshold;  # threshold
end
nanswers(pep::LinearThreshold, μ) = 2^length(µ);
narms(pep::LinearThreshold, µ) = length(pep.arms);
istar(pep::LinearThreshold, μ) = 1 + sum([(µ[k] > pep.threshold) * 2^(k-1) for k in 1:length(µ)]);
getexpfam(pep::LinearThreshold, k) = pep.expfam;
############################################ Required by CG-C and Lk-C implemented by Xuedong Shang ############################################
function alt_min_cgc(pep::LinearThreshold, w, µ, k)
    sum_w = sum(w)
    w = w/sum(w)  # avoid dividing by small quantities
    K = narms(pep, µ)
    # Construction of the matrix V = (sum_k w_k a_k a_k^T)^{-1}
    sum_arms_matrix = zeros(length(µ),length(µ))
    for j in 1:K
        sum_arms_matrix .+= w[j].* (pep.arms[j]*transpose(pep.arms[j]))
    end
    Vinv = inv(sum_arms_matrix)
    # Closest point
    η = (µ[k] - pep.threshold) / Vinv[k,k]
    λ = µ .- η * Vinv[:,k]
    # Divergence to that point
    val = .5 * sum_w * (µ[k] - pep.threshold)^2 / Vinv[k,k]
    return val, λ, k
end
function alt_min_cgc(pep::LinearThreshold, w, µ)
    minimum(alt_min_cgc(pep, w, µ, i) for i in 1:length(µ))
end
function glrt_cgc(pep::LinearThreshold, w::Vector, μ)
    @assert length(size(μ)) == 1
    star = istar(pep, µ)
    val, λ, k = alt_min_cgc(pep, w, µ)
    answer = star
    if µ[k]>pep.threshold
        answer -= 2^(k-1)
    else
        answer += 2^(k-1)
    end
    #Return: distance to closest alternative, answer and closest λ for that alternative, answer and vector for closest point in model
    val, (answer, λ), (star, μ);
end
function glrt_cgc(pep::LinearThreshold, P::Matrix, μ)
    # P is a matrix of size nb_answers*nb_arms (both of which may be different from length(µ)).
    # In the BAI case, nb_answers = nb_arms != length(µ)
    nb_answers = nanswers(pep, µ)
    @assert length(size(μ)) == 1
    val = 0
    λs = zeros(nb_answers, length(µ))
    star = istar(pep, µ)
    for i in 1:nb_answers
        if i != star
            λs[i, :] = copy(µ)  # µ belongs to ¬i
        else
            val_i, λ_i, _ = alt_min_cgc(pep, P[i,:], µ)
            λs[i, :] = λ_i
            val += val_i
        end
    end
    val, λs, (star, μ);
end
################################################################################################################################################
############################################## Our elegant implementation by Envelope Theorem ##################################################
function glrt(pep::LinearThreshold, N, μ, Vxinv)
    @assert length(size(μ)) == 1
    answer = [((pep.arms[k]'μ>pep.threshold) ? 1 : 0) for k=1:length(pep.arms)];
    val, _, _ = alt_min_linear_threshold(N, μ, pep.arms, Vxinv);
    return val, answer, μ;
end
function oracle_linear_threshold(pep::LinearThreshold, μ)
    K = length(pep.arms); dim = length(μ); reward = [μ'pep.arms[k] for k=1:K]; μi = argmax(reward);
    @assert all(hμ_in_lambda_threshold(reward, K, pep.threshold)) "the difference between μ'arm and threshold should be greater than $(eps())";
    x = [1.0/K for k=1:K]; Vxinv = zeros(Float64, dim,dim);
    for k in 1:K
        Vxinv += pep.arms[k]*(pep.arms[k]')/K;
    end
    for t=1:10000
        if is_complete_square(floor(Int, t/K))
            z = [1.0/K for k=1:K];
        else
            f, ∇f, fidx = compute_f_∇f_linear_threshold(pep, x, μ, t^(-0.9)/K, Vxinv);
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
        x = t/(t+1.0) * x + z/(t+1.0);
        for k=1:K
            Vxinv = sherman_morrison(Vxinv*t/(t+1.0), z[k]*pep.arms[k]/(t+1.0));
        end
    end
    value, _, _ = alt_min_linear_threshold(x, μ, pep.arms, Vxinv);
    return 1/value, x;
end
################################################################################################################################################

"""
Lipschitz Structure.
We use Envelope Theorem to simplify `glrt` function and to implement `oracle_lipschitz_bai`.
"""
struct LipschitzBestArm
    expfam; # common exponential family
    arms;  # array of size K of arms
    μ; # reward of the arms
    L; # lipschitz constant
end
nanswers(pep::LipschitzBestArm, μ) = length(μ);
istar(pep::LipschitzBestArm, μ) = argmax(μ);
getexpfam(pep::LipschitzBestArm, k) = pep.expfam;
getlipschconst(pep::LipschitzBestArm) = pep.L;
############################################## Our elegant implementation by Envelope Theorem ##################################################
function glrt(pep::LipschitzBestArm, N, μ, L)
    @assert length(size(μ)) == 1
    ⋆ = argmax(μ); # index of best arm among μ
    val, _, _, _ = alt_min_lipschitz(N, μ, pep.arms, L);
    val, ⋆, μ;
end
function oracle_lipschitz_bai(pep::LipschitzBestArm)
    K = length(pep.arms); μi = argmax(pep.μ);
    @assert all(hμ_in_lambda(pep.μ, μi, K) && is_lipschitz(pep.μ, pep.arms, pep.L)) "μ has >=2 best arms or μ violates $(pep.L)-Lipschitz assumption.";
    x = [1.0/K for k=1:K];
    for t=1:100000 # depending on the number of arms
        if is_complete_square(floor(Int, t/K))
            z = [1.0/K for k=1:K];
        else
            f, ∇f, fidx = compute_f_∇f_lipschitz_bai(x, pep.μ, t^(-0.9)/K, pep.arms, pep.L);
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
        x = t/(t+1.0) * x + z/(t+1.0);
    end
    value, _, _, _ = alt_min_lipschitz(x, pep.μ, pep.arms, pep.L);
    return 1/value, x;
end
#########################################################################################################################################
