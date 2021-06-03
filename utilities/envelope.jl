######################################################################################################
# Functions for implementing our sampling rules as well as generic way to compute f and ∇f           #
######################################################################################################
using JuMP;
import Tulip;

function hμ_in_lambda(hμ, hi, K)
    for i=1:K
        if ((i!=hi) && (hμ[hi]-hμ[i])<=eps())
            return false;
        end
    end
    return true;
end
function hμ_in_lambda_threshold(reward, K, threshold)
    for i=1:K
        if abs(reward[i]-threshold)<=eps()
            return false;
        end
    end
    return true;
end
function is_complete_square(n)
    p = floor(Int, sqrt(n));
    return p*p == n;
end
function pseudo_lipschitz(hμ, arms, L)
    K = length(arms);
    maxR = maximum([abs(hμ[i]-hμ[j])/(L*abs(arms[i]-arms[j])) for i=1:K for j=(i+1):K]);
    return maximum([maxR*L, L]);
end
function is_lipschitz(hμ, arms, L)
    R = pseudo_lipschitz(hμ, arms, L);
    return (R <= L+eps());
end

"""
Required by our method: solving the LP formulation of mixed-strategy 2-player zero-sum game
"""
function solveZeroSumGame(M_payoff, K, n_row)
    m = Model(with_optimizer(Tulip.Optimizer));
    @variable(m, x[1:K] >= 0)
    @variable(m, w)
    for j in 1:n_row
        @constraint(m, sum(M_payoff[j][k]*x[k] for k=1:K) >= w)
    end
    @constraint(m, sum(x[i] for i=1:K) == 1)
    @objective(m, Max, w)
    optimize!(m);
    f_success = termination_status(m);
    z = JuMP.value.(x);
    return z;
end

"""
Standard BAI: computing f and ∇f by our Proposition 1
"""
function compute_f_∇f_standard_bai(hw, hμ, ξ, hi, r, K)
    μbar = [[(hw[i]*ξ[i]+hw[j]*ξ[j])/(hw[i]+hw[j]) for j=1:K] for i=1:K];
    suboptimal = [i for i=1:K if i!=hi];
    # construct ∇f
    ∇f = [[0.0 for j=1:K] for i=1:K];
    for j in suboptimal
        ∇f[j][hi] = d(getexpfam(pep,hi),hμ[hi],μbar[hi][j]);
        ∇f[j][j] = d(getexpfam(pep,j),hμ[j],μbar[hi][j]);
    end
    # construct f
    f = [hw'∇f[j] for j in suboptimal];
    fmin = minimum(f);
    if r > eps()
        fidx = [j for (idxj,j) in enumerate(suboptimal) if (f[idxj]<fmin+r)]
    elseif abs(r)<eps()
        fidx = [suboptimal[argmin(f)]];
    else
        fidx = suboptimal;
    end
    return f, ∇f, fidx;
end

"""
Linear BAI: computing f and ∇f by our Proposition 1
"""
# Envelope theorem simplifies the computation of the confusing parameters
function alt_min_linear_bai(hw, hμ, arms, Vxinv)
    K = length(arms); dim = length(hμ);
    hr = [hμ'arms[k] for k=1:K];
    hi = argmax(hr);
    suboptimal = [i for i=1:K if i!=hi];
    # construct ∇f
    λ = zeros(dim,K);
    for k in suboptimal
        direction = arms[hi]-arms[k];
        λ[:,k] = hμ - (direction'hμ / ((direction')*Vxinv*direction)) * Vxinv*(direction);
    end
    ∇f = [[0.0 for i=1:K] for j=1:K];
    for k in suboptimal
        for i=1:K
            ∇f[k][i] = ((arms[i]')*(hμ-λ[:,k]))^2 / 2;
        end
    end
    # construct f
    f = [hw'∇f[j] for j in suboptimal];
    return minimum(f), f, ∇f, suboptimal;
end
function compute_f_∇f_linear_bai(hw, hμ, r, arms, Vxinv)
    fmin, f, ∇f, suboptimal = alt_min_linear_bai(hw, hμ, arms, Vxinv);
    if r > eps()
        fidx = [j for (idxj,j) in enumerate(suboptimal) if (f[idxj]<fmin+r)]
    elseif abs(r)<eps()
        fidx = [suboptimal[argmin(f)]];
    else
        fidx = suboptimal;
    end
    return f, ∇f, fidx;
end

"""
Linear Threshold: computing f and ∇f by our Proposition 1
"""
# Envelope theorem simplifies the computation of the confusing parameters
function alt_min_linear_threshold(hw, hμ, arms, Vxinv)
    K = length(pep.arms); dim = length(hμ);
    # construct ∇f
    λ = zeros(dim,K);
    for k=1:K
        x = pep.threshold - (hμ')*pep.arms[k]; ak = pep.arms[k];
        λ[:,k] = hμ + sign(x) * (x/((ak')*Vxinv*ak)) * Vxinv * ak;
    end
    ∇f = [[0.0 for i=1:K] for j=1:K];
    for k=1:K
        for i=1:K
            ∇f[k][i] = ((pep.arms[i]')*(hμ-λ[:,k]))^2 / 2;
        end
    end
    # construct f
    f = [hw'∇f[j] for j=1:K];
    return minimum(f), f, ∇f;
end
function compute_f_∇f_linear_threshold(pep, hw, hμ, r, Vxinv)
    fmin, f, ∇f = alt_min_linear_threshold(hw, hμ, pep.arms, Vxinv);
    if r > eps()
        fidx = [j for j=1:length(pep.arms) if (f[j]<fmin+r)]
    else
        fidx = [argmin(f)];
    end
    return f, ∇f, fidx;
end


"""
Lipschitz BAI: computing f and ∇f by our Proposition 1
"""
function confusing_parameter_function(θ, j, hw, hμ, hi, arms, L)
    K = length(arms);
    return sum([2*hw[k]*maximum([θ-L*abs(arms[k]-arms[j])-hμ[k], 0]) for k=1:K]) - sum([2*hw[k]*maximum([hμ[k]-θ-L*abs(arms[k]-arms[hi]), 0]) for k=1:K]);
end
# Envelope theorem simplifies the computation of the confusing parameters
function alt_min_lipschitz(hw, hμ, arms, L)
    K = length(arms); hi = argmax(hμ);
    suboptimal = [i for i=1:K if i!=hi];
    # confusing parameters
    Θ = zeros(K);
    for j in suboptimal
        Θ[j] = binary_search(θ -> confusing_parameter_function(θ, j, hw, hμ, hi, arms, L), hμ[j], hμ[hi]);
    end
    # construct ∇f
    λ = [[0.0 for i=1:K] for j=1:K];
    for j in suboptimal
        for k=1:K
            λ[j][k] = minimum([maximum([Θ[j]-L*abs(arms[k]-arms[j]), hμ[k]]), Θ[j]+L*abs(arms[k]-arms[hi])]);
        end
    end
    ∇f = [[0.0 for i=1:K] for j=1:K];
    for j in suboptimal
        for k=1:K
            ∇f[j][k] = (hμ[k]-λ[j][k])^2/2.0;
        end
    end
    # construct f
    f = [hw'∇f[j] for j in suboptimal];
    return minimum(f), f, ∇f, suboptimal;
end
function compute_f_∇f_lipschitz_bai(hw, hμ, r, arms, L)
    fmin, f, ∇f, suboptimal = alt_min_lipschitz(hw, hμ, arms, L);
    if r > eps()
        fidx = [j for (idxj,j) in enumerate(suboptimal) if (f[idxj]<fmin+r)]
    elseif abs(r)<eps()
        fidx = [suboptimal[argmin(f)]];
    else
        fidx = suboptimal;
    end
    return f, ∇f, fidx;
end
