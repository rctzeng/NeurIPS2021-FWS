######################################################################################################
# Copy from https://github.com/xuedong/LinBAI.jl  by Xuedong Shang                                   #
######################################################################################################
using Random;
using CPUTime;
using LinearAlgebra;
include("../utilities/peps.jl");
include("../utilities/expfam.jl");
include("samplingrules.jl");

# Run the learning algorithm, paramterised by a sampling rule
# The stopping and recommendation rules are common
#
# βs must be a list of thresholds *in increasing order*

function play!(i, k, rng, pep, µ, S, N, P, Vinv)
    arm = pep.arms[k]
    Y = sample(rng, getexpfam(pep, 1), arm'µ)
    S .+= Y .* arm
    Vinv .= sherman_morrison(Vinv, arm)
    N[k] += 1
end

function runit(seed, sr, μs, pep::LinearThreshold, βs)
    # seed: random seed. UInt.
    # sr: sampling rule.
    # µs: mean vector.
    # pep: pure exploration problem.
    # βs: list of thresholds.
    convex_sr = (typeof(sr) == ConvexGame) || (typeof(sr) == LearnerK);  # test if P is needed.
    qbc_sr = typeof(sr) == FWSampling;
    βs = collect(βs) # mutable copy
    rng = MersenneTwister(seed)
    K = narms(pep, µs)
    nb_I = nanswers(pep, µs)
    dim = length(μs)
    convex_sr ? P = ones(Int64, (nb_I, K)) : P = ones(Int64, (1, 1))  # counts detailed by answer
    N = zeros(Int64, K)              # counts
    S = zeros(dim)                     # sum of samples
    Vinv = Matrix{Float64}(I, dim, dim)  # inverse of the design matrix

    if qbc_sr # initialize Vxinv matrix
        Vxinv = zeros(Float64, dim,dim);
        for k in 1:K
            Vxinv += pep.arms[k]*(pep.arms[k]')/K;
        end
    end

    baseline = CPUtime_us()
    for k = 1:K
        play!(1, k, rng, pep, µs, S, N, P, Vinv)
    end

    # start sampling rules
    if qbc_sr
        state = start(sr, N, Vxinv)
    elseif convex_sr
        state = start(sr, N, P);
    else
        state = start(sr, N);
    end

    R = Tuple{Array{Int64,1},Array{Int64,1},UInt64}[] # collect return values
    while true
        t = sum(N)
        hµ = Vinv * S  # emp. estimates
        # test stopping criterion
        Z, answer, ξ = glrt(pep, N, hμ, Vinv);

        while Z > βs[1](t)
            popfirst!(βs)
            push!(R, (copy(answer), copy(N), CPUtime_us() - baseline));
            if isempty(βs)
                return R
            end
        end
        # invoke sampling rule
        if convex_sr
            i, k = nextsample(state, pep, argmax([hμ'pep.arms[k] for k=1:K]), ξ, N, P, S, Vinv);
        else
            i, k = nextsample(state, pep, N, S, Vinv)
        end
        play!(i, k, rng, pep, µ, S, N, P, Vinv)
        convex_sr ? P[i, k] += 1 : nothing
        t += 1
    end
end
