######################################################################################################
# Copy from https://github.com/xuedong/LinBAI.jl  by Xuedong Shang                                   #
# We replace Xuedong's `glrt` with our elegant `glrt` that is based on Envelope Theorem.             #
######################################################################################################
using Random;
using CPUTime;
using LinearAlgebra;
include("../utilities/peps.jl");
include("../utilities/expfam.jl");
include("samplingrules.jl");

# Run the learning algorithm, paramterised by a sampling rule
# The stopping and recommendation rules are common
# βs must be a list of thresholds *in increasing order*

function play!(i, k, rng, pep, µ, S, N, Vinv)
    arm = pep.arms[k]
    Y = sample(rng, getexpfam(pep, 1), arm'µ)
    S .+= Y .* arm
    Vinv .= sherman_morrison(Vinv, arm)
    N[k] += 1
end

function runit(seed, sr, μ, pep::LinearBestArm, βs, δs)
    # seed: random seed. UInt.
    # sr: sampling rule.
    # μ: mean vector.
    # pep: pure exploration problem.
    # βs: list of thresholds.
    convex_sr = (typeof(sr) == ConvexGame) || (typeof(sr) == LearnerK);
    xya_sr = typeof(sr) == XYAdaptive;
    qbc_sr = typeof(sr) == FWSampling;
    lztas_sr = typeof(sr) == LazyTrackAndStop;
    gap_sr = typeof(sr) == LinGapE;

    βs = collect(βs) # mutable copy
    δs = collect(δs); # mutable copy
    rng = MersenneTwister(seed)
    K = narms(pep, μ);
    nb_I = nanswers(pep, μ)
    dim = length(μ)
    N = zeros(Int64, K) # allocations
    S = zeros(dim) # sum of samples
    Vinv = Matrix{Float64}(I, dim, dim)  # inverse of the design matrix
    (convex_sr) ? P = ones(Int64, (nb_I, K)) : nothing; # P is used only for (Degenne et al. 2020)

    if xya_sr
        ρ = 1; ρ_old = 1; Xactive = copy(pep.arms); α = 0.1; # setting by Soare'14
    elseif qbc_sr # initialize Vxinv matrix
        Vxinv = zeros(Float64, dim,dim);
        for k in 1:K
            Vxinv += pep.arms[k]*(pep.arms[k]')/K;
        end
    end

    # Force exploration for initialization
    baseline = CPUtime_us();
    if lztas_sr # the strategy in Lazy T&S (Jedra and Proutiere, 2020)
        lztas_A = sum([N[k]*pep.arms[k]*(pep.arms[k]') for k=1:K]);
        lztas_A0 = zeros(Int64, dim);
        r = 0;
        while r < dim
            k = rand(rng, 1:K);
            play!(1, k, rng, pep, μ, S, N, Vinv);
            if rank(lztas_A + pep.arms[k]*(pep.arms[k]')) > r
                lztas_A += pep.arms[k]*(pep.arms[k]');
                lztas_A0[r+1] = k; # P represents A0
                r += 1;
            end
        end
        lztas_c0 = minimum(eigvals(lztas_A)) / sqrt(dim);
        lztas_c1 = minimum(eigvals(lztas_A));
        lztas_c2 = 1.1; # (1+u) * (sigma^2), where u=0.1, sigma=1
        lztas_c3 = dim*log(sqrt(11)) # dim * log(sqrt(u^-1 + 1)), used by DesignType="Heuristic"
        lztas_i0 = 0;
        lztas_w = [1.0/K for i=1:K];
    else # pull each arm once.
        for k = 1:K
            play!(1, k, rng, pep, μ, S, N, Vinv);
        end
    end

    # start sampling rules
    if qbc_sr
        state = start(sr, N, Vxinv)
    elseif convex_sr # keep as the implementations by Xuedong
        state = start(sr, N, P);
    else
        state = start(sr, N);
    end

    t_old = sum(N) # required by XY and RAGE
    R = Tuple{Int64,Array{Int64,1},UInt64}[] # collect return values
    while true
        t = sum(N)
        hµ = Vinv * S  # emp. estimates
        # test stopping criterion
        if xya_sr
            # invoke sampling rule
            i, k, Xactive, ρ, ρ_old, t_old = nextsample(state, pep, N, S, Vinv, Xactive, α, ρ, ρ_old, t_old, δs[1]);
            # test stopping criterion
            while length(Xactive) <= 1
                popfirst!(δs)
                push!(R, (i, copy(N), CPUtime_us() - baseline))
                if isempty(δs)
                    return R
                end
            end
        elseif gap_sr
            _, star, ξ = glrt(pep, N, hμ, Vinv)
            # invoke sampling rule
            i, k, ucb = nextsample(state, pep, star, ξ, N, S, Vinv, βs[1](t))
            while ucb <= 0
                popfirst!(βs)
                push!(R, (star, copy(N), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
        else
            Z, star, ξ = glrt(pep, N, hμ, Vinv);
            while Z > βs[1](t)
                popfirst!(βs)
                push!(R, (star, copy(N), CPUtime_us() - baseline))
                if isempty(βs)
                    return R
                end
            end
            # invoke sampling rule
            if convex_sr
                i, k = nextsample(state, pep, star, ξ, N, P, S, Vinv);
            elseif lztas_sr
                i, k, lztas_i0 = nextsample(state, pep, N, S, Vinv, lztas_A, lztas_A0, lztas_i0, lztas_c0);
                lztas_A += pep.arms[k]*(pep.arms[k]');
            else
                i, k = nextsample(state, pep, N, S, Vinv)
            end
        end
        # play the choosen arm
        play!(i, k, rng, pep, μ, S, N, Vinv);
        convex_sr ? P[i, k] += 1 : nothing;
        t += 1;
    end
end
