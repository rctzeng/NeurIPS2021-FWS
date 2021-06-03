######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
using Random;
using CPUTime;
include("../utilities/peps.jl");
include("../utilities/expfam.jl");
include("samplingrules.jl");

# Run the learning algorithm, paramterised by a sampling rule
# The stopping and recommendation rules are common
# βs must be a list of thresholds *in increasing order*

function runit(seed, sr, μs, pep, βs)
    βs = collect(βs); # mutable copy
    rng = MersenneTwister(seed);
    K = length(μs);
    N = zeros(Int64, K); # counts
    S = zeros(K);        # sum of samples
    tas_sr = typeof(sr) == TrackAndStop;
    baseline = CPUtime_us();
    classical_pep = BestArm(pep.expfam);
    # pull each arm once
    for k in 1:K
        S[k] += sample(rng, getexpfam(pep, k), μs[k]); N[k] += 1;
    end

    state = start(sr, N);
    R = Tuple{Int64, Array{Int64,1}, UInt64}[]; # collect return values
    while true
        t = sum(N);
        hμ = S./N; # emp. estimates
        hw = N./t;

        # test stopping criterion
        if tas_sr
            Z, (_, _), (istar, ξ) = glrt(classical_pep, N, hμ); # stopping rule for classical setting
        else
            pseudoL = pseudo_lipschitz(hμ, pep.arms, pep.L)
            Z, istar, ξ = glrt(pep, N, hμ, pseudoL); # stopping rule for Lipschitz structure
        end

        while Z > βs[1](t)
            popfirst!(βs);
            push!(R, (istar, copy(N), CPUtime_us()-baseline));
            if isempty(βs)
                return R;
            end
        end
        # invoke sampling rule
        k = nextsample(state, pep, istar, ξ, N, S);
        # and actually sample
        S[k] += sample(rng, getexpfam(pep, k), μs[k]);
        N[k] += 1;
        t += 1;
    end
end
