######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
using Plots;
using Distributed;
using Printf;
@everywhere include("runit.jl");
include("experiment_helpers.jl");

# exercise all code paths
function test_all()
    δ = 10^-2; # confidence
    problems = (((μs, pep)
        for dist in (Gaussian(), Bernoulli(), Exponential())
        for (μs, pep) in (([.5, .6, .7, .8, .9],    MinimumThreshold(dist, .08)),
                          ([.5, .6, .7, .8, .9],    BestArm(dist)),
                          ([.5, .6, .5, .6, .55],   SameSign(dist, .45)),
                          ([.1 .2 .3; .6 .5 .4][:], LargestProfit(dist))
                          ))
                ...,
                ([.4, .4], HeteroSkedastic((Gaussian(1), Gaussian(2)), .5))
                );
    configs = ((μs, sr, pep)
               for (μs, pep) in problems
               for sr in everybody(pep, μs)
               );
    β = (t) -> -log(δ); # gross cheat here, no log(log(1/\δ)) or log(log(t)) terms
    for (μs, sr, pep) in configs
        Tstar, _ = oracle(pep, μs);
        ((irec, N, time),) = runit(1, sr, μs, pep, (β,));
        ⋆ = istar(pep, μs);
        println("problem $pep\nsampling $sr\ninstance $μs\nTstar $(Tstar == nothing ? "nope" : Tstar*d(Bernoulli(), δ, 1-δ))  τ $(sum(N))\n⋆ $(⋆)  rec $irec\ntime $(time/1000000.0)\n");
    end
end





function launch()
    δ = 10^-4; # confidence
    dist = Gaussian(1);
    case = 1;
    if case == 1
        μs = [.3, .6]; # parameters
        if false
            γ = .08;
        else
            γ = .6;
        end
        pep = MinimumThreshold(dist, γ);
    elseif case == 2
        μs = [.5, .6, .7, .8, .9]; # parameters
        pep = BestArm(dist);
    elseif case == 3
        μs = [.5, .6, .5, .6, .55]; # parameters
        pep = SameSign(dist, .45);
    elseif case == 4
        μs = [.1 .2 .3; .6 .5 .4][:]
        pep = LargestProfit(dist);
    elseif case == 5
        μs = [.5, .5]
        pep = HeteroSkedastic((Gaussian(1), Gaussian(5)), .3);
    else
        @assert false "unknown case $case"
    end

    K = length(μs)
    Tstar, wstar = oracle(pep, μs);
    β = (t) -> -log(δ); # gross cheat here, no log(K), log(log(1/\δ)) or log(log(t)) terms
    # correct answer
    ⋆ = istar(pep, μs);

    nreps = 1000;
    @time runs = pmap(seed -> begin
                      ((irec, N, H, A),) = runit(seed, DaBomb(DTracking, nanswers(pep, μs)), μs, pep, (β,));
                      if irec != ⋆
                      println("error $irec != $(⋆) at τ $(sum(N))  N $N");
                      end
                      irec, N
                      end,
                      1:nreps);


    if Tstar != nothing
        lbd = Tstar*d(Bernoulli(), δ, 1-δ);
        println("w⋆ = $(wstar)");
        println("lbd = $lbd");
    end

    println("E[N] = ", [sum(x->x[2][k], runs)/nreps for k in 1:K])

    Eτ = sum(sum(x[2] for x in runs))/nreps;
    perr = sum(first.(runs) .!= ⋆)/length(runs);

    if perr > δ
        @warn "too many errors: $perr > $δ";
    end

    println("E[τ] ≈ $Eτ");
    println("P(err) ≈ $(perr)");

    # histogram of run times
    stephist([sum(x[2]) for x in runs], normalize=:pdf, bins = 50, label="tau");
    if Tstar != nothing
        vline!([lbd], label="lower bd");
    end
    vline!([Eτ], label="emp. mean");

    gui();
    runs
end
