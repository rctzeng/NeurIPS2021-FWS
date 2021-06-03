using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions
@everywhere include("runit.jl");
@everywhere include("../utilities/thresholds.jl");
include("../utilities/experiment_helpers.jl");

function f(x)
    return 9cos(x)/(x^2+10)
end

dist = Gaussian();
L = 0.9; # Lipschitz constant
arms = [1.25,1.5,
        1.75,2,2.25,2.5,2.75,
        3,3.25,3.5,3.75,4,
        4.25,4.5,4.75,5,5.25,
        5.5,5.75,6];
μ = [f(k) for k in arms];
pep = LipschitzBestArm(dist, arms, μ, L);
# methods to be compared
srs = [FWSampling(), Menard(CTracking, 1/oracle(pep, μ)[1]), TrackAndStop(DTracking)];

δs =  (0.1,0.01); # confidence
βs = GK16.(δs);
repeats = 100;
seed = 1234;


println("arms=$arms, μ=$μ, repeats=$repeats");
# compute
@time data = pmap(
    ((sr, i),) -> runit(seed + i, sr, μ, pep, βs),
    Iterators.product(srs, 1:repeats),
);
dump_stats(pep, μ, δs, βs, srs, data, repeats);
# save
@save isempty(ARGS) ? "BAI1.dat" : ARGS[1] dist μ pep srs data δs βs repeats seed
# visualise by loading viz_bai.jl
