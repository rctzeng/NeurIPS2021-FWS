using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions
@everywhere include("runit.jl");
@everywhere include("../utilities/thresholds.jl");
include("../utilities/experiment_helpers.jl");


dist = Gaussian();
L = 0.01; # Lipschitz constant
arms = [0,96,97,98, 99, 100, 101, 102,103,104]
μ = [ 1.06,0.99, 0.99, 0.99, 0.99, 1, 0.99,0.99,0.99,0.99];

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
@save isempty(ARGS) ? "BAI2.dat" : ARGS[1] dist μ pep srs data δs βs repeats seed
# visualise by loading viz_bai.jl
