######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
using JLD2;
using Distributed;
using Printf;
@everywhere include("runit.jl");
@everywhere include("../utilities/thresholds.jl");
include("../utilities/experiment_helpers.jl");

# setup of the 2nd experiment from Optimal Best Arm Identification with Fixed Confidence (Garivier and Kaufmann 2016)
dist = Bernoulli();
μ = [.3, .21, .2, .19, .18];
δs = (0.1, 0.01, 0.001, 0.0001);
βs = GK16.(δs); # Recommended in Section 6 of Wouter Koolen's paper
N = 3000;
seed = 1234;
pep = BestArm(dist);
# sampling rules to be compared
srs = [
    FWSampling(),
    TrackAndStop(DTracking),
    DaBomb(CTracking, nanswers(pep, μ)),
    Menard(CTracking, 1/oracle(pep, μ)[1]),
    OptimisticTrackAndStop(CTracking),
    RoundRobin()
];

println("μ=$μ, N=$N");
# compute
@time data = pmap(
    ((sr,i),) -> runit(seed+i, sr, μ, pep, βs),
    Iterators.product(srs, 1:N)
);

dump_stats(pep, μ, δs, βs, srs, data, N);
# save
@save isempty(ARGS) ? "BAI1.dat" : ARGS[1]  dist μ pep srs data δs βs N seed

# visualise by loading viz_bai.jl
