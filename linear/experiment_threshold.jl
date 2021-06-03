######################################################################################################
# Copy from https://github.com/xuedong/LinBAI.jl  by Xuedong Shang                                   #
######################################################################################################
using JLD2;
using Distributed;
using Printf;
@everywhere include("runit_threshold.jl");
@everywhere include("../utilities/thresholds.jl");
include("../utilities/experiment_helpers.jl");

# setup
dist = Gaussian();
dim = 6;
μ = zeros(dim); µ[1] = 1.;

arms = Vector{Float64}[]
for k in 1:dim
	v = zeros(dim); v[k] = 1.; push!(arms, v);
end
ω = 0.01; v = zeros(dim); v[1] = cos(ω); v[2] = sin(ω); push!(arms, v)
threshold = 0.9;

pep = LinearThreshold(dist, arms, threshold);
srs = [FWSampling(), ConvexGame(CTracking), LearnerK(CTracking), RoundRobin()];

δs = (0.1,0.01,0.001,0.0001); # confidence
βs = GK16.(δs);
repeats = 1000;
seed = 1234;

# compute
@time data = pmap(
    ((sr,i),) -> runit(seed+i, sr, μ, pep, βs),
    Iterators.product(srs, 1:repeats)
);
dump_stats(pep, μ, δs, βs, srs, data, repeats);
# save
@save isempty(ARGS) ? "Threshold.dat" : ARGS[1]  dist μ pep srs data δs βs repeats seed
# visualise by loading viz.jl
