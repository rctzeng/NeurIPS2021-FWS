######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
using JLD2;
using Printf;
using StatsPlots;
include("runit.jl"); # for types
include("../utilities/experiment_helpers.jl");
include("../utilities/thresholds.jl");

name = ARGS[1];

@load "$name.dat" dist μ pep srs data δs βs N seed

dump_stats(pep, μ, δs, βs, srs, data, N);

for i in 1:length(δs)
    plot(_boxes(pep, μ, δs[i], βs[i], srs, getindex.(data, i), N));
    savefig("$(name)_$(δs[i])_no_oulier.pdf");
end
