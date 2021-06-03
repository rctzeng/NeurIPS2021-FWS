######################################################################################################
# Copy from https://github.com/xuedong/LinBAI.jl  by Xuedong Shang                                   #
######################################################################################################
using JLD2;
using Printf;
using StatsPlots;
using LaTeXStrings;
include("runit.jl"); # for types
include("../utilities/experiment_helpers.jl");
include("../utilities/thresholds.jl");

name = ARGS[1];

@load "$(name).dat" dist μ pep srs data δs βs repeats seed

dump_stats(pep, μ, δs, βs, srs, data, repeats);

for i in 1:length(δs)
    plot(_boxes(pep, μ, δs[i], βs[i], srs, getindex.(data, i), repeats));
    savefig("$(name)_$(δs[i])_no_outlier.pdf");
end
