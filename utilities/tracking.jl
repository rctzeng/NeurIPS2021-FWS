######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
# Tracking helper types

struct CTracking
    sumw;
    CTracking(N) = new(
        Float64.(N)
    ); # makes a copy of the starting situation
end
abbrev(_::Type{CTracking}) = "C";
# add a weight vector and track it
function track(t::CTracking, N, w)
    @assert all(N .≤ t.sumw.+1) "N $N  sumw $(t.sumw)";
    @assert all(N .≥ t.sumw.-log(length(N))) "N $N  sumw $(t.sumw)"; # This line is removed in https://github.com/xuedong/LinBAI.jl
    @assert sum(N) ≈ sum(t.sumw);
    t.sumw .+= w;
    argmin(N .- t.sumw);
end


struct DTracking
    DTracking(N) = new();
end
abbrev(_::Type{DTracking}) = "D";
function track(t::DTracking, N, w)
    argmin(N .- sum(N).*w);
end


# Wrapper to add forced exploration to a tracking rule
struct ForcedExploration
    t;
end
function track(fe::ForcedExploration, N, w)
    t = sum(N);
    K = length(N);
    undersampled = N .≤ sqrt(t) .- K/2;
    if any(undersampled)
        track(fe.t, N, undersampled/sum(undersampled));
    else
        track(fe.t, N, w);
    end
end
