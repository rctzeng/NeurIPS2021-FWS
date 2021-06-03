######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
# binary search for zero of continuous function f
# with one zero-crossing in [lo, hi]
function binary_search(f, lo, hi; ϵ = 1e-10, maxiter = 100)
    for i in 1:maxiter
        mid = (lo+hi)/2;
        if mid == lo || mid == hi
            return mid;
        end
        fmid = f(mid);
        if fmid < -ϵ
            lo = mid
        elseif fmid > ϵ
            hi = mid;
        else
            return mid;
        end
    end
    @warn "binary_search did not reach tolerance $ϵ in $maxiter iterations.\nf($(lo)) = $(f(lo))\nf($(hi)) = $(f(hi)),\nmid would be $((lo+hi)/2)";
    return (lo+hi)/2;
end
