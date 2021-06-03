######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
struct GK16
    δ;
end

function (β::GK16)(t) # Recommended in Sec 6 (Garivier and Kaufmann 2016)
    log((log(t)+1)/β.δ);
end
