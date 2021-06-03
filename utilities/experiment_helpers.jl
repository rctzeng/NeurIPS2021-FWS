######################################################################################################
# Merge from                                                                                         #
# * Wouter Koolen: https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/, and       #
# * Xuedong Shang: https://github.com/xuedong/LinBAI.jl                                              #
# We only modify the `dump_stats` with our added `oracle` for the linear and Lipschitz structure.    #
######################################################################################################
using Statistics;

function solve(pep, μ, δ, β)
    Tstar, wstar = oracle(pep, μ);
    ⋆ = istar(pep, μ);
    # lower bound
    kl = (1-2δ)*log((1-δ)/δ);
    lbd = Tstar*kl;
    # more practical lower bound with the employed threshold β
    practical = binary_search(t -> t-Tstar*β(t), max(1, lbd), 1e10);
    Tstar, wstar, ⋆, lbd, practical;
end

function dump_stats(pep, μ, δs, βs, srs, datas, repeats)
    best_arm = typeof(pep) == BestArm;
    linear_bai = typeof(pep) == LinearBestArm;
    linear_threshold = typeof(pep) == LinearThreshold;
    linear = linear_bai || linear_threshold;
    lipschitz = typeof(pep) == LipschitzBestArm;
    linear ? K=length(pep.arms) : K=length(μ);
    if lipschitz
        value_lipschitz, wstar_lipschitz = oracle_lipschitz_bai(pep);
        value_standard, wstar_standard = oracle(pep, μ);
        println("[optimal value] standard=$(value_standard), Lipschitz=$(value_lipschitz)");
    elseif linear
        if linear_bai
            value_linear, wstar_linear = oracle_linear_bai(pep, μ);
            value_standard, wstar_standard = oracle(pep, [μ'pep.arms[k] for k=1:K]);
            println("[optimal value] standard=$(value_standard), linear=$(value_linear)");
        elseif linear_threshold
            value_linear, wstar_linear = oracle_linear_threshold(pep, μ);
            println("[optimal value] linear=$(value_linear)");
        end
    end

    for i in 1:length(δs)
        δ = δs[i];
        β = βs[i];
        data = getindex.(datas, i);
        ⋆ = istar(pep, µ);
        if best_arm
            Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β);
        end
        rule = repeat("-", 60);
        println("");
        println(rule);
        println("$pep at δ = $δ");
        println(@sprintf("%27s", "samples"), " ",
                @sprintf("%6s", "err"), " ",
                @sprintf("%5s", "time"), " ",
                join(map(k -> @sprintf("%15s", k), 1:length(μ))),
        );
        if best_arm
            println(@sprintf("%-42s", "μ"), join(map(x -> @sprintf("%0.4f      ", x), μ)));
            println(@sprintf("%-42s", "w⋆"), join(map(x -> @sprintf("%0.4f      ", x), wstar)));
            println(rule);
            println(@sprintf("%-20s", "oracle"), @sprintf("%7.2f", lbd), @sprintf("%-13s", " "), join(map(w -> @sprintf("%6.0f(%.2f)", lbd*w, w), wstar)), " ");
            println(@sprintf("%-20s", "practical"), @sprintf("%7.2f", practical), @sprintf("%-13s", " "), join(map(w -> @sprintf("%6.0f(%.2f)", practical*w, w), wstar)), " ");
        elseif linear
            println(@sprintf("%-42s", "a\'μ"), join(map(x -> @sprintf("%9.4f", x'µ), pep.arms)));
            println(@sprintf("%-20s", "w⋆_linear"), " ",
                    @sprintf("%7.0f", (1-2δ)*log((1-δ)/δ)*value_linear), " ",
                    @sprintf("%-11s", "--"), " ",
                    join(map(x -> @sprintf("%0.4f      ", x), wstar_linear))
            );
            if linear_bai
                println(@sprintf("%-20s", "w⋆_standard"), " ",
                        @sprintf("%7.0f", (1-2δ)*log((1-δ)/δ)*value_standard), " ",
                        @sprintf("%-11s", "--"), " ",
                        join(map(x -> @sprintf("%0.4f      ", x), wstar_standard))
                );
            end
        elseif lipschitz
            println(@sprintf("%-42s", "μ"), join(map(x -> @sprintf("%0.8f      ", x), μ)));
            println(@sprintf("%-20s", "w⋆_lipschitz"), " ",
                    @sprintf("%7.0f", (1-2δ)*log((1-δ)/δ)*value_lipschitz), " ",
                    @sprintf("%-11s", "--"), " ",
                    join(map(x -> @sprintf("%0.4f      ", x), wstar_lipschitz))
            );
            println(@sprintf("%-20s", "w⋆_standard"), " ",
                    @sprintf("%7.0f", (1-2δ)*log((1-δ)/δ)*value_standard), " ",
                    @sprintf("%-11s", "--"), " ",
                    join(map(x -> @sprintf("%0.4f      ", x), wstar_standard))
            );
        end
        println(rule);

        for r in eachindex(srs)
            Eτ = sum(x->sum(x[2]), data[r,:])/repeats;
            Estd = std(map(x->sum(x[2]), data[r,:]));
            if linear_threshold
                answer = [((pep.arms[k]'µ > pep.threshold) ? 1 : 0) for k=1:length(pep.arms)];
                err = sum(x->(sum(x[1].!=answer)>0 ? 1 : 0), data[r,:])/repeats;
            else # bai
                err = sum(x->x[1].!=⋆, data[r,:])/repeats;
            end
            tim = sum(x->x[3],data[r,:])/repeats;
            println(@sprintf("%-20s", long(srs[r])),
                    @sprintf("%7.0f", Eτ), " ",
                    @sprintf("%0.4f", err), " ",
                    @sprintf("%3.1f", tim/1e6),
                    join(map(k -> @sprintf("%6.0f(%.4f)", sum(x->x[2][k], data[r,:])/repeats, sum(x->x[2][k], data[r,:])/(repeats*Eτ)),
                        1:K)), " "
                    );
            if err > δ
                @warn "too many errors for $(srs[r])";
            end
        end
        println(rule);
    end
end

function τhist(pep, μ, δ, β, srs, data)
    Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)
    stephist(map(x -> sum(x[2]), data)', label=permutedims(collect(abbrev.(srs))));
    vline!([lbd], label="lower bd");
    vline!([practical], label="practical");
end

function _boxes(pep, μ, δ, β, srs, data, repeats)
    xs = permutedims(collect(abbrev.(srs)));
    means = sum(sum.(getindex.(data,2)),dims=2)/repeats;
    boxplot(xs, map(x -> sum(x[2]), data)', label="", notch=true, outliers=false, xtickfontsize=15, ytickfontsize=15); # yaxis=:log) # xguidefontsize=30, yguidefontsize=30, legendfontsize=30
    plot!(xs, means', marker=(:star4,10,:black), label="");
end
