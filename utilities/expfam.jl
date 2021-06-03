######################################################################################################
# Copy from https://bitbucket.org/wmkoolen/tidnabbil/src/master/purex_games_paper/ by Wouter Koolen  #
######################################################################################################
include("binary_search.jl");

struct Gaussian
    σ2;
end

# convenience
Gaussian() = Gaussian(1);

struct Bernoulli
end

struct Exponential
end

struct Poisson
end


# KL divergence
rel_entr(x, y) = x==0 ? 0. : x*log(x/y);
dx_rel_entr(x, y) = x==0 ? 0. : log(x/y);
dy_rel_entr(x, y) = -x/y;

d(expfam::Gaussian,    μ, λ) = (μ-λ)^2/(2*expfam.σ2);
d(expfam::Bernoulli,   μ, λ) = max(0, rel_entr(μ, λ) + rel_entr(1-μ, 1-λ));
d(expfam::Exponential, μ, λ) = λ == 0 ? Inf : μ/λ - log(μ/λ) - 1;
d(expfam::Poisson,     μ, λ) = rel_entr(μ, λ) - μ + λ;

dµ_d(expfam::Gaussian,    μ, λ) = (µ-λ)/expfam.σ2
dµ_d(expfam::Bernoulli,   μ, λ) = dx_rel_entr(μ, λ) - dx_rel_entr(1-μ, 1-λ);
dµ_d(expfam::Exponential, μ, λ) = 1/λ - 1/µ;
dµ_d(expfam::Poisson,     μ, λ) = dx_rel_entr(μ, λ) - 1;

dλ_d(expfam::Gaussian,    μ, λ) = (λ-µ)/expfam.σ2
dλ_d(expfam::Bernoulli,   μ, λ) = dy_rel_entr(μ, λ) - dy_rel_entr(1-μ, 1-λ);
dλ_d(expfam::Exponential, μ, λ) = -µ/λ^2 + 1/λ;
dλ_d(expfam::Poisson,     μ, λ) = dy_rel_entr(μ, λ) + 1;
# solution for λ in \min_λ d(μ, λ) - λ x
# i.e. λ satisfying (λ - μ)ϕ''(λ) == x

invh(expfam::Gaussian,    μ, x) = μ + x*expfam.σ2;
invh(expfam::Bernoulli,   μ, x) = 2μ/(1-x+sqrt((x-1)^2 + 4*x*μ));
invh(expfam::Exponential, μ, x) = x > 0 ? Inf : 2μ/(1 + sqrt(1 - 4*x*μ));
invh(expfam::Poisson,     μ, x) = μ/(1-x);

sample(rng, expfam::Gaussian,    μ) = μ + sqrt(expfam.σ2)*randn(rng);
sample(rng, expfam::Bernoulli,   μ) = rand(rng) ≤ μ;
sample(rng, expfam::Exponential, μ) = randexp(rng)*μ;

# upward and downward confidence intervals
dup(expfam::Gaussian, μ, v) = μ + sqrt(2*expfam.σ2*v);
ddn(expfam::Gaussian, μ, v) = μ - sqrt(2*expfam.σ2*v);

function dup(expfam::Bernoulli, μ, v)
    μ == 1 ? 1. : binary_search(λ -> d(expfam, μ, λ) - v, μ, 1);
end

function ddn(expfam::Bernoulli, μ, v)
    μ == 0 ? 0. : binary_search(λ -> v - d(expfam, μ, λ), 0, μ);
end

function dup(expfam::Exponential, μ, v)
    binary_search(λ -> d(expfam, μ, λ) - v, μ, μ*exp(v+1));
end

function ddn(expfam::Exponential, μ, v)
    binary_search(λ -> v - d(expfam, μ, λ), 0, μ);
end
