# sweep_common.jl -- shared code for the sweep-analysis notebooks (sweep_opt*_*.ipynb).
# Every notebook `include`s this, so the significance criteria are defined once.

using NuGridJl, DataFrames, CSV
import CairoMakie as CM

const ILIADIS_THRESHOLD = log10(1.1)     # ratio X_var/X_base > 1.1 or < 1/1.1 (10 %)
const RESIDUAL_THRESHOLD = 0.1           # |X_var - X_base| / X_base > 10 %

# ---------------------------------------------------------------- reaction lists

"Kept (non-excluded) reactions of `sweep_lists/opt<N>_reactions.csv`, as name => (index, source, starlib)."
function kept_reactions(case_dir, opt)
    df = CSV.read(joinpath(case_dir, "sweep_lists", "opt$(opt)_reactions.csv"), DataFrame)
    df = filter(r -> ismissing(r.excluded) || isempty(r.excluded), df)
    return Dict(String(r.name) => (index = r.index, source = String(r.source), starlib = r.starlib) for r in eachrow(df))
end

# ---------------------------------------------------------------- variants

struct Variant
    reaction::String
    label::String        # "-2σ", "+1σ", "×0.01", ...
    value::Float64       # sigma, or the factor itself
    run::PPNRun
    base::PPNRun
    group::String        # "STARLIB" / "flat ×10" / "factor"
    source::String       # networksetup.txt rate-source label of the factored reaction ("" if unknown)
end

sigma_label(s) = (s > 0 ? "+" : "") * string(Int(s)) * "σ"

function sigma_variants(root, names, opt, group; sigmas = [-2, -1, 1, 2], sources = Dict{String,String}())
    return [Variant(n, sigma_label(s), Float64(s),
                    PPNRun(joinpath(root, n, "run_$(s)sigma_opt$opt")),
                    PPNRun(joinpath(root, n, "baseline_opt$opt")), group, get(sources, n, ""))
            for n in sort(collect(names)) for s in sigmas]
end

function factor_variants(root, names, baseline; factors = [0.01, 0.1, 0.5, 2.0, 10.0, 100.0], sources = Dict{String,String}())
    return [Variant(n, "×" * string(f), f, PPNRun(joinpath(root, n, "run_f$(f)_opt0")), baseline, "factor", get(sources, n, ""))
            for n in sort(collect(names)) for f in factors]
end

# ---------------------------------------------------------------- tables

"""
    sweep_tables(variants) -> (ratio_table, residual_table)

Both are long tables, one row per (reaction, variant, isotope), final abundances:
  ratio_table    : isotopes with X_var/X_base > 1.1 or < 1/1.1
  residual_table : isotopes with |X_var - X_base|/X_base > 0.1
"""
function sweep_tables(variants)
    rows = NamedTuple[]
    for v in variants
        ab_b, ab_v = abundances(v.base, :final), abundances(v.run, :final)
        ch = changed_isotopes(ab_v, ab_b; threshold = ILIADIS_THRESHOLD)   # X1 = variant, X2 = baseline
        for r in eachrow(ch)
            (isfinite(r.ratio) && r.X2 > 0) || continue
            push!(rows, (reaction = v.reaction, source = v.source, group = v.group, variant = v.label, value = v.value, isotope = r.isotope,
                         X_base = r.X2, X_var = r.X1, ratio = r.ratio, log_ratio = r.log_ratio,
                         residual = (r.X1 - r.X2) / r.X2))
        end
    end
    tbl = isempty(rows) ? DataFrame(reaction = String[], source = String[], group = String[], variant = String[], value = Float64[],
                                    isotope = String[], X_base = Float64[], X_var = Float64[], ratio = Float64[],
                                    log_ratio = Float64[], residual = Float64[]) : DataFrame(rows)
    ratio_table = tbl
    residual_table = filter(:residual => r -> abs(r) > RESIDUAL_THRESHOLD, tbl)
    return ratio_table, residual_table
end

"Per (reaction, variant): how many isotopes crossed the bar and the biggest swing."
function affected_by_variant(tbl)
    isempty(tbl) && return DataFrame()
    g = combine(groupby(tbl, [:reaction, :source, :group, :variant, :value]),
                nrow => :n_isotopes,
                :log_ratio => (x -> maximum(abs.(x))) => :max_abs_log_ratio,
                :isotope => (x -> join(sort(x), ", ")) => :isotopes)
    return sort(g, [:reaction, :value])
end

"Ranking: reactions ordered by their single biggest swing over all their variants."
function rank_reactions(tbl, all_names; sources = Dict{String,String}())
    ranked = isempty(tbl) ? DataFrame(reaction = String[], source = String[], n_isotopes = Int[], max_abs_log_ratio = Float64[]) :
        combine(groupby(tbl, [:reaction, :source]), :isotope => (x -> length(unique(x))) => :n_isotopes,
                :log_ratio => (x -> maximum(abs.(x))) => :max_abs_log_ratio)
    missing_names = setdiff(all_names, ranked.reaction)
    ranked = vcat(ranked, DataFrame(reaction = collect(missing_names), source = [get(sources, n, "") for n in missing_names],
                                    n_isotopes = 0, max_abs_log_ratio = 0.0))
    ranked.max_pct_change = 100 .* (10.0 .^ ranked.max_abs_log_ratio .- 1)
    return sort(ranked, :max_abs_log_ratio; rev = true)
end

"Reaction x variant grid of max |log10(X_var/X_base)| (0 = nothing crossed 10 %)."
function ranking_grid(tbl, variants)
    labels = unique([(v.value, v.label) for v in variants]) |> x -> [l for (_, l) in sort(x)]
    grid = DataFrame(reaction = sort(unique(v.reaction for v in variants)))
    src = Dict(v.reaction => v.source for v in variants)
    grid.source = [src[n] for n in grid.reaction]
    for l in labels
        grid[!, l] = [begin
                          sub = filter(r -> r.reaction == n && r.variant == l, tbl)
                          isempty(sub) ? 0.0 : maximum(abs.(sub.log_ratio))
                      end for n in grid.reaction]
    end
    return grid
end

function ranking_barplot(ranked; title = "sensitivity ranking")
    shown = filter(:n_isotopes => >(0), ranked)
    with_nugrid_theme() do
        fig = CM.Figure(size = (900, max(400, 24 * nrow(shown) + 120)))
        ax = CM.Axis(fig[1, 1]; xlabel = "max |log10(X_var / X_base)|  (any isotope, any variant)", ylabel = "reaction",
                     title = title, yticks = (1:nrow(shown), shown.reaction), yreversed = true)
        CM.barplot!(ax, 1:nrow(shown), shown.max_abs_log_ratio; direction = :x, color = NUGRID_PALETTE[1])
        fig
    end
end

# ---------------------------------------------------------------- drill-down

hottest_cycle(run) = run.cycles[argmax([abundances(run, c).t9 for c in run.cycles])]

"""
    inspect_variant(v, reaction_index)

Charts for one factored run against its baseline: final abundances, ratio chart, residual chart,
and the flux charts (baseline, then factored) over the Z range around the reaction.
"""
function inspect_variant(v::Variant, reaction_index::Integer)
    ab_b, ab_v = abundances(v.base, :final), abundances(v.run, :final)
    net = network(v.base)
    r = reaction_by_index(net, reaction_index)
    zs = [i.Z for i in vcat(collect(r.reactants), collect(r.products))]
    zr = (max(0, minimum(zs) - 1), maximum(zs) + 1)
    hc = hottest_cycle(v.base)
    ttl = "$(v.reaction) $(v.label)"
    println(ttl, ": Z range ", zr, ", hottest baseline cycle ", hc)
    display(abundance_chart(ab_b; title = "baseline final abundance", hide_below_tolerance = true, tolerance = 1e-13))
    display(abundance_chart(ab_v; title = "$ttl final abundance", hide_below_tolerance = true, tolerance = 1e-13))
    display(ratio_chart(ab_v, ab_b; title = "$ttl : X_var / X_base", hide_below_tolerance = true, tolerance = 1e-20))
    display(residual_chart(ab_b, ab_v; title = "$ttl : present only in baseline (red) / only in factored (blue)", hide_below_tolerance = true))
    display(flux_chart(v.base, hc; net = net, z_range = zr, tolerance = 1e-10, hide_below_tolerance = true,
                       title = "baseline flux, cycle $hc"))
    display(flux_chart(v.run, hc; net = network(v.run), z_range = zr, tolerance = 1e-10, hide_below_tolerance = true,
                       title = "$ttl flux, cycle $hc"))
    return nothing
end

# ---------------------------------------------------------------- STARLIB factoring check

"""
    verify_starlib_factoring(root, name, species, opt; sigmas = [-2,-1,1,2]) -> DataFrame

Reads the STARLIB data file the sweep actually rewrote (the one real, non-symlink file under
run_<σ>sigma_opt<N>/NPDATA) next to the untouched original, and compares
r_σ(T9) / r_median(T9) with f(T9)^σ at every tabulated T9 -- the equation the sweep is meant to apply.
"""
function verify_starlib_factoring(root, name, species, opt; sigmas = [-2, -1, 1, 2])
    rundir(s) = joinpath(root, name, "run_$(s)sigma_opt$opt")
    npdata = joinpath(rundir(sigmas[1]), "NPDATA")
    rel = nothing
    for (dir, _, files) in walkdir(npdata), f in files
        p = joinpath(dir, f)
        islink(p) || (rel = relpath(p, npdata))
    end
    rel === nothing && error("no rewritten data file under $npdata")
    base = starlib_rate_curve(joinpath(root, name, "baseline_opt$opt", "NPDATA", rel), species)
    out = DataFrame(T9 = base.T9, rate_median = base.rate, f_u = base.factor)
    for s in sigmas
        cur = starlib_rate_curve(joinpath(rundir(s), "NPDATA", rel), species)
        out[!, "r/r_med  (σ=$s)"] = cur.rate ./ base.rate
        out[!, "f^σ  (σ=$s)"] = base.factor .^ s
    end
    println("rewritten file: ", rel)
    return out
end

"max relative error between the applied factor and f^σ over all T9 and σ in a `verify_starlib_factoring` table."
function max_factoring_error(tbl, sigmas = [-2, -1, 1, 2])
    return maximum(maximum(abs.(tbl[!, "r/r_med  (σ=$s)"] ./ tbl[!, "f^σ  (σ=$s)"] .- 1)) for s in sigmas)
end

function plot_factored_curves(tbl, name; sigmas = [-2, -1, 1, 2])
    with_nugrid_theme() do
        fig = CM.Figure(size = (800, 500))
        ax = CM.Axis(fig[1, 1]; xlabel = "T9", ylabel = "rate  [cm³ mol⁻¹ s⁻¹]", yscale = log10, xscale = log10,
                     title = "$name: STARLIB median and factored rate curves")
        CM.lines!(ax, tbl.T9, tbl.rate_median; color = :black, label = "median (0σ)", linewidth = 3)
        for (k, s) in enumerate(sigmas)
            CM.lines!(ax, tbl.T9, tbl.rate_median .* tbl[!, "r/r_med  (σ=$s)"]; label = "$(s)σ", color = NUGRID_PALETTE[k])
        end
        CM.axislegend(ax; position = :lt)
        fig
    end
end

# ---------------------------------------------------------------- era comparison helpers

"`sweep_lists/opt<N>_starlib.txt` as name => species tokens."
function read_starlib_list(case_dir, opt)
    d = Dict{String,Vector{String}}()
    for l in eachline(joinpath(case_dir, "sweep_lists", "opt$(opt)_starlib.txt"))
        (isempty(strip(l)) || startswith(l, "#")) && continue
        n, sp = split(l, ':'; limit = 2)
        d[strip(n)] = String.(strip.(split(sp, ',')))
    end
    return d
end

"""
    starlib_fu_at(root, name, species, opt, T9) -> Float64 or missing

The STARLIB factor uncertainty f.u.(T9) of one swept reaction at temperature `T9` (linear interpolation in
log10 T9 between tabulated points, clamped at the ends), read from the untouched original data file.
"""
function starlib_fu_at(root, name, species, opt, T9)
    npdata = joinpath(root, name, "run_1sigma_opt$opt", "NPDATA")
    rel = nothing
    for (dir, _, files) in walkdir(npdata), f in files
        p = joinpath(dir, f)
        islink(p) || (rel = relpath(p, npdata))
    end
    rel === nothing && return missing
    c = starlib_rate_curve(joinpath(root, name, "baseline_opt$opt", "NPDATA", rel), species)
    x, y, xt = log10.(c.T9), c.factor, log10(T9)
    xt <= x[1] && return y[1]
    xt >= x[end] && return y[end]
    k = searchsortedlast(x, xt)
    w = (xt - x[k]) / (x[k+1] - x[k])
    return y[k] * (1 - w) + y[k+1] * w
end

"Uncertainty level a variant represents: |σ| for σ sweeps; 1 for ×0.1/×10 and 2 for ×0.01/×100 in the pure-factor sweep; 0 for ×0.5/×2."
function level_of(group, value)
    group == "factor" || return abs(Int(value))
    return value in (0.1, 10.0) ? 1 : value in (0.01, 100.0) ? 2 : 0
end
