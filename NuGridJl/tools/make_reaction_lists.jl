# make_reaction_lists.jl — per-option sweep lists from finished baselines.
#
# For each starlib_option N, reads <case_dir>/run_opt<N>/ppn (see build_baselines.jl),
# takes every reaction with flux >= --threshold at any cycle of the trajectory, drops
# the ones that should not be factored, and writes:
#
#   <out_dir>/opt<N>_reactions.csv   every listed reaction + whether/why it was excluded
#   <out_dir>/opt<N>_flat.txt        "name: index"   -> --rate-index-file lists
#   <out_dir>/opt<N>_starlib.txt     "name: species" -> --reactions-file lists (N >= 1 only:
#                                    reactions whose source in that network is STL01/STL02)
#
# Excluded from every sweep:
#   * weak decays -- rtype (+,g), (-,g), (b,p): measured half-lives, few-percent uncertainty
#   * 8B(g,a)4He (the 8B -> 2a breakup)
#   * a non-STARLIB row that duplicates an active STARLIB row (same reactants and products)
#     -- the network double-counts that reaction (e.g. 25Mg(p,g)26Al's ILI01 row)
#
# Usage: julia --project=<NuGridJl> make_reaction_lists.jl <case_dir> <out_dir>
#            [--options 0,1,2] [--threshold 1e-10]

if abspath(PROGRAM_FILE) == (@__FILE__) && (isempty(ARGS) || ARGS[1] in ("-h", "--help"))
    println("Usage: julia make_reaction_lists.jl <case_dir> <out_dir> [--options 0,1,2] [--threshold 1e-10]")
    exit(0)
end

using NuGridJl, DataFrames, CSV

const WEAK_RTYPES = Set(["(+,g)", "(-,g)", "(b,p)"])
const STARLIB_SOURCES = Set(["STL01", "STL02"])

function starlib_token(iso::Isotope)
    (iso.Z, iso.A) == (1, 1) && return "p"
    (iso.Z, iso.A) == (0, 1) && return "n"
    (iso.Z, iso.A) == (1, 2) && return "d"
    return lowercase(element_symbol(iso.Z)) * string(iso.A)
end

function reaction_name(r::Reaction)
    m = match(r"^(\S+?)\((\w+),(\w+)\)(\S+)$", label(r))
    m === nothing && return replace(label(r), r"[^A-Za-z0-9]" => "_")
    return string(m.captures[1], "_", m.captures[2], m.captures[3], "_", m.captures[4])
end

pair_key(r::Reaction) = (sort(collect(r.reactants); by = i -> (i.Z, i.A, i.isomer)),
                         sort(collect(r.products); by = i -> (i.Z, i.A, i.isomer)))

function make_lists(case_dir, out_dir, option; threshold = 1e-10)
    run = PPNRun(joinpath(case_dir, "run_opt$option", "ppn"))
    net = network(run)
    df = flux_reaction_list(run, run.cycles; threshold)
    by_index = Dict(r.index => r for r in net.reactions)
    starlib_pairs = Set(pair_key(by_index[row.index]) for row in eachrow(df) if row.source in STARLIB_SOURCES)

    reasons = String[]; names = String[]
    for row in eachrow(df)
        r = by_index[row.index]
        push!(names, reaction_name(r))
        reason =
            row.rtype in WEAK_RTYPES ? "weak decay" :
            (label(r) == "8B(g,a)4He") ? "8B -> 2a breakup" :
            (!(row.source in STARLIB_SOURCES) && pair_key(r) in starlib_pairs) ? "duplicates an active STARLIB row" : ""
        push!(reasons, reason)
    end
    # names must be unique directory names
    seen = Dict{String,Int}()
    for (k, n) in enumerate(names)
        seen[n] = get(seen, n, 0) + 1
        seen[n] > 1 && (names[k] = string(n, "_i", df.index[k]))
    end
    df.name = names; df.excluded = reasons
    df.starlib = [s in STARLIB_SOURCES for s in df.source]

    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "opt$(option)_reactions.csv"), df)
    kept = filter(:excluded => isempty, df)
    open(joinpath(out_dir, "opt$(option)_flat.txt"), "w") do io
        println(io, "# opt$option: reactions with flux >= $threshold, without a tabulated STARLIB uncertainty; name: networksetup.txt index")
        for row in eachrow(option == 0 ? kept : filter(:starlib => !, kept))
            println(io, row.name, ": ", row.index)
        end
    end
    if option >= 1
        open(joinpath(out_dir, "opt$(option)_starlib.txt"), "w") do io
            println(io, "# opt$option: STARLIB-sourced reactions (name: species tokens)")
            for row in eachrow(filter(:starlib => identity, kept))
                r = by_index[row.index]
                println(io, row.name, ": ", join(starlib_token.(vcat(collect(r.reactants), collect(r.products))), ","))
            end
        end
    end
    println("opt$option: $(nrow(df)) listed, $(nrow(kept)) kept ($(count(kept.starlib)) STARLIB); excluded: ",
            join(["$k=$v" for (k, v) in sort(collect(pairs(Dict(r => count(==(r), reasons) for r in unique(reasons) if r != ""))))], ", "))
    return df
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    let options = [0, 1, 2], threshold = 1e-10, positional = String[], i = 1
        while i <= length(ARGS)
            if ARGS[i] == "--options"; options = parse.(Int, split(ARGS[i+1], ',')); i += 2
            elseif ARGS[i] == "--threshold"; threshold = parse(Float64, ARGS[i+1]); i += 2
            else push!(positional, ARGS[i]); i += 1 end
        end
        length(positional) == 2 || (println(stderr, "expected <case_dir> <out_dir>"); exit(1))
        foreach(o -> make_lists(positional[1], positional[2], o; threshold), options)
    end
end
