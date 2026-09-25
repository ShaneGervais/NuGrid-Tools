# build_factor_sweep.jl — pure multiplicative-factor sweep, addressed by reaction index.
#
# For every "name: index" line in --rate-index-file, builds
#   <out_dir_root>/<name>/run_f<factor>_opt<N>/ppn/
# for each factor: a copy of <template_dir> with only starlib_option = N and
# rate_index(1)/rate_factor(1) = (index, factor) patched in ppn_physics.input -- ppn's own
# apply_rate_factors mechanism: rate(T) -> factor * rate(T) at every T9 and every timestep.
# No per-reaction baseline is built; compare against the option's own run_opt<N> baseline.
#
# Usage: julia --project=<NuGridJl> build_factor_sweep.jl <template_dir> <out_dir_root>
#            --rate-index-file FILE [--factors 0.01,0.1,0.5,2,10,100] [--options 0] [--jobs N] [--dry-run]

include("build_sigma_sweep.jl")

factor_label(f::Real) = "f" * string(f)

function build_factor_sweep(template_dir, out_dir, index::Integer; factors, option::Integer)
    return [build_flat_factor_run!(template_dir, joinpath(out_dir, "run_$(factor_label(f))_opt$option"), option;
                                    factor_spec = (index = index, value = f)) for f in factors]
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    let file = nothing, factors = [0.01, 0.1, 0.5, 2.0, 10.0, 100.0], options = [0], jobs = 4, dry_run = false,
        positional = String[], i = 1
        while i <= length(ARGS)
            a = ARGS[i]
            if a == "--rate-index-file"; file = ARGS[i+1]; i += 2
            elseif a == "--factors"; factors = parse_real_list(ARGS[i+1]); i += 2
            elseif a == "--options"; options = parse_int_list(ARGS[i+1]); i += 2
            elseif a in ("--jobs", "-j"); jobs = parse(Int, ARGS[i+1]); i += 2
            elseif a == "--dry-run"; dry_run = true; i += 1
            elseif a in ("-h", "--help"); println("see the header of build_factor_sweep.jl"); exit(0)
            else push!(positional, a); i += 1 end
        end
        (length(positional) == 2 && file !== nothing) || (println(stderr, "usage: <template_dir> <out_dir_root> --rate-index-file FILE"); exit(1))
        template_dir, out_root = positional
        all_dirs = String[]
        reactions = read_flat_factor_reactions_file(file)
        for (k, (name, index)) in enumerate(reactions), option in options
            println("[$k/$(length(reactions))] $name (index $index), option $option")
            append!(all_dirs, build_factor_sweep(template_dir, joinpath(out_root, name), index; factors, option))
        end
        if !dry_run
            results = run_parallel(all_dirs; jobs)
            failed = [d for (d, ok) in results if !ok]
            isempty(failed) || @warn "some runs failed" failed
        end
        println(length(all_dirs), " run(s) at ", out_root)
    end
end
