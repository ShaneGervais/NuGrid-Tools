# build_baselines.jl — one unmodified baseline run per starlib_option.
#
# Builds <out_dir>/run_opt<N>/ppn/ (+ run_opt<N>/NPDATA symlink) for every N in
# --options, each a copy of <template_dir> with only `starlib_option = N`
# patched in ppn_physics.input, then runs them in parallel. Option 0 = no
# STARLIB, 1 = MC10+MC13, 2 = ETR25 layered on MC10+MC13.
#
# Usage:
#   julia --project=<NuGridJl> build_baselines.jl <template_dir> <out_dir>
#         [--options 0,1,2] [--jobs N] [--dry-run]

include("build_sigma_sweep.jl")

function usage()
    println("""
Usage:
  julia build_baselines.jl <template_dir> <out_dir> [options]

Build and run one unmodified baseline per starlib_option at
<out_dir>/run_opt<N>/ (ppn.exe runs inside run_opt<N>/ppn/).

Options:
  --options N,...   starlib_option values (default: 0,1,2)
  --jobs N, -j N    ppn.exe runs in parallel (default: 3)
  --dry-run         Build directories without launching ppn.exe
  -h, --help        Show this help

Example:
  julia --project=NuGridJl NuGridJl/tools/build_baselines.jl ppn . --jobs 3
""")
end

"""
    build_baselines(template_dir, out_dir; options = [0, 1, 2], jobs = 3, dry_run = false) -> Vector{String}

Build (and, unless `dry_run`, run) `out_dir/run_opt<N>/ppn` for each `N` in
`options`. Returns the run directories (the `ppn/` subdirectories).
"""
function build_baselines(template_dir::AbstractString, out_dir::AbstractString;
                          options::Vector{<:Integer} = [0, 1, 2], jobs::Integer = 3, dry_run::Bool = false)
    ppn_dirs = String[]
    for option in options
        run_dir = joinpath(out_dir, "run_opt$option")
        isdir(run_dir) && throw(ArgumentError("$run_dir already exists -- remove it first"))
        push!(ppn_dirs, build_flat_factor_run!(template_dir, run_dir, option))
    end
    dry_run || run_parallel(ppn_dirs; jobs)
    return ppn_dirs
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    let jobs = 3, dry_run = false, options = Int[0, 1, 2], positional = String[], i = 1
        if isempty(ARGS) || ARGS[1] in ("-h", "--help")
            usage(); exit(0)
        end
        while i <= length(ARGS)
            a = ARGS[i]
            if a in ("-h", "--help")
                usage(); exit(0)
            elseif a == "--options"
                i == length(ARGS) && error("--options requires a value")
                options = parse_int_list(ARGS[i + 1]); i += 2
            elseif a in ("--jobs", "-j")
                i == length(ARGS) && error("--jobs requires a value")
                jobs = parse(Int, ARGS[i + 1]); i += 2
            elseif a == "--dry-run"
                dry_run = true; i += 1
            elseif startswith(a, "-")
                println(stderr, "Unknown option: $a\n"); usage(); exit(1)
            else
                push!(positional, a); i += 1
            end
        end
        if length(positional) != 2
            println(stderr, "Expected 2 positional arguments (<template_dir> <out_dir>), got $(length(positional)).\n")
            usage(); exit(1)
        end
        built = build_baselines(positional[1], positional[2]; options, jobs, dry_run)
        println(length(built), " baseline run(s) at ", positional[2])
    end
end
