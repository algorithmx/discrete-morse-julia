# Multi-round test runner: sweep surfaces across multiple resolutions

using Printf
using Dates
using Plots

# Bring in the algorithm and surface generators
include(joinpath(@__DIR__, "..", "discrete_morse_critical_points.jl"))
include(joinpath(@__DIR__, "..", "morse_smale_surface_segmentation.jl"))
include(joinpath(@__DIR__, "test_surface_generator.jl"))
include(joinpath(@__DIR__, "run_discrete_morse_sinusoidal.jl")) # brings draw_... helpers

struct RunResult
    round::Int
    surface::String
    resx::Int
    resy::Int
    nverts::Int
    ntris::Int
    n_min::Int
    n_sad::Int
    n_max::Int
    elapsed_ms::Int
end

function write_csv(path::String, rows::Vector{RunResult})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "round,surface,resx,resy,nverts,ntris,n_min,n_sad,n_max,elapsed_ms")
        for r in rows
            println(io, @sprintf("%d,%s,%d,%d,%d,%d,%d,%d,%d,%d",
                r.round, r.surface, r.resx, r.resy, r.nverts, r.ntris,
                r.n_min, r.n_sad, r.n_max, r.elapsed_ms))
        end
    end
end


function run_multi_round_tests(; rounds::Int=5,
    persistence_threshold::Float64=0.05,
    rebuild_gradient_after_simplification::Bool=true,
    build_network::Bool=false,
    save_objs::Bool=true,
    save_images::Bool=true,
    results_dir::String=joinpath(@__DIR__, "results"),
    resolutions::Vector{Tuple{Int,Int}}=[(48, 48), (96, 96), (128, 128), (160, 160)])

    # Define surfaces repertoire
    surfaces = [
        (
            name="eggbox",
            planar=true,
            gen=(nx, ny) -> begin
                width, height = 20.0, 20.0
                cycles = 6.0
                generate_sinusoidal_surface(nx, ny, width, height;
                    frequency1=2π * cycles / width,
                    frequency2=2π * cycles / height,
                    amplitude1=3.0,
                    amplitude2=0.0,
                    phase_shift=0.0)
            end,
            dims=(20.0, 20.0),
            resolutions=resolutions
        ),
        (
            name="sinusoidal",
            planar=true,
            gen=(nx, ny) -> generate_sinusoidal_surface(nx, ny, 25.0, 25.0;
                    frequency1=2π / (25.0 / 2), 
                    frequency2=2π / (25.0 / 3), 
                    amplitude1=3.0, 
                    amplitude2=1.6, 
                    phase_shift=2π * rand()),
            dims=(25.0, 25.0),
            resolutions=resolutions
        ),
        (
            name="gaussian_bumps",
            planar=true,
            gen=(nx, ny) -> generate_gaussian_bumps(nx, ny, 20.0, 20.0;
                    bump_positions=[(5.0, 5.0), (12.0, 7.0), (15.0, 14.0)], 
                    bump_heights=[4.2, 2.3, 3.1], 
                    bump_widths=[1.3, 0.8, 1.1]),
            dims=(20.0, 20.0),
            resolutions=resolutions
        ),
        (
            name="saddle",
            planar=true,
            gen=(nx, ny) -> generate_saddle_surface(nx, ny, 10.0, 10.0; curvature=0.5),
            dims=(10.0, 10.0),
            resolutions=resolutions
        ),
        (
            name="complex",
            planar=true,
            gen=(nx, ny) -> generate_complex_surface(nx, ny, 24.0, 24.0),
            dims=(24.0, 24.0),
            resolutions=resolutions
        ),
    ]

    results = RunResult[]
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    println("Starting multi-round tests at $ts\n")
    img_dir = joinpath(results_dir, "images")
    obj_dir = joinpath(results_dir, "objs")
    if save_images
        mkpath(img_dir)
    end
    if save_objs
        mkpath(obj_dir)
    end

    # No torus helper needed anymore

    for r in 1:rounds
        s = surfaces[((r-1)%length(surfaces))+1]
        println("Round $r / $rounds: Surface = $(s.name)")

        for (resx, resy) in s.resolutions
            # Generate mesh (generators return 0-based tris)
            verts, tris0 = s.gen(resx, resy)
            tris1 = tris0 .+ 1

            mesh = TriangleMesh(verts, tris1)
            scalar_field = vec(verts[3, :])

            t0 = time_ns()
            results_ms = compute_surface_morse_smale_from_scalar(
                mesh, scalar_field;
                persistence_threshold=persistence_threshold,
                rebuild_gradient_after_simplification=rebuild_gradient_after_simplification,
                build_network=build_network,
            )
            elapsed_ms = Int(Base.round((time_ns() - t0) / 1e6))

            n_min = length(results_ms.minima)
            n_sad = length(results_ms.saddles)
            n_max = length(results_ms.maxima)

            # Render image per mesh
            if save_images
                ms_labels = results_ms.segmentation.morseSmale_
                if s.planar
                    # For saddle, its coordinates span [-w/2,w/2]x[-h/2,h/2]. Shift a copy to [0,w]x[0,h] for visualization.
                    if s.name == "saddle"
                        minx = minimum(mesh.vertices[1, :])
                        maxx = maximum(mesh.vertices[1, :])
                        miny = minimum(mesh.vertices[2, :])
                        maxy = maximum(mesh.vertices[2, :])
                        widthp = maxx - minx
                        heightp = maxy - miny
                        verts_plot = copy(mesh.vertices)
                        verts_plot[1, :] .-= minx
                        verts_plot[2, :] .-= miny
                        mesh_plot = TriangleMesh(verts_plot, mesh.triangles)
                        outpath = joinpath(img_dir, @sprintf("%s_r%d_%dx%d.png", s.name, r, resx, resy))
                        mkpath(dirname(outpath))
                        draw_heightfield_with_segmentation_and_critical_points(
                            mesh_plot, results_ms.minima, results_ms.saddles, results_ms.maxima,
                            resx, resy, widthp, heightp, scalar_field, ms_labels;
                            jitter_frac=0.0,
                            min_sep_frac=0.0,
                            seg_alpha=0.35,
                            seg_palette=:tab20,
                            outpath=outpath,
                        )
                    else
                        width, height = s.dims
                        outpath = joinpath(img_dir, @sprintf("%s_r%d_%dx%d.png", s.name, r, resx, resy))
                        mkpath(dirname(outpath))
                        draw_heightfield_with_segmentation_and_critical_points(
                            mesh, results_ms.minima, results_ms.saddles, results_ms.maxima,
                            resx, resy, width, height, scalar_field, ms_labels;
                            jitter_frac=0.0,
                            min_sep_frac=0.0,
                            seg_alpha=0.35,
                            seg_palette=:tab20,
                            outpath=outpath,
                        )
                    end
                end
            end

            push!(results, RunResult(r, s.name, resx, resy, size(verts, 2), size(tris1, 2), n_min, n_sad, n_max, elapsed_ms))

            @printf("  - %s %dx%d | V=%d T=%d | mins=%d saddles=%d maxs=%d | %d ms\n",
                s.name, resx, resy, size(verts, 2), size(tris1, 2), n_min, n_sad, n_max, elapsed_ms)

            if save_objs
                objname = joinpath(obj_dir, @sprintf("%s_r%d_%dx%d.obj", s.name, r, resx, resy))
                mkpath(dirname(objname))
                save_mesh_obj(verts, tris0, objname)
            end
        end
        println("")
    end

    # Write CSV summary
    csv_path = joinpath(results_dir, @sprintf("multi_round_results_%s.csv", ts))
    write_csv(csv_path, results)
    println("Summary written to $(csv_path)")

    return results
end

function main()
    run_multi_round_tests(; rounds=5, persistence_threshold=0.05, rebuild_gradient_after_simplification=true, build_network=false, save_objs=true)
    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
