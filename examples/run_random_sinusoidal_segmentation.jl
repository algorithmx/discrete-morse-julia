# Runner: Random sinusoidal surface with critical points and segmentation overlay

using Printf
using Random
using Statistics
using Plots

# Bring in core algorithms and surface generator (files live one level up)
include(joinpath(@__DIR__, "..", "discrete_morse_critical_points.jl"))
include(joinpath(@__DIR__, "..", "morse_smale_surface_segmentation.jl"))
include(joinpath(@__DIR__, "test_surface_generator.jl"))

"""
Generate a randomized smooth height field using a sum of random sinusoidal components
plus a few random Gaussian bumps, then normalize.
Returns a ny x nx matrix Z.
"""
function generate_random_heightfield(xvals::Vector{Float64}, yvals::Vector{Float64};
    K::Int=7,
    amp_range::Tuple{Float64,Float64}=(0.3, 1.2),
    k_range::Tuple{Float64,Float64}=(0.6, 4.0),   # cycles across domain
    envelope_amp::Float64=0.35,
    bump_count::Int=6,
    bump_amp::Float64=0.8,
    bump_sigma_range::Tuple{Float64,Float64}=(0.06, 0.18), # fraction of domain
    seed::Int=1234)

    Random.seed!(seed)

    nx = length(xvals)
    ny = length(yvals)
    width = xvals[end] - xvals[1]
    height = yvals[end] - yvals[1]

    # Random sinusoidal components: A_k * sin(2π * (kx*x/width + ky*y/height) + phi)
    comps = [(rand() * (amp_range[2]-amp_range[1]) + amp_range[1],
              (rand() * (k_range[2]-k_range[1]) + k_range[1]),
              2π * rand(),
              2π * rand()) for _ in 1:K]
    # comps: (A, k, theta, phi)

    # Envelope parameters
    env_rx = rand(1.0:0.1:2.2)
    env_ry = rand(0.6:0.1:1.8)
    env_phi = 2π * rand()

    # Random bumps: (Ab, cx, cy, sigma_x, sigma_y)
    bumps = [( (2rand()-1) * bump_amp,
               xvals[1] + rand() * width,
               yvals[1] + rand() * height,
               (rand()*(bump_sigma_range[2]-bump_sigma_range[1]) + bump_sigma_range[1]) * width,
               (rand()*(bump_sigma_range[2]-bump_sigma_range[1]) + bump_sigma_range[1]) * height)
             for _ in 1:bump_count]

    Z = Matrix{Float64}(undef, ny, nx)
    for (j, y) in enumerate(yvals)
        for (i, x) in enumerate(xvals)
            # Sum of random directional sinusoids
            s = 0.0
            for (A, k, theta, phi) in comps
                kx = k * cos(theta)
                ky = k * sin(theta)
                arg = 2π * (kx * (x - xvals[1]) / width + ky * (y - yvals[1]) / height) + phi
                s += A * sin(arg)
            end

            # Smooth envelope to vary amplitude spatially
            env = 1.0 + envelope_amp * sin(2π * (env_rx * (x - xvals[1]) / width + env_ry * (y - yvals[1]) / height) + env_phi)

            # Add random bumps (positive and negative)
            bsum = 0.0
            for (Ab, cx, cy, sx, sy) in bumps
                dx = x - cx; dy = y - cy
                bsum += Ab * exp(-0.5 * ((dx/sx)^2 + (dy/sy)^2))
            end

            Z[j,i] = env * s + bsum
        end
    end

    # Normalize to zero mean and unit stddev, then scale to typical amplitude ~ 1.2
    m = mean(Z)
    sd = std(Z)
    sd = sd > 0 ? sd : 1.0
    Z .= 1.2 .* ((Z .- m) ./ sd)
    return Z
end

# Draw heightfield with segmentation and critical points overlay
function draw_heightfield_with_segmentation_and_critical_points(
    mesh::TriangleMesh,
    minima::Vector{Int},
    saddles::Vector{Int},
    maxima::Vector{Int},
    nx::Int, ny::Int, width::Float64, height::Float64,
    scalar_field::Vector{Float64},
    segmentation::Vector{Int};
    outpath::String="random_height_seg_crit.png",
    seg_alpha::Float64=0.35,
    seg_palette=:tab20,
    jitter_frac::Float64=0.0,
    fig_size::Tuple{Int,Int}=(900, 800),
    min_ms::Int=5, sad_ms::Int=5, max_ms::Int=6,
    min_alpha::Float64=0.95, sad_alpha::Float64=0.95, max_alpha::Float64=0.95,
    min_sep_frac::Float64=0.0
)
    xvals = collect(range(0, width, length=nx))
    yvals = collect(range(0, height, length=ny))

    Z = reshape(scalar_field, nx, ny)'

    segF = Float64.(segmentation)
    replace!(segF, -1 => NaN)
    Zseg = reshape(segF, nx, ny)'

    plt = heatmap(
        xvals, yvals, Z,
        aspect_ratio=:equal,
        color=:viridis,
        framestyle=:box,
        xlabel="x",
        ylabel="y",
        title="Random sinusoidal: segmentation and Morse critical points",
        size=fig_size,
        colorbar=false,
    )

    heatmap!(plt, xvals, yvals, Zseg; color=seg_palette, alpha=seg_alpha, colorbar=false)

    # Critical points
    xmin = [mesh.vertices[1, v] for v in minima]
    ymin = [mesh.vertices[2, v] for v in minima]
    xsad = Float64[]; ysad = Float64[]
    for e in saddles
        p = get_critical_point_location(Cell(1, e), mesh)
        push!(xsad, p[1]); push!(ysad, p[2])
    end
    xmax = Float64[]; ymax = Float64[]
    for t in maxima
        p = get_critical_point_location(Cell(2, t), mesh)
        push!(xmax, p[1]); push!(ymax, p[2])
    end

    dx = (nx > 1 ? width / (nx - 1) : width)
    dy = (ny > 1 ? height / (ny - 1) : height)
    jx = jitter_frac * dx
    jy = jitter_frac * dy
    xminp = copy(xmin); yminp = copy(ymin)
    xsadp = copy(xsad); ysadp = copy(ysad)
    xmaxp = copy(xmax); ymaxp = copy(ymax)
    if jitter_frac > 0
        @inbounds for i in eachindex(xminp); xminp[i] -= 0.7*jx; end
        @inbounds for i in eachindex(yminp); yminp[i] -= 0.7*jy; end
        @inbounds for i in eachindex(xsadp); xsadp[i] += 0.25*jx; end
        @inbounds for i in eachindex(ysadp); ysadp[i] -= 0.25*jy; end
        @inbounds for i in eachindex(xmaxp); xmaxp[i] += 0.7*jx; end
        @inbounds for i in eachindex(ymaxp); ymaxp[i] += 0.7*jy; end
    end

    function thin_points(x::Vector{Float64}, y::Vector{Float64}, min_sep_x::Float64, min_sep_y::Float64)
        if isempty(x) || (min_sep_x <= 0 && min_sep_y <= 0)
            return x, y
        end
        kept_x = Float64[]; kept_y = Float64[]
        occupied = Set{Tuple{Int,Int}}()
        for i in eachindex(x)
            col = min_sep_x > 0 ? Int(floor((x[i] - 0.0) / min_sep_x)) : 0
            row = min_sep_y > 0 ? Int(floor((y[i] - 0.0) / min_sep_y)) : 0
            key = (col, row)
            if !(key in occupied)
                push!(occupied, key)
                push!(kept_x, x[i])
                push!(kept_y, y[i])
            end
        end
        return kept_x, kept_y
    end

    if min_sep_frac > 0
        min_sep_x = min_sep_frac * dx
        min_sep_y = min_sep_frac * dy
        xminp, yminp = thin_points(xminp, yminp, min_sep_x, min_sep_y)
        xsadp, ysadp = thin_points(xsadp, ysadp, min_sep_x, min_sep_y)
        xmaxp, ymaxp = thin_points(xmaxp, ymaxp, min_sep_x, min_sep_y)
    end

    if !isempty(xminp)
        scatter!(plt, xminp, yminp; markershape=:circle, markerstrokecolor=:white, markerstrokewidth=0.6,
                 color=:dodgerblue, label="minima", ms=min_ms, alpha=min_alpha)
    end
    if !isempty(xsadp)
        scatter!(plt, xsadp, ysadp; markershape=:square, markerstrokecolor=:black, markerstrokewidth=0.6,
                 color=:gold, label="saddles", ms=sad_ms, alpha=sad_alpha)
    end
    if !isempty(xmaxp)
        scatter!(plt, xmaxp, ymaxp; markershape=:utriangle, markerstrokecolor=:black, markerstrokewidth=0.6,
                 color=:red, label="maxima", ms=max_ms, alpha=max_alpha)
    end

    savefig(plt, outpath)
    println("Saved segmentation+overlay plot to $(outpath)")
end

function main()
    println("Generating random sinusoidal surface and running Discrete Morse with segmentation…")

    # Resolution and domain
    nx, ny = 160, 160
    width, height = 25.0, 25.0

    verts, tris0 = generate_sinusoidal_surface(
        nx, ny, width, height;
        frequency1=2π/ (width/2),
        frequency2=2π/ (height/3),
        amplitude1=1.0,
        amplitude2=0.6,
        phase_shift=2π*rand(),
    )

    # Randomize height field
    xvals = collect(range(0, width, length=nx))
    yvals = collect(range(0, height, length=ny))
    seed_val = Int(mod(time_ns(), 0x7fffffff))
    Z = generate_random_heightfield(xvals, yvals; K=9, amp_range=(0.25,1.3), k_range=(0.6,5.0),
                                    envelope_amp=0.45, bump_count=8, bump_amp=0.9,
                                    bump_sigma_range=(0.05, 0.16), seed=seed_val)

    # Update vertex elevations
    vidx = 1
    for j in 1:ny
        for i in 1:nx
            verts[3, vidx] = Z[j,i]
            vidx += 1
        end
    end

    # Build mesh (convert triangles from 0-based to 1-based)
    tris1 = tris0 .+ 1
    mesh = TriangleMesh(verts, tris1)

    # Scalar field from z
    scalar_field = vec(verts[3, :])

    # Compute discrete Morse + segmentation (final MS segmentation)
    results = compute_surface_morse_smale_from_scalar(
        mesh, scalar_field;
        persistence_threshold=0.05,
        rebuild_gradient_after_simplification=true,
        build_network=false,
    )

    minima = results.minima
    saddles = results.saddles
    maxima = results.maxima
    ms_labels = results.segmentation.morseSmale_

    # Print summary
    print_critical_point_summary(minima, saddles, maxima, mesh)

    # Save mesh and separatrices
    save_mesh_obj(verts, tris0, "random_sinusoidal_surface.obj")
    export_separatrices_vtk(results.separatrices_points, "random_sinusoidal_separatrices.vtk")

    # Draw overlay
    draw_heightfield_with_segmentation_and_critical_points(
        mesh, minima, saddles, maxima,
        nx, ny, width, height, scalar_field, ms_labels;
        jitter_frac=0.0,
        min_sep_frac=0.0,
        seg_alpha=0.35,
        seg_palette=:tab20,
        outpath="random_sinusoidal_height_seg_crit.png",
    )

    println("\nDone. OBJ saved as random_sinusoidal_surface.obj, VTK saved as random_sinusoidal_separatrices.vtk, and image saved as random_sinusoidal_height_seg_crit.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
