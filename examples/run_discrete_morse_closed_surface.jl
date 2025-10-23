# Runner: Discrete Morse on a closed 2-manifold (wrapped torus) with random smooth variations

using Printf
using Random
using Statistics
using LinearAlgebra
using Plots

# Bring in the algorithm and mesh utilities (OBJ writer)
include(joinpath(@__DIR__, "..", "discrete_morse_critical_points.jl"))
include(joinpath(@__DIR__, "test_surface_generator.jl"))

# ==============================================
# Torus generator with periodic connectivity
# ==============================================

"""
Generate a wrapped torus surface with periodic connectivity (closed 2-manifold).

Returns:
- vertices :: 3 x (nu*nv) matrix, ordered with u varying fastest
- triangles :: 3 x (2*nu*nv) matrix of 0-based indices (for OBJ writer compatibility)
- u_vals :: vector of length nu with parameter values in [0, 2π)
- v_vals :: vector of length nv with parameter values in [0, 2π)
"""
function generate_wrapped_torus_surface(nu::Int, nv::Int;
    major_radius::Float64=3.0, minor_radius::Float64=1.0)

    @assert nu >= 3 && nv >= 3 "nu and nv must be >= 3"

    # Parameters in [0, 2π), periodic in both directions
    u_vals = collect(range(0, 2π, length=nu+1))[1:nu]
    v_vals = collect(range(0, 2π, length=nv+1))[1:nv]

    # Vertex layout: i in 1..nu (u index), j in 1..nv (v index), vertex id = (j-1)*nu + i
    vertices = Matrix{Float64}(undef, 3, nu*nv)
    @inbounds for j in 1:nv
        v = v_vals[j]
        cv, sv = cos(v), sin(v)
        for i in 1:nu
            u = u_vals[i]
            cu, su = cos(u), sin(u)
            x = (major_radius + minor_radius * cv) * cu
            y = (major_radius + minor_radius * cv) * su
            z = minor_radius * sv
            vid = (j-1)*nu + i
            vertices[1, vid] = x
            vertices[2, vid] = y
            vertices[3, vid] = z
        end
    end

    # Triangles: wrap indices periodically
    ntris = 2 * nu * nv
    triangles = Matrix{Int}(undef, 3, ntris)
    tri_idx = 1
    vid = (i, j) -> (j-1)*nu + i

    @inbounds for j in 1:nv
        jp = (j % nv) + 1
        for i in 1:nu
            ip = (i % nu) + 1
            v1 = vid(i, j)
            v2 = vid(ip, j)
            v3 = vid(i, jp)
            v4 = vid(ip, jp)

            # Two triangles per quad (use 0-based indexing for storage)
            triangles[1, tri_idx] = v1 - 1
            triangles[2, tri_idx] = v2 - 1
            triangles[3, tri_idx] = v3 - 1
            triangles[1, tri_idx + 1] = v2 - 1
            triangles[2, tri_idx + 1] = v4 - 1
            triangles[3, tri_idx + 1] = v3 - 1
            tri_idx += 2
        end
    end

    return vertices, triangles, u_vals, v_vals
end

# ==============================================
# Smooth random scalar field on the torus (u,v)
# ==============================================

"""
Build a smooth random scalar field over the torus parameters (u, v) using a small
random Fourier basis. Output is an nv x nu matrix Z with v as rows and u as columns.
"""
function generate_random_scalar_on_torus(u_vals::Vector{Float64}, v_vals::Vector{Float64};
    K::Int=10,
    k_range_u::UnitRange{Int}=1:4,
    k_range_v::UnitRange{Int}=1:4,
    amp_range::Tuple{Float64,Float64}=(0.3, 1.0),
    seed::Int=0)

    if seed != 0
        Random.seed!(seed)
    end

    nu = length(u_vals)
    nv = length(v_vals)

    modes = [(rand(k_range_u), rand(k_range_v), 2π*rand(), 2π*rand(), rand()*(amp_range[2]-amp_range[1]) + amp_range[1])
             for _ in 1:K]
    # tuple: (ku, kv, phi_u, phi_v, A)

    Z = Matrix{Float64}(undef, nv, nu)
    @inbounds for j in 1:nv
        v = v_vals[j]
        for i in 1:nu
            u = u_vals[i]
            s = 0.0
            for (ku, kv, phiu, phiv, A) in modes
                s += A * sin(ku*u + phiu) * cos(kv*v + phiv)
            end
            Z[j, i] = s
        end
    end

    # Normalize to zero mean, unit std, then scale
    m = mean(Z)
    sd = std(Z)
    sd = sd > 0 ? sd : 1.0
    Z .= 1.2 .* ((Z .- m) ./ sd)
    return Z
end

# ==============================================
# UV heatmap with critical point overlay
# ==============================================

"""
Draw a UV heatmap of the scalar field and overlay critical points (minima, saddles, maxima).
u is on the x-axis, v on the y-axis. For saddle/maximum markers, uv positions are derived
from the corresponding cells by averaging the parameters of incident vertices on the torus
with proper angle-averaging.
"""
function draw_uv_heatmap_with_critical_points(
    mesh::TriangleMesh,
    minima::Vector{Int},
    saddles::Vector{Int},
    maxima::Vector{Int},
    nu::Int, nv::Int,
    u_vals::Vector{Float64}, v_vals::Vector{Float64},
    scalar_field::Vector{Float64};
    outpath::String="torus_uv_crit.png",
    fig_size::Tuple{Int,Int}=(980, 820),
    jitter::Bool=false,
    ms_min::Int=5, ms_sad::Int=5, ms_max::Int=6)

    # Reshape scalar to nv x nu grid (rows=v, cols=u)
    Z = reshape(scalar_field, nu, nv)'

    plt = heatmap(
        u_vals, v_vals, Z,
        aspect_ratio=:equal,
        color=:viridis,
        framestyle=:box,
        xlabel="u (rad)", ylabel="v (rad)",
        title="UV scalar field with discrete Morse critical points",
        size=fig_size,
    )

    # Helpers for angle means on circle, normalized to [0, 2π)
    ang_mean(a::Vector{Float64}) = atan(sum(sin.(a)), sum(cos.(a)))
    norm_0_2pi(theta::Float64) = mod(theta, 2π)
    ang_mean_02pi(a::Vector{Float64}) = norm_0_2pi(ang_mean(a))

    # Vertex positions in parameter domain
    uv_of_vertex = function(vid::Int)
        i = ((vid - 1) % nu) + 1
        j = ((vid - 1) ÷ nu) + 1
        return (u_vals[i], v_vals[j])
    end

    # Minima (vertex-based)
    u_min = Float64[]; v_min = Float64[]
    for v in minima
        u, vv = uv_of_vertex(v)
        push!(u_min, u); push!(v_min, vv)
    end

    # Saddles (edge midpoints in UV by circular mean)
    u_sad = Float64[]; v_sad = Float64[]
    for e in saddles
        v1, v2 = mesh.edge_vertices[:, e]
        u1, v1p = uv_of_vertex(v1)
        u2, v2p = uv_of_vertex(v2)
        push!(u_sad, ang_mean_02pi([u1, u2]))
        push!(v_sad, ang_mean_02pi([v1p, v2p]))
    end

    # Maxima (triangle centroids in UV by circular mean)
    u_max = Float64[]; v_max = Float64[]
    for t in maxima
        a, b, c = mesh.triangles[:, t]
        uva = uv_of_vertex(a)
        uvb = uv_of_vertex(b)
        uvc = uv_of_vertex(c)
        push!(u_max, ang_mean_02pi([uva[1], uvb[1], uvc[1]]))
        push!(v_max, ang_mean_02pi([uva[2], uvb[2], uvc[2]]))
    end

    # Optional tiny jitter for visibility
    if jitter
        jitter_scale_u = (nu > 1 ? (u_vals[2]-u_vals[1]) : 0.0) * 0.15
        jitter_scale_v = (nv > 1 ? (v_vals[2]-v_vals[1]) : 0.0) * 0.15
        randj() = 2*(rand()-0.5)
        for arr in (u_min, u_sad, u_max)
            @inbounds for i in eachindex(arr); arr[i] += randj()*jitter_scale_u; end
        end
        for arr in (v_min, v_sad, v_max)
            @inbounds for i in eachindex(arr); arr[i] += randj()*jitter_scale_v; end
        end
    end

    if !isempty(u_min)
        scatter!(plt, u_min, v_min; markershape=:circle, markerstrokecolor=:white, markerstrokewidth=0.6,
                 color=:dodgerblue, label="minima", ms=ms_min, alpha=0.95)
    end
    if !isempty(u_sad)
        scatter!(plt, u_sad, v_sad; markershape=:square, markerstrokecolor=:black, markerstrokewidth=0.6,
                 color=:gold, label="saddles", ms=ms_sad, alpha=0.95)
    end
    if !isempty(u_max)
        scatter!(plt, u_max, v_max; markershape=:utriangle, markerstrokecolor=:black, markerstrokewidth=0.6,
                 color=:red, label="maxima", ms=ms_max, alpha=0.95)
    end

    savefig(plt, outpath)
    println("Saved UV overlay plot to $(outpath)")
end


function main()
    println("Generating wrapped torus (closed 2-manifold) and running Discrete Morse analysis…")

    # Torus resolution and radii
    nu, nv = 160, 120  # moderate resolution
    R, r = 3.0, 1.1

    # Build wrapped torus
    verts, tris0, u_vals, v_vals = generate_wrapped_torus_surface(nu, nv; major_radius=R, minor_radius=r)

    # Random smooth scalar over (u,v)
    seed_val = Int(mod(time_ns(), 0x7fffffff))
    Z = generate_random_scalar_on_torus(u_vals, v_vals; K=12, k_range_u=1:6, k_range_v=1:6,
                                        amp_range=(0.25, 1.0), seed=seed_val)

    # Scalar field laid out in vertex order (u fastest)
    scalar_field = collect(vec(Z'))  # transpose so that i (u) varies fastest across columns

    # Convert triangles to 1-based for TriangleMesh
    tris1 = tris0 .+ 1
    mesh = TriangleMesh(verts, tris1)

    # Run discrete Morse critical point detection with optional persistence simplification
    minima, saddles, maxima, gradient = find_critical_points_discrete_morse(
        mesh, scalar_field;
        # sos_offsets = some_int_vector,
        persistence_threshold = 0.06,
        rebuild_gradient_after_simplification = true,
    )

    # Print results
    print_critical_point_summary(minima, saddles, maxima, mesh)

    # Save mesh for external visualization (OBJ expects 0-based triangles)
    save_mesh_obj(verts, tris0, "wrapped_torus_random_scalar.obj")

    # Plot UV heatmap with critical points overlay
    draw_uv_heatmap_with_critical_points(
        mesh, minima, saddles, maxima,
        nu, nv, u_vals, v_vals, scalar_field;
        outpath="wrapped_torus_uv_crit.png",
        jitter=false,
    )

    println("\nDone. OBJ saved as wrapped_torus_random_scalar.obj and plot saved as wrapped_torus_uv_crit.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
