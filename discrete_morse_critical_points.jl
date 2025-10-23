"""
Discrete Morse Theory Critical Point Detection

Implementation of ttkDiscreteGradient critical point search algorithm in Julia.
Based on the ProcessLowerStars algorithm from Robins, Wood, and Sheppard (2000).

Author: Generated from TTK architecture analysis
Date: October 2025
"""

using Printf
using DataStructures
using LinearAlgebra
using Base.Threads

include(joinpath(@__DIR__, "core_data_structures.jl"))
include(joinpath(@__DIR__, "core_algorithms.jl"))
include(joinpath(@__DIR__, "critical_point.jl"))

# ==============================================================================
# MAIN CRITICAL POINT SEARCH FUNCTION
# ==============================================================================

"""
Main function to find critical points on a triangle surface mesh using discrete Morse theory.
Follows C++ TTK buildGradient algorithm structure.

Arguments:
- `mesh`: TriangleMesh containing the surface geometry
- `scalar_field`: Vector of scalar values at each vertex

Returns:
- `minima`: Vector of vertex indices that are local minima
- `saddles`: Vector of edge indices that are saddle points
- `maxima`: Vector of triangle indices that are local maxima
- `gradient`: Complete gradient field for further analysis

Example:
# Create a simple triangulated surface
vertices = [0.0 1.0 0.5; 0.0 0.0 1.0; 0.0 0.0 0.0]  # 3x3 matrix
triangles = [1 2; 2 3; 3 1]  # Single triangle
mesh = TriangleMesh(vertices, triangles)

# Define scalar field (e.g., height)
scalar_field = [0.0, 1.0, 0.5]

# Find critical points
minima, saddles, maxima, gradient = find_critical_points_discrete_morse(mesh, scalar_field)

println("Found \$(length(minima)) minima, \$(length(saddles)) saddles, \$(length(maxima)) maxima")
"""
function find_critical_points_discrete_morse(
    mesh::TriangleMesh,
    scalar_field::AbstractVector{<:Real};
    sos_offsets::Union{Nothing,AbstractVector{Int}}=nothing,
    persistence_threshold::Real=0.0, # tau
    rebuild_gradient_after_simplification::Bool=true,
    validate_gradient::Bool=true,
)
    # Compute vertex order using TTK OrderDisambiguation semantics
    # Primary: scalar; tie-breaker: provided SoS offsets if given, else vertex id.
    vertex_order = sort_vertices_order(scalar_field, sos_offsets)

    # Optionally perform scalar-field simplification by persistence, then
    # rebuild vertex order and gradient to reflect true topological simplification
    local_scalars = Vector{Float64}(scalar_field)
    if persistence_threshold > 0 && rebuild_gradient_after_simplification
        # adjust scalars in place
        _ = simplify_scalar_field_by_persistence!(mesh, local_scalars, vertex_order, persistence_threshold)
        # recompute order with same SoS tie-breaker (if provided)
        vertex_order = sort_vertices_order(local_scalars, sos_offsets)
    end

    # Initialize gradient field - matches C++ initMemory
    gradient = GradientField(
        size(mesh.vertices, 2),
        size(mesh.edge_vertices, 2),
        size(mesh.triangles, 2))
    # Build discrete gradient field using ProcessLowerStars with (possibly) updated order
    build_gradient_field!(mesh, vertex_order, gradient)

    # Optional consistency validation (useful with threading)
    if validate_gradient
        assert_gradient_consistency(mesh, gradient)
    end

    # Find critical points - matches C++ getCriticalPoints
    crit_verts, crit_edges, crit_tris = find_critical_points(mesh, gradient)

    # Classify critical points
    minima, saddles, maxima = classify_critical_points(crit_verts, crit_edges, crit_tris, mesh)

    # Optional post-filtering of critical lists (does not change gradient).
    # If we rebuilt the gradient after simplification, lists should already be cleaner.
    if persistence_threshold > 0 && !rebuild_gradient_after_simplification
        filter_critical_by_persistence!(minima, saddles, maxima, mesh, local_scalars, vertex_order, persistence_threshold)
    end

    return minima, saddles, maxima, gradient
end


# ==============================================================================
# ORDER DISAMBIGUATION (SoS) — port of TTK core/base/common/OrderDisambiguation.h
# ==============================================================================

"""
Sort vertices according to scalars disambiguated by SoS offsets.

This mirrors ttk::sortVertices in core/base/common/OrderDisambiguation.h:
- primary key: scalar value (ascending)
- tie-breaker: sos_offsets[a] < sos_offsets[b] when provided, otherwise index a < b

Returns an `order` array where order[v] is the rank of vertex v in ascending
order of (scalar, tie-breaker), 1-based to match Julia indexing.
"""
function sort_vertices_order(scalars::AbstractVector{T},
    sos_offsets::Union{AbstractVector{Int},Nothing}) where {T<:Real}
    nVerts = length(scalars)
    sortedVertices = collect(1:nVerts)
    if sos_offsets !== nothing
        sort!(sortedVertices, by=i -> (scalars[i], sos_offsets[i]))
    else
        # Match C++ fallback: tie-break by vertex id
        sort!(sortedVertices, by=i -> (scalars[i], i))
    end
    order = Vector{Int}(undef, nVerts)
    @inbounds for (rank, v) in enumerate(sortedVertices)
        order[v] = rank
    end
    return order
end

"""
Precondition an order array to be consumed by the base layer API.

Equivalent of ttk::preconditionOrderArray: compute total order by scalar with
index as tie-breaker (no external SoS).
"""
function precondition_order_array(scalars::AbstractVector{T}) where {T<:Real}
    return sort_vertices_order(scalars, nothing)
end

"""
Lookup edge ID between two vertices using precomputed edge lookup dictionary.
Returns -1 if edge does not exist.

This provides O(1) edge lookup performance compared to O(degree) linear search.
"""
function lookup_edge_id(mesh::TriangleMesh, u::Int, v::Int)
    edge_key = (min(u, v), max(u, v))
    return get(mesh.edge_lookup, edge_key, -1)
end


"""
Flood-fill vertices in the connected component of `start` using a threshold predicate.

Parameters:
- `mode`: :below for sublevel-set flood (scalar < threshold)
          :above for superlevel-set flood (scalar > threshold)

Returns the list of visited vertex indices.
"""
function flood_component_vertices(mesh::TriangleMesh,
    scalars::AbstractVector{<:Real},
    start::Int,
    threshold::Float64,
    mode::Symbol,
    neigh::Union{Nothing,Vector{Vector{Int}}}=nothing
)
    neigh === nothing && (neigh = mesh.vertex_neighbors)
    n = length(scalars)
    visited = falses(n)
    q = Int[]

    # Early exit if start does not satisfy predicate
    if mode === :below
        if !(scalars[start] < threshold)
            return Int[]
        end
    elseif mode === :above
        if !(scalars[start] > threshold)
            return Int[]
        end
    else
        error("Unsupported flood mode: $mode")
    end

    push!(q, start)
    visited[start] = true
    comp = Int[]

    while !isempty(q)
        v = pop!(q)
        push!(comp, v)
        for u in neigh[v]
            if !visited[u]
                if mode === :below
                    if scalars[u] < threshold
                        visited[u] = true
                        push!(q, u)
                    end
                else # :above
                    if scalars[u] > threshold
                        visited[u] = true
                        push!(q, u)
                    end
                end
            end
        end
    end

    return comp
end

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

"""
Get the vertex coordinates of a critical point for visualization.
"""
function get_critical_point_location(cell::Cell, mesh::TriangleMesh)
    if cell.dim == 0  # Vertex
        return mesh.vertices[:, cell.id]
    elseif cell.dim == 1  # Edge midpoint
        v1, v2 = mesh.edge_vertices[:, cell.id]
        return (mesh.vertices[:, v1] + mesh.vertices[:, v2]) / 2.0
    elseif cell.dim == 2  # Triangle centroid
        v1, v2, v3 = mesh.triangles[:, cell.id]
        return (mesh.vertices[:, v1] + mesh.vertices[:, v2] + mesh.vertices[:, v3]) / 3.0
    end
    return zeros(3)
end

"""
Build a dictionary containing critical point summary statistics.
Returns a Dict with organized summary data.
"""
function build_critical_point_summary(minima::Vector{Int}, saddles::Vector{Int},
    maxima::Vector{Int}, mesh::TriangleMesh)
    n_verts = size(mesh.vertices, 2)
    n_edges = size(mesh.edge_vertices, 2)
    n_tris = size(mesh.triangles, 2)

    # Euler characteristic verification
    euler_char = n_verts - n_edges + n_tris
    morse_euler = length(minima) - length(saddles) + length(maxima)

    return Dict(
        :mesh_info => Dict(
            :vertices => n_verts,
            :edges => n_edges,
            :triangles => n_tris
        ),
        :critical_points => Dict(
            :minima => length(minima),
            :saddles => length(saddles),
            :maxima => length(maxima)
        ),
        :euler_characteristic => Dict(
            :topological => euler_char,
            :morse => morse_euler,
            :match => euler_char == morse_euler
        )
    )
end

"""
Pretty print a critical point summary dictionary.
"""
function print_critical_point_summary(summary::Dict)
    println("=== Discrete Morse Theory Critical Point Analysis ===")

    mesh_info = summary[:mesh_info]
    println("Mesh: $(mesh_info[:vertices]) vertices, $(mesh_info[:edges]) edges, $(mesh_info[:triangles]) triangles")

    crit_points = summary[:critical_points]
    println("Critical Points:")
    println("  Minima (0-cells):     $(crit_points[:minima])")
    println("  Saddles (1-cells):    $(crit_points[:saddles])")
    println("  Maxima (2-cells):     $(crit_points[:maxima])")

    euler_char = summary[:euler_characteristic]
    println("\nEuler Characteristic:")
    println("  Topological:    $(euler_char[:topological])")
    println("  Morse:          $(euler_char[:morse])")
    match_symbol = euler_char[:match] ? "✓" : "✗"
    println("  Match: $match_symbol")
end

"""
Print summary statistics of critical points.
This function now builds a summary dictionary and then prints it.
"""
function print_critical_point_summary(minima::Vector{Int}, saddles::Vector{Int},
    maxima::Vector{Int}, mesh::TriangleMesh)
    summary = build_critical_point_summary(minima, saddles, maxima, mesh)
    print_critical_point_summary(summary)
end

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

"""
Validate discrete gradient array consistency.

Checks reciprocal pointers and admissible pairings:
- If vertex_to_edge[v] = e, then edge_to_vertex[e] = v, and v is incident to e.
- An edge cannot be paired with both a vertex and a triangle simultaneously.
- If edge_to_triangle[e] = t, then triangle_to_edge[t] = e, and e is a face of t.
- If triangle_to_edge[t] = e, then edge_to_triangle[e] = t.

Returns (ok::Bool, errors::Vector{String}).
"""
function validate_gradient_consistency(mesh::TriangleMesh, gradient::GradientField)
    nV = size(mesh.vertices, 2)
    nE = size(mesh.edge_vertices, 2)
    nT = size(mesh.triangles, 2)

    # Thread-local error buffers for safe parallel accumulation
    # Use explicit initialization to avoid any undefined slots in rare contexts
    errs_tls = Vector{Vector{String}}(undef, nthreads())
    @inbounds for i in 1:length(errs_tls)
        errs_tls[i] = String[]
    end

    # Helper to safely access thread-local buffer even if thread id exceeds length
    get_tls() = errs_tls[min(threadid(), length(errs_tls))]

    # Vertex -> Edge reciprocal and incidence
    @threads for v in 1:nV
        @inbounds begin
            e = gradient.vertex_to_edge[v]
            if e != -1
                local_errs = get_tls()
                if !(1 <= e <= nE)
                    push!(local_errs, "vertex_to_edge[$v]=$e out of bounds 1..$nE")
                else
                    if gradient.edge_to_vertex[e] != v
                        push!(local_errs, "edge_to_vertex[$e]!=$v (got $(gradient.edge_to_vertex[e]))")
                    end
                    ev1, ev2 = mesh.edge_vertices[:, e]
                    if v != ev1 && v != ev2
                        push!(local_errs, "vertex_to_edge[$v]=$e but v is not incident to edge")
                    end
                end
            end
        end
    end

    # Edge constraints and reciprocity
    @threads for e in 1:nE
        @inbounds begin
            v = gradient.edge_to_vertex[e]
            t = gradient.edge_to_triangle[e]
            if v != -1 && t != -1
                local_errs = get_tls()
                push!(local_errs, "edge $e paired to both vertex $v and triangle $t")
            end
            if v != -1
                local_errs = get_tls()
                if !(1 <= v <= nV)
                    push!(local_errs, "edge_to_vertex[$e]=$v out of bounds 1..$nV")
                else
                    if gradient.vertex_to_edge[v] != e
                        push!(local_errs, "vertex_to_edge[$v]!=$e (got $(gradient.vertex_to_edge[v]))")
                    end
                end
            end
            if t != -1
                local_errs = get_tls()
                if !(1 <= t <= nT)
                    push!(local_errs, "edge_to_triangle[$e]=$t out of bounds 1..$nT")
                else
                    if gradient.triangle_to_edge[t] != e
                        push!(local_errs, "triangle_to_edge[$t]!=$e (got $(gradient.triangle_to_edge[t]))")
                    end
                    ev1, ev2 = mesh.edge_vertices[:, e]
                    tv1, tv2, tv3 = mesh.triangles[:, t]
                    if !((ev1==tv1 || ev1==tv2 || ev1==tv3) && (ev2==tv1 || ev2==tv2 || ev2==tv3))
                        push!(local_errs, "edge $e not a face of triangle $t but paired")
                    end
                end
            end
        end
    end

    # Triangle -> Edge reciprocity
    @threads for t in 1:nT
        @inbounds begin
            e = gradient.triangle_to_edge[t]
            if e != -1
                local_errs = get_tls()
                if !(1 <= e <= nE)
                    push!(local_errs, "triangle_to_edge[$t]=$e out of bounds 1..$nE")
                else
                    if gradient.edge_to_triangle[e] != t
                        push!(local_errs, "edge_to_triangle[$e]!=$t (got $(gradient.edge_to_triangle[e]))")
                    end
                end
            end
        end
    end

    errors = isempty(errs_tls) ? String[] : reduce(vcat, errs_tls)
    sort!(errors)  # provide deterministic ordering independent of thread scheduling
    return isempty(errors), errors
end

"""
Assert discrete gradient consistency. Throws an error with details if inconsistencies are found.
Returns true when the gradient is consistent.
"""
function assert_gradient_consistency(mesh::TriangleMesh, gradient::GradientField)
    ok, errs = validate_gradient_consistency(mesh, gradient)
    if !ok
        error("Gradient consistency check failed:\n" * join(errs, "\n"))
    end
    return true
end
