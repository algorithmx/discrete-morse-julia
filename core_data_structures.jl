"""
Discrete Morse Theory Critical Point Detection -- Core Data Structures

Implementation of ttkDiscreteGradient critical point search algorithm in Julia.
Based on the ProcessLowerStars algorithm from Robins, Wood, and Sheppard (2000).

Author: Generated from TTK architecture analysis
Date: October 2025
"""

using Printf
using DataStructures
using LinearAlgebra


# ==============================================================================
# CORE DATA STRUCTURES
# ==============================================================================

"""
Basic cell representation in the discretization.
A cell is defined by its topological dimension and its identifier.
"""
struct Cell
    dim::Int          # Topological dimension (0=vertex, 1=edge, 2=triangle)
    id::Int           # Cell identifier in the mesh

    Cell() = new(-1, -1)
    Cell(dim::Int, id::Int) = new(dim, id)
end

# Comparison operators for cells
Base.isless(c1::Cell, c2::Cell) = (c1.dim < c2.dim) || (c1.dim == c2.dim && c1.id < c2.id)
Base.:(==)(c1::Cell, c2::Cell) = (c1.dim == c2.dim && c1.id == c2.id)
Base.hash(c::Cell) = hash(c.dim) ⊻ hash(c.id)

"""
Extended cell structure for ProcessLowerStars algorithm.
Contains additional information for gradient pairing.
"""
mutable struct CellExt
    dim::Int                     # Topological dimension
    id::Int                      # Cell identifier
    lower_verts::NTuple{3,Int}  # Lower vertex scalar values (sorted)
    faces::NTuple{3,Int}        # Face indices in lower star (1-based indices into ls.edges, -1 when unused)
    paired::Bool                 # Whether cell has been paired

    function CellExt(dim::Int, id::Int)
    new(dim, id, (-1, -1, -1), (-1, -1, -1), false)
    end

    function CellExt(dim::Int, id::Int, lower_verts::NTuple{3,Int}, faces::NTuple{3,Int})
        new(dim, id, lower_verts, faces, false)
    end
end

# Min-heap ordering for CellExt based on lexicographic order of lower_verts
struct LowerVertsOrdering <: Base.Order.Ordering end
Base.Order.lt(::LowerVertsOrdering, a::CellExt, b::CellExt) = a.lower_verts < b.lower_verts
const LV_ORD = LowerVertsOrdering()

# Comparison for priority queues (descending order for min extraction)
Base.isless(c1::CellExt, c2::CellExt) = c1.lower_verts > c2.lower_verts

"""
Lower star representation for a vertex.
Stores all cells in the lower star organized by dimension.
"""
struct LowerStar
    vertices::Vector{CellExt}    # 0-dimensional cells (just the vertex itself)
    edges::Vector{CellExt}       # 1-dimensional cells
    triangles::Vector{CellExt}   # 2-dimensional cells
    tetrahedra::Vector{CellExt}  # 3-dimensional cells (empty for surface meshes)

    LowerStar() = new(CellExt[], CellExt[], CellExt[], CellExt[])
end

"""
Gradient field storage.
Follows C++ TTK structure: gradient_[2*dim][cell_id] stores paired cell information.
"""
struct GradientField
    vertex_to_edge::Vector{Int}     # gradient_[0] - vertex -> edge pairs
    edge_to_vertex::Vector{Int}     # gradient_[1] - edge -> vertex pairs
    edge_to_triangle::Vector{Int}   # gradient_[2] - edge -> triangle pairs
    triangle_to_edge::Vector{Int}   # gradient_[3] - triangle -> edge pairs

    function GradientField(n_verts::Int, n_edges::Int, n_triangles::Int)
        new(
            fill(-1, n_verts),        # vertex_to_edge
            fill(-1, n_edges),        # edge_to_vertex
            fill(-1, n_edges),        # edge_to_triangle
            fill(-1, n_triangles)     # triangle_to_edge
        )
    end
end

"""
Triangle surface mesh representation.
Minimal interface for ProcessLowerStars algorithm.
"""
struct TriangleMesh
    vertices::Matrix{Float64}        # 3 x n_vertices matrix of vertex positions
    triangles::Matrix{Int}           # 3 x n_triangles matrix of triangle vertex indices
    vertex_to_edges::Vector{Vector{Int}}   # Adjacency: vertex -> incident edges
    edge_to_triangles::Vector{Vector{Int}} # Adjacency: edge -> incident triangles
    edge_vertices::Matrix{Int}       # 2 x n_edges matrix of edge vertex indices
    vertex_neighbors::Vector{Vector{Int}} # Adjacency: vertex -> neighboring vertices (1-ring)
    edge_lookup::Dict{Tuple{Int,Int},Int} # Edge lookup: (min_vertex, max_vertex) -> edge_id
    vertex_to_triangles::Vector{Vector{Int}} # Adjacency: vertex -> incident triangles

    function TriangleMesh(verts::Matrix{Float64}, tris::Matrix{Int})
        # Build edge connectivity
        n_verts = size(verts, 2)
        n_tris = size(tris, 2)

        # Extract unique edges from triangles
        edge_map = Dict{Tuple{Int,Int}, Int}()
        edge_list = Tuple{Int,Int}[]

        for i in 1:n_tris
            v0, v1, v2 = tris[:, i]

            # Create directed edges with consistent ordering
            edges = [(min(v0, v1), max(v0, v1)),
                     (min(v1, v2), max(v1, v2)),
                     (min(v2, v0), max(v2, v0))]

            for edge in edges
                if !haskey(edge_map, edge)
                    push!(edge_list, edge)
                    edge_map[edge] = length(edge_list)
                end
            end
        end

        n_edges = length(edge_list)
        edge_vertices = hcat([collect(edge) for edge in edge_list]...)

        # Build vertex to edge adjacency
        vertex_to_edges = [Int[] for _ in 1:n_verts]
        for (edge_idx, (v1, v2)) in enumerate(edge_list)
            push!(vertex_to_edges[v1], edge_idx)
            push!(vertex_to_edges[v2], edge_idx)
        end

        # Build edge to triangle adjacency
        edge_to_triangles = [Int[] for _ in 1:n_edges]
        for tri_idx in 1:n_tris
            v0, v1, v2 = tris[:, tri_idx]

            tri_edges = [(min(v0, v1), max(v0, v1)),
                        (min(v1, v2), max(v1, v2)),
                        (min(v2, v0), max(v2, v0))]

            for edge in tri_edges
                edge_idx = edge_map[edge]
                push!(edge_to_triangles[edge_idx], tri_idx)
            end
        end

        # Build vertex neighbors (1-ring) from edge connectivity
        vertex_neighbors = [Int[] for _ in 1:n_verts]
        for e in 1:n_edges
            u, v = edge_vertices[:, e]
            push!(vertex_neighbors[u], v)
            push!(vertex_neighbors[v], u)
        end

        # Build edge lookup dictionary for O(1) edge_id lookup
        edge_lookup = Dict{Tuple{Int,Int},Int}()
        for e in 1:n_edges
            u, v = edge_vertices[:, e]
            edge_key = (min(u, v), max(u, v))
            edge_lookup[edge_key] = e
        end

        # Build vertex to triangle adjacency for O(degree(v)) triangle lookup
        vertex_to_triangles = [Int[] for _ in 1:n_verts]
        for tri_idx in 1:n_tris
            v0, v1, v2 = tris[:, tri_idx]
            push!(vertex_to_triangles[v0], tri_idx)
            push!(vertex_to_triangles[v1], tri_idx)
            push!(vertex_to_triangles[v2], tri_idx)
        end

        new(verts, tris, vertex_to_edges, edge_to_triangles, edge_vertices, vertex_neighbors, edge_lookup, vertex_to_triangles)
    end
end


# ==============================================================================
# PERSISTENCE PAIRS (0–1 and 1–2) — union-find like TTK PersistenceDiagram
# ==============================================================================

# Pair types, mirroring ttk::lts::LocalizedTopologicalSimplification::PAIR_TYPE
const PAIR_MINIMUM_SADDLE = 1  # MINIMUM_SADDLE
const PAIR_MAXIMUM_SADDLE = 2  # MAXIMUM_SADDLE

struct PersistencePair
    pair_type::Int      # PAIR_MINIMUM_SADDLE or PAIR_MAXIMUM_SADDLE
    extremum::Int       # vertex id of minimum (for MINIMUM_SADDLE) or maximum (for MAXIMUM_SADDLE)
    saddle_edge::Int    # edge id (1-saddle)
    birth_value::Float64
    death_value::Float64
    persistence::Float64
end


# ==============================================================================
# Simple union-find (disjoint set) with component birth tracking (elder rule).
# ==============================================================================

mutable struct UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
    comp_birth_vertex::Vector{Int}  # representative vertex at birth (minimum or maximum)
end

function UnionFind(n::Int)
    parent = collect(1:n)
    rank = fill(0, n)
    comp_birth_vertex = collect(1:n)
    return UnionFind(parent, rank, comp_birth_vertex)
end

function uf_find!(uf::UnionFind, x::Int)
    p = uf.parent[x]
    if p != x
        uf.parent[x] = uf_find!(uf, p)
    end
    return uf.parent[x]
end

function uf_union_elder!(uf::UnionFind, a::Int, b::Int, birth_order::AbstractVector{Int})
    ra = uf_find!(uf, a)
    rb = uf_find!(uf, b)
    if ra == rb
        return ra
    end
    # elder = component with older birth (smaller order)
    elder = (birth_order[uf.comp_birth_vertex[ra]] <= birth_order[uf.comp_birth_vertex[rb]]) ? ra : rb
    other = (elder == ra) ? rb : ra
    # union by rank under elder
    if uf.rank[elder] < uf.rank[other]
        elder, other = other, elder
    end
    uf.parent[other] = elder
    if uf.rank[elder] == uf.rank[other]
        uf.rank[elder] += 1
    end
    # the elder's birth vertex remains
    return elder
end
