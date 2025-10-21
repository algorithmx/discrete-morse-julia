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

        new(verts, tris, vertex_to_edges, edge_to_triangles, edge_vertices)
    end
end

# ==============================================================================
# ORDER DISAMBIGUATION (SoS) — port of TTK core/base/common/OrderDisambiguation.h
# ==============================================================================

"""
Sort vertices according to scalars disambiguated by offsets (SoS).

This mirrors ttk::sortVertices in core/base/common/OrderDisambiguation.h:
- primary key: scalar value (ascending)
- tie-breaker: offsets[a] < offsets[b] when provided, otherwise index a < b

Returns an `order` array where order[v] is the rank of vertex v in ascending
order of (scalar, tie-breaker), 1-based to match Julia indexing.
"""
function sort_vertices_order(scalars::AbstractVector{T},
                             offsets::Union{AbstractVector{Int},Nothing}) where {T<:Real}
    nVerts = length(scalars)
    sortedVertices = collect(1:nVerts)
    if offsets !== nothing
        sort!(sortedVertices, by = i -> (scalars[i], offsets[i]))
    else
        # Match C++ fallback: tie-break by vertex id
        sort!(sortedVertices, by = i -> (scalars[i], i))
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

"""
Simple union-find (disjoint set) with component birth tracking (elder rule).
"""
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

"""
Build vertex neighbors (1-ring) from mesh connectivity.
"""
function build_vertex_neighbors(mesh::TriangleMesh)
    n = size(mesh.vertices, 2)
    neigh = [Int[] for _ in 1:n]
    for e in 1:size(mesh.edge_vertices, 2)
        u, v = mesh.edge_vertices[:, e]
        push!(neigh[u], v)
        push!(neigh[v], u)
    end
    return neigh
end

"""
Compute 0D persistence pairs between minima and 1-saddles (edges) via elder rule.
Pairing semantics:
- iterate vertices by ascending (scalar, SoS)
- at each vertex v, consider lower neighbors (with lower order)
- among distinct neighbor components, keep the elder component; each other component C forms a pair
  (min of C, saddle edge (v, any u in C)), with death at f(v).

Returns a vector of PersistencePair of type PAIR_MINIMUM_SADDLE.
"""
function compute_persistence_pairs_min_sad(mesh::TriangleMesh,
                                           scalars::AbstractVector{<:Real},
                                           order::AbstractVector{Int})
    n = length(scalars)
    neigh = build_vertex_neighbors(mesh)
    # map vertex id to its position in the sorted order (1..n)
    invOrder = order
    # vertices sorted ascending by order
    verts_sorted = sort(collect(1:n), by = i -> invOrder[i])
    uf = UnionFind(n)
    active = falses(n)
    # track which component each active vertex currently belongs to
    pairs = PersistencePair[]

    for v in verts_sorted
        active[v] = true
        uf.comp_birth_vertex[v] = v
        # gather neighbors that are already active and have lower order
        comps = Dict{Int,Int}()  # root -> sample neighbor vertex in that comp
        for u in neigh[v]
            if active[u] && invOrder[u] < invOrder[v]
                r = uf_find!(uf, u)
                comps[r] = u
            end
        end
        if isempty(comps)
            # new component born at v (minimum)
            continue
        end
        # choose elder component to survive
        roots = collect(keys(comps))
        elder = roots[1]
        for r in roots[2:end]
            if invOrder[uf.comp_birth_vertex[r]] < invOrder[uf.comp_birth_vertex[elder]]
                elder = r
            end
        end
        # pair all other components with current vertex v (death)
        for r in roots
            if r == elder; continue; end
            minVertex = uf.comp_birth_vertex[r]
            # find an edge (v, u) that connects to this component
            u = comps[r]
            # retrieve edge id for (v,u)
            edge_id = -1
            # lookup via mesh.vertex_to_edges
            for e in mesh.vertex_to_edges[v]
                a, b = mesh.edge_vertices[:, e]
                if (a == v && b == u) || (a == u && b == v)
                    edge_id = e
                    break
                end
            end
            if edge_id == -1
                continue  # should not happen
            end
            birth = float(scalars[minVertex])
            death = float(scalars[v])
            push!(pairs, PersistencePair(PAIR_MINIMUM_SADDLE, minVertex, edge_id, birth, death, death - birth))
        end
        # merge all components with elder
        for r in roots
            uf_union_elder!(uf, elder, r, invOrder)
        end
        # finally union v into elder as well
        uf_union_elder!(uf, elder, v, invOrder)
    end

    return pairs
end

"""
Compute 2D dual pairs (maxima with 1-saddles) by applying the same procedure
to the superlevel sets (i.e., run on -scalars). Returns PAIR_MAXIMUM_SADDLE pairs.
"""
function compute_persistence_pairs_max_sad(mesh::TriangleMesh,
                                           scalars::AbstractVector{<:Real},
                                           order::AbstractVector{Int})
    neg = [-float(s) for s in scalars]
    # reuse the same order on vertices to disambiguate ties consistently with TTK
    pairs0 = compute_persistence_pairs_min_sad(mesh, neg, order)
    # convert type to MAXIMUM_SADDLE and flip birth/death signs back
    pairs = PersistencePair[]
    for p in pairs0
        birth = -p.birth_value
        death = -p.death_value
        push!(pairs, PersistencePair(PAIR_MAXIMUM_SADDLE, p.extremum, p.saddle_edge, birth, death, birth - death))
    end
    return pairs
end

"""
Filter critical points using a persistence threshold:
- remove minima that appear in MINIMUM_SADDLE pairs with persistence < tau
- remove saddles that appear in any pair (min or max) with persistence < tau
Maxima (triangles) are left untouched to avoid breaking Morse/Euler counts
without recomputing the gradient.
"""
function filter_critical_by_persistence!(minima::Vector{Int}, saddles::Vector{Int}, maxima::Vector{Int},
                                         mesh::TriangleMesh,
                                         scalars::AbstractVector{<:Real},
                                         order::AbstractVector{Int},
                                         tau::Real # persistence_threshold
)
    if tau <= 0
        return
    end
    pairs_min = compute_persistence_pairs_min_sad(mesh, scalars, order)

    # collect to-remove sets
    rm_min = Set{Int}()
    rm_sad = Set{Int}()
    for p in pairs_min
        if p.persistence < tau
            push!(rm_min, p.extremum)
            push!(rm_sad, p.saddle_edge)
        end
    end
    # Do NOT remove saddles for max-saddle pairs without rebuilding the gradient,
    # since maxima in discrete Morse are 2-cells (triangles) and a direct mapping
    # from vertex maxima to triangle maxima is non-trivial. Rebuilding handles this.

    # filter vectors in-place
    filter!((v)->!(v in rm_min), minima)
    filter!((e)->!(e in rm_sad), saddles)
    # leave maxima as-is for consistency
    return
end

# ==============================================================================
# TOPOLOGICAL SIMPLIFICATION VIA SCALAR ADJUSTMENT AND REBUILD
# ==============================================================================

"""
Simplify the scalar field in-place by canceling pairs below a threshold:
- For each MINIMUM_SADDLE pair with persistence < tau, raise the minimum's
  scalar to the pair's death value.
- For each MAXIMUM_SADDLE pair with persistence < tau, lower the maximum's
  scalar to the pair's death value.

Returns the list of modified vertex indices.
This mirrors TTK's LTS intent (local value edits leading to cancellations)
without the full propagation machinery.
"""
function simplify_scalar_field_by_persistence!(mesh::TriangleMesh,
                                               scalars::Vector{Float64},
                                               order::AbstractVector{Int},
                                               tau::Real)
    if tau <= 0
        return Int[]
    end
    pairs_min = compute_persistence_pairs_min_sad(mesh, scalars, order)
    pairs_max = compute_persistence_pairs_max_sad(mesh, scalars, order)
    modified = Set{Int}()
    for p in pairs_min
        if p.persistence < tau
            v = p.extremum
            # lift minimum to the merge level (death)
            newv = max(scalars[v], p.death_value)
            if newv != scalars[v]
                scalars[v] = newv
                push!(modified, v)
            end
        end
    end
    for p in pairs_max
        if p.persistence < tau
            v = p.extremum
            # lower maximum to the merge level (death)
            newv = min(scalars[v], p.death_value)
            if newv != scalars[v]
                scalars[v] = newv
                push!(modified, v)
            end
        end
    end
    return collect(modified)
end

# ==============================================================================
# CORE ALGORITHM FUNCTIONS
# ==============================================================================

"""
Compute the lower star of vertex `a` with respect to scalar field `offsets`.
The lower star contains all cells where vertex `a` has the maximum scalar value.
Follows C++ TTK implementation exactly.
"""
function compute_lower_star!(ls::LowerStar, a::Int, offsets::Vector{Int}, mesh::TriangleMesh)
    # Clear previous lower star
    empty!(ls.vertices)
    empty!(ls.edges)
    empty!(ls.triangles)

    # a belongs to its lower star
    push!(ls.vertices, CellExt(0, a))

    # Store lower edges - pre-allocate for performance
    nedges = length(mesh.vertex_to_edges[a])
    sizehint!(ls.edges, nedges)

    for edge_idx in mesh.vertex_to_edges[a]
        v1, v2 = mesh.edge_vertices[:, edge_idx]
        other_vertex = (v1 == a) ? v2 : v1

        if offsets[other_vertex] < offsets[a]
            lower_verts = (offsets[other_vertex], -1, -1)
            push!(ls.edges, CellExt(1, edge_idx, lower_verts, (-1, -1, -1)))
        end
    end

    # At least two edges in lower star for one triangle
    if length(ls.edges) < 2
        return
    end

    # Process triangles - use optimized approach like C++ version
    function process_triangle(triangle_id::Int, v0::Int, v1::Int, v2::Int)
        low_verts = (-1, -1, -1)
        if v0 == a
            low_verts = (offsets[v1], offsets[v2], -1)
        elseif v1 == a
            low_verts = (offsets[v0], offsets[v2], -1)
        elseif v2 == a
            low_verts = (offsets[v0], offsets[v1], -1)
        end

        # Higher order vertex first
        if low_verts[1] < low_verts[2]
            low_verts = (low_verts[2], low_verts[1], -1)
        end

        if offsets[a] > low_verts[1]  # Triangle in lower star
            # Find the two edge indices in lower star corresponding to the two lower vertices
            face1 = -1
            face2 = -1
            for (edge_idx, edge) in enumerate(ls.edges)
                e_v1, e_v2 = mesh.edge_vertices[:, edge.id]
                e_other = (e_v1 == a) ? e_v2 : e_v1
                e_offset = offsets[e_other]

                if e_offset == low_verts[1] && face1 == -1
                    face1 = edge_idx
                elseif e_offset == low_verts[2] && face2 == -1
                    face2 = edge_idx
                end

                if face1 != -1 && face2 != -1
                    break
                end
            end

            if face1 != -1 && face2 != -1
                push!(ls.triangles, CellExt(2, triangle_id, low_verts, (face1, face2, -1)))
            end
        end
    end

    # Find triangles containing vertex a
    for tri_idx in 1:size(mesh.triangles, 2)
        v0, v1, v2 = mesh.triangles[:, tri_idx]
        if v0 == a || v1 == a || v2 == a
            process_triangle(tri_idx, v0, v1, v2)
        end
    end
end

"""
Pair two cells in the discrete gradient field.
Establishes gradient relationship from alpha (lower dimension) to beta (higher dimension).
Follows C++ TTK pairing logic with consistent 0-based indexing.
"""
function pair_cells!(alpha::CellExt, beta::CellExt, gradient::GradientField)
    alpha.paired = true
    beta.paired = true

    if alpha.dim == 0 && beta.dim == 1
        # gradient_[2*alpha.dim_][alpha.id_] = beta.id_
            gradient.vertex_to_edge[alpha.id] = beta.id
            # gradient_[2*alpha.dim_+1][beta.id_] = alpha.id_
            gradient.edge_to_vertex[beta.id] = alpha.id
    elseif alpha.dim == 1 && beta.dim == 2
        # gradient_[2*alpha.dim_][alpha.id_] = beta.id_
            gradient.edge_to_triangle[alpha.id] = beta.id
            # gradient_[2*alpha.dim_+1][beta.id_] = alpha.id_
            gradient.triangle_to_edge[beta.id] = alpha.id
    end
end

"""
Count unpaired faces of a triangle in the lower star.
Returns (count, face_id) where face_id is an unpaired face if count > 0.
Matches C++ numUnpairedFacesTriangle implementation.
"""
function count_unpaired_faces_triangle(triangle::CellExt, ls::LowerStar)
    count = 0
    face_id = -1

    # Check the two edge faces of the triangle (faces stored 1-based into ls.edges)
    for i in 1:2
        edge_idx_in_ls = triangle.faces[i]
        if edge_idx_in_ls > 0 && edge_idx_in_ls <= length(ls.edges) && !ls.edges[edge_idx_in_ls].paired
            count += 1
            face_id = edge_idx_in_ls
        end
    end

    return count, face_id
end

"""
Process the lower star of a single vertex to build gradient pairs.
Implements the core ProcessLowerStars algorithm.
"""
function process_lower_star_vertex!(ls::LowerStar, a::Int, offsets::Vector{Int},
                                   mesh::TriangleMesh, gradient::GradientField)
    # Compute lower star for vertex a
    compute_lower_star!(ls, a, offsets, mesh)

    # If no edges in lower star, vertex is a local minimum
    if isempty(ls.edges)
        return  # Vertex remains critical (no outgoing gradient)
    end

    # Find edge with minimal scalar value (steepest gradient)
    min_edge_idx = 1
    min_scalar = ls.edges[1].lower_verts[1]

    for i in 2:length(ls.edges)
        if ls.edges[i].lower_verts[1] < min_scalar
            min_edge_idx = i
            min_scalar = ls.edges[i].lower_verts[1]
        end
    end

    # Pair vertex with steepest edge
    delta_edge = ls.edges[min_edge_idx]
    pair_cells!(ls.vertices[1], delta_edge, gradient)

    # Initialize priority queues using DataStructures.BinaryHeap with custom ordering
    pq_zero = BinaryHeap{CellExt}(LV_ORD)  # For critical cells
    pq_one = BinaryHeap{CellExt}(LV_ORD)   # For cells with one unpaired face

    # Add all other edges to pq_zero
    for (i, edge) in enumerate(ls.edges)
        if i != min_edge_idx
            push!(pq_zero, edge)
        end
    end

    # Insert cofaces function matching C++ insertCofacets
    function insert_cofacets(ca::CellExt, ls_local::LowerStar)
        if ca.dim == 1  # Edge -> add triangles
            for beta in ls_local.triangles
                # Check if edge ca belongs to triangle beta
                edge_in_triangle = false
                for face_idx in 1:2
                    edge_id_in_ls = beta.faces[face_idx]
                    if edge_id_in_ls > 0 && edge_id_in_ls <= length(ls_local.edges) &&
                       ls_local.edges[edge_id_in_ls].id == ca.id
                        edge_in_triangle = true
                        break
                    end
                end

                if edge_in_triangle
                    count, _ = count_unpaired_faces_triangle(beta, ls_local)
                    if count == 1
                        push!(pq_one, beta)
                    end
                end
            end
        end
    end

    # Push into pq_one every coface of delta in Lx such that numUnpairedFaces == 1
    insert_cofacets(delta_edge, ls)

    # Main processing loop following C++ algorithm exactly
    while !isempty(pq_one) || !isempty(pq_zero)
        # Process pq_one first (cells with one unpaired face)
        while !isempty(pq_one)
            alpha = pop!(pq_one)
            unpaired_faces = count_unpaired_faces_triangle(alpha, ls)

            if unpaired_faces[1] == 0
                push!(pq_zero, alpha)
            else
                # Get pair_alpha from unpaired face (face index is 1-based into ls.edges)
                pair_alpha = ls.edges[unpaired_faces[2]]

                # Store (pair_alpha) -> (alpha) V-path
                pair_cells!(pair_alpha, alpha, gradient)

                # Add cofaces of both cells to pq_one
                insert_cofacets(alpha, ls)
                insert_cofacets(pair_alpha, ls)
            end
        end

        # Skip pair_alpha from pq_zero: cells in pq_zero are not critical if already paired
        while !isempty(pq_zero) && first(pq_zero).paired
            pop!(pq_zero)
        end

        if !isempty(pq_zero)
            gamma = pop!(pq_zero)
            # Gamma is a critical cell - mark gamma as paired
            gamma.paired = true

            # Add cofacets to pq_one
            insert_cofacets(gamma, ls)
        end
    end
end

"""
Build the complete discrete gradient field using ProcessLowerStars algorithm.
Matches C++ processLowerStars implementation.
"""
function build_gradient_field!(mesh::TriangleMesh, offsets::Vector{Int}, gradient::GradientField)
    n_vertices = size(mesh.vertices, 2)
    ls = LowerStar()

    # Process each vertex - matches C++ for loop structure
    for vertex_id in 1:n_vertices
        process_lower_star_vertex!(ls, vertex_id, offsets, mesh, gradient)
    end
end

# ==============================================================================
# CRITICAL POINT DETECTION
# ==============================================================================

"""
Check if a cell is critical (has no gradient connections).
Matches C++ isCellCritical implementation.
"""
function is_critical_cell(cell::Cell, gradient::GradientField, mesh::TriangleMesh)
    if cell.dim == 0  # Vertex
        return gradient.vertex_to_edge[cell.id] == -1
    elseif cell.dim == 1  # Edge
        return gradient.edge_to_vertex[cell.id] == -1 &&
               gradient.edge_to_triangle[cell.id] == -1
    elseif cell.dim == 2  # Triangle
        return gradient.triangle_to_edge[cell.id] == -1
    end
    return false
end

"""
Find all critical points in the gradient field.
Returns arrays of critical cells organized by dimension.
Matches C++ getCriticalPoints implementation.
"""
function find_critical_points(mesh::TriangleMesh, gradient::GradientField)
    n_vertices = size(mesh.vertices, 2)
    n_edges = size(mesh.edge_vertices, 2)
    n_triangles = size(mesh.triangles, 2)

    critical_vertices = Int[]
    critical_edges = Int[]
    critical_triangles = Int[]

    # Pre-allocate for performance
    sizehint!(critical_vertices, n_vertices ÷ 10)
    sizehint!(critical_edges, n_edges ÷ 10)
    sizehint!(critical_triangles, n_triangles ÷ 10)

    # Check vertices
    for v_id in 1:n_vertices
        if is_critical_cell(Cell(0, v_id), gradient, mesh)
            push!(critical_vertices, v_id)
        end
    end

    # Check edges
    for e_id in 1:n_edges
        if is_critical_cell(Cell(1, e_id), gradient, mesh)
            push!(critical_edges, e_id)
        end
    end

    # Check triangles
    for t_id in 1:n_triangles
        if is_critical_cell(Cell(2, t_id), gradient, mesh)
            push!(critical_triangles, t_id)
        end
    end

    return critical_vertices, critical_edges, critical_triangles
end

"""
Classify critical points by type based on their dimension.
"""
function classify_critical_points(vertices::Vector{Int}, edges::Vector{Int},
                                triangles::Vector{Int}, mesh::TriangleMesh)
    minima = vertices
    saddles = edges
    maxima = triangles

    return minima, saddles, maxima
end

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
    persistence_threshold::Real=0.0,
    rebuild_gradient_after_simplification::Bool=true,
)
    # Compute vertex order (offsets) using TTK OrderDisambiguation semantics
    # Primary: scalar; tie-breaker: provided SoS offsets if given, else vertex id.
    offsets = sort_vertices_order(scalar_field, sos_offsets)

    # Initialize gradient field - matches C++ initMemory
    n_vertices = size(mesh.vertices, 2)
    n_edges = size(mesh.edge_vertices, 2)
    n_triangles = size(mesh.triangles, 2)
    gradient = GradientField(n_vertices, n_edges, n_triangles)

    # Optionally perform scalar-field simplification by persistence, then
    # rebuild offsets and gradient to reflect true topological simplification
    local_scalars = Vector{Float64}(scalar_field)
    if persistence_threshold > 0 && rebuild_gradient_after_simplification
        # adjust scalars in place
        _ = simplify_scalar_field_by_persistence!(mesh, local_scalars, offsets, persistence_threshold)
        # recompute order with same SoS tie-breaker (if provided)
        offsets = sort_vertices_order(local_scalars, sos_offsets)
    end

    # Build discrete gradient field using ProcessLowerStars with (possibly) updated order
    build_gradient_field!(mesh, offsets, gradient)

    # Find critical points - matches C++ getCriticalPoints
    crit_verts, crit_edges, crit_tris = find_critical_points(mesh, gradient)

    # Classify critical points
    minima, saddles, maxima = classify_critical_points(crit_verts, crit_edges, crit_tris, mesh)

    # Optional post-filtering of critical lists (does not change gradient).
    # If we rebuilt the gradient after simplification, lists should already be cleaner.
    if persistence_threshold > 0 && !rebuild_gradient_after_simplification
        filter_critical_by_persistence!(minima, saddles, maxima, mesh, local_scalars, offsets, persistence_threshold)
    end

    return minima, saddles, maxima, gradient
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
Print summary statistics of critical points.
"""
function print_critical_point_summary(minima::Vector{Int}, saddles::Vector{Int},
                                    maxima::Vector{Int}, mesh::TriangleMesh)
    n_verts = size(mesh.vertices, 2)
    n_tris = size(mesh.triangles, 2)

    println("=== Discrete Morse Theory Critical Point Analysis ===")
    println("Mesh: $(n_verts) vertices, $(size(mesh.edge_vertices, 2)) edges, $(n_tris) triangles")
    println("Critical Points:")
    println("  Minima (0-cells):     $(length(minima))")
    println("  Saddles (1-cells):    $(length(saddles))")
    println("  Maxima (2-cells):     $(length(maxima))")

    # Euler characteristic verification
    euler_char = n_verts - size(mesh.edge_vertices, 2) + n_tris
    morse_euler = length(minima) - length(saddles) + length(maxima)
    println("\nEuler Characteristic:")
    println("  Topological:    $euler_char")
    println("  Morse:          $morse_euler")
    match_symbol = euler_char == morse_euler ? "✓" : "✗"
    println("  Match: $match_symbol")
end

# ==============================================================================
# TEST CASE
# ==============================================================================

"""
Create a test case with a simple triangulated surface to verify the algorithm.
"""
function create_test_mesh()
    # Create a simple 2x2 grid of vertices
    vertices = [0.0 1.0 0.0 1.0;
                0.0 0.0 1.0 1.0;
                0.0 0.0 0.0 0.0]

    # Create two triangles forming a square (3 x n_triangles)
    triangles = [1 2;
                 2 3;
                 3 4]

    return TriangleMesh(vertices, triangles)
end

"""
Run a basic test of the critical point detection algorithm.
"""
function test_critical_point_detection()
    println("Testing Discrete Morse Theory Critical Point Detection")
    println("=" ^ 50)

    # Create test mesh
    mesh = create_test_mesh()

    # Define a simple scalar field (e.g., distance from origin)
    scalar_field = [sqrt(sum(mesh.vertices[:,i].^2)) for i in 1:size(mesh.vertices, 2)]

    # Find critical points
    minima, saddles, maxima, gradient = find_critical_points_discrete_morse(mesh, scalar_field)

    # Print results
    print_critical_point_summary(minima, saddles, maxima, mesh)

    # Print critical point locations
    println("\nCritical Point Locations:")
    println("Minima:")
    for v_id in minima
        pos = get_critical_point_location(Cell(0, v_id), mesh)
        println("  Vertex $v_id: ($(pos[1]), $(pos[2]), $(pos[3]))")
    end

    println("Saddles:")
    for e_id in saddles
        pos = get_critical_point_location(Cell(1, e_id), mesh)
        println("  Edge $e_id: ($(pos[1]), $(pos[2]), $(pos[3]))")
    end

    println("Maxima:")
    for t_id in maxima
        pos = get_critical_point_location(Cell(2, t_id), mesh)
        println("  Triangle $t_id: ($(pos[1]), $(pos[2]), $(pos[3]))")
    end

    return mesh, minima, saddles, maxima, gradient
end

# Run test if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_critical_point_detection()
end