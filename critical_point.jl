
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
    order::AbstractVector{Int}
)
    n = length(scalars)
    neigh = mesh.vertex_neighbors
    # map vertex id to its position in the sorted order (1..n)
    invOrder = order
    # vertices sorted ascending by order
    verts_sorted = sort(collect(1:n), by=i -> invOrder[i])
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
            if r == elder
                continue
            end
            minVertex = uf.comp_birth_vertex[r]
            # find an edge (v, u) that connects to this component
            u = comps[r]
            # retrieve edge id for (v,u) using O(1) lookup
            edge_id = lookup_edge_id(mesh, v, u)
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
    order::AbstractVector{Int}
)
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
    filter!((v) -> !(v in rm_min), minima)
    filter!((e) -> !(e in rm_sad), saddles)
    # leave maxima as-is for consistency
    return
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

    # Idiomatic threaded scan: fill boolean masks in parallel, then collect indices.
    vmask = falses(n_vertices)
    emask = falses(n_edges)
    tmask = falses(n_triangles)

    @threads for v_id in 1:n_vertices
        @inbounds vmask[v_id] = is_critical_cell(Cell(0, v_id), gradient, mesh)
    end

    @threads for e_id in 1:n_edges
        @inbounds emask[e_id] = is_critical_cell(Cell(1, e_id), gradient, mesh)
    end

    @threads for t_id in 1:n_triangles
        @inbounds tmask[t_id] = is_critical_cell(Cell(2, t_id), gradient, mesh)
    end

    critical_vertices = findall(vmask)
    critical_edges = findall(emask)
    critical_triangles = findall(tmask)

    return critical_vertices, critical_edges, critical_triangles
end

"""
Classify critical points by type based on cell dimension.

This is intentionally simple: in 2D discrete Morse on triangle meshes,
- 0-cells (vertices) are minima,
- 1-cells (edges) are saddles,
- 2-cells (triangles) are maxima.

The heavy lifting is done in `find_critical_points`, which determines which
cells are critical by inspecting the discrete gradient pairings. Here we just
partition those critical cells by type, while also sanitizing and sorting the
outputs for deterministic downstream use.

Returns three sorted, de-duplicated vectors of valid indices:
`(minima::Vector{Int}, saddles::Vector{Int}, maxima::Vector{Int})`.
"""
function classify_critical_points(vertices::Vector{Int},
    edges::Vector{Int}, triangles::Vector{Int}, mesh::TriangleMesh)

    nv = size(mesh.vertices, 2)
    ne = size(mesh.edge_vertices, 2)
    nt = size(mesh.triangles, 2)

    sanitize(list::Vector{Int}, upper::Int) = begin
        # keep indices in range, drop duplicates, and sort for determinism
        uniq = unique(filter(i -> 1 <= i <= upper, list))
        sort!(uniq)
        uniq
    end

    minima  = sanitize(vertices, nv)
    saddles = sanitize(edges, ne)
    maxima  = sanitize(triangles, nt)

    return minima, saddles, maxima
end
