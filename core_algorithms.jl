
# ==============================================================================
# CORE ALGORITHM FUNCTIONS
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
    tau::Real # persistence_threshold
)
    if tau <= 0
        return Int[]
    end
    pairs_min = compute_persistence_pairs_min_sad(mesh, scalars, order)
    pairs_max = compute_persistence_pairs_max_sad(mesh, scalars, order)
    modified = Set{Int}()

    # Reuse neighbors for all floods
    neigh = mesh.vertex_neighbors

    # Process minima cancellations first (in increasing death value)
    sort!(pairs_min, by=p -> p.death_value)
    for p in pairs_min
        if p.persistence < tau
            # flood sublevel component strictly below the merge level (death)
            comp = flood_component_vertices(mesh, scalars, p.extremum, p.death_value, :below, neigh)
            if !isempty(comp)
                for v in comp
                    if scalars[v] < p.death_value
                        scalars[v] = p.death_value
                        push!(modified, v)
                    end
                end
            end
        end
    end

    # Then process maxima cancellations (in increasing death value)
    sort!(pairs_max, by=p -> p.death_value)
    for p in pairs_max
        if p.persistence < tau
            # flood superlevel component strictly above the merge level (death)
            comp = flood_component_vertices(mesh, scalars, p.extremum, p.death_value, :above, neigh)
            if !isempty(comp)
                for v in comp
                    if scalars[v] > p.death_value
                        scalars[v] = p.death_value
                        push!(modified, v)
                    end
                end
            end
        end
    end

    return collect(modified)
end

"""
Compute the lower star of vertex `a` with respect to vertex ordering.
The lower star contains all cells where vertex `a` has the maximum order value.
Follows C++ TTK implementation exactly.
"""
function compute_lower_star!(ls::LowerStar, a::Int, vertex_order::Vector{Int}, mesh::TriangleMesh)
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

        if vertex_order[other_vertex] < vertex_order[a]
            lower_verts = (vertex_order[other_vertex], -1, -1)
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
            low_verts = (vertex_order[v1], vertex_order[v2], -1)
        elseif v1 == a
            low_verts = (vertex_order[v0], vertex_order[v2], -1)
        elseif v2 == a
            low_verts = (vertex_order[v0], vertex_order[v1], -1)
        end

        # Higher order vertex first
        if low_verts[1] < low_verts[2]
            low_verts = (low_verts[2], low_verts[1], -1)
        end

        if vertex_order[a] > low_verts[1]  # Triangle in lower star
            # Find the two edge indices in lower star corresponding to the two lower vertices
            face1 = -1
            face2 = -1
            for (edge_idx, edge) in enumerate(ls.edges)
                e_v1, e_v2 = mesh.edge_vertices[:, edge.id]
                e_other = (e_v1 == a) ? e_v2 : e_v1
                e_offset = vertex_order[e_other]

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

    # Find triangles containing vertex a - optimized using pre-computed adjacency
    for tri_idx in mesh.vertex_to_triangles[a]
        v0, v1, v2 = mesh.triangles[:, tri_idx]
        process_triangle(tri_idx, v0, v1, v2)
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
function process_lower_star_vertex!(ls::LowerStar, 
    a::Int, 
    vertex_order::Vector{Int},
    mesh::TriangleMesh, 
    gradient::GradientField
)
    # Compute lower star for vertex a
    compute_lower_star!(ls, a, vertex_order, mesh)

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

    # Initialize priority queues using DataStructures.BinaryHeap
    # TTK uses std::priority_queue with a comparator equivalent to a min-heap
    # on the lexicographic low-vertex tuples. Use LV_ORD to match that behavior.
    pq_zero = BinaryHeap{CellExt}(LV_ORD)  # For critical cells (min-heap)
    pq_one = BinaryHeap{CellExt}(LV_ORD)   # For cells with one unpaired face (min-heap)

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
function build_gradient_field!(mesh::TriangleMesh, vertex_order::Vector{Int}, gradient::GradientField)
    n_vertices = size(mesh.vertices, 2)

    # Parallelize over vertices with thread-local LowerStar state.
    # Safety: Each simplex (vertex/edge/triangle) belongs to the lower star of
    # exactly one vertex (the one with max total order among its vertices).
    # Therefore, gradient writes are disjoint across vertices; no locks needed.
    tls = [LowerStar() for _ in 1:nthreads()]  # pre-allocate per-thread to reuse buffers
    @threads for vertex_id in 1:n_vertices
        # Guard against rare mismatches between threadid() and tls length
        # (observed in some CI/test contexts). Fallback allocates a local buffer.
        local_ls = (tid -> (1 <= tid <= length(tls) ? tls[tid] : LowerStar()))(threadid())
        ls = local_ls
        process_lower_star_vertex!(ls, vertex_id, vertex_order, mesh, gradient)
    end
end
