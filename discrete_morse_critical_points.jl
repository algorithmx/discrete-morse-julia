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
function find_critical_points_discrete_morse(mesh::TriangleMesh, scalar_field::Vector{Float64})
    # Compute orderings (offsets) - sort vertices by scalar value
    # Matches C++ TTK offset computation
    vertex_order = sortperm(scalar_field)
    offsets = zeros(Int, length(scalar_field))
    for (new_id, old_id) in enumerate(vertex_order)
        offsets[old_id] = new_id
    end

    # Initialize gradient field - matches C++ initMemory
    n_vertices = size(mesh.vertices, 2)
    n_edges = size(mesh.edge_vertices, 2)
    n_triangles = size(mesh.triangles, 2)
    gradient = GradientField(n_vertices, n_edges, n_triangles)

    # Build discrete gradient field using ProcessLowerStars
    # Matches C++ processLowerStars call
    build_gradient_field!(mesh, offsets, gradient)

    # Find critical points - matches C++ getCriticalPoints
    crit_verts, crit_edges, crit_tris = find_critical_points(mesh, gradient)

    # Classify critical points
    minima, saddles, maxima = classify_critical_points(crit_verts, crit_edges, crit_tris, mesh)

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