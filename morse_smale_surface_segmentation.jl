"""
Morse–Smale Surface Segmentation (TTK-aligned data structures)

This file implements the first slice of functionality needed to build
Morse–Smale networks and surface segmentations in Julia, following the
algorithms and data structures from TTK's MorseSmaleComplex and
DiscreteGradient modules.

Scope (dim = 2 surfaces):
- V-path tracing (ascending and descending) on triangulated surfaces
- 1-separatrices (ascending) from 1-saddles (edges) to maxima (triangles)
- Ascending and descending segmentations
- Final Morse–Smale segmentation (ascending ⊗ descending)

Data structures mirror TTK naming and grouping where practical.

Dependencies: expects TriangleMesh, Cell, GradientField to be defined
in the workspace (see discrete_morse_critical_points.jl). This file
does not assume a module; functions can be imported directly.

Author: GitHub Copilot (auto-generated, aligned with TTK C++)
Date: October 2025
"""

# ==========================
#  Data Structures (TTK-like)
# ==========================

"""
Separatrix edge (1-separatrix) between critical points.
Matches ttk::MorseSmaleComplex::Separatrix for the fields we use.
"""
struct Separatrix
    source_::Cell                 # starting critical cell (e.g., saddle edge)
    destination_::Cell            # ending critical cell (e.g., maximum triangle)
    geometry_::Vector{Cell}       # full V-path geometry (cells along the path)

    function Separatrix()
        new(Cell(), Cell(), Cell[])
    end

    function Separatrix(source_::Cell, destination_::Cell, geometry_::Vector{Cell})
        new(source_, destination_, geometry_)
    end
end

"""
Critical point node for an MS network graph (2D).
Types: 0 = minimum (vertex), 1 = saddle (edge), 2 = maximum (triangle)
"""
struct CriticalPointNode
    id::Int              # node id (0-based indexing for parity with TTK style)
    cell::Cell           # underlying critical cell
    cptype::Int8         # 0,1,2 as above
    point::NTuple{3,Float32}
end

"""
Separatrix edge in an MS network graph (2D).
edgeType: 0 = descending to minima, 1 = ascending to maxima
"""
struct NetworkEdge
    id::Int
    edgeType::Int8       # 0 or 1
    sourceNode::Int      # node id (0-based)
    destinationNode::Int # node id (0-based)
    geometry::Vector{Cell}
    isOnBoundary::Int8
end

"""
Lightweight Morse–Smale Network container for 2D surfaces.
"""
struct MSNetwork
    nodes::Vector{CriticalPointNode}
    edges::Vector{NetworkEdge}
end

"""
Output arrays for 1-separatrices, organized as in TTK (pt & cl groups).
Only a subset of arrays are populated initially for 2D surfaces.
"""
mutable struct Output1Separatrices
    # point data arrays
    pt_numberOfPoints_::Int
    pt_points_::Vector{Float32}          # flat array [x0,y0,z0, x1,y1,z1, ...]
    pt_smoothingMask_::Vector{UInt8}
    pt_cellDimensions_::Vector{Int8}
    pt_cellIds_::Vector{Int}

    # cell data arrays
    cl_numberOfCells_::Int
    cl_connectivity_::Vector{Int}        # line segments connectivity [u0,v0, u1,v1, ...]
    cl_sourceIds_::Vector{Int}
    cl_destinationIds_::Vector{Int}
    cl_separatrixIds_::Vector{Int}
    cl_separatrixTypes_::Vector{Int8}
    cl_isOnBoundary_::Vector{Int8}

    # per-separatrix function extrema (store vertex ids, like TTK)
    cl_sepFuncMaxId_::Vector{Int}
    cl_sepFuncMinId_::Vector{Int}

    function Output1Separatrices()
        new(0, Float32[], UInt8[], Int8[], Int[],
            0, Int[], Int[], Int[], Int[], Int8[], Int8[],
            Int[], Int[])
    end
end

"""
Output arrays for segmentation (ascending, descending, final).
Matches ttk::MorseSmaleComplex::OutputManifold (pointer group) conceptually.
"""
mutable struct OutputManifold
    ascending_::Vector{Int}      # per-vertex ascending manifold id
    descending_::Vector{Int}     # per-vertex descending manifold id
    morseSmale_::Vector{Int}     # per-vertex final MS cell id

    function OutputManifold(nVerts::Int)
        new(fill(-1, nVerts), fill(-1, nVerts), fill(-1, nVerts))
    end
end


# =====================
#  Helper / Primitives
# =====================

"""
Return whether a cell is critical with respect to the gradient field.
This mirrors the semantics of TTK's DiscreteGradient::isCellCritical for dim=2.
"""
function is_cell_critical(cell::Cell, gradient::GradientField)::Bool
    if cell.dim == 0
        return gradient.vertex_to_edge[cell.id] == -1
    elseif cell.dim == 1
        # In discrete Morse, a 1-cell may be paired with either a 0-cell or a 2-cell.
        return gradient.edge_to_vertex[cell.id] == -1 && gradient.edge_to_triangle[cell.id] == -1
    elseif cell.dim == 2
        return gradient.triangle_to_edge[cell.id] == -1
    else
        return true
    end
end

"""
Return the gradient-paired neighbor of a cell.
Equivalent of TTK's DiscreteGradient::getPairedCell with reverse flag:
- If reverse=true, return face (lower-dim) paired to cell
- If reverse=false, return coface (higher-dim) paired to cell

For dim=2 surfaces, the only meaningful cases are:
- reverse=true: triangle -> paired edge, edge -> paired vertex
- reverse=false: edge -> paired triangle, vertex -> paired edge
Unknown/invalid returns -1.
"""
function get_paired_cell_id(cell::Cell, gradient::GradientField; reverse::Bool=false)::Int
    if cell.dim == 2
        return reverse ? gradient.triangle_to_edge[cell.id] : -1
    elseif cell.dim == 1
        return reverse ? gradient.edge_to_vertex[cell.id] : gradient.edge_to_triangle[cell.id]
    elseif cell.dim == 0
        return reverse ? -1 : gradient.vertex_to_edge[cell.id]
    else
        return -1
    end
end


"""
Get the list of vertex ids belonging to a cell.
"""
function cell_vertices(cell::Cell, mesh::TriangleMesh)::Vector{Int}
    if cell.dim == 0
        return [cell.id]
    elseif cell.dim == 1
        return [mesh.edge_vertices[1, cell.id], mesh.edge_vertices[2, cell.id]]
    elseif cell.dim == 2
        return [mesh.triangles[1, cell.id], mesh.triangles[2, cell.id], mesh.triangles[3, cell.id]]
    else
        return Int[]
    end
end

"""
Return the vertex id with minimal offset within a cell (SoS lower vertex).
"""
function get_cell_lower_vertex(cell::Cell, mesh::TriangleMesh, offsets::AbstractVector{Int})::Int
    vs = cell_vertices(cell, mesh)
    return argmin(i -> offsets[i], vs)
end

"""
Return the vertex id with maximal offset within a cell (SoS greater vertex).
"""
function get_cell_greater_vertex(cell::Cell, mesh::TriangleMesh, offsets::AbstractVector{Int})::Int
    vs = cell_vertices(cell, mesh)
    return argmax(i -> offsets[i], vs)
end

"""
Boundary predicates.
"""
is_edge_on_boundary(mesh::TriangleMesh, e::Int) = (length(mesh.edge_to_triangles[e]) == 1)

function is_triangle_on_boundary(mesh::TriangleMesh, t::Int)
    # Triangle on boundary if any of its edges is boundary
    # find its edges by scanning edge_to_triangles membership (cheap for tests)
    for e in 1:size(mesh.edge_vertices, 2)
        tris = mesh.edge_to_triangles[e]
        if any(==(t), tris) && length(tris) == 1
            return true
        end
    end
    return false
end

function is_vertex_on_boundary(mesh::TriangleMesh, v::Int)
    for e in mesh.vertex_to_edges[v]
        if is_edge_on_boundary(mesh, e)
            return true
        end
    end
    return false
end

function is_boundary(cell::Cell, mesh::TriangleMesh)
    if cell.dim == 0
        return is_vertex_on_boundary(mesh, cell.id)
    elseif cell.dim == 1
        return is_edge_on_boundary(mesh, cell.id)
    elseif cell.dim == 2
        return is_triangle_on_boundary(mesh, cell.id)
    else
        return false
    end
end

"""
Compute geometric representative point for a cell:
- vertex: the vertex position
- edge: midpoint
- triangle: incenter (barycenter fallback)
"""
function cell_point(cell::Cell, mesh::TriangleMesh)::NTuple{3,Float32}
    if cell.dim == 0
        v = cell.id
        return (Float32(mesh.vertices[1, v]), Float32(mesh.vertices[2, v]), Float32(mesh.vertices[3, v]))
    elseif cell.dim == 1
        v0 = mesh.edge_vertices[1, cell.id]
        v1 = mesh.edge_vertices[2, cell.id]
        x = (mesh.vertices[1, v0] + mesh.vertices[1, v1]) / 2
        y = (mesh.vertices[2, v0] + mesh.vertices[2, v1]) / 2
        z = (mesh.vertices[3, v0] + mesh.vertices[3, v1]) / 2
        return (Float32(x), Float32(y), Float32(z))
    elseif cell.dim == 2
        v0 = mesh.triangles[1, cell.id]
        v1 = mesh.triangles[2, cell.id]
        v2 = mesh.triangles[3, cell.id]
        # barycenter as a robust representative
        x = (mesh.vertices[1, v0] + mesh.vertices[1, v1] + mesh.vertices[1, v2]) / 3
        y = (mesh.vertices[2, v0] + mesh.vertices[2, v1] + mesh.vertices[2, v2]) / 3
        z = (mesh.vertices[3, v0] + mesh.vertices[3, v1] + mesh.vertices[3, v2]) / 3
        return (Float32(x), Float32(y), Float32(z))
    else
        return (0f0, 0f0, 0f0)
    end
end


# ==============================
#  V-Path Tracing (dim = 2 only)
# ==============================

"""
Trace an ascending V-path starting from a 2D cell (triangle) on a surface.
Mirrors TTK's DiscreteGradient::getAscendingPath for dimensionality_ == 2.
The path alternates: triangle -> edge -> triangle -> ... until a critical cell or boundary.

Arguments:
- cell: starting cell (should be Cell(2, triangle_id))
- gradient, mesh: supporting data
- cycle_detector: when true, break on detected cycles (safety)

Returns: Vector{Cell} with the visited cells in order (including start)
"""
function trace_ascending_path(cell::Cell,
                              gradient::GradientField,
                              mesh::TriangleMesh;
                              cycle_detector::Bool=false)::Vector{Cell}
    @assert cell.dim == 2 "trace_ascending_path expects a triangle cell on surfaces (dim=2)"
    path = Cell[]

    nTri = size(mesh.triangles, 2)
    seen = cycle_detector ? falses(nTri) : Bool[]

    currentId = cell.id
    oldId = -1
    while currentId != oldId
        oldId = currentId

        # add triangle
        push!(path, Cell(2, currentId))
        if is_cell_critical(Cell(2, currentId), gradient)
            break
        end

        # triangle -> paired edge (reverse=true)
        connectedEdgeId = get_paired_cell_id(Cell(2, currentId), gradient; reverse=true)
        if connectedEdgeId == -1
            break
        end

        # add edge
        push!(path, Cell(1, connectedEdgeId))
        if is_cell_critical(Cell(1, connectedEdgeId), gradient)
            break
        end

        # edge -> opposite triangle across edge
        tris = mesh.edge_to_triangles[connectedEdgeId]
        nextTri = currentId
        if !isempty(tris)
            if length(tris) == 1
                # boundary edge, convergence
                nextTri = currentId
            else
                a, b = tris[1], tris[2]
                nextTri = (a == currentId) ? b : a
            end
        end

        if cycle_detector
            if !isempty(seen)
                if seen[nextTri]
                    break
                else
                    seen[nextTri] = true
                end
            end
        end

        currentId = nextTri
    end

    return path
end

"""
Trace a descending V-path starting from a 2D cell (triangle) on a surface.
This follows decreasing dimension (triangle -> edge -> vertex) until critical.
This is a simplified counterpart to TTK's getDescendingPath for 2D.
"""
function trace_descending_path(cell::Cell,
                              gradient::GradientField,
                              mesh::TriangleMesh)::Vector{Cell}
    @assert cell.dim == 2 "trace_descending_path expects a triangle cell on surfaces (dim=2)"
    path = Cell[]

    # add triangle
    push!(path, Cell(2, cell.id))
    if is_cell_critical(cell, gradient)
        return path
    end

    # triangle -> paired edge (reverse=true)
    eId = get_paired_cell_id(cell, gradient; reverse=true)
    if eId == -1
        return path
    end
    push!(path, Cell(1, eId))
    if is_cell_critical(Cell(1, eId), gradient)
        return path
    end

    # edge -> paired vertex (reverse=true)
    vId = get_paired_cell_id(Cell(1, eId), gradient; reverse=true)
    if vId == -1
        return path
    end
    push!(path, Cell(0, vId))
    # vertex is terminal in 2D
    return path
end


"""
Trace a descending V-path starting from a 1D cell (edge) on a surface.
Alternates: edge -> vertex -> edge -> ... until reaching a critical vertex or boundary.
"""
function trace_descending_path_from_edge(edgeCell::Cell,
                                         gradient::GradientField,
                                         mesh::TriangleMesh)::Vector{Cell}
    @assert edgeCell.dim == 1 "trace_descending_path_from_edge expects an edge cell"
    path = Cell[edgeCell]
    # edge -> vertex (reverse=true)
    eId = edgeCell.id
    while true
        vId = get_paired_cell_id(Cell(1, eId), gradient; reverse=true)
        if vId == -1
            break
        end
        push!(path, Cell(0, vId))
        if is_cell_critical(Cell(0, vId), gradient)
            break
        end
        # vertex -> edge (reverse=false)
        eNext = get_paired_cell_id(Cell(0, vId), gradient; reverse=false)
        if eNext == -1
            break
        end
        push!(path, Cell(1, eNext))
        eId = eNext
    end
    return path
end


# ===========================================
#  1-Separatrices (ascending) from 1-saddles
# ===========================================

"""
Compute ascending 1-separatrices on a surface (dim = 2).
Mirrors ttk::MorseSmaleComplex::getAscendingSeparatrices1 for 2D,
by seeding from each 1-saddle (edge) and following V-paths to maxima.
"""
function get_ascending_separatrices1(saddles::Vector{Int},
                                     gradient::GradientField,
                                     mesh::TriangleMesh)::Vector{Separatrix}
    seps = Separatrix[]
    for sId in saddles
        saddle = Cell(1, sId)
        # star of edge: incident triangles
        for tId in mesh.edge_to_triangles[sId]
            # vpath starts with the saddle, then follow from adjacent triangle
            vpath = Cell[saddle]
            vpath_tail = trace_ascending_path(Cell(2, tId), gradient, mesh)
            append!(vpath, vpath_tail)
            last = vpath[end]
            if last.dim == 2 && is_cell_critical(last, gradient)
                push!(seps, Separatrix(saddle, last, vpath))
            end
        end
    end
    return seps
end


"""
Compute descending 1-separatrices on a surface (dim = 2).
Seed from 1-saddles (edges) and follow descending V-paths to minima (vertices).
"""
function get_descending_separatrices1(saddles::Vector{Int},
                                      gradient::GradientField,
                                      mesh::TriangleMesh)::Vector{Separatrix}
    seps = Separatrix[]
    for sId in saddles
        saddle = Cell(1, sId)
        vpath = trace_descending_path_from_edge(saddle, gradient, mesh)
        last = vpath[end]
        if last.dim == 0 && is_cell_critical(last, gradient)
            push!(seps, Separatrix(saddle, last, vpath))
        end
    end
    return seps
end


# ======================
#  Segmentation (dim=2)
# ======================

"""
Ascending segmentation: assign each vertex to a maximum basin.
Follows ttk::MorseSmaleComplex::setAscendingSegmentation logic for 2D:
1) propagate manifold ids on triangles following V-paths
2) pull labels back to vertices via star mapping
"""
function set_ascending_segmentation(maxima_triangles::Vector{Int},
                                    gradient::GradientField,
                                    mesh::TriangleMesh)::Vector{Int}
    nVerts = size(mesh.vertices, 2)
    nTri = size(mesh.triangles, 2)
    labels_cells = fill(-1, nTri)

    # mark maxima (triangles) with unique ids
    nextId = 0
    for t in maxima_triangles
        labels_cells[t] = nextId
        nextId += 1
    end

    isMarked = falses(nTri)
    visited = Int[]
    for i in 1:nTri
        if isMarked[i] == 1
            continue
        end
        empty!(visited)
        curr = i
        # walk until reaching a marked cell (or boundary/critical)
        while labels_cells[curr] == -1
            if isMarked[curr] == 1
                break
            end
            # follow triangle -> paired edge -> opposite triangle
            paired_edge = get_paired_cell_id(Cell(2, curr), gradient; reverse=true)
            nextTri = curr
            if paired_edge != -1
                tris = mesh.edge_to_triangles[paired_edge]
                if !isempty(tris)
                    if length(tris) == 1
                        # boundary convergence
                        nextTri = curr
                    else
                        a, b = tris[1], tris[2]
                        nextTri = (a == curr) ? b : a
                    end
                end
            end
            push!(visited, curr)
            if nextTri == curr
                # no progress (boundary or degenerate)
                break
            end
            curr = nextTri
        end
        for el in visited
            labels_cells[el] = labels_cells[curr]
            isMarked[el] = 1
        end
    end

    # map triangle labels back to vertex labels using a quick vertex->triangle star
    labels_vertices = fill(-1, nVerts)
    for v in 1:nVerts
        # derive a triangle in the vertex star via incident edges
        assigned = false
        for e in mesh.vertex_to_edges[v]
            if assigned; break; end
            for t in mesh.edge_to_triangles[e]
                labels_vertices[v] = labels_cells[t]
                assigned = true
                break
            end
        end
    end

    return labels_vertices
end

"""
Descending segmentation: assign each vertex to a minimum basin.
Follows ttk::MorseSmaleComplex::setDescendingSegmentation for 2D.
"""
function set_descending_segmentation(minima_vertices::Vector{Int},
                                     gradient::GradientField,
                                     mesh::TriangleMesh)::Vector{Int}
    nVerts = size(mesh.vertices, 2)

    if length(minima_vertices) == 1
        return fill(0, nVerts)
    end

    labels = fill(-1, nVerts)
    # mark minima
    nextId = 0
    for v in minima_vertices
        labels[v] = nextId
        nextId += 1
    end

    visited = Int[]
    for i in 1:nVerts
        if labels[i] != -1
            continue
        end
        empty!(visited)
        curr = i
        while labels[curr] == -1
            # vertex -> paired edge
            paired_edge = get_paired_cell_id(Cell(0, curr), gradient; reverse=false)
            if paired_edge == -1
                # dead end
                break
            end
            # edge -> the other vertex
            v0 = mesh.edge_vertices[1, paired_edge]
            v1 = mesh.edge_vertices[2, paired_edge]
            nextV = (v0 == curr) ? v1 : v0
            push!(visited, curr)
            curr = nextV
        end
        for el in visited
            labels[el] = labels[curr]
        end
    end

    return labels
end

"""
Combine ascending and descending segmentations into final Morse–Smale cells.
Mirrors ttk::MorseSmaleComplex::setFinalSegmentation: regionId = a * nMax + d,
then compress to dense ids.
Inputs:
- ascending: per-vertex labels in [0, nMax) or -1
- descending: per-vertex labels in [0, nMin) or -1
"""
function set_final_segmentation(ascending::Vector{Int},
                                descending::Vector{Int},
                                nMax::Int)::Vector{Int}
    @assert length(ascending) == length(descending)
    n = length(ascending)
    ms_sparse = Vector{Int}(undef, n)
    for i in 1:n
        a = ascending[i]; d = descending[i]
        ms_sparse[i] = (a == -1 || d == -1) ? -1 : (a * nMax + d)
    end

    # compress sparse ids to dense [0..k)
    uniq = sort(unique(ms_sparse))
    # ensure -1 stays mapped to -1 (unlabeled)
    mapdict = Dict{Int,Int}()
    dense = 0
    for id in uniq
        if id == -1
            mapdict[id] = -1
        else
            mapdict[id] = dense
            dense += 1
        end
    end
    ms_dense = [mapdict[x] for x in ms_sparse]
    return ms_dense
end


# =====================
#  High-level pipeline
# =====================

"""
Compute ascending 1-separatrices and segmentations given critical points.
This mirrors the sequencing inside ttk::MorseSmaleComplex::execute for dim=2.

Arguments:
- minima: vector of vertex ids
- saddles: vector of edge ids (1-saddles)
- maxima: vector of triangle ids

Returns: (separatrices1::Vector{Separatrix}, outManifold::OutputManifold)
"""
function compute_surface_morse_smale(
    mesh::TriangleMesh,
    gradient::GradientField,
    minima::Vector{Int},
    saddles::Vector{Int},
    maxima::Vector{Int},
)::Tuple{Vector{Separatrix},OutputManifold}
    # 1-separatrices (ascending)
    seps1 = get_ascending_separatrices1(saddles, gradient, mesh)

    # segmentations
    asc = set_ascending_segmentation(maxima, gradient, mesh)
    desc = set_descending_segmentation(minima, gradient, mesh)
    nMax = maximum(asc) + 1
    final = set_final_segmentation(asc, desc, nMax)

    out = OutputManifold(length(final))
    out.ascending_ .= asc
    out.descending_ .= desc
    out.morseSmale_ .= final
    return (seps1, out)
end


"""
End-to-end pipeline from scalar field:
- Reuse discrete_morse_critical_points.jl to build gradient and detect critical points
- Optionally simplify by persistence (threshold), matching TTK's approach
- Compute ascending/descending 1-separatrices and segmentations
- Optionally build an MS network

Returns a NamedTuple with fields:
  minima, saddles, maxima, gradient,
  asc_seps, desc_seps, separatrices_points (Output1Separatrices),
  segmentation (OutputManifold), network (MSNetwork or nothing), node_map (Dict or nothing)
"""
function compute_surface_morse_smale_from_scalar(
    mesh::TriangleMesh,
    scalar_field::AbstractVector{<:Real};
    sos_offsets::Union{Nothing,AbstractVector{Int}}=nothing,
    persistence_threshold::Real=0.0,
    rebuild_gradient_after_simplification::Bool=true,
    build_network::Bool=false,
)
    # Build gradient and detect critical points (with optional persistence simplification)
    minima, saddles, maxima, gradient = find_critical_points_discrete_morse(
        mesh, scalar_field;
        sos_offsets=sos_offsets,
        persistence_threshold=persistence_threshold,
        rebuild_gradient_after_simplification=rebuild_gradient_after_simplification,
    )

    # 1-separatrices
    asc_seps = get_ascending_separatrices1(saddles, gradient, mesh)
    desc_seps = get_descending_separatrices1(saddles, gradient, mesh)

    # Build Output1Separatrices (use SoS order derived from input scalars)
    # Note: if scalar field was simplified internally, these offsets may not exactly match
    # the modified scalars; this only affects sepFunc* attributes, not geometry/segmentation.
    offsets = sort_vertices_order(scalar_field, sos_offsets)
    out1 = Output1Separatrices()
    set_separatrices1!(out1, asc_seps, offsets, mesh)
    set_separatrices1!(out1, desc_seps, offsets, mesh)

    # Segmentations
    asc = set_ascending_segmentation(maxima, gradient, mesh)
    desc = set_descending_segmentation(minima, gradient, mesh)
    nMax = isempty(asc) ? 0 : (maximum(asc) + 1)
    final = nMax == 0 ? asc : set_final_segmentation(asc, desc, nMax)
    outSeg = OutputManifold(length(asc))
    outSeg.ascending_ .= asc
    outSeg.descending_ .= desc
    outSeg.morseSmale_ .= final

    # Optional network
    net = nothing; node_map = nothing
    if build_network
        net, node_map = build_ms_network(mesh, gradient, minima, saddles, maxima, asc_seps, desc_seps)
    end

    return (
        minima=minima, saddles=saddles, maxima=maxima, gradient=gradient,
        asc_seps=asc_seps, desc_seps=desc_seps, separatrices_points=out1,
        segmentation=outSeg, network=net, node_map=node_map,
    )
end


# ==========================================
#  Output builders and simple VTK exporters
# ==========================================

"""
Append 1-separatrices to Output1Separatrices (TTK-aligned builder).
This mirrors ttk::MorseSmaleComplex::setSeparatrices1 for dim=2.
"""
function set_separatrices1!(out::Output1Separatrices,
                            separatrices::Vector{Separatrix},
                            offsets::AbstractVector{Int},
                            mesh::TriangleMesh)
    # determine starting separatrix id
    sepIdStart = isempty(out.cl_separatrixIds_) ? 0 : (maximum(out.cl_separatrixIds_) + 1)

    # current counts
    npoints = out.pt_numberOfPoints_
    ncells = out.cl_numberOfCells_

    # precompute begin indices per separatrix
    geomPointsBegId = [npoints]
    geomCellsBegId = [ncells]
    for sep in separatrices
        sz = length(sep.geometry_)
        npoints += sz
        ncells += max(0, sz - 1)
        push!(geomPointsBegId, npoints)
        push!(geomCellsBegId, ncells)
    end

    # resize arrays
    resize!(out.pt_points_, 3 * npoints)
    resize!(out.pt_smoothingMask_, npoints)
    resize!(out.pt_cellDimensions_, npoints)
    resize!(out.pt_cellIds_, npoints)
    resize!(out.cl_connectivity_, 2 * ncells)
    resize!(out.cl_sourceIds_, ncells)
    resize!(out.cl_destinationIds_, ncells)
    resize!(out.cl_separatrixIds_, ncells)
    resize!(out.cl_separatrixTypes_, ncells)
    resize!(out.cl_isOnBoundary_, ncells)
    resize!(out.cl_sepFuncMaxId_, sepIdStart + length(separatrices))
    resize!(out.cl_sepFuncMinId_, sepIdStart + length(separatrices))

    for (i, sep) in enumerate(separatrices)
        sepId = sepIdStart + (i - 1)
        src = sep.source_
        dst = sep.destination_
        sepGeom = sep.geometry_

        # separatrix type per TTK rule (no saddle connectors in 2D)
        sepType::Int8 = Int8(min(dst.dim, 1))

        # sep function extrema in terms of vertex ids
        gVerts = (get_cell_greater_vertex(src, mesh, offsets), get_cell_greater_vertex(dst, mesh, offsets))
        lVerts = (get_cell_lower_vertex(src, mesh, offsets), get_cell_lower_vertex(dst, mesh, offsets))
        sepFuncMax = offsets[gVerts[1]] >= offsets[gVerts[2]] ? gVerts[1] : gVerts[2]
        sepFuncMin = offsets[lVerts[1]] <= offsets[lVerts[2]] ? lVerts[1] : lVerts[2]
        out.cl_sepFuncMaxId_[sepId + 1] = sepFuncMax
        out.cl_sepFuncMinId_[sepId + 1] = sepFuncMin

        # boundary flag
        onBoundary::Int8 = Int8(is_boundary(src, mesh)) + Int8(is_boundary(dst, mesh))

        # fill points and cells
        for j in 1:length(sepGeom)
            cell = sepGeom[j]
            (px, py, pz) = cell_point(cell, mesh)
            k = geomPointsBegId[i] + j
            out.pt_points_[3 * k - 2] = px
            out.pt_points_[3 * k - 1] = py
            out.pt_points_[3 * k - 0] = pz
            out.pt_smoothingMask_[k] = (j == 1 || j == length(sepGeom)) ? UInt8(0) : UInt8(1)
            out.pt_cellDimensions_[k] = Int8(cell.dim)
            out.pt_cellIds_[k] = cell.id

            if j == 1
                continue
            end
            l = geomCellsBegId[i] + (j - 2) + 1
            out.cl_connectivity_[2 * l - 1] = k - 1
            out.cl_connectivity_[2 * l - 0] = k
            out.cl_sourceIds_[l] = src.id
            out.cl_destinationIds_[l] = dst.id
            out.cl_separatrixIds_[l] = sepId
            out.cl_separatrixTypes_[l] = sepType
            out.cl_isOnBoundary_[l] = onBoundary
        end
    end

    out.pt_numberOfPoints_ = npoints
    out.cl_numberOfCells_ = ncells
    return out
end

"""
Export Output1Separatrices to a simple legacy VTK PolyData (.vtk) with lines.
"""
function export_separatrices_vtk(out::Output1Separatrices, filepath::AbstractString)
    open(filepath, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "Separatrices")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS $(out.pt_numberOfPoints_) float")
        for i in 1:out.pt_numberOfPoints_
            x = out.pt_points_[3*i-2]; y = out.pt_points_[3*i-1]; z = out.pt_points_[3*i]
            println(io, "$(x) $(y) $(z)")
        end
        nlines = out.cl_numberOfCells_
        println(io, "LINES $nlines $(3*nlines)")
        for i in 1:nlines
            a = out.cl_connectivity_[2*i-1]
            b = out.cl_connectivity_[2*i]
            println(io, "2 $a $b")
        end
    end
    return filepath
end

"""
Build an MS network (nodes+edges) from critical point sets and computed separatrices.
Returns MSNetwork and a mapping from critical Cell(dim,id) to node id (0-based).
"""
function build_ms_network(
    mesh::TriangleMesh,
    gradient::GradientField,
    minima::Vector{Int},
    saddles::Vector{Int},
    maxima::Vector{Int},
    asc_seps::Vector{Separatrix},
    desc_seps::Vector{Separatrix},
)
    nodes = CriticalPointNode[]
    cellToNode = Dict{Tuple{Int,Int},Int}()

    nid = 0
    # minima (vertices)
    for v in minima
        c = Cell(0, v)
        push!(nodes, CriticalPointNode(nid, c, Int8(0), cell_point(c, mesh)))
        cellToNode[(0, v)] = nid
        nid += 1
    end
    # saddles (edges)
    for e in saddles
        c = Cell(1, e)
        push!(nodes, CriticalPointNode(nid, c, Int8(1), cell_point(c, mesh)))
        cellToNode[(1, e)] = nid
        nid += 1
    end
    # maxima (triangles)
    for t in maxima
        c = Cell(2, t)
        push!(nodes, CriticalPointNode(nid, c, Int8(2), cell_point(c, mesh)))
        cellToNode[(2, t)] = nid
        nid += 1
    end

    edges = NetworkEdge[]
    eid = 0
    # ascending edges (to maxima), type=1
    for sep in asc_seps
        srcCell = sep.source_; dstCell = sep.destination_
        hasSrc = haskey(cellToNode, (srcCell.dim, srcCell.id))
        hasDst = haskey(cellToNode, (dstCell.dim, dstCell.id))
        if !hasSrc || !hasDst
            continue
        end
        srcN = cellToNode[(srcCell.dim, srcCell.id)]
        dstN = cellToNode[(dstCell.dim, dstCell.id)]
        onBoundary = Int8(is_boundary(srcCell, mesh)) + Int8(is_boundary(dstCell, mesh))
        push!(edges, NetworkEdge(eid, Int8(1), srcN, dstN, sep.geometry_, onBoundary))
        eid += 1
    end
    # descending edges (to minima), type=0
    for sep in desc_seps
        srcCell = sep.source_; dstCell = sep.destination_
        hasSrc = haskey(cellToNode, (srcCell.dim, srcCell.id))
        hasDst = haskey(cellToNode, (dstCell.dim, dstCell.id))
        if !hasSrc || !hasDst
            continue
        end
        srcN = cellToNode[(srcCell.dim, srcCell.id)]
        dstN = cellToNode[(dstCell.dim, dstCell.id)]
        onBoundary = Int8(is_boundary(srcCell, mesh)) + Int8(is_boundary(dstCell, mesh))
        push!(edges, NetworkEdge(eid, Int8(0), srcN, dstN, sep.geometry_, onBoundary))
        eid += 1
    end

    return MSNetwork(nodes, edges), cellToNode
end


# =====================
#  Minimal smoke test
# =====================

"""
Minimal non-executed example of usage (smoke test template):

# assuming you have: mesh::TriangleMesh, gradient::GradientField
# and critical points lists (minima, saddles, maxima)

seps, seg = compute_surface_morse_smale(mesh, gradient, minima, saddles, maxima)

"""
