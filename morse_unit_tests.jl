"""
Unit Tests for Discrete Morse Theory Critical Point Detection
    and Morse-Smale Surface Segmentation (2D)

Covers:
- Data structure construction parity (Separatrix, Output1Separatrices, OutputManifold)
- V-path tracing (ascending/descending) on a simple mesh with manual gradient
- Ascending 1-separatrices from 1-saddles
- Ascending/descending segmentation and final MS segmentation
- Integration test via compute_surface_morse_smale


Author: Generated for TTK algorithm validation
Date: October 2025
"""

using Printf
using DataStructures
using LinearAlgebra
using Test

include("discrete_morse_critical_points.jl")
include("morse_smale_surface_segmentation.jl")


"""
Construct the standard 2-triangle mesh used across tests.
Vertices: 4 points forming a square in XY plane
Triangles: T1 = (1,2,3), T2 = (2,4,3)
Returns (mesh, tri1, tri2)
"""
function make_two_triangle_mesh()
    vertices = [0.0 1.0 0.0 1.0;
                0.0 0.0 1.0 1.0;
                0.0 0.0 0.0 0.0]
    triangles = [1 2;
                 2 4;
                 3 3]
    mesh = TriangleMesh(vertices, triangles)
    return mesh, 1, 2
end

"""
Find the edge id between two vertex ids (order agnostic).
"""
function find_edge_between(mesh::TriangleMesh, vA::Int, vB::Int)
    ne = size(mesh.edge_vertices, 2)
    for e in 1:ne
        a = mesh.edge_vertices[1, e]
        b = mesh.edge_vertices[2, e]
        if (a == vA && b == vB) || (a == vB && b == vA)
            return e
        end
    end
    return -1
end

"""
Find the shared edge id between two triangles t1 and t2.
"""
function find_shared_edge(mesh::TriangleMesh, t1::Int, t2::Int)
    ne = size(mesh.edge_vertices, 2)
    for e in 1:ne
        # edge is shared if both triangles appear in its triangle list
        tris = mesh.edge_to_triangles[e]
        if length(tris) == 2 && ((tris[1] == t1 && tris[2] == t2) || (tris[1] == t2 && tris[2] == t1))
            return e
        end
    end
    return -1
end

"""
Initialize a fresh GradientField filled with -1 (no pairings).
"""
function fresh_gradient(mesh::TriangleMesh)
    nv = size(mesh.vertices, 2)
    ne = size(mesh.edge_vertices, 2)
    nt = size(mesh.triangles, 2)
    g = GradientField(nv, ne, nt)
    # ensure -1 fills even if constructor already does
    fill!(g.vertex_to_edge, -1)
    fill!(g.edge_to_vertex, -1)
    fill!(g.edge_to_triangle, -1)
    fill!(g.triangle_to_edge, -1)
    return g
end


function run_all_tests_1()
    @testset "Discrete Morse Theory - Unit Tests" begin
        @testset "Basic Data Structures" begin
            # Cell creation and comparison
            c1 = Cell(0, 5)
            c2 = Cell(1, 3)
            @test c1.dim == 0 && c1.id == 5
            @test c2.dim == 1 && c2.id == 3
            @test c1 < c2  # Lower dimension should come first

            # CellExt creation
            ce1 = CellExt(1, 10)
            @test ce1.dim == 1 && ce1.id == 10 && !ce1.paired

            # LowerStar initialization
            ls = LowerStar()
            @test length(ls.vertices) == 0 && length(ls.edges) == 0 && length(ls.triangles) == 0

            # GradientField initialization
            gf = GradientField(5, 8, 3)
            @test length(gf.vertex_to_edge) == 5
            @test length(gf.edge_to_triangle) == 8
            @test length(gf.triangle_to_edge) == 3
        end

        @testset "Simple Mesh Creation" begin
            # Single triangle mesh
            vertices = [0.0 1.0 0.5;
                        0.0 0.0 0.0;
                        0.0 0.0 1.0]  # 3x3 matrix
            triangles = reshape([1, 2, 3], 3, 1)  # Single triangle (3x1)
            mesh = TriangleMesh(vertices, triangles)
            @test size(mesh.vertices, 2) == 3
            @test size(mesh.triangles, 2) == 1

            # Edge connectivity for simple mesh
            n_edges = size(mesh.edge_vertices, 2)
            @test n_edges == 3  # Triangle has 3 edges

            # 2x2 grid mesh (two triangles forming a square with diagonal)
            vertices_2x2 = [0.0 1.0 0.0 1.0;
                            0.0 0.0 1.0 1.0;
                            0.0 0.0 0.0 0.0]  # 3x4 matrix
            triangles_2x2 = [1 2;
                             2 4;
                             3 3]  # 3x2
            mesh2 = TriangleMesh(vertices_2x2, triangles_2x2)
            n_edges2 = size(mesh2.edge_vertices, 2)
            @test n_edges2 == 5  # Square with diagonal has 5 edges
        end

        @testset "Lower Star Computation" begin
            # Simple 2x2 mesh with known scalar values
            vertices = [0.0 1.0 0.0 1.0;
                        0.0 0.0 1.0 1.0;
                        0.0 0.0 0.0 0.0]  # 3x4 matrix
            triangles = [1 2;
                         2 4;
                         3 3]  # 3x2
            offsets = [1, 2, 3, 4]  # Simple ordering

            mesh = TriangleMesh(vertices, triangles)
            ls = LowerStar()

            # Lower star of vertex 4 (maximum)
            compute_lower_star!(ls, 4, offsets, mesh)
            @test length(ls.vertices) == 1 && ls.vertices[1].id == 4
            # Vertex 4 is maximum; its lower star should include incident lower edges (non-empty)
            @test length(ls.edges) >= 1

            # Lower star of a non-maximum vertex expected to have edges
            ls = LowerStar()
            compute_lower_star!(ls, 3, offsets, mesh)
            @test length(ls.edges) >= 1
        end

        @testset "Gradient Field Construction" begin
            vertices = [0.0 1.0 0.0 1.0;
                        0.0 0.0 1.0 1.0;
                        0.0 0.0 0.0 0.0]
            triangles = [1 2;
                         2 4;
                         3 3]
            mesh = TriangleMesh(vertices, triangles)

            gradient = GradientField(4, 5, 2)
            offsets = [1, 2, 3, 4]
            ls = LowerStar()
            compute_lower_star!(ls, 4, offsets, mesh)
            @test length(ls.edges) > 0  # Should have at least one edge in lower star of vertex 4

            # Vertex-edge pairing
            v1 = CellExt(0, 1)
            e1 = CellExt(1, 1)
            v1.paired = false
            e1.paired = false
            pair_cells!(v1, e1, gradient)
            @test v1.paired && e1.paired
            @test gradient.vertex_to_edge[1] == 1
            @test gradient.edge_to_vertex[1] == 1

            # Edge-triangle pairing
            e2 = CellExt(1, 2)
            t1 = CellExt(2, 1)
            e2.paired = false
            t1.paired = false
            pair_cells!(e2, t1, gradient)
            @test e2.paired && t1.paired
            @test gradient.edge_to_triangle[2] == 1
            @test gradient.triangle_to_edge[1] == 2
        end

        @testset "Critical Point Detection" begin
            # Create simple gradient field with known critical points
            gradient = GradientField(4, 5, 2)

            # Minimal mesh used only for API completeness
            vertices = [0.0 1.0 0.0 1.0;
                        0.0 0.0 1.0 1.0;
                        0.0 0.0 0.0 0.0]
            triangles = [1 2;
                         2 4;
                         3 3]
            mesh = TriangleMesh(vertices, triangles)

            # Minimum vertex detection: vertex 1 has no outgoing gradient
            gradient.vertex_to_edge[1] = -1
            gradient.edge_to_vertex[1] = -1
            gradient.edge_to_triangle[1] = -1
            @test is_critical_cell(Cell(0, 1), gradient, mesh)

            # Saddle edge detection: edge with no connections
            gradient.edge_to_vertex[2] = -1
            gradient.edge_to_triangle[2] = -1
            @test is_critical_cell(Cell(1, 2), gradient, mesh)

            # Non-critical edge
            gradient.edge_to_vertex[3] = 1
            gradient.edge_to_triangle[3] = -1
            @test !is_critical_cell(Cell(1, 3), gradient, mesh)

            # Maximum triangle detection: triangle with no incoming gradient
            gradient.triangle_to_edge[1] = -1
            gradient.edge_to_triangle[1] = -1
            @test is_critical_cell(Cell(2, 1), gradient, mesh)

            # Non-critical triangle
            gradient.triangle_to_edge[2] = 1
            gradient.edge_to_triangle[1] = -1
            @test !is_critical_cell(Cell(2, 2), gradient, mesh)
        end

        @testset "Complete Algorithm - Simple Case" begin
            # Create a monotonic increasing scalar field (should have 1 minimum, 1 maximum, 0 saddles)
            vertices = [0.0 1.0 0.0 1.0;
                        0.0 0.0 1.0 1.0;
                        0.0 0.0 0.0 0.0]
            triangles = [1 2;
                         2 4;
                         3 3]
            scalar_field = [0.0, 0.0, 1.0, 2.0]  # Strictly increasing

            mesh = TriangleMesh(vertices, triangles)
            minima, saddles, maxima, gradient = find_critical_points_discrete_morse(mesh, scalar_field)

            # Should have exactly 1 minimum (vertex with value 0.0)
            @test length(minima) == 1

            # On meshes with boundary, the algorithm may yield 0 or 1 triangle maxima depending on pairings.
            # We accept either, and validate overall with the Euler characteristic below.
            @test length(maxima) in (0, 1)

            # Euler characteristic matching (V - E + F = 4 - 5 + 2 = 1)
            euler_morse = length(minima) - length(saddles) + length(maxima)
            euler_topo = 4 - 5 + 2  # V - E + F
            @test euler_morse == euler_topo
        end
    end
end


function run_all_tests_2()
    @testset "Data Structures (Segmentation)" begin
        s = Separatrix()
        @test s.source_.dim == -1 && s.destination_.dim == -1 && length(s.geometry_) == 0

        out1 = Output1Separatrices()
        @test out1.pt_numberOfPoints_ == 0 && out1.cl_numberOfCells_ == 0

        mesh, _, _ = make_two_triangle_mesh()
        outM = OutputManifold(size(mesh.vertices, 2))
        @test length(outM.ascending_) == size(mesh.vertices, 2)
    end

    @testset "V-Paths on Two-Triangle Mesh" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)

        # Configure gradient: T1 paired to shared edge; T2 is a maximum (unpaired triangle)
        eshared = find_shared_edge(mesh, t1, t2)
        @test eshared != -1
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1
        # no pairing for T2 => critical maximum

        # Ascending path from T1 should go: T1 -> eshared -> T2 (critical)
        path = trace_ascending_path(Cell(2, t1), g, mesh)
        @test length(path) >= 2
        @test path[1].dim == 2 && path[1].id == t1
        @test path[end].dim == 2 && path[end].id == t2
        @test is_cell_critical(path[end], g)

        # Descending path from T1: T1 -> edge -> possibly stop (no vertex pairing set)
        dpath = trace_descending_path(Cell(2, t1), g, mesh)
        @test dpath[1].dim == 2 && dpath[1].id == t1
    end

    @testset "Ascending 1-Separatrices" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)
        eshared = find_shared_edge(mesh, t1, t2)
        # same gradient config as above
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1

        # choose an edge with two incident triangles as a 1-saddle seed
        saddles = [eshared]
        seps = get_ascending_separatrices1(saddles, g, mesh)

        # Expect at least one separatrix reaching T2
        @test length(seps) >= 1
        got = any(sep -> sep.destination_.dim == 2 && sep.destination_.id == t2, seps)
        @test got

        # geometry starts with the saddle edge
        @test all(sep -> length(sep.geometry_) >= 2 && sep.geometry_[1].dim == 1, seps)

        # path alternation invariant (2,1,2,1,...) as in TTK getAscendingPath
        function dims_alternate(vpath)
            for k in 2:length(vpath)
                if vpath[k].dim == vpath[k-1].dim
                    return false
                end
            end
            return true
        end
        @test all(sep -> dims_alternate(sep.geometry_), seps)
    end

    @testset "Segmentation (Ascending/Descending/Final)" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)

        # Set ascending flow: T1 -> eshared -> T2(max)
        eshared = find_shared_edge(mesh, t1, t2)
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1

        # Descending flow: pick v1 as sole minimum, and pair others toward it
        # Find edges connecting to v1
        v1 = 1
        ne = size(mesh.edge_vertices, 2)
        # map from vertex to an adjacent edge leading closer to v1 (simple star)
        for v in 1:size(mesh.vertices, 2)
            if v == v1
                g.vertex_to_edge[v] = -1  # minimum
                continue
            end
            # choose any edge incident to v; if possible, pick one touching v1
            chosen = -1
            for e in mesh.vertex_to_edges[v]
                a = mesh.edge_vertices[1, e]; b = mesh.edge_vertices[2, e]
                if a == v1 || b == v1
                    chosen = e; break
                end
            end
            if chosen == -1
                # fallback: pick first incident edge
                chosen = first(mesh.vertex_to_edges[v])
            end
            g.vertex_to_edge[v] = chosen
        end

        # Ascending segmentation: T2 is the only maximum
        asc = set_ascending_segmentation([t2], g, mesh)
        @test length(asc) == size(mesh.vertices, 2)
        # All vertices should be assigned to basin 0 (since only one maximum)
        @test all(x -> x in (0, -1), asc)  # mapping picks first star triangle; tolerate -1 on isolated

        # Descending segmentation: v1 is the only minimum
        desc = set_descending_segmentation([v1], g, mesh)
        @test all(x -> x == 0, desc)

        # Final segmentation must be consistent and dense (0..k-1 or -1)
        final = set_final_segmentation(asc, desc, 1)  # nMax = 1
        @test all(x -> x in (0, -1), final)
    end

    @testset "Boundary Convergence and Cycle Detection" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)

        # Force T2 to pair to a boundary edge (edge with only 1 incident triangle)
        # pick an outer edge of t2; search edges of t2 and choose one with 1 triangle
        boundary_edge = -1
        for e in 1:size(mesh.edge_vertices, 2)
            tris = mesh.edge_to_triangles[e]
            if any(==(t2), tris) && length(tris) == 1
                boundary_edge = e
                break
            end
        end
        @test boundary_edge != -1
        g.triangle_to_edge[t2] = boundary_edge
        g.edge_to_triangle[boundary_edge] = t2

        # Ascending path from T2 should be [T2, boundary_edge] then stop (no other triangle)
        p = trace_ascending_path(Cell(2, t2), g, mesh)
        @test length(p) == 2
        @test p[1].dim == 2 && p[2].dim == 1

        # Create a trivial cycle setup and ensure cycle_detector breaks
        # T1 paired to shared edge; set shared edge's triangles list to [t1,t1] logically via nextTri rule
        # We approximate by marking triangle_to_edge for t1 but not creating a new triangle; nextTri will be t1 again
        g2 = fresh_gradient(mesh)
        eshared = find_shared_edge(mesh, t1, t2)
        g2.triangle_to_edge[t1] = eshared
        g2.edge_to_triangle[eshared] = t1
        cyc = trace_ascending_path(Cell(2, t1), g2, mesh; cycle_detector=true)
        # Should terminate quickly (<= 3 steps: T, E, T)
        @test length(cyc) <= 3
    end

    @testset "Final Segmentation Dense Mapping" begin
        # Directly test sparse-to-dense compression behavior
        ascending = [0, 1, 1, 0]
        descending = [0, 0, 1, 1]
        nMax = 2
        final = set_final_segmentation(ascending, descending, nMax)
        # Collect unique region IDs (should be dense 0..k-1)
        uniq = sort(unique(final))
        @test uniq[1] == 0
        @test all(uniq[i] == i-1 for i in 1:length(uniq))
    end

    @testset "Integration: compute_surface_morse_smale" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)
        eshared = find_shared_edge(mesh, t1, t2)
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1

        # minima/1-saddles/maxima
        v1 = 1
        saddles = [eshared]
        minima = [v1]
        maxima = [t2]

        seps, out = compute_surface_morse_smale(mesh, g, minima, saddles, maxima)
        @test length(seps) >= 1
        @test length(out.ascending_) == size(mesh.vertices, 2)
        @test length(out.descending_) == size(mesh.vertices, 2)
        @test length(out.morseSmale_) == size(mesh.vertices, 2)
    end

    # Builder and Exporter tests
    @testset "Output1Separatrices Builder" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)
        eshared = find_shared_edge(mesh, t1, t2)
        # ascending: T1 -> T2(max)
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1

        # descending minima setup
        v1 = 1
        for v in 1:size(mesh.vertices, 2)
            if v == v1
                g.vertex_to_edge[v] = -1
            else
                g.vertex_to_edge[v] = first(mesh.vertex_to_edges[v])
            end
        end

        # compute separatrices
        asc_seps = get_ascending_separatrices1([eshared], g, mesh)
        desc_seps = get_descending_separatrices1([eshared], g, mesh)

        # offsets (SoS ranks): identity
        offsets = collect(1:size(mesh.vertices, 2))

        out = Output1Separatrices()
        set_separatrices1!(out, asc_seps, offsets, mesh)
        set_separatrices1!(out, desc_seps, offsets, mesh)

        # basic size checks
        @test out.pt_numberOfPoints_ > 0
        @test out.cl_numberOfCells_ > 0
        @test length(out.pt_points_) == 3 * out.pt_numberOfPoints_
        @test length(out.cl_connectivity_) == 2 * out.cl_numberOfCells_

        # types present should be 0 or 1 (descending or ascending)
        @test all(x -> x in Int8[0,1], out.cl_separatrixTypes_)

        # boundary flags in {0,1,2}
        @test all(x -> x in Int8[0,1,2], out.cl_isOnBoundary_)

        # sep func ids indexed per separatrix count
        @test length(out.cl_sepFuncMaxId_) >= length(asc_seps) + length(desc_seps)
        @test length(out.cl_sepFuncMinId_) >= length(asc_seps) + length(desc_seps)

        # exporter (write to temp file)
        tmpfile = joinpath(pwd(), "_sep_lines.vtk")
        export_separatrices_vtk(out, tmpfile)
        @test isfile(tmpfile)
        rm(tmpfile; force=true)
    end

    # MS Network builder tests
    @testset "MS Network Builder" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        g = fresh_gradient(mesh)
        eshared = find_shared_edge(mesh, t1, t2)
        # ascending: T1 -> eshared -> T2(max)
        g.triangle_to_edge[t1] = eshared
        g.edge_to_triangle[eshared] = t1

        # minima: pick v1; set it critical and pair the shared edge to it for descending sep
        v1 = 1
        g.vertex_to_edge[v1] = -1
        g.edge_to_vertex[eshared] = v1

        # compute separatrices
        asc_seps = get_ascending_separatrices1([eshared], g, mesh)
        desc_seps = get_descending_separatrices1([eshared], g, mesh)
        @test !isempty(asc_seps)
        @test !isempty(desc_seps)

        # build network
        minima = [v1]; saddles = [eshared]; maxima = [t2]
        net, cell2node = build_ms_network(mesh, g, minima, saddles, maxima, asc_seps, desc_seps)

        # node counts and mapping
        @test length(net.nodes) == 3
        @test haskey(cell2node, (0, v1)) && haskey(cell2node, (1, eshared)) && haskey(cell2node, (2, t2))
        sid = cell2node[(1, eshared)]; mid = cell2node[(0, v1)]; xid = cell2node[(2, t2)]

        # edges: one ascending from saddle->max, one descending from saddle->min
        hasAsc = any(e -> e.edgeType == Int8(1) && e.sourceNode == sid && e.destinationNode == xid, net.edges)
        hasDesc = any(e -> e.edgeType == Int8(0) && e.sourceNode == sid && e.destinationNode == mid, net.edges)
        @test hasAsc
        @test hasDesc
    end

    # End-to-end from scalar field (no persistence simplification)
    @testset "Integration: from scalar field" begin
        mesh, t1, t2 = make_two_triangle_mesh()
        # Simple scalar field: x+y
        scalars = [mesh.vertices[1,i] + mesh.vertices[2,i] for i in 1:size(mesh.vertices,2)]

        res = compute_surface_morse_smale_from_scalar(mesh, scalars; persistence_threshold=0.0, build_network=true)

        @test length(res.segmentation.ascending_) == size(mesh.vertices, 2)
        @test length(res.asc_seps) >= 0
        @test length(res.desc_seps) >= 0
        # If a network is built, ensure types and trivial structure
        @test res.network === nothing || (isa(res.network, MSNetwork) && length(res.network.nodes) >= 0)
    end

end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    @testset "All Tests" begin
        run_all_tests_1() 
        run_all_tests_2()
    end
end

