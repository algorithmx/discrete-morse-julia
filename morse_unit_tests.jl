"""
Unit Tests for Discrete Morse Theory Critical Point Detection

This file provides comprehensive unit tests for the ProcessLowerStars algorithm
implementation, testing each component systematically before integration.

Author: Generated for TTK algorithm validation
Date: October 2025
"""

using Printf
using DataStructures
using LinearAlgebra
using Test

# Include the main implementation
include("discrete_morse_critical_points.jl")

"""
Test Suite 1: Basic Data Structures
"""
function test_data_structures()
    println("\n" * "="^50)
    println("TEST SUITE 1: BASIC DATA STRUCTURES")
    println("="^50)

    passed = 0
    total = 0

    # Test Cell creation and comparison
    total += 1
    c1 = Cell(0, 5)
    c2 = Cell(1, 3)
    if c1.dim == 0 && c1.id == 5 && c2.dim == 1 && c2.id == 3
        println("✓ Cell creation: PASSED")
        passed += 1
    else
        println("✗ Cell creation: FAILED")
    end

    # Test Cell comparison
    total += 1
    if c1 < c2  # Lower dimension should come first
        println("✓ Cell comparison: PASSED")
        passed += 1
    else
        println("✗ Cell comparison: FAILED")
    end

    # Test CellExt creation
    total += 1
    ce1 = CellExt(1, 10)
    if ce1.dim == 1 && ce1.id == 10 && !ce1.paired
        println("✓ CellExt creation: PASSED")
        passed += 1
    else
        println("✗ CellExt creation: FAILED")
    end

    # Test LowerStar
    total += 1
    ls = LowerStar()
    if length(ls.vertices) == 0 && length(ls.edges) == 0 && length(ls.triangles) == 0
        println("✓ LowerStar initialization: PASSED")
        passed += 1
    else
        println("✗ LowerStar initialization: FAILED")
    end

    # Test GradientField
    total += 1
    gf = GradientField(5, 8, 3)
    if length(gf.vertex_to_edge) == 5 && length(gf.edge_to_triangle) == 8 && length(gf.triangle_to_edge) == 3
        println("✓ GradientField initialization: PASSED")
        passed += 1
    else
        println("✗ GradientField initialization: FAILED")
    end

    println("\nData Structure Tests: $passed/$total passed")
    return passed == total
end

"""
Test Suite 2: Simple Mesh Creation
"""
function test_simple_mesh_creation()
    println("\n" * "="^50)
    println("TEST SUITE 2: SIMPLE MESH CREATION")
    println("="^50)

    passed = 0
    total = 0

    # Create simplest possible mesh - single triangle
    total += 1
    vertices = [0.0 1.0 0.5;
                0.0 0.0 0.0;
                0.0 0.0 1.0]  # 3x3 matrix
    triangles = reshape([1, 2, 3], 3, 1)  # Single triangle (3x1)
    try
        mesh = TriangleMesh(vertices, triangles)
        if size(mesh.vertices, 2) == 3 && size(mesh.triangles, 2) == 1
            println("✓ Single triangle mesh: PASSED")
            passed += 1
        else
            println("✗ Single triangle mesh: FAILED - wrong dimensions")
        end
    catch e
        println("✗ Single triangle mesh: FAILED - Exception: $e")
    end

    # Test edge connectivity for simple mesh
    total += 1
    try
        mesh = TriangleMesh(vertices, triangles)
        n_edges = size(mesh.edge_vertices, 2)
        if n_edges == 3  # Triangle has 3 edges
            println("✓ Edge count for single triangle: PASSED (3 edges)")
            passed += 1
        else
            println("✗ Edge count for single triangle: FAILED (got $n_edges, expected 3)")
        end
    catch e
        println("✗ Edge count test: FAILED - Exception: $e")
    end

    # Test 2x2 grid mesh
    total += 1
    vertices_2x2 = [0.0 1.0 0.0 1.0;
                    0.0 0.0 1.0 1.0;
                    0.0 0.0 0.0 0.0]  # 3x4 matrix
    triangles_2x2 = [1 2;
                     2 4;
                     3 3]  # 3x2: two triangles forming a square with diagonal
    try
        mesh = TriangleMesh(vertices_2x2, triangles_2x2)
        n_edges = size(mesh.edge_vertices, 2)
        if n_edges == 5  # Square with diagonal has 5 edges
            println("✓ 2x2 grid mesh: PASSED (5 edges)")
            passed += 1
        else
            println("✗ 2x2 grid mesh: FAILED (got $n_edges, expected 5)")
        end
    catch e
        println("✗ 2x2 grid mesh: FAILED - Exception: $e")
    end

    println("\nMesh Creation Tests: $passed/$total passed")
    return passed == total
end

"""
Test Suite 3: Lower Star Computation
"""
function test_lower_star_computation()
    println("\n" * "="^50)
    println("TEST SUITE 3: LOWER STAR COMPUTATION")
    println("="^50)

    passed = 0
    total = 0

    # Create simple 2x2 mesh with known scalar values
        vertices = [0.0 1.0 0.0 1.0;
                    0.0 0.0 1.0 1.0;
                    0.0 0.0 0.0 0.0]  # 3x4 matrix
        triangles = [1 2;
                     2 4;
                     3 3]  # 3x2: two triangles forming a square with diagonal

    # Scalar field: vertex 1 is minimum (0.0), vertex 4 is maximum (2.0)
    scalar_field = [0.0, 0.0, 1.0, 2.0]  # Heights
    offsets = [1, 2, 3, 4]  # Simple ordering

    try
        mesh = TriangleMesh(vertices, triangles)
        ls = LowerStar()

        # Test lower star of vertex 4 (maximum)
        total += 1
        compute_lower_star!(ls, 4, offsets, mesh)
        if length(ls.vertices) == 1 && ls.vertices[1].id == 4
            println("✓ Lower star vertex creation: PASSED")
            passed += 1
        else
            println("✗ Lower star vertex creation: FAILED")
        end

        # Test lower star contains correct structure
        total += 1
        # Vertex 4 is maximum; its lower star should include incident lower edges (non-empty)
        if length(ls.edges) >= 1
            println("✓ Maximum vertex lower star: PASSED (has lower edges)")
            passed += 1
        else
            println("✗ Maximum vertex lower star: FAILED (expected edges)")
        end

        # Test lower star of a non-maximum vertex expected to have edges
        total += 1
        ls = LowerStar()
        compute_lower_star!(ls, 3, offsets, mesh)
        if length(ls.edges) >= 1  # Should have incident lower edges
            println("✓ Non-maximum vertex lower star: PASSED (has edges)")
            passed += 1
        else
            println("✗ Non-maximum vertex lower star: FAILED (no edges found)")
        end

    catch e
        println("✗ Lower star computation: FAILED - Exception: $e")
    end

    println("\nLower Star Tests: $passed/$total passed")
    return passed == total
end

"""
Test Suite 4: Gradient Field Construction
"""
function test_gradient_field_construction()
    println("\n" * "="^50)
    println("TEST SUITE 4: GRADIENT FIELD CONSTRUCTION")
    println("="^50)

    passed = 0
    total = 0

    # Test simple case where gradient pairing should be deterministic
    vertices = [0.0 1.0 0.0 1.0;
                0.0 0.0 1.0 1.0;
                0.0 0.0 0.0 0.0]
    triangles = [1 2;
                 2 4;
                 3 3]
    scalar_field = [0.0, 0.0, 1.0, 2.0]

    try
        mesh = TriangleMesh(vertices, triangles)
    gradient = GradientField(4, 5, 2)

        # Simple manual pairing: vertex 1 should pair with edge to vertex 2 (steepest)
        total += 1
        offsets = [1, 2, 3, 4]  # Simple ordering
        ls = LowerStar()
        compute_lower_star!(ls, 4, offsets, mesh)

        if length(ls.edges) > 0  # Should have at least one edge in lower star of vertex 4
            println("✓ Gradient field setup: PASSED")
            passed += 1
        else
            println("✗ Gradient field setup: FAILED (no edges found)")
        end

        # Test that gradient pairing function works
        total += 1
        v1 = CellExt(0, 1)
        e1 = CellExt(1, 1)
        v1.paired = false
        e1.paired = false
        pair_cells!(v1, e1, gradient)

        if v1.paired && e1.paired && gradient.vertex_to_edge[1] == 1 && gradient.edge_to_vertex[1] == 1
            println("✓ Vertex-edge pairing: PASSED")
            passed += 1
        else
            println("✗ Vertex-edge pairing: FAILED")
        end

        # Test edge-triangle pairing
        total += 1
        e2 = CellExt(1, 2)
        t1 = CellExt(2, 1)
        e2.paired = false
        t1.paired = false
        pair_cells!(e2, t1, gradient)

        if e2.paired && t1.paired && gradient.edge_to_triangle[2] == 1 && gradient.triangle_to_edge[1] == 2
            println("✓ Edge-triangle pairing: PASSED")
            passed += 1
        else
            println("✗ Edge-triangle pairing: FAILED")
        end

    catch e
        println("✗ Gradient field construction: FAILED - Exception: $e")
    end

    println("\nGradient Field Tests: $passed/$total passed")
    return passed == total
end

"""
Test Suite 5: Critical Point Detection Logic
"""
function test_critical_point_detection()
    println("\n" * "="^50)
    println("TEST SUITE 5: CRITICAL POINT DETECTION")
    println("="^50)

    passed = 0
    total = 0

    try
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

        # Set up known state: vertex 1 is minimum (no outgoing gradient)
        total += 1
        gradient.vertex_to_edge[1] = -1
        gradient.edge_to_vertex[1] = -1
        gradient.edge_to_triangle[1] = -1

        if is_critical_cell(Cell(0, 1), gradient, mesh)
            println("✓ Minimum vertex detection: PASSED")
            passed += 1
        else
            println("✗ Minimum vertex detection: FAILED")
        end

        # Set up saddle: edge with no connections
        total += 1
        gradient.edge_to_vertex[2] = -1
        gradient.edge_to_triangle[2] = -1

        if is_critical_cell(Cell(1, 2), gradient, mesh)
            println("✓ Saddle edge detection: PASSED")
            passed += 1
        else
            println("✗ Saddle edge detection: FAILED")
        end

        # Set up non-critical edge
        total += 1
        gradient.edge_to_vertex[3] = 1
        gradient.edge_to_triangle[3] = -1

        if !is_critical_cell(Cell(1, 3), gradient, mesh)
            println("✓ Non-critical edge detection: PASSED")
            passed += 1
        else
            println("✗ Non-critical edge detection: FAILED")
        end

        # Set up maximum: triangle with no incoming gradient
        total += 1
        gradient.triangle_to_edge[1] = -1
        gradient.edge_to_triangle[1] = -1

        if is_critical_cell(Cell(2, 1), gradient, mesh)
            println("✓ Maximum triangle detection: PASSED")
            passed += 1
        else
            println("✗ Maximum triangle detection: FAILED")
        end

        # Set up non-critical triangle
        total += 1
        gradient.triangle_to_edge[2] = 1
        gradient.edge_to_triangle[1] = -1

        if !is_critical_cell(Cell(2, 2), gradient, mesh)
            println("✓ Non-critical triangle detection: PASSED")
            passed += 1
        else
            println("✗ Non-critical triangle detection: FAILED")
        end

    catch e
        println("✗ Critical point detection: FAILED - Exception: $e")
    end

    println("\nCritical Point Detection Tests: $passed/$total passed")
    return passed == total
end

"""
Test Suite 6: Complete Algorithm on Simple Case
"""
function test_complete_algorithm_simple()
    println("\n" * "="^50)
    println("TEST SUITE 6: COMPLETE ALGORITHM - SIMPLE CASE")
    println("="^50)

    passed = 0
    total = 0

    try
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

        total += 1
        # Should have exactly 1 minimum (vertex with value 0.0)
        if length(minima) == 1
            println("✓ Single minimum detected: PASSED ($(length(minima)) found)")
            passed += 1
        else
            println("✗ Single minimum detected: FAILED ($(length(minima)) found, expected 1)")
        end

        total += 1
        # On meshes with boundary, the algorithm may yield 0 or 1 triangle maxima depending on pairings.
        # We accept either, and validate overall with the Euler characteristic below.
        if length(maxima) in (0, 1)
            println("✓ Maximum count acceptable: PASSED ($(length(maxima)) found)")
            passed += 1
        else
            println("✗ Maximum count acceptable: FAILED ($(length(maxima)) found, expected 0 or 1)")
        end

        total += 1
        # Should have Euler characteristic matching (V - E + F = 4 - 5 + 2 = 1)
        euler_morse = length(minima) - length(saddles) + length(maxima)
        euler_topo = 4 - 5 + 2  # V - E + F
        if euler_morse == euler_topo
            println("✓ Euler characteristic: PASSED ($euler_morse = $euler_topo)")
            passed += 1
        else
            println("✗ Euler characteristic: FAILED ($euler_morse ≠ $euler_topo)")
        end

    catch e
        println("✗ Complete algorithm test: FAILED - Exception: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
        end
    end

    println("\nComplete Algorithm Tests: $passed/$total passed")
    return passed == total
end

"""
Main Test Runner
"""
function run_all_tests()
    println("RUNNING COMPREHENSIVE UNIT TESTS FOR DISCRETE MORSE THEORY")
    println("="^70)

    test_results = []

    # Run all test suites
    push!(test_results, ("Data Structures", test_data_structures()))
    push!(test_results, ("Mesh Creation", test_simple_mesh_creation()))
    push!(test_results, ("Lower Star Computation", test_lower_star_computation()))
    push!(test_results, ("Gradient Field Construction", test_gradient_field_construction()))
    push!(test_results, ("Critical Point Detection", test_critical_point_detection()))
    push!(test_results, ("Complete Algorithm", test_complete_algorithm_simple()))

    # Summary
    println("\n" * "="^70)
    println("UNIT TEST SUMMARY")
    println("="^70)

    total_passed = 0
    total_tests = length(test_results)

    for (test_name, result) in test_results
        if result
            total_passed += 1
            println("✓ $test_name: PASSED")
        else
            println("✗ $test_name: FAILED")
        end
    end

    println("\nOverall Results: $total_passed/$total_tests test suites passed")

    if total_passed == total_tests
        println("🎉 ALL UNIT TESTS PASSED - Implementation appears correct!")
        return true
    else
        println("❌ SOME UNIT TESTS FAILED - Implementation needs fixes!")
        return false
    end
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = run_all_tests()
    exit(success ? 0 : 1)
end