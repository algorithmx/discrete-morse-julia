"""
Test Surface Generator for Discrete Morse Theory

Generates triangulated surfaces with controlled critical points
to test the discrete gradient critical point detection algorithm.

Author: Generated for TTK algorithm testing
Date: October 2025
"""

using Printf
using LinearAlgebra

"""
Generate a mesh with sinusoidal landscape containing predictable critical points.
Creates a rectangular mesh with a combination of sine waves.
"""
function generate_sinusoidal_surface(nx::Int, ny::Int,
                                   width::Float64=10.0, height::Float64=10.0;
                                   frequency1::Float64=2π/width,
                                   frequency2::Float64=2π/height,
                                   amplitude1::Float64=1.0,
                                   amplitude2::Float64=0.5,
                                   phase_shift::Float64=π/4)

    # Create grid of vertices
    x_range = range(0, width, length=nx)
    y_range = range(0, height, length=ny)

    vertices = zeros(3, nx * ny)
    vertex_idx = 1

    for j in 1:ny
        for i in 1:nx
            x, y = x_range[i], y_range[j]

            # Compute height using combination of sine waves
            z = amplitude1 * sin(frequency1 * x) * sin(frequency1 * y) +
                amplitude2 * sin(frequency2 * x + phase_shift) * cos(frequency2 * y)

            vertices[:, vertex_idx] = [x, y, z]
            vertex_idx += 1
        end
    end

    # Create triangles (regular grid triangulation)
    triangles = Matrix{Int}(undef, 3, 2 * (nx-1) * (ny-1))
    tri_idx = 1

    for j in 1:(ny-1)
        for i in 1:(nx-1)
            # Vertex indices for current grid cell
            v1 = (j-1) * nx + i      # bottom-left
            v2 = (j-1) * nx + i + 1  # bottom-right
            v3 = j * nx + i          # top-left
            v4 = j * nx + i + 1      # top-right

            # Two triangles per grid cell (using 0-based indexing adjustment)
            triangles[:, tri_idx] = [v1-1, v2-1, v3-1]  # Triangle 1
            triangles[:, tri_idx + 1] = [v2-1, v4-1, v3-1]  # Triangle 2

            tri_idx += 2
        end
    end

    return vertices, triangles
end

"""
Generate a surface with isolated Gaussian bumps (creates multiple maxima).
"""
function generate_gaussian_bumps(nx::Int, ny::Int,
                               width::Float64=10.0, height::Float64=10.0;
                               bump_positions::Vector{Tuple{Float64,Float64}}=[(2.5, 2.5), (7.5, 7.5)],
                               bump_heights::Vector{Float64}=[2.0, 1.5],
                               bump_widths::Vector{Float64}=[1.0, 1.2])

    if length(bump_positions) != length(bump_heights) || length(bump_positions) != length(bump_widths)
        error("bump_positions, bump_heights, and bump_widths must have the same length")
    end

    # Create grid of vertices
    x_range = range(0, width, length=nx)
    y_range = range(0, height, length=ny)

    vertices = zeros(3, nx * ny)
    vertex_idx = 1

    for j in 1:ny
        for i in 1:nx
            x, y = x_range[i], y_range[j]

            # Start with base height
            z = 0.0

            # Add contribution from each Gaussian bump
            for (pos, height, width) in zip(bump_positions, bump_heights, bump_widths)
                dist_sq = (x - pos[1])^2 + (y - pos[2])^2
                z += height * exp(-dist_sq / (2 * width^2))
            end

            vertices[:, vertex_idx] = [x, y, z]
            vertex_idx += 1
        end
    end

    # Create triangles (same as sinusoidal surface)
    triangles = Matrix{Int}(undef, 3, 2 * (nx-1) * (ny-1))
    tri_idx = 1

    for j in 1:(ny-1)
        for i in 1:(nx-1)
            v1 = (j-1) * nx + i
            v2 = (j-1) * nx + i + 1
            v3 = j * nx + i
            v4 = j * nx + i + 1

            triangles[:, tri_idx] = [v1-1, v2-1, v3-1]
            triangles[:, tri_idx + 1] = [v2-1, v4-1, v3-1]

            tri_idx += 2
        end
    end

    return vertices, triangles
end

"""
Generate a saddle surface (hyperbolic paraboloid).
"""
function generate_saddle_surface(nx::Int, ny::Int,
                               width::Float64=4.0, height::Float64=4.0;
                               curvature::Float64=0.5)

    # Create grid of vertices
    x_range = range(-width/2, width/2, length=nx)
    y_range = range(-height/2, height/2, length=ny)

    vertices = zeros(3, nx * ny)
    vertex_idx = 1

    for j in 1:ny
        for i in 1:nx
            x, y = x_range[i], y_range[j]

            # Hyperbolic paraboloid: z = curvature * (x^2 - y^2)
            z = curvature * (x^2 - y^2)

            vertices[:, vertex_idx] = [x, y, z]
            vertex_idx += 1
        end
    end

    # Create triangles
    triangles = Matrix{Int}(undef, 3, 2 * (nx-1) * (ny-1))
    tri_idx = 1

    for j in 1:(ny-1)
        for i in 1:(nx-1)
            v1 = (j-1) * nx + i
            v2 = (j-1) * nx + i + 1
            v3 = j * nx + i
            v4 = j * nx + i + 1

            triangles[:, tri_idx] = [v1-1, v2-1, v3-1]
            triangles[:, tri_idx + 1] = [v2-1, v4-1, v3-1]

            tri_idx += 2
        end
    end

    return vertices, triangles
end

"""
Generate a complex surface with multiple types of critical points.
Combines multiple functions to create a rich topological landscape.
"""
function generate_complex_surface(nx::Int, ny::Int,
                                width::Float64=12.0, height::Float64=12.0)

    # Create grid of vertices
    x_range = range(0, width, length=nx)
    y_range = range(0, height, length=ny)

    vertices = zeros(3, nx * ny)
    vertex_idx = 1

    for j in 1:ny
        for i in 1:nx
            x, y = x_range[i], y_range[j]

            # Complex combination of functions
            # Base sinusoidal pattern
            z1 = 1.5 * sin(2π * x / width) * sin(2π * y / height)

            # Diagonal ridge
            z2 = 0.8 * sin(π * (x + y) / (width + height))

            # Local Gaussian bumps
            z3 = 1.2 * exp(-((x - width*0.3)^2 + (y - height*0.3)^2) / 2.0)
            z4 = 0.9 * exp(-((x - width*0.7)^2 + (y - height*0.7)^2) / 1.5)

            # Gaussian depression (negative bump)
            z5 = -0.7 * exp(-((x - width*0.5)^2 + (y - height*0.5)^2) / 1.0)

            # Saddle component
            z6 = 0.3 * ((x - width/2)^2 - (y - height/2)^2) / (width^2 + height^2)

            z = z1 + z2 + z3 + z4 + z5 + z6

            vertices[:, vertex_idx] = [x, y, z]
            vertex_idx += 1
        end
    end

    # Create triangles
    triangles = Matrix{Int}(undef, 3, 2 * (nx-1) * (ny-1))
    tri_idx = 1

    for j in 1:(ny-1)
        for i in 1:(nx-1)
            v1 = (j-1) * nx + i
            v2 = (j-1) * nx + i + 1
            v3 = j * nx + i
            v4 = j * nx + i + 1

            triangles[:, tri_idx] = [v1-1, v2-1, v3-1]
            triangles[:, tri_idx + 1] = [v2-1, v4-1, v3-1]

            tri_idx += 2
        end
    end

    return vertices, triangles
end

"""
Generate a torus surface embedded in 3D.
Creates a surface with the topology of a torus (genus 1).
"""
function generate_torus_surface(nu::Int, nv::Int,
                               major_radius::Float64=3.0, minor_radius::Float64=1.0)

    vertices = zeros(3, nu * nv)
    vertex_idx = 1

    for j in 1:nv
        v = 2π * (j-1) / (nv-1)

        for i in 1:nu
            u = 2π * (i-1) / (nu-1)

            # Parametric torus equations
            x = (major_radius + minor_radius * cos(v)) * cos(u)
            y = (major_radius + minor_radius * cos(v)) * sin(u)
            z = minor_radius * sin(v)

            vertices[:, vertex_idx] = [x, y, z]
            vertex_idx += 1
        end
    end

    # Create triangles
    triangles = Matrix{Int}(undef, 3, 2 * (nu-1) * (nv-1))
    tri_idx = 1

    for j in 1:(nv-1)
        for i in 1:(nu-1)
            v1 = (j-1) * nu + i
            v2 = (j-1) * nu + i + 1
            v3 = j * nu + i
            v4 = j * nu + i + 1

            triangles[:, tri_idx] = [v1-1, v2-1, v3-1]
            triangles[:, tri_idx + 1] = [v2-1, v4-1, v3-1]

            tri_idx += 2
        end
    end

    return vertices, triangles
end

"""
Save mesh to OBJ file format for visualization.
"""
function save_mesh_obj(vertices::Matrix{Float64}, triangles::Matrix{Int}, filename::String)
    open(filename, "w") do f
        # Write vertices
        for i in 1:size(vertices, 2)
            v = vertices[:, i]
            println(f, "v $(v[1]) $(v[2]) $(v[3])")
        end

        # Write faces (convert to 1-based indexing for OBJ format)
        for i in 1:size(triangles, 2)
            t = triangles[:, i] .+ 1  # Convert to 1-based indexing
            println(f, "f $(t[1]) $(t[2]) $(t[3])")
        end
    end
    println("Mesh saved to $filename")
end

"""
Test function to generate and save various test surfaces.
"""
function test_surface_generation()
    println("Generating Test Surfaces for Critical Point Analysis")
    println("=" ^ 55)

    # Test 1: Simple sinusoidal surface
    println("\n1. Generating sinusoidal surface...")
    verts1, tris1 = generate_sinusoidal_surface(20, 20)
    save_mesh_obj(verts1, tris1, "sinusoidal_surface.obj")
    println("   Expected: Multiple maxima, minima, and saddle points")

    # Test 2: Gaussian bumps
    println("\n2. Generating surface with Gaussian bumps...")
    verts2, tris2 = generate_gaussian_bumps(25, 25,
                                           bump_positions=[(3.0, 3.0), (7.0, 7.0), (5.0, 9.0)],
                                           bump_heights=[2.0, 1.5, 1.8],
                                           bump_widths=[1.0, 1.2, 0.8])
    save_mesh_obj(verts2, tris2, "gaussian_bumps.obj")
    println("   Expected: Multiple isolated maxima (the bumps)")

    # Test 3: Saddle surface
    println("\n3. Generating saddle surface...")
    verts3, tris3 = generate_saddle_surface(15, 15)
    save_mesh_obj(verts3, tris3, "saddle_surface.obj")
    println("   Expected: One saddle point at the center")

    # Test 4: Complex surface
    println("\n4. Generating complex surface...")
    verts4, tris4 = generate_complex_surface(30, 30)
    save_mesh_obj(verts4, tris4, "complex_surface.obj")
    println("   Expected: Multiple critical points of all types")

    # Test 5: Torus
    println("\n5. Generating torus surface...")
    verts5, tris5 = generate_torus_surface(20, 15)
    save_mesh_obj(verts5, tris5, "torus_surface.obj")
    println("   Expected: Toroidal topology (genus 1)")

    println("\n" * "=" ^ 55)
    println("All test surfaces generated and saved as OBJ files.")
    println("You can now use these surfaces to test the critical point detection algorithm.")

    return [(verts1, tris1), (verts2, tris2), (verts3, tris3), (verts4, tris4), (verts5, tris5)]
end

# Run test if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_surface_generation()
end