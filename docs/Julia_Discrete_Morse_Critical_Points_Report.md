# Comparative Report: Julia Implementation of Discrete Morse Critical Point Search vs. TTK C++ DiscreteGradient

## Overview

- Goal: The Julia file `discrete_morse_critical_points.jl` implements 2D discrete Morse gradient construction with critical point detection using the ProcessLowerStars algorithm, then classifies minima/saddles/maxima.
- Scope: 2D triangle surface meshes only (0-, 1-, 2-cells). No 3D (tetrahedra) support.
- Reference: The C++ reference is the TTK `ttk::dcg::DiscreteGradient` ProcessLowerStars implementation as summarized in `ttkDiscreteGradient_Architecture_Analysis.md`.

High-level verdict: The Julia code captures the core ProcessLowerStars mechanics and critical point detection for 2D surfaces, with correct core pairing logic and lower-star construction. However, there are notable differences that can change ordering/tie-breaking and performance characteristics, and some features from TTK are not implemented (3D, preconditioning, parallelism, boundary/meta outputs, path extraction, persistence/simplification). With a few adjustments, compliance can be tightened.

## Data structures and their alignment

- Cell/CellExt:
  - Julia: `Cell` (dim, id) and `CellExt` (dim, id, lower_verts, faces, paired).
  - C++: Similar `Cell` and `CellExt` with lower-star-local metadata.
  - Compliance: Good. Julia’s `lower_verts` encodes the “lower” vertex orders (offsets). For edges, it stores one rank; for triangles, two ranks sorted descending. This matches the TTK intent for lexicographic ordering.

- Lower star (per vertex):
  - Julia: `LowerStar` holds vectors for vertices, edges, triangles (tetrahedra vector exists but unused).
  - C++: A 4-slot array of vectors by dimension.
  - Compliance: Good for 2D. Storage is per-dimension with local per-vertex lifetime.

- Gradient storage:
  - Julia: `GradientField` with 4 arrays for 2D: vertex_to_edge, edge_to_vertex, edge_to_triangle, triangle_to_edge, filled with -1 to indicate unpaired.
  - C++: `gradient_[2*dim][id]` with -1 for unpaired.
  - Compliance: Good for 2D. No 3D arrays.

- Mesh connectivity:
  - Julia: `TriangleMesh` builds unique edges, `edge_vertices`, `vertex_to_edges`, `edge_to_triangles`; triangle list is 3×n (1-based).
  - C++: Uses triangulation interface with rich preconditioning (stars, links, neighbors).
  - Compliance: Functionally OK, but Julia recomputes lower stars by scanning all triangles to find those containing the vertex (O(n_tris) per vertex), whereas TTK uses preconditioned vertex stars. This impacts performance and scalability.

## Offsets, total order, and tie-breaking

- Julia computes `offsets` as rank of scalar values via `sortperm`, assigning unique integers 1..n. This induces a strict total order across vertices. Lower-star conditions check `offsets[neighbor] < offsets[a]`.
- TTK uses offsets to enforce a strict total order as well (often derived from scalar + secondary offset to break ties).
- Compliance: Good. The Julia approach yields a deterministic total order. If scalar values have ties, `sortperm` still produces a total order, but it’s not explicitly tied to a second “offset array” as in TTK. For practical purposes this matches the intent.

## Lower star construction

- Edges:
  - Julia includes any incident edge (a, b) where rank(b) < rank(a), storing the neighbor’s rank in `lower_verts = (rank(b), -1, -1)`.
  - TTK logic is equivalent: edges in the lower star have all vertices lower than ‘a’, which for edges is just the other vertex.

- Triangles:
  - Julia builds triangles containing ‘a’ and sets `low_verts` to the ranks of the two other vertices, then sorts them descending so `low_verts[1] ≥ low_verts[2]`, and includes the triangle if `rank(a) > low_verts[1]`. It also finds the two lower-star edge positions to store in `faces` as indices into `ls.edges`.
  - TTK does the same conceptually, including faces to support fast “unpaired faces” counting.
  - Compliance: Good. The details of lexicographic sorting and the “max equals a” criterion are consistent.

- Performance note: Julia scans all triangles to find those containing ‘a’. TTK uses vertex-star adjacency. This is a performance deviation, not a correctness issue.

## Priority queues and processing loop

- Queues:
  - Julia uses two binary heaps:
    - `pq_zero`: cells with zero unpaired faces (initially all edges except the steepest one are inserted).
    - `pq_one`: cells with exactly one unpaired face (populated via `insert_cofacets`).
  - Ordering:
    - Julia defines a custom ordering `LowerVertsOrdering` with `lt(a,b) = a.lower_verts < b.lower_verts` and constructs `BinaryHeap{CellExt}(LV_ORD)`, which yields a min-heap by that ordering.
    - In TTK, `std::priority_queue` with `std::less` and their `operator<` often results in a max-heap behavior (top is lexicographically greatest unless they invert the comparator).
  - Compliance and impact:
    - The exact heap polarity (min vs max) affects processing order but the ProcessLowerStars algorithm is designed to be order-independent in terms of producing a valid discrete gradient, given a strict total order on vertices. However, TTK’s implementation is deterministic and may rely on a specific queue order to maintain exact reproducibility. Julia’s different heap polarity could yield different—but still valid—pairings, i.e., not bit-for-bit identical to TTK.
    - Recommendation: If exact compliance/reproducibility versus TTK is desired, align the heap “top” semantics with TTK (max-heap on the same lexicographic comparator), or explicitly document the minor divergence.

- Initialization and pairing:
  - Julia pairs the pivot vertex with the steepest lower edge (the one with minimal neighbor rank), matching TTK’s “delta” logic.
  - It then pushes other edges into `pq_zero` and cofacets of delta with one unpaired face into `pq_one`.
  - While `pq_one` non-empty: pop triangle, count unpaired faces; if 1, pair its unique unpaired edge with it; if 0, move to `pq_zero`.
  - While `pq_zero` non-empty: pop gamma; if it’s already paired, skip; else mark as critical in the local processing (Julia sets `gamma.paired = true`) and insert its cofacets into `pq_one`.
  - Compliance: The control flow mirrors ProcessLowerStars. The key method `count_unpaired_faces_triangle` and face bookkeeping are implemented as in TTK (using triangle.faces indices into the lower-star edge array and the `paired` markers on local `CellExt` edges).

- Marking “paired” for gamma from pq_zero:
  - Julia sets `gamma.paired = true` when popped from `pq_zero` to prevent reprocessing; they do not write any gradient pair for gamma (i.e., it becomes a lower-star-local critical cell).
  - TTK also treats such cells as critical in the lower star. Using the local flag to avoid re-queueing is fine; final criticality is determined from the global `gradient` arrays, not these local flags.
  - Compliance: Acceptable. Does not affect final gradient correctness because global gradient arrays are the single source of truth.

## Pairing logic and gradient storage

- Julia’s `pair_cells!`:
  - Marks both local CellExts as `.paired = true`.
  - Writes the gradient pair into the correct arrays:
    - vertex->edge and edge->vertex
    - edge->triangle and triangle->edge
  - There’s no out-of-range or 0-based indexing risk because all IDs in this mesh are 1-based and arrays are 1-based; `-1` is used as the sentinel value, not as an index.
- TTK uses the same logic in its `gradient_[2*dim + {0,1}]` arrays with -1 sentinel.
- Compliance: Good.

## Critical point detection and classification

- Julia’s `is_critical_cell`:
  - Vertex critical if vertex_to_edge[id] == -1.
  - Edge critical if both edge_to_vertex[id] and edge_to_triangle[id] == -1.
  - Triangle critical if triangle_to_edge[id] == -1.
- Classification:
  - minima = critical vertices, saddles = critical edges, maxima = critical triangles.
- TTK 2D classification matches this mapping (0, 1, and 2-dimensional critical cells).
- Compliance: Good for 2D.

## Dimensionality and features beyond 2D

- 3D support:
  - Julia: not implemented (no tetrahedra logic, no arrays for 3D gradient pairs).
  - TTK: Fully supports 3D, including 2-saddles, 3D separatrices, walls, and segmentation.
  - Compliance: Out of scope in Julia. Documented limitation.

- Path tracing, separatrices, segmentation, persistence/simplification:
  - Julia: Not present.
  - TTK: Extensive implementation and options.
  - Compliance: Intended subset; fine for a focused critical-points implementation, but not feature-parity.

- Boundary and metadata:
  - Julia: No explicit boundary classification or metadata (e.g., `isOnBoundary`, manifold sizes, PL identifiers).
  - TTK: Rich metadata and boundary handling in outputs.
  - Compliance: Subset only.

## Correctness notes and edge cases

- Total order and ties:
  - Julia’s `offsets` are unique ranks from `sortperm`, so there are no rank ties. This is compliant with the requirement for a strict total order.
  - If an application expects TTK’s tie-breaking scheme (value + offset array), results should still align because Julia’s ranks impose a strict order.

- Queue order polarity (min vs max):
  - Potentially different deterministic ordering than TTK due to heap semantics; the gradient should remain valid, but may differ in specific pairings versus TTK. If exact reproducibility is a requirement (e.g., for unit tests mirroring TTK), align heap polarity.

- Local paired flags:
  - Julia uses `CellExt.paired` to drive lower-star processing and to count unpaired triangle faces. This matches the standard approach and is consistent with TTK’s local bookkeeping.

- Mesh traversal:
  - Julia scans all triangles to locate those containing a vertex during lower-star computation; this is correct but suboptimal for large meshes. TTK uses preconditioned star queries.

- Manifold/boundary conditions:
  - No special-casing of boundaries. For manifold surfaces, ProcessLowerStars is robust without special boundary logic; however, explicit boundary awareness (as in TTK) can be useful for classification and downstream products.

## Performance and scalability

- Julia:
  - Single-threaded.
  - Lower-star triangle discovery by full scan: O(n_tris) per vertex yields O(n_verts·n_tris) worst-case behavior, which is significantly slower than star-based adjacency on large meshes.
  - Priority queues are standard binary heaps; good asymptotics.
- TTK:
  - Threaded with OpenMP, vertex-star preconditioning, and careful memory layout for cache locality. Strong/weak scaling is a design goal.
- Compliance: Functional but not performance-equivalent. For parity, add vertex-star adjacency and parallelization.

## Complexity

- Theoretical algorithmic complexity (per ProcessLowerStars) is O(sum over vertices of star size × log star size) with priority queues. The Julia implementation adds an O(n_tris) factor per vertex due to the brute-force triangle scan.

## Gaps vs. C++ and recommendations for tighter compliance

1. Heap polarity and order determinism (DONE)
  - Confirmation: TTK uses `std::priority_queue` with a comparator equivalent to a min-heap on the lexicographic `lowVerts_` tuple (see `DiscreteGradient_Template.h`: `orderCells(a,b) { return a.lowVerts_ > b.lowVerts_; }`).
  - Status in Julia: The implementation uses a min-heap via `BinaryHeap{CellExt}(LV_ORD)` where `lt` is lexicographic ascending, matching TTK’s effective semantics.
  - Action: None required for parity; both use min-heap behavior. Keep comparator consistent and documented.

2. Lower-star triangle discovery (DONE)
   - Difference: Julia scans all triangles to find those incident to the pivot vertex.
   - Impact: Performance.
   - Action: Precompute and store vertex-to-triangle (vertex star) adjacency; or use `edge_to_triangles` plus incident edges to derive local triangles quickly.

3. 3D support (NOT PLANNED)
   - Difference: Not implemented in Julia.
   - Action: If needed, extend `LowerStar` and `GradientField` to 3D cells; implement tetrahedra support with 3-face bookkeeping and 2-saddle/maximum classification.

4. Parallelization and preconditioning
   - Difference: Single-threaded; no triangulation preconditioning.
   - Action: Add precomputation hooks (stars, links) and leverage Julia threading for vertex loop parallelism.

5. Boundary/meta outputs and downstream features
   - Difference: Only critical points are returned; no boundary flags, no separatrices, no segmentation.
   - Action: Add optional boundary detection and metadata; implement ascending/descending path extraction using the gradient arrays for parity with TTK features (optional if scope remains critical points only).

6. Tests for reproducibility vs TTK
   - Action: Create small fixtures that run both Julia and TTK on the same meshes and scalars; compare gradient arrays and critical sets. If differences exist solely due to queue order, document or align ordering.

## Summary

- What’s compliant:
  - Core ProcessLowerStars ideas are implemented correctly for 2D: lower-star construction, queue-based processing, pairing rules, and gradient storage. Critical point detection and 2D classification match TTK’s design.
- What diverges:
  - Not heap polarity (now confirmed aligned). Differences may still arise from tie-breaking or local queue content timing, but not from heap min/max semantics.
  - Performance: Julia lacks triangulation preconditioning and parallelism; lower-star triangle discovery is O(n_tris) per vertex.
  - Feature coverage: 3D, path/separatrix/segmentation, persistence, and boundary metadata are not implemented (expected given scope).
- Practical guidance:
  - If exact reproducibility with TTK is desired, align queue top semantics and add vertex-star adjacency.
  - For scalability, add star/neighbor preconditioning and parallelize vertex processing.
  - For completeness, extend to 3D and add path/separatrix routines as needed.

With these adjustments, the Julia implementation can be both faithful to TTK’s algorithmic behavior and performant on larger meshes, while remaining a clean, educational translation of the discrete Morse lower-star method.
