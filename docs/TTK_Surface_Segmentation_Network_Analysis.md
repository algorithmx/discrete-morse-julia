# TTK Surface Segmentation Network Analysis

## Executive Summary

After investigating the TTK C++ codebase, particularly the `ttkMorseSmaleComplex` and `ttkDiscreteGradient` modules, I've identified how TTK builds networks of critical points to perform surface segmentation. The system uses a sophisticated multi-layered approach that constructs **separatrices** (integral lines connecting critical points) and then uses these to partition the surface into topologically meaningful regions.

## 1. Core Network Construction Architecture

### **1.1 The Morse-Smale Complex Network**

TTK builds a network through the following hierarchical structure:

```
Critical Points (0-cells, 1-cells, 2-cells)
        ↓
    Separatrices (1D and 2D integral manifolds)  
        ↓
   Surface Segmentation (Morse-Smale cells)
```

### **1.2 Key Network Components**

#### **A. Critical Points as Network Nodes**
- **Minima (0D)**: Vertices with no outgoing gradient paths
- **1-Saddles (1D)**: Edges connecting different regions 
- **Maxima (2D)**: Triangles with no incoming gradient paths

#### **B. Separatrices as Network Edges**
- **1-Separatrices**: Integral lines connecting saddles to extrema
- **2-Separatrices**: Integral surfaces separating 3D regions (for 3D data)
- **Saddle Connectors**: Special paths connecting different saddles

## 2. Separatrix Computation Algorithms

### **2.1 1-Separatrix Construction**

```cpp
// From MorseSmaleComplex.h lines 700-730
template <typename triangulationType>
int getAscendingSeparatrices1(const std::vector<SimplexId> &saddles,
                              std::vector<Separatrix> &separatrices,
                              const triangulationType &triangulation) const {
  // For each 1-saddle (edge)
  for(SimplexId i = 0; i < numberOfSaddles; ++i) {
    const Cell saddle{dim - 1, saddles[i]};
    
    // Get all adjacent cells in the star
    const auto starNumber{(triangulation.*getFaceStarNumber)(saddle.id_)};
    for(SimplexId j = 0; j < starNumber; ++j) {
      SimplexId sId{};
      (triangulation.*getFaceStar)(saddle.id_, j, sId);
      
      // Trace ascending path from adjacent cell to maximum
      std::vector<Cell> vpath{saddle};
      discreteGradient_.getAscendingPath(Cell(dim, sId), vpath, triangulation);
      
      // If path reaches a critical maximum, create separatrix
      const Cell &lastCell = vpath.back();
      if(lastCell.dim_ == dim and discreteGradient_.isCellCritical(lastCell)) {
        separatrices.emplace_back();
        separatrices.back().source_ = saddle;
        separatrices.back().destination_ = lastCell;
        separatrices.back().geometry_ = std::move(vpath);
      }
    }
  }
}
```

**Key Algorithm Features:**
- **V-Path Following**: Uses discrete gradient field to trace integral lines
- **Bidirectional Tracing**: Computes both ascending and descending separatrices
- **Parallel Execution**: Each saddle processed independently
- **Robust Termination**: Paths terminate at critical points or boundaries

### **2.2 Separatrix Data Structure**

```cpp
struct Separatrix {
  Cell source_;              // Starting critical point (saddle)
  Cell destination_;         // Ending critical point (extremum)
  std::vector<Cell> geometry_; // Complete path geometry
};

struct Output1Separatrices {
  struct {
    std::vector<float> points_;           // 3D coordinates along paths
    std::vector<char> cellDimensions_;    // Cell types (0D, 1D, 2D)
    std::vector<SimplexId> cellIds_;      // Original mesh cell IDs
  } pt; // Point data
  
  struct {
    std::vector<SimplexId> connectivity_;     // Line connectivity
    std::vector<SimplexId> sourceIds_;        // Source critical points
    std::vector<SimplexId> destinationIds_;   // Destination critical points
    std::vector<SimplexId> separatrixIds_;    // Unique separatrix identifiers
    std::vector<char> separatrixTypes_;       // Ascending/descending/connector
    std::vector<char> isOnBoundary_;         // Boundary classification
  } cl; // Cell data
};
```

## 3. Advanced Network Features

### **3.1 Saddle Connectors (3D)**

For 3D datasets, TTK computes special **saddle connectors** that link 1-saddles to 2-saddles:

```cpp
template <typename triangulationType>
int getSaddleConnectors(const std::vector<SimplexId> &saddles2,
                       std::vector<Separatrix> &separatrices,
                       const triangulationType &triangulation) const {
  
  for(size_t i = 0; i < saddles2.size(); ++i) {
    const Cell s2{dim - 1, saddles2[i]};  // 2-saddle (triangle)
    
    // Compute descending wall from 2-saddle
    VisitedMask mask{isVisited, visitedTriangles};
    discreteGradient_.getDescendingWall(s2, mask, triangulation, nullptr, &saddles1);
    
    // For each 1-saddle in the wall
    for(const auto saddle1Id : saddles1) {
      const Cell s1{1, saddle1Id};
      
      // Find path through the wall
      std::vector<Cell> vpath;
      const bool isMultiConnected = discreteGradient_.getAscendingPathThroughWall(
        s1, s2, isVisited, &vpath, triangulation);
      
      // Create connector if unique path exists
      if(!isMultiConnected && vpath.back().id_ == s2.id_) {
        separatrices.emplace_back();
        separatrices.back().source_ = s1;
        separatrices.back().destination_ = s2;
        separatrices.back().geometry_ = std::move(vpath);
      }
    }
  }
}
```

### **3.2 Wall Computation (2D Separatrices)**

TTK computes **2D separatrices** using wall-tracing algorithms:

```cpp
template <typename triangulationType>
int getAscendingWall(const Cell &cell, VisitedMask &mask,
                     const triangulationType &triangulation,
                     std::vector<Cell> *const wall,
                     std::vector<SimplexId> *const saddles) const {
  
  if(cell.dim_ == 1) {  // Starting from 1-saddle (edge)
    const SimplexId originId = cell.id_;
    std::queue<SimplexId> bfs;
    bfs.push(originId);
    
    // BFS traversal following gradient pairs
    while(!bfs.empty()) {
      const SimplexId edgeId = bfs.front();
      bfs.pop();
      
      if(!mask.isVisited_[edgeId]) {
        mask.isVisited_[edgeId] = true;
        wall->push_back(Cell(1, edgeId));
        
        // Find paired triangles and continue traversal
        const SimplexId triangleNumber = triangulation.getEdgeTriangleNumber(edgeId);
        for(SimplexId k = 0; k < triangleNumber; ++k) {
          SimplexId triangleId;
          triangulation.getEdgeTriangle(edgeId, k, triangleId);
          
          const SimplexId pairedCellId = getPairedCell(Cell(2, triangleId), triangulation);
          if(pairedCellId != -1) {
            bfs.push(pairedCellId);
          }
        }
      }
    }
  }
}
```

## 4. Surface Segmentation Through Networks

### **4.1 Ascending and Descending Manifolds**

TTK partitions the surface using **ascending and descending manifolds**:

```cpp
template <typename triangulationType>
int setAscendingSegmentation(const std::vector<SimplexId> &maxima,
                            SimplexId *const morseSmaleManifold,
                            const triangulationType &triangulation) const {
  
  // Mark all maxima with unique IDs
  size_t nMax{};
  for(const auto &id : maxima) {
    morseSmaleManifoldOnCells[id] = nMax++;
  }
  
  // Propagate manifold IDs following gradient paths
  for(SimplexId i = 0; i < nCells; ++i) {
    if(isMarked[i] == 1) continue;
    
    auto curr{i};
    while(morseSmaleManifoldOnCells[curr] == -1) {
      // Follow V-path till marked cell is reached
      const auto paired{discreteGradient_.getPairedCell(Cell{dim, curr}, triangulation, true)};
      
      // Find next cell in star of paired cell
      SimplexId next{curr};
      const auto nStars{triangulation.getFaceStarNumber(paired)};
      for(SimplexId j = 0; j < nStars; ++j) {
        triangulation.getFaceStar(paired, j, next);
        if(next != curr) break;
      }
      
      visited.emplace_back(curr);
      curr = next;
    }
    
    // Assign manifold ID to all visited cells
    for(const auto el : visited) {
      morseSmaleManifoldOnCells[el] = morseSmaleManifoldOnCells[curr];
    }
  }
}
```

### **4.2 Final Morse-Smale Segmentation**

The final segmentation combines ascending and descending manifolds:

```cpp
template <typename triangulationType>
int setFinalSegmentation(const SimplexId numberOfMaxima,
                        const SimplexId *const ascendingManifold,
                        const SimplexId *const descendingManifold,
                        SimplexId *const morseSmaleManifold,
                        const triangulationType &triangulation) const {
  
  // Create unique region ID for each (ascending, descending) pair
  for(size_t i = 0; i < nVerts; ++i) {
    const auto d = ascendingManifold[i];
    const auto a = descendingManifold[i];
    if(a == -1 || d == -1) {
      morseSmaleManifold[i] = -1;
    } else {
      morseSmaleManifold[i] = a * numberOfMaxima + d;  // Unique pairing
    }
  }
  
  // Compress sparse region IDs to dense range
  std::vector<SimplexId> sparseRegionIds(morseSmaleManifold, morseSmaleManifold + nVerts);
  TTK_PSORT(threadNumber_, sparseRegionIds.begin(), sparseRegionIds.end());
  
  auto last = std::unique(sparseRegionIds.begin(), sparseRegionIds.end());
  sparseRegionIds.erase(last, sparseRegionIds.end());
  
  // Map sparse IDs to dense range [0, numRegions)
  std::map<SimplexId, size_t> sparseToDenseRegionId{};
  for(size_t i = 0; i < sparseRegionIds.size(); ++i) {
    sparseToDenseRegionId[sparseRegionIds[i]] = i;
  }
  
  // Update all vertices with dense region IDs
  for(size_t i = 0; i < nVerts; ++i) {
    morseSmaleManifold[i] = sparseToDenseRegionId[morseSmaleManifold[i]];
  }
}
```

## 5. Key Network Properties

### **5.1 Topological Guarantees**

1. **Completeness**: Every point belongs to exactly one Morse-Smale cell
2. **Connectivity**: Each region is connected through gradient flow
3. **Boundary Consistency**: Separatrices form proper boundaries between regions
4. **Hierarchical Structure**: Regions can be merged based on persistence

### **5.2 Network Characteristics**

#### **Network Topology**
- **Nodes**: Critical points of dimensions 0, 1, 2
- **Edges**: Separatrices (1D integral lines)
- **Faces**: Separating surfaces (2D integral manifolds)
- **Regions**: Morse-Smale cells (connected components)

#### **Network Properties**
- **Planar Embedding**: For 2D surfaces, network forms planar graph
- **Persistence Hierarchy**: Network can be simplified by persistence thresholds
- **Boundary Handling**: Proper treatment of domain boundaries
- **Multi-scale**: Supports hierarchical segmentation

## 6. Implementation Insights for Julia Translation

### **6.1 Essential Data Structures Needed**

```julia
# Network node representation
struct CriticalPointNode
    id::Int                    # Critical point identifier
    type::Int                  # 0=minimum, 1=saddle, 2=maximum
    location::Vector{Float64}  # 3D coordinates
    manifold_id::Int          # Segmentation region ID
end

# Network edge representation  
struct SeparatrixEdge
    source::Int               # Source critical point ID
    destination::Int          # Destination critical point ID
    path::Vector{Cell}        # Complete geometric path
    separatrix_type::Int      # 0=ascending, 1=descending, 2=connector
    persistence::Float64      # Topological persistence
end

# Segmentation region
struct MorseSmaleRegion
    id::Int                   # Unique region identifier
    vertices::Vector{Int}     # Vertices in this region
    boundary_separatrices::Vector{Int}  # Bounding separatrices
    area::Float64             # Region area/volume
    extrema::Vector{Int}      # Associated critical points
end

# Complete network
struct MorseSmaleNetwork
    critical_points::Vector{CriticalPointNode}
    separatrices::Vector{SeparatrixEdge}
    regions::Vector{MorseSmaleRegion}
    adjacency_matrix::Matrix{Bool}  # Region adjacency
end
```

### **6.2 Key Algorithms to Implement**

1. **V-Path Tracing**: Follow gradient field from cells to critical points
2. **Wall Computation**: BFS traversal to build 2D separatrices
3. **Manifold Construction**: Region growing from critical points
4. **Network Assembly**: Connect critical points via separatrices
5. **Segmentation Computation**: Partition mesh into Morse-Smale cells

### **6.3 Algorithm Complexity**

- **Separatrix Computation**: O(n log n) per separatrix where n is path length
- **Wall Computation**: O(m) where m is number of cells in wall
- **Segmentation**: O(V) where V is number of vertices
- **Total Complexity**: O(V + E + S·P) where S is separatrices count, P is average path length

## 7. Julia Implementation Strategy

### **7.1 Phase 1: Basic Separatrix Tracing**
```julia
function trace_ascending_separatrices(mesh::TriangleMesh, 
                                     gradient::GradientField,
                                     saddles::Vector{Int})
    separatrices = SeparatrixEdge[]
    
    for saddle_id in saddles
        saddle = Cell(1, saddle_id)  # 1-saddle (edge)
        
        # Find adjacent triangles  
        adjacent_triangles = mesh.edge_to_triangles[saddle_id]
        
        for triangle_id in adjacent_triangles
            # Trace ascending path from triangle
            path = trace_gradient_path(Cell(2, triangle_id), gradient, mesh, :ascending)
            
            # Check if path reaches critical maximum
            if is_critical_cell(path[end], gradient, mesh)
                push!(separatrices, SeparatrixEdge(
                    saddle_id, path[end].id, path, 0, compute_persistence(saddle, path[end])
                ))
            end
        end
    end
    
    return separatrices
end
```

### **7.2 Phase 2: Surface Segmentation**
```julia
function compute_morse_smale_segmentation(mesh::TriangleMesh,
                                         network::MorseSmaleNetwork)
    n_verts = size(mesh.vertices, 2)
    segmentation = fill(-1, n_verts)
    
    # Assign each maximum a unique region ID
    for (region_id, critical_point) in enumerate(network.critical_points)
        if critical_point.type == 2  # Maximum
            segmentation[critical_point.id] = region_id
        end
    end
    
    # Propagate region IDs via gradient flow
    for vertex_id in 1:n_verts
        if segmentation[vertex_id] == -1
            path = trace_gradient_path(Cell(0, vertex_id), gradient, mesh, :ascending)
            target_vertex = path[end].id
            segmentation[vertex_id] = segmentation[target_vertex]
        end
    end
    
    return segmentation
end
```

## 8. Conclusion

TTK's surface segmentation network is a sophisticated system that:

1. **Constructs topological networks** connecting critical points via separatrices
2. **Partitions surfaces** into meaningful Morse-Smale cells
3. **Provides hierarchical decomposition** based on topological persistence
4. **Handles complex topology** including boundaries and multi-connected regions
5. **Scales efficiently** to large datasets through parallelization

The key insight is that **separatrices act as the network edges** that connect critical point nodes, and the **segmentation emerges naturally** from the gradient flow structure. This provides a principled, topology-based approach to surface partitioning that is both mathematically rigorous and computationally efficient.

For our Julia implementation, we should focus on:
- Implementing robust V-path tracing algorithms
- Building separatrix networks from critical points  
- Computing manifold-based segmentation
- Adding persistence-based simplification
- Ensuring proper boundary handling

This will provide the foundation for topology-driven surface analysis and segmentation in our discrete Morse theory toolkit.