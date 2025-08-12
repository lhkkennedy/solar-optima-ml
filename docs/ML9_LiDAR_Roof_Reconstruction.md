## ML-9: LiDAR-guided roof segmentation and structured reconstruction (notes + theory)

### Scope

- Summarize practical methods from “Semantic Segmentation and Roof Reconstruction of Urban Buildings Based on LiDAR Point Clouds” (Sun et al., 2024) and map them to our system.
- Focus: robust roof plane extraction and topology-correct 3D reconstruction from LiDAR (or nDSM surrogate), and how to incrementally integrate with our image-first pipeline.

### High-level pipeline (paper → our system)

- **Semantic segmentation (point clouds)**
  - Paper: ELFA‑RandLA‑Net (encoder–decoder, KNN neighborhoods, Position Encoding + Semantic Encoding, mixed pooling: max + attention) for efficient city‑scale point clouds.
  - Our plan: add a LiDAR path that can ingest LAS/LAZ, compute features (height normalization, surface change rate z‑normal), and train a RandLA‑like lightweight model. Until then, reuse our image path and nDSM surrogate for roof extraction.

- **Building instance extraction**
  - Paper: cluster building points to isolate singles (DBSCAN).
  - Our plan: DBSCAN per semantic building class in either 3D point cloud or 2.5D nDSM‑lifted points.

- **Plane segmentation + refinement**
  - Paper: RANSAC plane extraction; fix merged planes via DBSCAN re‑segmentation; filter tiny planes.
  - Our plan: apply RANSAC to either LiDAR or nDSM‑lifted 3D points; resegment with DBSCAN; drop small primitives.

- **Roof plane recognition (CSF)**
  - Paper: Cloth Simulation Filter on inverted point clouds; planes close to the cloth surface are roofs. Single parameter set works across buildings.
  - Our plan: implement CSF on LiDAR when present; provide nDSM surrogate (robust morphological surface + distance threshold) when LiDAR absent.

- **Vertical plane inference (handles missing wall data)**
  - Paper: infer vertical roof planes between pairs of roof planes with adjacent XY projections and large height differences; fit vertical planes using midpoint/edge projections. Ensures topology even when wall points are missing.
  - Our plan: adopt same inference on plane pairs from RANSAC (LiDAR) or from nDSM‑lifted planes.

- **Roof topology graph + polygon regularization**
  - Paper: nodes=planes; edges if outward‑expanded polygons intersect with sufficient overlap. Compute inner points (triple‑plane intersections via least squares) and outer points (projections on intersection lines). Correct per‑plane polygons to nearest inner/outer points; merge into closed roof polygon; project outer contour to ground for full 3D.
  - Our plan: add outward expansion, overlap‑based adjacency, intersection‑driven vertex correction, closed polygon assembly; export GeoJSON + glTF.

### Theory highlights (why these steps work)

- **ELFA local feature aggregation**
  - KNN neighborhoods capture local geometry; PE encodes absolute/relative positions and distances; SE encodes per‑point semantics and relative feature deltas; mixed pooling (max + attention) preserves salient and discriminative details; random downsampling enables large‑scale processing.

- **RANSAC + DBSCAN**
  - RANSAC is robust to outliers but may fuse near‑coplanar facets under a loose threshold; density‑aware re‑segmentation (DBSCAN) splits such fusions; removing tiny planes reduces topological noise.

- **CSF roof identification**
  - By inverting the scene and simulating a draped cloth, the algorithm approximates an upper envelope of building surfaces. Roof planes are those with small distance to the cloth. Unlike purely normal‑based filtering, CSF avoids per‑building height heuristics and handles vertical planes on roofs.

- **Vertical plane inference**
  - Many airborne scans miss vertical walls. The structural prior: a vertical plane typically connects two roof facets with large height difference and adjacent projections. Fitting the inferred plane restores topological consistency and enables closed roof polygons.

- **Topology graph + polygon correction**
  - Outward expansion compensates regularization shrinkage of plane footprints; intersecting expanded polygons imply adjacency; triple‑plane intersections yield inner roof vertices; projected endpoints yield outer vertices. Correcting raw plane polygons to these vertices produces watertight, semantically consistent roof boundaries.

### Metrics for QA

- **P2M** (point→model): distance from LiDAR points to nearest model plane; completeness/fit.
- **M2P** (model→point): distance from model vertices to nearest point; geometric accuracy of vertices.
- Report Max/Mean/RMS per building and region to compare parameterizations and regressions.

### Integration plan (incremental)

1) nDSM‑first improvements (no LiDAR required)
   - CSF‑like roof filter on nDSM surface.
   - RANSAC planes on nDSM‑lifted points; DBSCAN refinement; overlap‑based adjacency.
   - Vertical‑plane inference; closed polygon assembly; P2M/M2P vs synthesized samples.

2) Optional LiDAR branch
   - Point‑cloud ingestion (Open3D/PDAL); semantic building segmentation (ELFA‑RandLA‑Net‑lite); DBSCAN monomers.
   - CSF roof recognition; full topology + polygon pipeline; P2M/M2P over regions.

3) API integration
   - `/model3d`: choose LiDAR path when point clouds available; otherwise use nDSM path with the same adjacency/topology logic. Persist metrics and artifacts.

### Dependencies

- Open3D/PDAL (point‑cloud I/O, RANSAC, DBSCAN); pyCSF or in‑repo CSF; NumPy/SciPy; shapely (2D polygon ops); our existing pyproj/raster stack.

### Risks and mitigations

- Missing or sparse wall data → vertical plane inference restores topology.
- Over‑aggressive plane thresholds fuse facets → DBSCAN refinement and overlap checks.
- Flat roofs and hipped roofs: accuracy dominated by boundary fitting → emphasize robust contour extraction and vertex correction.

### References

- Sun, X. et al., 2024. Semantic Segmentation and Roof Reconstruction of Urban Buildings Based on LiDAR Point Clouds. ISPRS IJGI, 13(1), 19. CC BY 4.0.
- RandLA‑Net, CSF, PolyFit and related works as cited in the paper.


