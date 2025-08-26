from __future__ import annotations

import os
import struct
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
try:
    from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Primitive
    from pygltflib import FLOAT, VEC3, SCALAR, ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER, UNSIGNED_SHORT
    _HAS_PYGLTF = True
except Exception:  # pragma: no cover
    GLTF2 = None  # type: ignore
    Scene = Node = Mesh = Buffer = BufferView = Accessor = Asset = Primitive = None  # type: ignore
    FLOAT = VEC3 = SCALAR = ELEMENT_ARRAY_BUFFER = ARRAY_BUFFER = UNSIGNED_SHORT = None  # type: ignore
    _HAS_PYGLTF = False

from app.services.procedural_roof.synthesis import ProceduralRoofModel


def _triangulate_quad(quad: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    # Quad as v0,v1,v2,v3 -> triangles (0,1,2) and (0,2,3)
    if len(quad) != 4:
        raise ValueError("quad must have 4 vertices")
    verts = [tuple(map(float, v)) for v in quad]
    indices = [0, 1, 2, 0, 2, 3]
    return verts, indices


def _pack_floats(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _pack_ushort(arr: np.ndarray) -> bytes:
    return arr.astype(np.uint16).tobytes()


def write_gltf(model: ProceduralRoofModel,
               out_dir: Optional[str] = None,
               filename: Optional[str] = None) -> str:
    """
    Create a minimal glTF binary (.glb) scene with:
      - part quads as flat meshes (triangulated)
      - ridge polylines as line lists
    Coordinates use (lon, lat, z) directly for now.
    """
    base_dir = Path(out_dir or os.getenv("ARTIFACT_DIR", "./artifacts"))
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = filename or f"roof_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    path = base_dir / f"{stem}.glb"

    if not _HAS_PYGLTF:
        # Write a minimal placeholder binary so downstream code has a file to reference
        with open(path, "wb") as f:
            f.write(b"glTF placeholder\n")
        return str(path)

    gltf = GLTF2(asset=Asset(version="2.0"))  # type: ignore[call-arg]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    nodes: List[Node] = []
    meshes: List[Mesh] = []

    # Aggregate buffers
    bin_chunks: List[bytes] = []
    accessors: List[Accessor] = []
    bufferViews: List[BufferView] = []

    def add_primitive(positions: np.ndarray, indices: Optional[np.ndarray], mode: int) -> int:
        # positions: (N,3) float32, indices: (M,) uint16 (optional)
        nonlocal bin_chunks, accessors, bufferViews
        pos_bytes = _pack_floats(positions)
        pos_bv = BufferView(buffer=0, byteOffset=sum(len(c) for c in bin_chunks), byteLength=len(pos_bytes), target=ARRAY_BUFFER)
        bufferViews.append(pos_bv)
        bin_chunks.append(pos_bytes)
        pos_accessor = Accessor(bufferView=len(bufferViews)-1, componentType=FLOAT, count=positions.shape[0], type=VEC3,
                                min=[float(np.min(positions[:,0])), float(np.min(positions[:,1])), float(np.min(positions[:,2]))],
                                max=[float(np.max(positions[:,0])), float(np.max(positions[:,1])), float(np.max(positions[:,2]))])
        accessors.append(pos_accessor)
        indices_accessor_idx = None
        if indices is not None:
            idx_bytes = _pack_ushort(indices)
            idx_bv = BufferView(buffer=0, byteOffset=sum(len(c) for c in bin_chunks), byteLength=len(idx_bytes), target=ELEMENT_ARRAY_BUFFER)
            bufferViews.append(idx_bv)
            bin_chunks.append(idx_bytes)
            indices_accessor = Accessor(bufferView=len(bufferViews)-1, componentType=UNSIGNED_SHORT, count=indices.shape[0], type=SCALAR)
            accessors.append(indices_accessor)
            indices_accessor_idx = len(accessors)-1
        mesh = Mesh(primitives=[Primitive(attributes={"POSITION": len(accessors)-1 if indices is None else len(accessors)-2}, indices=indices_accessor_idx, mode=mode)])
        meshes.append(mesh)
        node = Node(mesh=len(meshes)-1)
        nodes.append(node)
        return len(nodes)-1

    # Parts as quads (flat, using first 4 points of rect ring)
    for part in model.parts:
        ring = part.get("rect_bbox", [])
        if not ring or len(ring) < 4:
            continue
        # Use lon,lat and zero z unless height exists; optional pitch not used for geometry yet
        quad_ll = [tuple(ring[i]) for i in range(4)]
        quad = [(lon, lat, float(part.get("height_stats_m", {}).get("mean", 0.0))) for lon, lat in quad_ll]
        verts, idx = _triangulate_quad(quad)
        positions = np.array(verts, dtype=np.float32)
        indices = np.array(idx, dtype=np.uint16)
        add_primitive(positions, indices, mode=4)  # TRIANGLES

    # Ridge polylines (as independent line strips)
    for part in model.parts:
        for ridge3d in part.get("ridges_3d", []):
            if len(ridge3d) < 2:
                continue
            positions = np.array([tuple(map(float, p)) for p in ridge3d], dtype=np.float32)
            # Represent as line strip using implicit ordering; indices optional
            add_primitive(positions, None, mode=3)  # LINE_STRIP

    # Assemble buffers
    bin_blob = b"".join(bin_chunks)
    gltf.buffers = [Buffer(byteLength=len(bin_blob))]  # type: ignore[list-item]
    gltf.bufferViews = bufferViews  # type: ignore[assignment]
    gltf.accessors = accessors  # type: ignore[assignment]
    gltf.meshes = meshes  # type: ignore[assignment]
    gltf.nodes = nodes  # type: ignore[assignment]
    gltf.scenes = gltf.scenes or [Scene(nodes=list(range(len(nodes))))]  # type: ignore[list-item]

    gltf.set_binary_blob(bin_blob)  # type: ignore[attr-defined]
    gltf.save_binary(str(path))  # type: ignore[attr-defined]
    return str(path)


__all__ = ["write_gltf"]

