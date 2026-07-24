<!-- README-ready gallery block. Merge into README.md (e.g. extend the Screenshots section).
     All frames: BonaFide CPU backend, 1280x720, procedural-sky IBL + CSM shadows.
     Reproduce: python examples/render_external.py  (see docs/GALLERY.md) -->

## 🖼️ External-Asset Gallery

Real-world models loaded straight from `external_assets/` and rendered by BonaFide —
auto-framed camera, procedural-sky IBL, `envmap` background, CSM shadows, 1280×720.
Full write-up + per-model code: [docs/GALLERY.md](docs/GALLERY.md).

| Stanford Bunny (PLY mesh) | Avocado (GLB) |
|---|---|
| ![Stanford Bunny rendered by BonaFide](docs/gallery/bunny.png) | ![Khronos Avocado rendered by BonaFide](docs/gallery/avocado.png) |
| *Stanford 3D Scanning Repository — ascii PLY, 69k tris* | *Khronos sample model, CC0 (Microsoft)* |

| BoomBox (GLB) | KayKit chest (OBJ) |
|---|---|
| ![Khronos BoomBox rendered by BonaFide](docs/gallery/boombox.png) | ![KayKit chest rendered by BonaFide](docs/gallery/chest.png) |
| *Khronos sample model, CC0 (Microsoft)* | *KayKit Dungeon pack, CC0 (Kay Lousberg)* |

| Dolphins point cloud (colored PLY) |
|---|
| ![Colored dolphin point cloud rendered by BonaFide](docs/gallery/dolphins.png) |
| *Ascii PLY point cloud, per-vertex RGB, LOD + surfels — MIT (three.js)* |
