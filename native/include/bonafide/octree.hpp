// bonafide/octree.hpp — internal octree node layout used by the kernels.
#pragma once

#include <cstdint>

namespace bonafide {

struct OctreeNode {
    float aabb_min[3];
    float aabb_max[3];
    std::int32_t first_index;     // offset into the flat index buffer
    std::int32_t count;            // 0 → internal node (children follow)
    std::int32_t child_base;       // index of first child node (or -1)
};

}  // namespace bonafide
