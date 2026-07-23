#pragma once
#include <cstddef>

namespace bonafide {
// Internal helper used by upload.cpp — guarded stream cache.
void *get_or_create_stream(const char *name);
}
