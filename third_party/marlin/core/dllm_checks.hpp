// dllm standalone adapter for host-only checks in the vendored scalar type.
#pragma once
#include <stdexcept>
#define STD_TORCH_CHECK(condition, ...) \
    do { if (!(condition)) throw std::invalid_argument("invalid Marlin scalar type"); } while (false)
