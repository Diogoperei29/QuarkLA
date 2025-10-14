[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Build](https://img.shields.io/github/actions/workflow/status/Diogoperei29/QuarkLA/cmake.yml?branch=main)](https://github.com/Diogoperei29/QuarkLA/actions)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/test_vec.cpp)

# QuarkLA

QuarkLA is a tiny, header-only C++20 linear algebra library for small, fixed-size problems. It provides `Vec<N, T>` and `Mat<R, C, T>` (planned) with constexpr-friendly operations, dimension-safe templates, and no heap usage.

---

## Overview

QuarkLA focuses on **compile-time linear algebra** for graphics, physics, and numerical computing where dimensions are known at compile time (e.g., 2D/3D vectors, small matrices). The library prioritizes:

- **Performance**: Zero-cost abstractions with aggressive inlining and loop unrolling
- **Safety**: Template-based dimension checking catches size mismatches at compile time
- **Simplicity**: Header-only design with minimal dependencies (just standard library)
- **Modern C++**: Leverages C++20 features like concepts, constexpr, and ranges


---

## Compatibility

- **Language**: C++20 or later
- **Compilers**: 
  - GCC
  - Clang
  - MSVC (Visual Studio 2019+)
- **Platforms**: Windows, Linux
- **Build System**: CMake 3.16+
- **Testing**: Google Test 1.17.0 (fetched automatically via CMake)

---

## Project Structure

```
QuarkLA/
├── include/
│   └── QuarkLA/
│       └── Vec.hpp          # Vector class implementation
├── src/
│   └── main.cpp             # Usage examples
├── tests/
│   └── test_vec.cpp         # Unit tests for Vec
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── LICENSE                  # MIT License
```

---

## Build Guide


### Build Steps

```bash
# Clone the repository
git clone https://github.com/Diogoperei29/QuarkLA.git
cd QuarkLA

# Create build directory
mkdir build

# Build
cmake --build build

# Run the example
.\build\QuarkLA.exe

# Run tests
.\build\test_all.exe
```

### CMake Options

- `CMAKE_CXX_STANDARD`: C++ standard (default: `20`)

---

## 🚀 Usage Guide

QuarkLA is **header-only**. Just include the header and start using it:

```cpp
#include <QuarkLA/Vec.hpp>
#include <iostream>

using namespace QuarkLA;

int main() {
    // Create vectors using type aliases
    Vec3f v1{1.0f, 2.0f, 3.0f};
    Vec3f v2{4.0f, 5.0f, 6.0f};
    
    // Arithmetic operations
    Vec3f sum = v1 + v2;              // {5, 7, 9}
    Vec3f scaled = v1 * 2.0f;         // {2, 4, 6}
    Vec3f diff = v2 - v1;             // {3, 3, 3}
    
    // Vector operations
    float dotProd = dot(v1, v2);      // 32.0
    float len = length(v1);           // ~3.742
    Vec3f normalized = normalize(v1); // Unit vector
    
    // Element access
    std::cout << "v1[0] = " << v1[0] << std::endl;
    v1[1] = 10.0f;  // Mutable access
    
    // Compile-time size checking
    constexpr size_t dims = Vec3f::size();  // dims = 3
    static_assert(dims == 3);
    
    // Cross product (3D only)
    Vec3f perpendicular = cross(v1, v2);
    
    return 0;
}
```

### Available Types

```cpp
// 2D vectors
Vec2f, Vec2d, Vec2i    // float, double, int

// 3D vectors
Vec3f, Vec3d, Vec3i

// 4D vectors
Vec4f, Vec4d, Vec4i

// Custom dimensions and types
Vec<5, double> v;      // 5D double vector
Vec<100, int> large;   // 100D integer vector
```

### Constexpr Support

Most operations work at compile time:

```cpp
constexpr Vec<3, int> v1{1, 2, 3};
constexpr Vec<3, int> v2{4, 5, 6};
constexpr auto v3 = v1 + v2;          // Computed at compile time
constexpr int d = dot(v1, v2);        // d = 32
static_assert(d == 32);
```

---

## Testing

QuarkLA uses Google Test for unit testing. Tests are automatically fetched and built by CMake.

### Running Tests

```bash
# From build directory
.\test_vec.exe

# Or using CTest
ctest --output-on-failure
```

### Test Coverage

Current tests cover:
- ✅ Constructors (default, fill, initializer list, copy, move)
- ✅ Element access (`operator[]`, `at()`)
- ✅ Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`)
- ✅ Compound assignment (`+=`, `-=`, `*=`, `/=`)
- ✅ Comparison (`==`, `!=`, `approx_equal`)
- ✅ Vector operations (dot, cross, length, normalize, distance)
- ✅ Utility functions (fill, swap, iterators)
- ✅ Constexpr operations
- ✅ Exception safety (bounds checking)


---

## Limitations

### Current Limitations

1. **Fixed Size Only**: Dimensions must be known at compile time. No dynamic resizing.
   ```cpp
   Vec<3, float> v;  // ✅ OK
   // Vec<n, float> v;  // ❌ n must be constexpr
   ```

2. **Matrix Not Implemented**: `Mat<R, C, T>` is planned but not yet available.

3. **No SIMD Optimization**: Currently relies on compiler auto-vectorization. Explicit SIMD (SSE, AVX) may be added later.

4. **Square Root Limitation**: `length()` and `normalize()` use a custom constexpr sqrt implementation that may be less accurate than `std::sqrt` for very large values. For runtime use, consider using `std::sqrt(length_squared(v))`.

5. **Limited Type Support**: Only arithmetic types (`int`, `float`, `double`, etc.) are supported via `static_assert`.

### Design Trade-offs

- **No Expression Templates**: Current implementation uses eager evaluation. This may create temporary objects but keeps the code simple and readable.
- **No Alignment Guarantees**: Uses `std::array` for storage.
- **Exception Safety**: Division operations throw on divide-by-zero. Use `noexcept` operations where exceptions are not desired.

### Planned Features

- [ ] Matrix class `Mat<R, C, T>`
- [ ] Matrix-vector and matrix-matrix operations
- [ ] Quaternion support
- [ ] Expression templates for lazy evaluation
- [ ] SIMD optimizations
- [ ] More vector operations (projection, reflection, lerp)

