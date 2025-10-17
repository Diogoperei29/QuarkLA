[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20) [![CI](https://github.com/Diogoperei29/QuarkLA/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/Diogoperei29/QuarkLA/actions/workflows/cmake-multi-platform.yml)

# QuarkLA

QuarkLA is a tiny, header-only C++20 linear algebra library for small, fixed-size problems. It provides `Vec<N, T>` and `Mat<R, C, T>` with constexpr-friendly operations, dimension-safe templates, and no heap usage.

---

## Overview

QuarkLA focuses on **compile-time linear algebra** for graphics, physics, and numerical computing where dimensions are known at compile time (e.g., 2D/3D vectors, matrices). The library prioritizes:

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
│       ├── Vec.hpp          # Vector class implementation
│       └── Mat.hpp          # Matrix class implementation
├── src/
│   └── main.cpp             # Usage examples
├── tests/
│   ├── test_vec.cpp         # Unit tests for Vec
│   └── test_mat.cpp         # Unit tests for Mat
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── LICENSE                  # Apache 2.0 License
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

## Usage Guide

QuarkLA is **header-only**. Just include the headers and start using them:

### Vector Operations

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

### Matrix Operations

```cpp
#include <QuarkLA/Mat.hpp>
#include <iostream>

using namespace QuarkLA;

int main() {
    // Create matrices
    Mat3f m1{1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 9.0f};
    
    // Identity and diagonal matrices
    Mat3f identity = Mat3f::identity();
    Mat3f diag = Mat3f::diagonal(5.0f);
    
    // Element access
    m1(0, 0) = 10.0f;           // Row 0, Col 0
    float val = m1[4];          // Linear index (row-major)
    
    // Matrix arithmetic
    Mat3f m2 = m1 + identity;
    Mat3f scaled = m1 * 2.0f;
    Mat3f negated = -m1;
    
    // Matrix-matrix multiplication
    Mat3f product = m1 * m2;
    
    // Matrix-vector multiplication
    Vec3f v{1.0f, 2.0f, 3.0f};
    Vec3f result = m1 * v;
    
    // Matrix operations
    Mat3f transposed = transpose(m1);
    m1.transpose_inplace();      // In-place (square matrices only)
    float tr = trace(m1);        // Sum of diagonal
    float det = determinant(m1); // Determinant
    Mat3f inv = inverse(m1);     // Matrix inverse
    float norm = norm_frobenius(m1); // Frobenius norm
    
    // Row and column access
    Vec3f row0 = m1.row(0);
    Vec3f col1 = m1.col(1);
    m1.set_row(0, Vec3f{1.0f, 2.0f, 3.0f});
    
    return 0;
}
```

### Available Types

```cpp
// Vectors
Vec2f, Vec2d, Vec2i    // 2D: float, double, int
Vec3f, Vec3d, Vec3i    // 3D: float, double, int
Vec4f, Vec4d, Vec4i    // 4D: float, double, int

// Square matrices
Mat2f, Mat2d, Mat2i    // 2×2: float, double, int
Mat3f, Mat3d, Mat3i    // 3×3: float, double, int
Mat4f, Mat4d, Mat4i    // 4×4: float, double, int

// Non-square matrices
Mat2x3f, Mat2x4f       // 2×3, 2×4
Mat3x2f, Mat3x4f       // 3×2, 3×4
Mat4x2f, Mat4x3f       // 4×2, 4×3

// Custom dimensions and types
Vec<5, double> v;      // 5D double vector
Mat<3, 5, float> m;    // 3×5 float matrix
```

### Constexpr Support

Most operations work at compile time:

```cpp
constexpr Vec<3, int> v1{1, 2, 3};
constexpr Vec<3, int> v2{4, 5, 6};
constexpr auto v3 = v1 + v2;
constexpr int d = dot(v1, v2);
static_assert(d == 32);
```

---

## Testing

QuarkLA uses Google Test for unit testing. Tests are automatically fetched and built by CMake.

> [!TIP]  
> Feel free to use these to test your own implementations!

### Running Tests

```bash
# From build directory
.\test_vec.exe          # Run Vec tests only
.\test_mat.exe          # Run Mat tests only
.\test_all.exe          # Run all tests

# Or using CTest
ctest --output-on-failure
```

### Test Coverage

**Vec Tests:**
- Constructors (default, fill, initializer list, copy, move)
- Element access (`operator[]`, `at()`)
- Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`)
- Compound assignment (`+=`, `-=`, `*=`, `/=`)
- Comparison (`==`, `!=`, `approx_equal`)
- Vector operations (dot, cross, length, normalize, distance)
- Utility functions (fill, swap, iterators)
- Constexpr operations
- Exception safety (bounds checking)

**Mat Tests:**
- Constructors (default, fill, initializer list, diagonal, identity)
- Element access (`operator()`, `operator[]`, `at()`)
- Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`)
- Matrix multiplication (matrix-matrix, matrix-vector)
- Hadamard product (element-wise multiplication)
- Compound assignment (`+=`, `-=`, `*=`, `/=`)
- Comparison (`==`, `!=`, `approx_equal`)
- ow and column access (`row()`, `col()`, `set_row()`, `set_col()`)
- Matrix operations (transpose, trace, determinant, inverse, norm)
- Utility functions (fill, swap, is_square)
- Constexpr operations
- Exception safety (bounds checking, singular matrix)


---


### Current Limitations

1. **Fixed Size Only**: Dimensions must be known at compile time. No dynamic resizing.
   ```cpp
   Vec<3, float> v;      //  OK
   Vec<n, float> v;      //  NOK: n must be constexpr
   ```

2. **Square Root Limitation**: some methods use a custom constexpr sqrt implementations that may be less accurate than `std::sqrt` for very large values. Consider using `std::sqrt(length_squared(v))` if your compiler supports these as constexpr.

3. **Limited Type Support**: Only arithmetic types (`int`, `float`, `double`, etc.) are supported via `static_assert`.

### Design Trade-offs

- **No Expression Templates**: Current implementation uses eager evaluation. This may create temporary objects but keeps the code simple and readable.
- **No Alignment Guarantees**: Uses `std::array` for storage.
- **Exception Safety**: Division operations throw on divide-by-zero. Use `noexcept` operations where exceptions are not desired.

### Planned Features

- [ ] Quaternion support
- [ ] More vector operations (projection, reflection, lerp)
- [ ] Matrix decompositions (LU, QR, SVD)

