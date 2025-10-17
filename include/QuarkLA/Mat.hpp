#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include "Vec.hpp"

namespace QuarkLA {



/**
 * Mat<R, C, T> - Fixed-size matrix with compile-time dimensions
 * 
 * @tparam R Number of rows
 * @tparam C Number of columns
 * @tparam T The scalar type (float, double, int, etc.)
 * 
 * Storage: Row-major order (rows stored contiguously in memory)
 */
template<std::size_t R, std::size_t C, typename T>
class Mat {
    static_assert(R > 0, "Matrix row count must be greater than zero");
    static_assert(C > 0, "Matrix column count must be greater than zero");
    static_assert(std::is_arithmetic_v<T>, "Matrix scalar type must be arithmetic");

private:
    std::array<T, R * C> data_;  // Row-major storage

public:
    using size_type = std::size_t;
    using row_type = Vec<C, T>;
    using col_type = Vec<R, T>;
    
    // -- Constructors and assignment --
    
    // Default constructor (zero-initialize)
    constexpr Mat() noexcept : data_{} {}
    
    // Fill constructor: all elements set to the same value
    explicit constexpr Mat(T value) noexcept {
        for (size_type i = 0; i < R*C; i++) data_[i] = value;
    }
    
    // Initializer list constructor (row-major order)
    constexpr Mat(std::initializer_list<T> init) : data_{} {
        if (init.size() > R*C) throw std::runtime_error("Too many initializers");
        size_type i = 0;
        for (auto it = init.begin(); it != init.end(); it++, i++) {
            data_[i] = *it;
        }
    }
    
    // Diagonal matrix constructor (for square matrices)
    static constexpr Mat diagonal(T value) noexcept requires (R == C) {
        Mat out;
        for (size_type j = 0; j < R; j++) {
            for (size_type i = 0; i < C; i++) {
                if (i==j)
                    out.data_[i + j*C] = T{value};
                else
                    out.data_[i + j*C] = T{0};
            }
        }
        return out;
    }
    
    // Identity matrix (for square matrices)
    static constexpr Mat identity() noexcept requires (R == C) {
        Mat out;
        for (size_type j = 0; j < R; j++) {
            for (size_type i = 0; i < C; i++) {
                if (i==j)
                    out.data_[i + j*C] = T{1};
                else
                    out.data_[i + j*C] = T{0};
            }
        }
        return out;
    }
    
    // Copy constructor
    constexpr Mat(const Mat&) noexcept = default;
    
    // Move constructor
    constexpr Mat(Mat&&) noexcept = default;
    
    // Copy assignment
    constexpr Mat& operator=(const Mat&) noexcept = default;
    
    // Move assignment
    constexpr Mat& operator=(Mat&&) noexcept = default;
    
    // Destructor
    ~Mat() = default;

    // -- Size and capacity (compile-time) --
    
    [[nodiscard]] static constexpr size_type rows() noexcept {
        return R;
    }
    [[nodiscard]] static constexpr size_type cols() noexcept {
        return C;
    }
    [[nodiscard]] static constexpr size_type size() noexcept {
        return (R*C);
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return false;
    }
    [[nodiscard]] constexpr bool is_square() const noexcept {
        return (R == C);
    }

    // -- Element access --
    
    // Unchecked access: mat(row, col)
    [[nodiscard]] constexpr T& operator()(size_type row, size_type col) noexcept {
        return data_[col + row*C];
    }
    [[nodiscard]] constexpr const T& operator()(size_type row, size_type col) const noexcept {
        return data_[col + row*C];
    }

    // Unchecked linear access (row-major): mat[index]
    [[nodiscard]] constexpr T& operator[](size_type index) noexcept {
        return data_[index];
    }
    [[nodiscard]] constexpr const T& operator[](size_type index) const noexcept {
        return data_[index];
    }
    
    // Checked access: mat.at(row, col)
    [[nodiscard]] constexpr T& at(size_type row, size_type col) {
        if (row >= R || col >= C) throw std::out_of_range("Index out of bounds");
        return data_[col + row*C];
    }
    [[nodiscard]] constexpr const T& at(size_type row, size_type col) const {
        if (row >= R || col >= C) throw std::out_of_range("Index out of bounds");
        return data_[col + row*C];
    }

    // Checked Linear access: mat.at(index)
    [[nodiscard]] constexpr T& at(size_type index) {
        if (index >= C*R) throw std::out_of_range("Index out of bounds");
        return data_[index];
    }
    [[nodiscard]] constexpr const T& at(size_type index) const {
        if (index >= C*R) throw std::out_of_range("Index out of bounds");
        return data_[index];
    }
    
    // Direct data pointer access
    [[nodiscard]] constexpr T* data() noexcept {
        return data_.data();
    }
    [[nodiscard]] constexpr const T* data() const noexcept {
        return data_.data();
    }

    // -- Row and column access --
    
    // Get row as a vector
    [[nodiscard]] constexpr row_type row(size_type r) const {
        if (r >= R) throw std::out_of_range("Index out of bounds");
        row_type out;
        for (size_type i = 0; i < C; i++)
            out[i] = data_[i + r*C];
        return out;
    }
    
    // Get column as a vector
    [[nodiscard]] constexpr col_type col(size_type c) const {
        if (c >= C) throw std::out_of_range("Index out of bounds");
        col_type out;
        for (size_type j = 0; j < R; j++) 
            out[j] = data_[c + j*C];
        return out;
    }
    
    // Set row from a vector
    constexpr void set_row(size_type r, const row_type& v) {
        if (r >= R) throw std::out_of_range("Index out of bounds");
        for (size_type i = 0; i < C; i++)
            data_[i + r*C] = v[i];
    }
    
    // Set column from a vector
    constexpr void set_col(size_type c, const col_type& v) {
        if (c >= C) throw std::out_of_range("Index out of bounds");
        for (size_type j = 0; j < R; j++)
            data_[c + j*C] = v[j];
    }

    // -- Unary arithmetic operators --
    
    // Unary plus (returns copy)
    constexpr Mat operator+() const noexcept {
        return *this; // Copy
    }
    
    // Unary minus (negation)
    constexpr Mat operator-() const noexcept {
        Mat mat;
        for (size_type i = 0; i < R*C; i++)
            mat.data_[i] = -data_[i];
        return mat;
    }

    // -- Compound assignment operators (matrix-matrix) --
    
    constexpr Mat& operator+=(const Mat& other) noexcept {
        for (size_type i = 0; i < R*C; i++)
            data_[i] += other.data_[i];
        return *this;
    }
    constexpr Mat& operator-=(const Mat& other) noexcept {
        for (size_type i = 0; i < R*C; i++)
            data_[i] -= other.data_[i];
        return *this;
    }

    // -- Compound assignment operators (matrix-scalar) --
    
    constexpr Mat& operator*=(T scalar) noexcept {
        for (size_type i = 0; i < R*C; i++)
            data_[i] *= scalar;
        return *this;
    }
    constexpr Mat& operator/=(T scalar) {
        if (scalar == 0) throw std::runtime_error("Cannot divide by zero");
        for (size_type i = 0; i < R*C; i++)
            data_[i] /= scalar;
        return *this;
    }
    // -- Utility methods --
    
    // Fill all elements with a value
    constexpr void fill(T value) noexcept {
        for (size_type i = 0; i < R*C; i++)
            data_[i] = value;
    }
    
    // Swap with another matrix
    constexpr void swap(Mat& other) noexcept {
        data_.swap(other.data_);
    }
    
    // Transpose (only for square matrices, in-place)
    constexpr void transpose_inplace() noexcept requires (R == C) {
        if (R <= 1) return;
        for (size_type j = 0; j < R; j++) {
            // top-right triangle
            for (size_type i = 1 + j; i < C; i++) { 
                helper::swap(data_[i + j*C], data_[j + i*C]);
            }
        }
    }
};

// -- Binary arithmetic operators (matrix-matrix) --

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator+(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = lhs[i] + rhs[i];
    return out;
}

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator-(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = lhs[i] - rhs[i];
    return out;
}

// Hadamard product (element-wise multiplication)
template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> hadamard(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = lhs[i] * rhs[i];
    return out;
}

// -- Binary arithmetic operators (matrix-scalar) --

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator*(const Mat<R, C, T>& mat, T scalar) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = mat[i] * scalar;
    return out;
}

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator*(T scalar, const Mat<R, C, T>& mat) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = scalar * mat[i];
    return out;
}

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator/(const Mat<R, C, T>& mat, T scalar) {
    if (scalar == 0) throw std::runtime_error("Cannot divide by zero");
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R*C; i++)
        out[i] = mat[i] / scalar;
    return out;
}

// -- Matrix-matrix multiplication --

template<std::size_t R, std::size_t N, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<R, C, T> operator*(const Mat<R, N, T>& lhs, const Mat<N, C, T>& rhs) noexcept {
    Mat<R, C, T> out;
    for (std::size_t i = 0; i < R; i++) {
        for (std::size_t j = 0; j < C; j++) {
            T sum = T(0);
            for (std::size_t k = 0; k < N; k++) {
                sum += lhs(i, k) * rhs(k, j);
            }
            out(i, j) = sum;
        }
    }
    return out;
}

// -- Matrix-vector multiplication --

template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Vec<R, T> operator*(const Mat<R, C, T>& mat, const Vec<C, T>& vec) noexcept {
    Vec<R, T> out;
    for (std::size_t i = 0; i < R; i++) {
        T sum = T(0);
        for (std::size_t j = 0; j < C; j++) {
            sum += mat(i, j) * vec[j];
        }
        out[i] = sum;
    }
    return out;
}

// -- Comparison operators --

template<std::size_t R, std::size_t C, typename T>
constexpr bool operator==(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs) noexcept {
    for (std::size_t i = 0; i < R*C; i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;    
}

template<std::size_t R, std::size_t C, typename T>
constexpr bool operator!=(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs) noexcept {
    return !(lhs == rhs);
}

// Epsilon-based comparison for floating point types
template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr bool approx_equal(const Mat<R, C, T>& lhs, const Mat<R, C, T>& rhs, 
                            T epsilon = T(1e-6)) noexcept {

    for (std::size_t i = 0; i < R*C; i++) {
        T diff = lhs[i] - rhs[i];
        if (diff < T(0)) 
            diff = -diff; 
        if (diff > epsilon) 
            return false;
    }
    return true;
}

                        

// -- Matrix operations --

// Transpose (returns new matrix)
template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr Mat<C, R, T> transpose(const Mat<R, C, T>& mat) noexcept {
    Mat<C, R, T> out;
    for (std::size_t i = 0; i < R; i++) {
        for (std::size_t j = 0; j < C; j++) {
            out(j, i) = mat(i, j);
        }
    }
    return out;
}

// Trace (sum of diagonal elements, square matrices only)
template<std::size_t N, typename T>
[[nodiscard]] constexpr T trace(const Mat<N, N, T>& mat) noexcept {
    T sum = T{0};
    for (std::size_t n = 0; n < N; n++) 
        sum += mat(n,n);
    return sum;
}

// Determinant (square matrices only)
template<std::size_t N, typename T>
[[nodiscard]] constexpr T determinant(const Mat<N, N, T>& mat) noexcept {
    // Base case: 1x1 matrix
    if constexpr (N == 1) {
        return mat(0, 0);
    }
    // Base case: 2x2 matrix
    else if constexpr (N == 2) {
        return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
    }
    // Base case: 3x3 matrix (Sarrus)
    else if constexpr (N == 3) {
        return mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(1, 2) * mat(2, 1))
             - mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0))
             + mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));
    }
    // General case: NxN matrix using cofactor expansion along first row
    else {
        T det = T(0);
        for (std::size_t j = 0; j < N; j++) {
            // Create (N-1)x(N-1) minor matrix
            Mat<N-1, N-1, T> minor;
            for (std::size_t mi = 0; mi < N-1; mi++) {
                for (std::size_t mj = 0; mj < N-1; mj++) {
                    std::size_t orig_j = (mj < j) ? mj : mj + 1;
                    minor(mi, mj) = mat(mi + 1, orig_j);
                }
            }
            // Cofactor expansion: det += (-1)^j * mat(0,j) * det(minor)
            T cofactor = ((j % 2 == 0) ? T(1) : T(-1)) * mat(0, j) * determinant(minor);
            det += cofactor;
        }
        return det;
    }
}

// Inverse (square matrices only)
template<std::size_t N, typename T>
[[nodiscard]] constexpr Mat<N, N, T> inverse(const Mat<N, N, T>& mat) {
    T det = determinant(mat);
    if (det == T(0)) {
        throw std::runtime_error("Matrix is singular, cannot invert");
    }
    
    // Base case: 1x1 matrix
    if constexpr (N == 1) {
        Mat<1, 1, T> result;
        result(0, 0) = T(1) / mat(0, 0);
        return result;
    }
    // Base case: 2x2 matrix
    else if constexpr (N == 2) {
        Mat<2, 2, T> result;
        T inv_det = T(1) / det;
        result(0, 0) =  mat(1, 1) * inv_det;
        result(0, 1) = -mat(0, 1) * inv_det;
        result(1, 0) = -mat(1, 0) * inv_det;
        result(1, 1) =  mat(0, 0) * inv_det;
        return result;
    }
    // General case: Adjugate matrix method (cofactor matrix transposed / determinant)
    else {
        Mat<N, N, T> adjugate;
        // Get cofactor matrix
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                // Create (N-1)x(N-1) minor by removing row i and column j
                Mat<N-1, N-1, T> minor;
                for (std::size_t mi = 0; mi < N-1; mi++) {
                    for (std::size_t mj = 0; mj < N-1; mj++) {
                        std::size_t orig_i = (mi < i) ? mi : mi + 1;
                        std::size_t orig_j = (mj < j) ? mj : mj + 1;
                        minor(mi, mj) = mat(orig_i, orig_j);
                    }
                }
                
                // Compute cofactor: (-1)^(i+j) * det(minor)
                T cofactor = (((i + j) % 2 == 0) ? T(1) : T(-1)) * determinant(minor);
                
                // Adjugate is the transpose of the cofactor matrix
                adjugate(j, i) = cofactor;
            }
        }
        
        // Inverse = adjugate / determinant
        return adjugate / det;
    }
}

// Frobenius norm (sqrt of sum of squared elements)
template<std::size_t R, std::size_t C, typename T>
[[nodiscard]] constexpr T norm_frobenius(const Mat<R, C, T>& mat) noexcept {
    T sum_sq = T(0);
    for (std::size_t i = 0; i < R * C; i++) {
        sum_sq += mat[i] * mat[i];
    }
    return helper::sqrt(sum_sq);
}

// -- Type traits and concepts (C++20) --
#if __cplusplus >= 202002L
// Check if a type is a Mat
template<typename T>
struct is_mat : std::false_type {};

// Specialization - Mat<R, C, T> is a Mat
template<std::size_t R, std::size_t C, typename T>
struct is_mat<Mat<R, C, T>> : std::true_type {};

template<typename T>
inline constexpr bool is_mat_v = is_mat<T>::value;

template<typename T>
concept MatType = is_mat_v<T>;
#endif

// Type aliases

// 2x2 matrices
using Mat2f = Mat<2, 2, float>;
using Mat2d = Mat<2, 2, double>;
using Mat2i = Mat<2, 2, int>;

// 3x3 matrices
using Mat3f = Mat<3, 3, float>;
using Mat3d = Mat<3, 3, double>;
using Mat3i = Mat<3, 3, int>;

// 4x4 matrices
using Mat4f = Mat<4, 4, float>;
using Mat4d = Mat<4, 4, double>;
using Mat4i = Mat<4, 4, int>;

// Non-square matrices
// float
using Mat2x2f = Mat<2, 2, float>;
using Mat2x3f = Mat<2, 3, float>;
using Mat2x4f = Mat<2, 4, float>;
using Mat3x2f = Mat<3, 2, float>;
using Mat3x3f = Mat<3, 3, float>;
using Mat3x4f = Mat<3, 4, float>;
using Mat4x2f = Mat<4, 2, float>;
using Mat4x3f = Mat<4, 3, float>;
using Mat4x4f = Mat<4, 4, float>;
// double
using Mat2x2d = Mat<2, 2, double>;
using Mat2x3d = Mat<2, 3, double>;
using Mat2x4d = Mat<2, 4, double>;
using Mat3x2d = Mat<3, 2, double>;
using Mat3x3d = Mat<3, 3, double>;
using Mat3x4d = Mat<3, 4, double>;
using Mat4x2d = Mat<4, 2, double>;
using Mat4x3d = Mat<4, 3, double>;
using Mat4x4d = Mat<4, 4, double>;
// int
using Mat2x2i = Mat<2, 2, int>;
using Mat2x3i = Mat<2, 3, int>;
using Mat2x4i = Mat<2, 4, int>;
using Mat3x2i = Mat<3, 2, int>;
using Mat3x3i = Mat<3, 3, int>;
using Mat3x4i = Mat<3, 4, int>;
using Mat4x2i = Mat<4, 2, int>;
using Mat4x3i = Mat<4, 3, int>;
using Mat4x4i = Mat<4, 4, int>;



} // namespace QuarkLA
