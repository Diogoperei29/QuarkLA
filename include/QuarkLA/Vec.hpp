#pragma once

#include <array>
#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>

namespace QuarkLA {



namespace helper {

// Constexpr sqrt using Newton-Raphson
template<typename T>
constexpr T constexpr_sqrt_impl(T x, T curr, T prev) noexcept {
    // Converged when difference is small enough
    return curr == prev ? curr : constexpr_sqrt_impl(x, T(0.5) * (curr + x / curr), curr);
}

template<typename T>
constexpr T constexpr_sqrt(T x) noexcept {
    if (x == T(0) || x == T(1)) 
        return x;
    if (x < T(0))
        return T(0);

    // constexpr sqrt with initial guess x/2
    return constexpr_sqrt_impl(x, x / T(2), T(0));
}

} // namespace helper



/**
 * Vec<N, T> - Fixed-size vector with compile-time dimensions
 * 
 * @tparam N The dimension (size) of the vector
 * @tparam T The scalar type (float, double, int, etc.)
 */
template<std::size_t N, typename T>
class Vec {
    static_assert(N > 0, "Vector dimension must be greater than zero");
    static_assert(std::is_arithmetic_v<T>, "Vector scalar type must be arithmetic");

private:
    std::array<T, N> data_;

public:
    using size_type = std::size_t;
    using iterator = typename std::array<T, N>::iterator;
    using const_iterator = typename std::array<T, N>::const_iterator;
    // Default constructor (zero-initialize)
    constexpr Vec() noexcept : data_{} {}
    
    // Fill constructor: all elements set to the same value
    explicit constexpr Vec(T value) noexcept {
        for(std::size_t i=0; i<N; i++) data_[i] = value;
    }
    
    // Initializer list constructor
    constexpr Vec(std::initializer_list<T> init) : data_{} {
        if (init.size() > N) throw std::runtime_error("Too many initializers");
        std::size_t i = 0;
        for (auto it = init.begin(); it != init.end(); it++, i++) {
            data_[i] = *it;
        }
    }
    
    // Copy constructor
    constexpr Vec(const Vec&) noexcept = default;
    
    // Move constructor
    constexpr Vec(Vec&&) noexcept = default;
    
    // Copy assignment
    constexpr Vec& operator=(const Vec&) noexcept = default;
    
    // Move assignment
    constexpr Vec& operator=(Vec&&) noexcept = default;
    
    // Destructor
    ~Vec() = default;

    // -- Size and capacity (compile-time) --
    
    static constexpr size_type size() noexcept {
        return N;
    }
    static constexpr size_type dimensions() noexcept {
        return N;
    }
    constexpr bool empty() const noexcept {
        return false; // can never be empty
    }

    // -- Element access --
    
    // Unchecked access 
    constexpr T& operator[](size_type index) noexcept {
        return data_[index];
    }
    constexpr const T& operator[](size_type index) const noexcept {
        return data_[index];
    }
    
    // Checked access (throws std::out_of_range if index >= N)
    constexpr T& at(size_type index) {
        if (index >= N) throw std::out_of_range("Index out of bounds");
        return data_[index];
    }
    constexpr const T& at(size_type index) const{
        if (index >= N) throw std::out_of_range("Index out of bounds");
        return data_[index];
    }
    
    // Direct data pointer access
    constexpr T* data() noexcept {
        return data_.data();
    }
    constexpr const T* data() const noexcept{
        return data_.data();
    }

    // -- Iterators --
    
    constexpr iterator begin() noexcept {
        return data_.begin();
    }
    constexpr const_iterator begin() const noexcept { 
        return data_.begin();
    }
    constexpr const_iterator cbegin() const noexcept {
        return data_.cbegin();
    }
    
    constexpr iterator end() noexcept {
        return data_.end();
    }
    constexpr const_iterator end() const noexcept {
        return data_.end();
    }
    constexpr const_iterator cend() const noexcept {
        return data_.cend();
    }

    // -- Unary arithmetic operators --
    
    // Unary plus (returns copy)
    constexpr Vec operator+() const noexcept {
        return *this;
    }
    
    // Unary minus (negation)
    constexpr Vec operator-() const noexcept {
        Vec out; 
        for (size_t i = 0; i < N; i++)
            out.data_[i] = -data_[i];
        return out;
    }

    // -- Compound assignment operators (vector-vector) -- 
    
    constexpr Vec& operator+=(const Vec& other) noexcept {
        for (size_t i = 0; i < N; i++)
            data_[i] += other[i];
        return *this;
    }
    constexpr Vec& operator-=(const Vec& other) noexcept {
        for (size_t i = 0; i < N; i++)
            data_[i] -= other[i];
        return *this;
    }
    
    // Hadamard product (element-wise multiplication)
    constexpr Vec& operator*=(const Vec& other) noexcept {
        for (size_t i = 0; i < N; i++)
            data_[i] *= other[i];
        return *this;
    }
    
    // Element-wise division
    constexpr Vec& operator/=(const Vec& other) {
        for (size_t i = 0; i < N; i++) {
            if (other[i] == 0) throw std::runtime_error("Cannot divide by zero");
            data_[i] /= other[i];
        }
        return *this;
    }

    // -- Compound assignment operators (vector-scalar) --
    
    constexpr Vec& operator*=(T scalar) noexcept  {
        for (size_t i = 0; i < N; i++)
            data_[i] *= scalar;
        return *this;
    }
    constexpr Vec& operator/=(T scalar)  {
        if (scalar == 0) throw std::runtime_error("Cannot divide by zero");
        for (size_t i = 0; i < N; i++)
            data_[i] /= scalar;
        return *this;
    }

    // -- Utility methods -- 
    
    // Fill all elements with a value
    constexpr void fill(T value) noexcept {
        for (size_t i = 0; i < N; i++) 
            data_[i] = value;
    }
    
    // Swap with another vector
    constexpr void swap(Vec& other) noexcept {
        data_.swap(other.data_);
    }
};

// -- Binary arithmetic operators (vector-vector) -- 

template<std::size_t N, typename T>
constexpr Vec<N, T> operator+(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = lhs[i] + rhs[i];
    return out;
}

template<std::size_t N, typename T>
constexpr Vec<N, T> operator-(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = lhs[i] - rhs[i];
    return out;
}

// Hadamard product (element-wise multiplication)
template<std::size_t N, typename T>
constexpr Vec<N, T> hadamard(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = lhs[i] * rhs[i];
    return out;
}

// -- Binary arithmetic operators (vector-scalar) -- 

template<std::size_t N, typename T>
constexpr Vec<N, T> operator*(const Vec<N, T>& vec, T scalar) noexcept {
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = vec[i] * scalar;
    return out;
}

template<std::size_t N, typename T>
constexpr Vec<N, T> operator*(T scalar, const Vec<N, T>& vec) noexcept {
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = scalar * vec[i];
    return out;
}

template<std::size_t N, typename T>
constexpr Vec<N, T> operator/(const Vec<N, T>& vec, T scalar) {
    if (scalar == 0) throw std::runtime_error("Cannot divide by zero");
    Vec<N, T> out;
    for (size_t i = 0; i < N; i++) 
        out[i] = vec[i] / scalar ;
    return out;
}

// -- Comparison operators -- 

template<std::size_t N, typename T>
constexpr bool operator==(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    for (size_t i = 0; i < N; i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

template<std::size_t N, typename T>
constexpr bool operator!=(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    return !(lhs == rhs);
}

// Epsilon-based comparison for floating point types
template<std::size_t N, typename T>
constexpr bool approx_equal(const Vec<N, T>& lhs, const Vec<N, T>& rhs, T epsilon = T(1e-6)) noexcept {
    for (size_t i = 0; i < N; i++) {
        T diff = lhs[i] - rhs[i];
        if (diff < T(0)) 
            diff = -diff; 
        if (diff > epsilon) 
            return false;
    }
    return true;
}

// -- Vector operations -- 

// Dot product (inner product)
template<std::size_t N, typename T>
constexpr T dot(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    T out = T(0);
    for (size_t i = 0; i < N; i++)
        out += (lhs[i] * rhs[i]);
    return out;
}

// Squared length (magnitude squared) - avoids sqrt for performance
template<std::size_t N, typename T>
constexpr T length_squared(const Vec<N, T>& vec) noexcept {
    T out = T(0);
    for (size_t i = 0; i < N; i++)
        out += (vec[i] * vec[i]);
    return out; 
}

// Length (for implementation sake) -- just use std::sqrt(length_squared)
template<std::size_t N, typename T>
constexpr T length(const Vec<N, T>& vec) noexcept {
    return helper::constexpr_sqrt(length_squared(vec));
}

// Normalize (for implementation sake) -- use normalize with length
template<std::size_t N, typename T>
constexpr Vec<N, T> normalize(const Vec<N, T>& vec) {
    T len = length(vec);
    if (len == T(0)) throw std::runtime_error("Cannot normalize zero vector");
    return vec / len;
}

// Normalize
template<std::size_t N, typename T>
constexpr Vec<N, T> normalize(const Vec<N, T>& vec, const T len) {
    if (len == T(0)) throw std::runtime_error("Cannot normalize zero vector");
    return vec / len;
}

// Distance between two vectors
template<std::size_t N, typename T>
constexpr T distance(const Vec<N, T>& lhs, const Vec<N, T>& rhs) noexcept {
    return length(lhs - rhs);
}

// Cross product (only Vec<3, T>)
template<typename T>
constexpr Vec<3, T> cross(const Vec<3, T>& lhs, const Vec<3, T>& rhs) noexcept {
    Vec<3, T> out;
    out[0] = lhs[1]*rhs[2] - lhs[2]*rhs[1];
    out[1] = lhs[2]*rhs[0] - lhs[0]*rhs[2];
    out[2] = lhs[0]*rhs[1] - lhs[1]*rhs[0];
    return out;
}

// -- Type traits and concepts (C++20) -- 

// Check if a type is a Vec
template<typename T>
struct is_vec : std::false_type {};

// Specialization - Vec<N,T> is a Vec
template<std::size_t N, typename T>
struct is_vec<Vec<N, T>> : std::true_type {};

template<typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

template<typename T>
concept Scalar = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPointScalar = std::is_floating_point_v<T>;

template<typename T>
concept VecType = is_vec_v<T>;

template<typename T, std::size_t N>
concept VecOfSize = is_vec_v<T> && T::size() == N;

// Type aliases

// 2D vectors
using Vec2f = Vec<2, float>;
using Vec2d = Vec<2, double>;
using Vec2i = Vec<2, int>;

// 3D vectors
using Vec3f = Vec<3, float>;
using Vec3d = Vec<3, double>;
using Vec3i = Vec<3, int>;

// 4D vectors
using Vec4f = Vec<4, float>;
using Vec4d = Vec<4, double>;
using Vec4i = Vec<4, int>;



} // namespace QuarkLA
