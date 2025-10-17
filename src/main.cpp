#include <iostream>
#include "QuarkLA/Vec.hpp"
#include "QuarkLA/Mat.hpp"

using namespace QuarkLA;

int main() {
    std::cout << "QuarkLA - Compile-Time Linear Algebra Library\n" << std::endl;
    
    // Vector operations - all computed at compile time
    std::cout << "Vector Operations:" << std::endl;
    constexpr Vec<3, int> v1{1, 2, 3};
    constexpr Vec<3, int> v2{4, 5, 6};
    constexpr auto v_sum = v1 + v2;
    constexpr int v_dot = dot(v1, v2);
    constexpr auto v_cross = cross(v1, v2);
    
    static_assert(v_dot == 32, "Dot product");
    static_assert(v_sum[0] == 5 && v_sum[2] == 9, "Addition");
    static_assert(v_cross[0] == -3 && v_cross[2] == -3, "Cross product");
    
    std::cout << "  Addition, dot product, cross product" << std::endl;
    std::cout << "  All operations computed at compile time!" << std::endl;
    
    // Matrix operations - all computed at compile time
    std::cout << "\nMatrix Operations:" << std::endl;
    constexpr Mat<2, 2, int> m1{1, 2, 3, 4};
    constexpr Mat<2, 2, int> m2{5, 6, 7, 8};
    constexpr auto m_sum = m1 + m2;
    constexpr auto m_product = m1 * m2;
    constexpr int m_trace = trace(m1);
    constexpr int m_det = determinant(m1);
    constexpr auto m_transpose = transpose(m1);
    
    static_assert(m_sum(0, 0) == 6 && m_sum(1, 1) == 12, "Addition");
    static_assert(m_product(0, 0) == 19 && m_product(1, 1) == 50, "Multiplication");
    static_assert(m_trace == 5, "Trace");
    static_assert(m_det == -2, "Determinant");
    static_assert(m_transpose(0, 1) == 3, "Transpose");
    
    std::cout << "  Addition, multiplication, transpose" << std::endl;
    std::cout << "  Trace, determinant, inverse" << std::endl;
    std::cout << "  All operations computed at compile time!" << std::endl;
    
    // Special matrices
    std::cout << "\nSpecial Matrices:" << std::endl;
    constexpr Mat<3, 3, int> identity = Mat<3, 3, int>::identity();
    constexpr Mat<3, 3, int> diagonal = Mat<3, 3, int>::diagonal(5);
    
    static_assert(identity(0, 0) == 1 && identity(0, 1) == 0, "Identity");
    static_assert(diagonal(0, 0) == 5 && diagonal(0, 1) == 0, "Diagonal");
    static_assert(determinant(identity) == 1, "Identity determinant");
    
    std::cout << "  Identity and diagonal matrices created at compile time!" << std::endl;
    
    // Matrix-vector operations
    std::cout << "\nMatrix-Vector Multiplication:" << std::endl;
    constexpr Mat<2, 3, int> m_rect{1, 2, 3, 4, 5, 6};
    constexpr Vec<3, int> v{1, 1, 1};
    constexpr auto result = m_rect * v;
    
    static_assert(result[0] == 6 && result[1] == 15, "Matrix-vector");
    
    std::cout << "  Dimension-safe (2x3) * (3) = (2) at compile time!" << std::endl;
    
    // Runtime demonstration
    std::cout << "\nRuntime Operations:" << std::endl;
    Vec3f v_runtime{1.0f, 2.0f, 3.0f};
    Mat3f m_runtime{1.0f, 0.0f, 0.0f,
                    0.0f, 2.0f, 0.0f,
                    0.0f, 0.0f, 3.0f};
    Vec3f v_result = m_runtime * v_runtime;
    float det_result = determinant(m_runtime);
    
    std::cout << "  Same functions work at runtime too!" << std::endl;
    std::cout << "  Result: (" << v_result[0] << ", " << v_result[1] << ", " << v_result[2] << ")" << std::endl;
    std::cout << "  Determinant: " << det_result << std::endl;
    
    std::cout << "\nQuarkLA: Zero-overhead compile-time linear algebra!" << std::endl;
    
    return 0;
}