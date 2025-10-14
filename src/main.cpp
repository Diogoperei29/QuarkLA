#include <iostream>
#include "QuarkLA/Vec.hpp"

using namespace QuarkLA;

int main() {
    std::cout << "QuarkLA - Simple Usage Example\n" << std::endl;
    
    // Runtime operations
    std::cout << "Runtime operations:" << std::endl;
    Vec3f v1{1.0f, 2.0f, 3.0f};
    Vec3f v2{4.0f, 5.0f, 6.0f};
    
    Vec3f sum = v1 + v2;
    float dotProd = dot(v1, v2);
    Vec3f cross_prod = cross(v1, v2);
    
    std::cout << "  v1 + v2 = (" << sum[0] << ", " << sum[1] << ", " << sum[2] << ")" << std::endl;
    std::cout << "  dot(v1, v2) = " << dotProd << std::endl;
    std::cout << "  cross(v1, v2) = (" << cross_prod[0] << ", " << cross_prod[1] << ", " << cross_prod[2] << ")\n" << std::endl;
    
    // Compile-time operations (constexpr)
    std::cout << "Compile-time operations (constexpr):" << std::endl;
    constexpr Vec<3, int> cv1{1, 2, 3};
    constexpr Vec<3, int> cv2{4, 5, 6};
    constexpr auto cv_sum = cv1 + cv2;
    constexpr int cv_dot = dot(cv1, cv2);
    constexpr auto cv_scaled = cv1 * 2;
    
    static_assert(cv_dot == 32, "Compile-time dot product");
    static_assert(cv_sum[0] == 5, "Compile-time addition");
    static_assert(cv_scaled[2] == 6, "Compile-time scalar multiplication");
    
    std::cout << "  constexpr cv1 + cv2 = (" << cv_sum[0] << ", " << cv_sum[1] << ", " << cv_sum[2] << ")" << std::endl;
    std::cout << "  constexpr dot(cv1, cv2) = " << cv_dot << std::endl;
    std::cout << "  constexpr cv1 * 2 = (" << cv_scaled[0] << ", " << cv_scaled[1] << ", " << cv_scaled[2] << ")" << std::endl;
    std::cout << "  ✓ All values computed at compile time and verified with static_assert!\n" << std::endl;
    
    std::cout << "Type aliases: Vec2f, Vec3f, Vec4f, Vec2d, Vec3d, Vec4d, Vec2i, Vec3i, Vec4i" << std::endl;
    std::cout << "All operations are constexpr-compatible for compile-time computation." << std::endl;
    
    return 0;
}