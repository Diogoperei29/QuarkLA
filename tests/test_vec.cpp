#include <gtest/gtest.h>
#include "QuarkLA/Vec.hpp"

using namespace QuarkLA;

TEST(VecTest, DefaultConstructor) {
    Vec<3, float> v;
    EXPECT_EQ(v.size(), 3);
    EXPECT_FALSE(v.empty());  // Fixed size, never empty

    EXPECT_EQ(v[0], 0.0f);
    EXPECT_EQ(v[1], 0.0f);
    EXPECT_EQ(v[2], 0.0f);
}

TEST(VecTest, FillConstructor) {
    Vec<3, float> v(2.0f);
    EXPECT_EQ(v[0], 2.0f);
    EXPECT_EQ(v[1], 2.0f);
    EXPECT_EQ(v[2], 2.0f);
}

TEST(VecTest, InitializerListConstructor) {
    Vec<3, int> v{1, 2, 3};
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
    
    // Test with fewer elements (rest should be zero)
    Vec<4, int> v2{5, 6};
    EXPECT_EQ(v2[0], 5);
    EXPECT_EQ(v2[1], 6);
    EXPECT_EQ(v2[2], 0);
    EXPECT_EQ(v2[3], 0);
}

TEST(VecTest, ElementAccess) {
    Vec<3, int> v{10, 20, 30};
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v.at(1), 20);
    EXPECT_EQ(v[2], 30);
    
    // Test const access
    const Vec<3, int> cv{1, 2, 3};
    EXPECT_EQ(cv[0], 1);
    EXPECT_EQ(cv.at(1), 2);
}

TEST(VecTest, CopyConstructor) {
    Vec<3, int> v1{1, 2, 3};
    Vec<3, int> v2(v1);
    EXPECT_EQ(v2.size(), v1.size());
    EXPECT_EQ(v2[0], v1[0]);
    EXPECT_EQ(v2[1], v1[1]);
    EXPECT_EQ(v2[2], v1[2]);
}

TEST(VecTest, AssignmentOperator) {
    Vec<3, int> v1{1, 2, 3};
    Vec<3, int> v2;
    v2 = v1;
    EXPECT_EQ(v2[0], 1);
    EXPECT_EQ(v2[1], 2);
    EXPECT_EQ(v2[2], 3);
}

TEST(VecTest, MoveSemantics) {
    Vec<3, int> v1{1, 2, 3};
    Vec<3, int> v2(std::move(v1));
    EXPECT_EQ(v2[0], 1);
    EXPECT_EQ(v2[1], 2);
    EXPECT_EQ(v2[2], 3);
}

TEST(VecTest, ArithmeticOperations) {
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{4.0f, 5.0f, 6.0f};
    
    Vec<3, float> v3 = v1 + v2;
    EXPECT_FLOAT_EQ(v3[0], 5.0f);
    EXPECT_FLOAT_EQ(v3[1], 7.0f);
    EXPECT_FLOAT_EQ(v3[2], 9.0f);
    
    Vec<3, float> v4 = v2 - v1;
    EXPECT_FLOAT_EQ(v4[0], 3.0f);
    EXPECT_FLOAT_EQ(v4[1], 3.0f);
    EXPECT_FLOAT_EQ(v4[2], 3.0f);
    
    Vec<3, float> v5 = v1 * 2.0f;
    EXPECT_FLOAT_EQ(v5[0], 2.0f);
    EXPECT_FLOAT_EQ(v5[1], 4.0f);
    EXPECT_FLOAT_EQ(v5[2], 6.0f);
    
    Vec<3, float> v6 = 3.0f * v1;
    EXPECT_FLOAT_EQ(v6[0], 3.0f);
    EXPECT_FLOAT_EQ(v6[1], 6.0f);
    EXPECT_FLOAT_EQ(v6[2], 9.0f);
    
    Vec<3, float> v7 = v2 / 2.0f;
    EXPECT_FLOAT_EQ(v7[0], 2.0f);
    EXPECT_FLOAT_EQ(v7[1], 2.5f);
    EXPECT_FLOAT_EQ(v7[2], 3.0f);
    
    Vec<3, float> v8 = -v1;
    EXPECT_FLOAT_EQ(v8[0], -1.0f);
    EXPECT_FLOAT_EQ(v8[1], -2.0f);
    EXPECT_FLOAT_EQ(v8[2], -3.0f);
}

TEST(VecTest, DotProduct) {
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{4.0f, 5.0f, 6.0f};
    float result = dot(v1, v2);  // 1*4 + 2*5 + 3*6 = 32
    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST(VecTest, Length) {
    Vec<3, float> v{3.0f, 4.0f, 0.0f};
    float len = length(v);  // sqrt(9 + 16) = 5
    EXPECT_FLOAT_EQ(len, 5.0f);
    
    float len_sq = length_squared(v);
    EXPECT_FLOAT_EQ(len_sq, 25.0f);
    
    Vec<2, double> unit{1.0, 0.0};
    EXPECT_DOUBLE_EQ(length(unit), 1.0);
}

TEST(VecTest, Normalize) {
    Vec<3, float> v{3.0f, 4.0f, 0.0f};
    Vec<3, float> normalized = normalize(v);
    EXPECT_FLOAT_EQ(length(normalized), 1.0f);
    EXPECT_FLOAT_EQ(normalized[0], 0.6f);
    EXPECT_FLOAT_EQ(normalized[1], 0.8f);
    EXPECT_FLOAT_EQ(normalized[2], 0.0f);
    
    // Test that zero vector throws
    Vec<3, float> zero{0.0f, 0.0f, 0.0f};
    // (void) avoid [[nodiscard]] warning
    EXPECT_THROW((void)normalize(zero), std::runtime_error); 
}

// Exception safety tests
TEST(VecTest, OutOfBoundsAccess) {
    Vec<3, int> v{1, 2, 3}; 
    // (void) avoid [[nodiscard]] warning
    EXPECT_THROW((void)v.at(5), std::out_of_range);
    EXPECT_THROW((void)v.at(3), std::out_of_range);
    EXPECT_NO_THROW((void)v.at(2));
}

// Constexpr tests
TEST(VecTest, ConstexprUsage) {
    constexpr Vec<3, int> v{1, 2, 3};
    constexpr int sum = v[0] + v[1] + v[2];
    static_assert(sum == 6, "Constexpr operations should work");
    EXPECT_EQ(sum, 6);
    
    // Test constexpr operations
    constexpr Vec<2, int> v1{1, 2};
    constexpr Vec<2, int> v2{3, 4};
    constexpr auto v3 = v1 + v2;
    EXPECT_EQ(v3[0], 4);
    EXPECT_EQ(v3[1], 6);
}

// Compound assignment tests
TEST(VecTest, CompoundAssignment) {
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{4.0f, 5.0f, 6.0f};
    
    v1 += v2;
    EXPECT_FLOAT_EQ(v1[0], 5.0f);
    EXPECT_FLOAT_EQ(v1[1], 7.0f);
    EXPECT_FLOAT_EQ(v1[2], 9.0f);
    
    v1 -= v2;
    EXPECT_FLOAT_EQ(v1[0], 1.0f);
    EXPECT_FLOAT_EQ(v1[1], 2.0f);
    EXPECT_FLOAT_EQ(v1[2], 3.0f);
    
    v1 *= 2.0f;
    EXPECT_FLOAT_EQ(v1[0], 2.0f);
    EXPECT_FLOAT_EQ(v1[1], 4.0f);
    EXPECT_FLOAT_EQ(v1[2], 6.0f);
    
    v1 /= 2.0f;
    EXPECT_FLOAT_EQ(v1[0], 1.0f);
    EXPECT_FLOAT_EQ(v1[1], 2.0f);
    EXPECT_FLOAT_EQ(v1[2], 3.0f);
}

// Comparison operators
TEST(VecTest, ComparisonOperators) {
    Vec<3, int> v1{1, 2, 3};
    Vec<3, int> v2{1, 2, 3};
    Vec<3, int> v3{4, 5, 6};
    
    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
}

// Approximate equality for floating point
TEST(VecTest, ApproximateEquality) {
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{1.0000001f, 2.0000001f, 3.0000001f};
    Vec<3, float> v3{1.1f, 2.1f, 3.1f};
    
    // Should be approximately equal with default epsilon
    EXPECT_TRUE(approx_equal(v1, v2));
    
    // Should not be approximately equal
    EXPECT_FALSE(approx_equal(v1, v3));
    
    // Should be equal with larger epsilon
    EXPECT_TRUE(approx_equal(v1, v3, 0.2f));
}

// Utility methods
TEST(VecTest, UtilityMethods) {
    Vec<3, int> v1{1, 2, 3};
    
    v1.fill(5);
    EXPECT_EQ(v1[0], 5);
    EXPECT_EQ(v1[1], 5);
    EXPECT_EQ(v1[2], 5);
    
    Vec<3, int> v2{7, 8, 9};
    v1.swap(v2);
    EXPECT_EQ(v1[0], 7);
    EXPECT_EQ(v1[1], 8);
    EXPECT_EQ(v1[2], 9);
    EXPECT_EQ(v2[0], 5);
    EXPECT_EQ(v2[1], 5);
    EXPECT_EQ(v2[2], 5);
}

// Hadamard product (element-wise multiplication)
TEST(VecTest, HadamardProduct) {
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{4.0f, 5.0f, 6.0f};
    
    Vec<3, float> result = hadamard(v1, v2);
    EXPECT_FLOAT_EQ(result[0], 4.0f);
    EXPECT_FLOAT_EQ(result[1], 10.0f);
    EXPECT_FLOAT_EQ(result[2], 18.0f);
    
    v1 *= v2;
    EXPECT_FLOAT_EQ(v1[0], 4.0f);
    EXPECT_FLOAT_EQ(v1[1], 10.0f);
    EXPECT_FLOAT_EQ(v1[2], 18.0f);
}

// Distance
TEST(VecTest, Distance) {
    Vec<2, float> v1{0.0f, 0.0f};
    Vec<2, float> v2{3.0f, 4.0f};
    
    float dist = distance(v1, v2);
    EXPECT_FLOAT_EQ(dist, 5.0f);
}

// Iterators
TEST(VecTest, Iterators) {
    Vec<3, int> v{1, 2, 3};
    
    int sum = 0;
    for (int val : v) {
        sum += val;
    }
    EXPECT_EQ(sum, 6);
    
    // Test begin/end
    EXPECT_EQ(*v.begin(), 1);
    EXPECT_EQ(*(v.end() - 1), 3);
}

// Type aliases
TEST(VecTest, TypeAliases) {
    Vec2f v2f{1.0f, 2.0f};
    EXPECT_EQ(v2f.size(), 2);
    
    Vec3d v3d{1.0, 2.0, 3.0};
    EXPECT_EQ(v3d.size(), 3);
    
    Vec4i v4i{1, 2, 3, 4};
    EXPECT_EQ(v4i.size(), 4);
}

// Cross product (only for 3D vectors)
TEST(VecTest, CrossProduct) {
    // Standard basis vectors
    Vec<3, float> i{1.0f, 0.0f, 0.0f};
    Vec<3, float> j{0.0f, 1.0f, 0.0f};
    Vec<3, float> k{0.0f, 0.0f, 1.0f};
    
    // i × j = k
    Vec<3, float> result1 = cross(i, j);
    EXPECT_FLOAT_EQ(result1[0], 0.0f);
    EXPECT_FLOAT_EQ(result1[1], 0.0f);
    EXPECT_FLOAT_EQ(result1[2], 1.0f);
    
    // j × k = i
    Vec<3, float> result2 = cross(j, k);
    EXPECT_FLOAT_EQ(result2[0], 1.0f);
    EXPECT_FLOAT_EQ(result2[1], 0.0f);
    EXPECT_FLOAT_EQ(result2[2], 0.0f);
    
    // k × i = j
    Vec<3, float> result3 = cross(k, i);
    EXPECT_FLOAT_EQ(result3[0], 0.0f);
    EXPECT_FLOAT_EQ(result3[1], 1.0f);
    EXPECT_FLOAT_EQ(result3[2], 0.0f);
    
    // Anti-commutativity: j × i = -k
    Vec<3, float> result4 = cross(j, i);
    EXPECT_FLOAT_EQ(result4[0], 0.0f);
    EXPECT_FLOAT_EQ(result4[1], 0.0f);
    EXPECT_FLOAT_EQ(result4[2], -1.0f);
    
    // Cross product of arbitrary vectors
    Vec<3, float> v1{1.0f, 2.0f, 3.0f};
    Vec<3, float> v2{4.0f, 5.0f, 6.0f};
    Vec<3, float> result5 = cross(v1, v2);
    // (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4)
    EXPECT_FLOAT_EQ(result5[0], -3.0f);
    EXPECT_FLOAT_EQ(result5[1], 6.0f);
    EXPECT_FLOAT_EQ(result5[2], -3.0f);
    
    // Cross product is perpendicular to both input vectors
    EXPECT_FLOAT_EQ(dot(result5, v1), 0.0f);
    EXPECT_FLOAT_EQ(dot(result5, v2), 0.0f);
    
    // Cross product of parallel vectors is zero 
    Vec<3, float> v3{2.0f, 4.0f, 6.0f};  // Parallel to v1
    Vec<3, float> result6 = cross(v1, v3);
    EXPECT_FLOAT_EQ(result6[0], 0.0f);
    EXPECT_FLOAT_EQ(result6[1], 0.0f);
    EXPECT_FLOAT_EQ(result6[2], 0.0f);
}