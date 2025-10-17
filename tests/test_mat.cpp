#include <gtest/gtest.h>
#include "QuarkLA/Mat.hpp"

using namespace QuarkLA;

TEST(MatTest, DefaultConstructor) {
    Mat<2, 3, float> m;
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.size(), 6);
    EXPECT_FALSE(m.empty());

    // All elements should be zero-initialized
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            EXPECT_EQ(m(i, j), 0.0f);
        }
    }
}

TEST(MatTest, FillConstructor) {
    Mat<2, 3, float> m(5.0f);
    for (std::size_t i = 0; i < m.rows(); i++) {
        for (std::size_t j = 0; j < m.cols(); j++) {
            EXPECT_EQ(m(i, j), 5.0f);
        }
    }
}

TEST(MatTest, InitializerListConstructor) {
    Mat<2, 3, int> m{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4);
    EXPECT_EQ(m(1, 1), 5);
    EXPECT_EQ(m(1, 2), 6);
    
    // Test with fewer elements (rest should be zero)
    Mat<2, 2, int> m2{1, 2};
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 2);
    EXPECT_EQ(m2(1, 0), 0);
    EXPECT_EQ(m2(1, 1), 0);
}

TEST(MatTest, DiagonalConstructor) {
    Mat<3, 3, float> m = Mat<3, 3, float>::diagonal(5.0f);
    EXPECT_EQ(m(0, 0), 5.0f);
    EXPECT_EQ(m(1, 1), 5.0f);
    EXPECT_EQ(m(2, 2), 5.0f);
    EXPECT_EQ(m(0, 1), 0.0f);
    EXPECT_EQ(m(1, 0), 0.0f);
}

TEST(MatTest, IdentityConstructor) {
    Mat<3, 3, float> m = Mat<3, 3, float>::identity();
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(1, 1), 1.0f);
    EXPECT_EQ(m(2, 2), 1.0f);
    EXPECT_EQ(m(0, 1), 0.0f);
    EXPECT_EQ(m(1, 0), 0.0f);
}

TEST(MatTest, ElementAccess) {
    Mat<2, 3, int> m{1, 2, 3, 4, 5, 6};
    
    // Test operator()
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 2), 6);
    
    // Test operator[]
    EXPECT_EQ(m[0], 1);
    EXPECT_EQ(m[5], 6);
    
    // Test at()
    EXPECT_EQ(m.at(0, 1), 2);
    EXPECT_EQ(m.at(4), 5);
    
    // Test const access
    const Mat<2, 2, int> cm{1, 2, 3, 4};
    EXPECT_EQ(cm(0, 0), 1);
    EXPECT_EQ(cm[1], 2);
    EXPECT_EQ(cm.at(1, 1), 4);
}

TEST(MatTest, CopyConstructor) {
    Mat<2, 2, int> m1{1, 2, 3, 4};
    Mat<2, 2, int> m2(m1);
    EXPECT_EQ(m2.rows(), m1.rows());
    EXPECT_EQ(m2.cols(), m1.cols());
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(1, 1), 4);
}

TEST(MatTest, AssignmentOperator) {
    Mat<2, 2, int> m1{1, 2, 3, 4};
    Mat<2, 2, int> m2;
    m2 = m1;
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 2);
    EXPECT_EQ(m2(1, 0), 3);
    EXPECT_EQ(m2(1, 1), 4);
}

TEST(MatTest, MoveSemantics) {
    Mat<2, 2, int> m1{1, 2, 3, 4};
    Mat<2, 2, int> m2(std::move(m1));
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(1, 1), 4);
}

TEST(MatTest, ArithmeticOperations) {
    Mat<2, 2, float> m1{1.0f, 2.0f, 3.0f, 4.0f};
    Mat<2, 2, float> m2{5.0f, 6.0f, 7.0f, 8.0f};
    
    // Addition
    Mat<2, 2, float> m3 = m1 + m2;
    EXPECT_FLOAT_EQ(m3(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(m3(1, 1), 12.0f);
    
    // Subtraction
    Mat<2, 2, float> m4 = m2 - m1;
    EXPECT_FLOAT_EQ(m4(0, 0), 4.0f);
    EXPECT_FLOAT_EQ(m4(1, 1), 4.0f);
    
    // Scalar multiplication
    Mat<2, 2, float> m5 = m1 * 2.0f;
    EXPECT_FLOAT_EQ(m5(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(m5(1, 1), 8.0f);
    
    Mat<2, 2, float> m6 = 3.0f * m1;
    EXPECT_FLOAT_EQ(m6(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(m6(1, 1), 12.0f);
    
    // Scalar division
    Mat<2, 2, float> m7 = m2 / 2.0f;
    EXPECT_FLOAT_EQ(m7(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(m7(1, 1), 4.0f);
    
    // Unary negation
    Mat<2, 2, float> m8 = -m1;
    EXPECT_FLOAT_EQ(m8(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(m8(1, 1), -4.0f);
}

TEST(MatTest, MatrixMultiplication) {
    // 2x3 * 3x2 = 2x2
    Mat<2, 3, float> m1{1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};
    Mat<3, 2, float> m2{7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f};
    Mat<2, 2, float> result = m1 * m2;
    
    // First element: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    EXPECT_FLOAT_EQ(result(0, 0), 58.0f);
    // result(0, 1): 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(result(0, 1), 64.0f);
    // result(1, 0): 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    EXPECT_FLOAT_EQ(result(1, 0), 139.0f);
    // result(1, 1): 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    EXPECT_FLOAT_EQ(result(1, 1), 154.0f);
    
    // Test identity matrix property: I * M = M
    Mat<2, 2, float> identity = Mat<2, 2, float>::identity();
    Mat<2, 2, float> m3{1.0f, 2.0f, 3.0f, 4.0f};
    Mat<2, 2, float> result2 = identity * m3;
    EXPECT_FLOAT_EQ(result2(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result2(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result2(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(result2(1, 1), 4.0f);
}

TEST(MatTest, MatrixVectorMultiplication) {
    Mat<2, 3, float> m{1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f};
    Vec<3, float> v{7.0f, 8.0f, 9.0f};
    Vec<2, float> result = m * v;
    
    // result[0]: 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    EXPECT_FLOAT_EQ(result[0], 50.0f);
    // result[1]: 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    EXPECT_FLOAT_EQ(result[1], 122.0f);
}

TEST(MatTest, HadamardProduct) {
    Mat<2, 2, float> m1{1.0f, 2.0f, 3.0f, 4.0f};
    Mat<2, 2, float> m2{5.0f, 6.0f, 7.0f, 8.0f};
    
    Mat<2, 2, float> result = hadamard(m1, m2);
    EXPECT_FLOAT_EQ(result(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 32.0f);
}

TEST(MatTest, CompoundAssignment) {
    Mat<2, 2, float> m1{1.0f, 2.0f, 3.0f, 4.0f};
    Mat<2, 2, float> m2{5.0f, 6.0f, 7.0f, 8.0f};
    
    m1 += m2;
    EXPECT_FLOAT_EQ(m1(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(m1(1, 1), 12.0f);
    
    m1 -= m2;
    EXPECT_FLOAT_EQ(m1(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m1(1, 1), 4.0f);
    
    m1 *= 2.0f;
    EXPECT_FLOAT_EQ(m1(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(m1(1, 1), 8.0f);
    
    m1 /= 2.0f;
    EXPECT_FLOAT_EQ(m1(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m1(1, 1), 4.0f);
}

TEST(MatTest, ComparisonOperators) {
    Mat<2, 2, int> m1{1, 2, 3, 4};
    Mat<2, 2, int> m2{1, 2, 3, 4};
    Mat<2, 2, int> m3{5, 6, 7, 8};
    
    EXPECT_TRUE(m1 == m2);
    EXPECT_FALSE(m1 == m3);
    EXPECT_FALSE(m1 != m2);
    EXPECT_TRUE(m1 != m3);
}

TEST(MatTest, ApproximateEquality) {
    Mat<2, 2, float> m1{1.0f, 2.0f, 3.0f, 4.0f};
    Mat<2, 2, float> m2{1.0000001f, 2.0000001f, 3.0000001f, 4.0000001f};
    Mat<2, 2, float> m3{1.1f, 2.1f, 3.1f, 4.1f};
    
    EXPECT_TRUE(approx_equal(m1, m2));
    EXPECT_FALSE(approx_equal(m1, m3));
    EXPECT_TRUE(approx_equal(m1, m3, 0.2f));
}

TEST(MatTest, RowAndColumnAccess) {
    Mat<2, 3, int> m{1, 2, 3,
                     4, 5, 6};
    
    // Get row
    auto row0 = m.row(0);
    EXPECT_EQ(row0[0], 1);
    EXPECT_EQ(row0[1], 2);
    EXPECT_EQ(row0[2], 3);
    
    auto row1 = m.row(1);
    EXPECT_EQ(row1[0], 4);
    EXPECT_EQ(row1[1], 5);
    EXPECT_EQ(row1[2], 6);
    
    // Get column
    auto col0 = m.col(0);
    EXPECT_EQ(col0[0], 1);
    EXPECT_EQ(col0[1], 4);
    
    auto col1 = m.col(1);
    EXPECT_EQ(col1[0], 2);
    EXPECT_EQ(col1[1], 5);
    
    auto col2 = m.col(2);
    EXPECT_EQ(col2[0], 3);
    EXPECT_EQ(col2[1], 6);
}

TEST(MatTest, SetRowAndColumn) {
    Mat<2, 3, int> m{1, 2, 3,
                     4, 5, 6};
    
    // Set row
    Vec<3, int> new_row{7, 8, 9};
    m.set_row(0, new_row);
    EXPECT_EQ(m(0, 0), 7);
    EXPECT_EQ(m(0, 1), 8);
    EXPECT_EQ(m(0, 2), 9);
    EXPECT_EQ(m(1, 0), 4);  // Second row unchanged
    
    // Set column
    Vec<2, int> new_col{10, 11};
    m.set_col(1, new_col);
    EXPECT_EQ(m(0, 1), 10);
    EXPECT_EQ(m(1, 1), 11);
    EXPECT_EQ(m(0, 0), 7);  // Other elements unchanged
}

TEST(MatTest, Transpose) {
    Mat<2, 3, int> m{1, 2, 3,
                     4, 5, 6};
    
    Mat<3, 2, int> mt = transpose(m);
    EXPECT_EQ(mt(0, 0), 1);
    EXPECT_EQ(mt(0, 1), 4);
    EXPECT_EQ(mt(1, 0), 2);
    EXPECT_EQ(mt(1, 1), 5);
    EXPECT_EQ(mt(2, 0), 3);
    EXPECT_EQ(mt(2, 1), 6);
}

TEST(MatTest, TransposeInPlace) {
    Mat<3, 3, int> m{1, 2, 3,
                     4, 5, 6,
                     7, 8, 9};
    
    m.transpose_inplace();
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 4);
    EXPECT_EQ(m(0, 2), 7);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(1, 1), 5);
    EXPECT_EQ(m(1, 2), 8);
    EXPECT_EQ(m(2, 0), 3);
    EXPECT_EQ(m(2, 1), 6);
    EXPECT_EQ(m(2, 2), 9);
}

TEST(MatTest, Trace) {
    Mat<3, 3, int> m{1, 2, 3,
                     4, 5, 6,
                     7, 8, 9};
    
    int tr = trace(m);
    EXPECT_EQ(tr, 15);  // 1 + 5 + 9 = 15
}

TEST(MatTest, Determinant) {
    // 1x1 matrix
    Mat<1, 1, float> m1{5.0f};
    EXPECT_FLOAT_EQ(determinant(m1), 5.0f);
    
    // 2x2 matrix
    Mat<2, 2, float> m2{1.0f, 2.0f,
                        3.0f, 4.0f};
    // det = 1*4 - 2*3 = 4 - 6 = -2
    EXPECT_FLOAT_EQ(determinant(m2), -2.0f);
    
    // 3x3 matrix
    Mat<3, 3, float> m3{1.0f, 2.0f, 3.0f,
                        0.0f, 1.0f, 4.0f,
                        5.0f, 6.0f, 0.0f};
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    EXPECT_FLOAT_EQ(determinant(m3), 1.0f);
    
    // Identity matrix has determinant 1
    Mat<3, 3, float> identity = Mat<3, 3, float>::identity();
    EXPECT_FLOAT_EQ(determinant(identity), 1.0f);
}

TEST(MatTest, Inverse) {
    // 2x2 matrix
    Mat<2, 2, float> m{4.0f, 7.0f,
                       2.0f, 6.0f};
    // det = 4*6 - 7*2 = 24 - 14 = 10
    // inv = (1/10) * [6, -7; -2, 4]
    Mat<2, 2, float> inv = inverse(m);
    EXPECT_FLOAT_EQ(inv(0, 0), 0.6f);
    EXPECT_FLOAT_EQ(inv(0, 1), -0.7f);
    EXPECT_FLOAT_EQ(inv(1, 0), -0.2f);
    EXPECT_FLOAT_EQ(inv(1, 1), 0.4f);
    
    // Test that m * inv = identity
    Mat<2, 2, float> identity = m * inv;
    EXPECT_TRUE(approx_equal(identity, Mat<2, 2, float>::identity(), 1e-5f));
    
    // Test singular matrix throws
    Mat<2, 2, float> singular{1.0f, 2.0f,
                              2.0f, 4.0f};  // det = 0
    EXPECT_THROW((void)inverse(singular), std::runtime_error);
}

TEST(MatTest, FrobeniusNorm) {
    Mat<2, 2, float> m{1.0f, 2.0f,
                       3.0f, 4.0f};
    // ||M||_F = sqrt(1 + 4 + 9 + 16) = sqrt(30)
    float norm = norm_frobenius(m);
    EXPECT_FLOAT_EQ(norm, std::sqrt(30.0f));
    
    // Identity matrix norm
    Mat<3, 3, float> identity = Mat<3, 3, float>::identity();
    EXPECT_FLOAT_EQ(norm_frobenius(identity), std::sqrt(3.0f));
}

TEST(MatTest, UtilityMethods) {
    Mat<2, 2, int> m1{1, 2, 3, 4};
    
    // Fill
    m1.fill(5);
    EXPECT_EQ(m1(0, 0), 5);
    EXPECT_EQ(m1(1, 1), 5);
    
    // Swap
    Mat<2, 2, int> m2{7, 8, 9, 10};
    m1.swap(m2);
    EXPECT_EQ(m1(0, 0), 7);
    EXPECT_EQ(m1(1, 1), 10);
    EXPECT_EQ(m2(0, 0), 5);
    EXPECT_EQ(m2(1, 1), 5);
}

TEST(MatTest, OutOfBoundsAccess) {
    Mat<2, 2, int> m{1, 2, 3, 4};
    EXPECT_THROW((void)m.at(2, 0), std::out_of_range);
    EXPECT_THROW((void)m.at(0, 2), std::out_of_range);
    EXPECT_THROW((void)m.at(4), std::out_of_range);
    EXPECT_NO_THROW((void)m.at(1, 1));
    EXPECT_NO_THROW((void)m.at(3));
}

TEST(MatTest, ConstexprUsage) {
    constexpr Mat<2, 2, int> m{1, 2, 3, 4};
    constexpr int sum = m(0, 0) + m(1, 1);
    static_assert(sum == 5, "Constexpr operations should work");
    EXPECT_EQ(sum, 5);
    
    // Test constexpr operations
    constexpr Mat<2, 2, int> m1{1, 2, 3, 4};
    constexpr Mat<2, 2, int> m2{5, 6, 7, 8};
    constexpr auto m3 = m1 + m2;
    EXPECT_EQ(m3(0, 0), 6);
    EXPECT_EQ(m3(1, 1), 12);
    
    // Test constexpr identity
    constexpr Mat<2, 2, int> identity = Mat<2, 2, int>::identity();
    static_assert(identity(0, 0) == 1, "Identity diagonal should be 1");
    static_assert(identity(0, 1) == 0, "Identity off-diagonal should be 0");
}

TEST(MatTest, TypeAliases) {
    Mat2f m2f{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(m2f.rows(), 2);
    EXPECT_EQ(m2f.cols(), 2);
    
    Mat3d m3d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_EQ(m3d.rows(), 3);
    EXPECT_EQ(m3d.cols(), 3);
    
    Mat4i m4i;
    EXPECT_EQ(m4i.rows(), 4);
    EXPECT_EQ(m4i.cols(), 4);
    
    Mat2x3f m23f;
    EXPECT_EQ(m23f.rows(), 2);
    EXPECT_EQ(m23f.cols(), 3);
}

TEST(MatTest, IsSquare) {
    Mat<2, 2, float> square;
    Mat<2, 3, float> rect;
    
    EXPECT_TRUE(square.is_square());
    EXPECT_FALSE(rect.is_square());
}

TEST(MatTest, DataPointer) {
    Mat<2, 2, int> m{1, 2, 3, 4};
    
    const int* ptr = m.data();
    EXPECT_EQ(ptr[0], 1);
    EXPECT_EQ(ptr[1], 2);
    EXPECT_EQ(ptr[2], 3);
    EXPECT_EQ(ptr[3], 4);
    
    // Modify through pointer
    int* mutable_ptr = m.data();
    mutable_ptr[0] = 10;
    EXPECT_EQ(m(0, 0), 10);
}
