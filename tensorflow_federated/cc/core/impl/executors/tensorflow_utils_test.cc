/* Copyright 2024, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

#include <complex>
#include <cstdint>
#include <string>

#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {
namespace {

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_fully_defined) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({2, 3});
  const tensorflow::TensorShape& expected_shape =
      tensorflow::TensorShape({2, 3});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_scalar) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({});
  const tensorflow::TensorShape& expected_shape = tensorflow::TensorShape({});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_partially_defined) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({2, -1});

  const absl::StatusOr<tensorflow::TensorShape>& result =
      TensorShapeFromArrayShape(shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_unknown) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({}, true);

  const absl::StatusOr<tensorflow::TensorShape>& result =
      TensorShapeFromArrayShape(shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

struct PartialTensorShapeFromArrayShapeTestCase {
  std::string test_name;
  const v0::ArrayShape shape_pb;
  const tensorflow::PartialTensorShape expected_shape;
};

using PartialTensorShapeFromArrayShapeTest =
    ::testing::TestWithParam<PartialTensorShapeFromArrayShapeTestCase>;

TEST_P(PartialTensorShapeFromArrayShapeTest, TestReturnsPartialTensorShape) {
  const PartialTensorShapeFromArrayShapeTestCase& test_case = GetParam();

  const tensorflow::PartialTensorShape& actual_shape =
      PartialTensorShapeFromArrayShape(test_case.shape_pb);

  EXPECT_TRUE(actual_shape.IsIdenticalTo(test_case.expected_shape));
}

INSTANTIATE_TEST_SUITE_P(
    PartialTensorShapeFromArrayShapeTestSuiteInstantiation,
    PartialTensorShapeFromArrayShapeTest,
    ::testing::ValuesIn<PartialTensorShapeFromArrayShapeTestCase>({
        {
            "fully_defined",
            testing::CreateArrayShape({2, 3}),
            tensorflow::PartialTensorShape({2, 3}),
        },
        {
            "partially_defined",
            testing::CreateArrayShape({2, -1}),
            tensorflow::PartialTensorShape({2, -1}),
        },
        {
            "unknown",
            testing::CreateArrayShape({}, true),
            tensorflow::PartialTensorShape(),
        },
        {
            "scalar",
            testing::CreateArrayShape({}),
            tensorflow::PartialTensorShape({}),
        },
    }),
    [](const ::testing::TestParamInfo<
        PartialTensorShapeFromArrayShapeTest::ParamType>& info) {
      return info.param.test_name;
    });

struct TensorFromArrayTestCase {
  std::string test_name;
  const v0::Array array_pb;
  const tensorflow::Tensor expected_tensor;
};

using TensorFromArrayTest = ::testing::TestWithParam<TensorFromArrayTestCase>;

TEST_P(TensorFromArrayTest, TestReturnsTensor) {
  const TensorFromArrayTestCase& test_case = GetParam();

  const tensorflow::Tensor& actual_tensor =
      TFF_ASSERT_OK(TensorFromArray(test_case.array_pb));

  tensorflow::test::ExpectEqual(actual_tensor, test_case.expected_tensor);
}

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayTestSuiteInstantiation, TensorFromArrayTest,
    ::testing::ValuesIn<TensorFromArrayTestCase>({
        {
            "bool",
            testing::CreateArray(v0::DataType::DT_BOOL,
                                 testing::CreateArrayShape({}), {true})
                .value(),
            tensorflow::test::AsScalar(true),
        },
        {
            "int8",
            testing::CreateArray(v0::DataType::DT_INT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<int8_t>(1),
        },
        {
            "int16",
            testing::CreateArray(v0::DataType::DT_INT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<int16_t>(1),
        },
        {
            "int32",
            testing::CreateArray(v0::DataType::DT_INT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<int32_t>(1),
        },
        {
            "int64",
            testing::CreateArray(v0::DataType::DT_INT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<int64_t>(1),
        },
        {
            "uint8",
            testing::CreateArray(v0::DataType::DT_UINT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            testing::CreateArray(v0::DataType::DT_UINT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            testing::CreateArray(v0::DataType::DT_UINT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            testing::CreateArray(v0::DataType::DT_UINT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            tensorflow::test::AsScalar<uint64_t>(1),
        },
        {
            "float16",
            testing::CreateArray(v0::DataType::DT_HALF,
                                 testing::CreateArrayShape({}),
                                 {Eigen::half{1.0}})
                .value(),
            tensorflow::test::AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            testing::CreateArray(v0::DataType::DT_FLOAT,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
            tensorflow::test::AsScalar<float>(1.0),
        },
        {
            "float64",
            testing::CreateArray(v0::DataType::DT_DOUBLE,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
            tensorflow::test::AsScalar<double>(1.0),
        },
        {
            "complex64",
            testing::CreateArray(v0::DataType::DT_COMPLEX64,
                                 testing::CreateArrayShape({}),
                                 {std::complex<float>(1.0, 1.0)})
                .value(),
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            testing::CreateArray(v0::DataType::DT_COMPLEX128,
                                 testing::CreateArrayShape({}),
                                 {std::complex<double>(1.0, 1.0)})
                .value(),
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "string",
            testing::CreateArray(v0::DataType::DT_STRING,
                                 testing::CreateArrayShape({}), {"a"})
                .value(),
            tensorflow::test::AsScalar<tensorflow::tstring>("a"),
        },
        {
            "array",
            testing::CreateArray(v0::DataType::DT_INT32,
                                 testing::CreateArrayShape({2, 3}),
                                 {1, 2, 3, 4, 5, 6})
                .value(),
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
        },
    }),
    [](const ::testing::TestParamInfo<TensorFromArrayTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace tensorflow_federated
