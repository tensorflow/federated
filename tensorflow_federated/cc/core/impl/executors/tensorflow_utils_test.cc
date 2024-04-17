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

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"
#include "tensorflow_federated/proto/v0/array.pb.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {
namespace {

using testing::EqualsProto;

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_FullyDefined) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({2, 3});
  const tensorflow::TensorShape& expected_shape =
      tensorflow::TensorShape({2, 3});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_Scalar) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({});
  const tensorflow::TensorShape& expected_shape = tensorflow::TensorShape({});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_PartiallyDefined) {
  const v0::ArrayShape& shape_pb = testing::CreateArrayShape({2, -1});

  const absl::StatusOr<tensorflow::TensorShape>& result =
      TensorShapeFromArrayShape(shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_Unknown) {
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

struct ArrayFromTensorTestCase {
  std::string test_name;
  const tensorflow::Tensor tensor;
  const v0::Array expected_array_pb;
};

using ArrayFromTensorTest = ::testing::TestWithParam<ArrayFromTensorTestCase>;

TEST_P(ArrayFromTensorTest, TestReturnsTensor) {
  const ArrayFromTensorTestCase& test_case = GetParam();

  const v0::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromTensor(test_case.tensor));

  EXPECT_THAT(actual_array_pb, EqualsProto(test_case.expected_array_pb));
}

INSTANTIATE_TEST_SUITE_P(
    ArrayFromTensorTestSuiteInstantiation, ArrayFromTensorTest,
    ::testing::ValuesIn<ArrayFromTensorTestCase>({
        {
            "bool",
            tensorflow::test::AsScalar(true),
            testing::CreateArray(v0::DataType::DT_BOOL,
                                 testing::CreateArrayShape({}), {true})
                .value(),
        },
        {
            "int8",
            tensorflow::test::AsScalar<int8_t>(1),
            testing::CreateArray(v0::DataType::DT_INT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int16",
            tensorflow::test::AsScalar<int16_t>(1),
            testing::CreateArray(v0::DataType::DT_INT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int32",
            tensorflow::test::AsScalar<int32_t>(1),
            testing::CreateArray(v0::DataType::DT_INT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int64",
            tensorflow::test::AsScalar<int64_t>(1),
            testing::CreateArray(v0::DataType::DT_INT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint8",
            tensorflow::test::AsScalar<uint8_t>(1),
            testing::CreateArray(v0::DataType::DT_UINT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint16",
            tensorflow::test::AsScalar<uint16_t>(1),
            testing::CreateArray(v0::DataType::DT_UINT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint32",
            tensorflow::test::AsScalar<uint32_t>(1),
            testing::CreateArray(v0::DataType::DT_UINT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint64",
            tensorflow::test::AsScalar<uint64_t>(1),
            testing::CreateArray(v0::DataType::DT_UINT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "float16",
            tensorflow::test::AsScalar(Eigen::half{1.0}),
            testing::CreateArray(v0::DataType::DT_HALF,
                                 testing::CreateArrayShape({}),
                                 {Eigen::half{1.0}})
                .value(),
        },
        {
            "float32",
            tensorflow::test::AsScalar<float>(1.0),
            testing::CreateArray(v0::DataType::DT_FLOAT,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
        },
        {
            "float64",
            tensorflow::test::AsScalar<double>(1.0),
            testing::CreateArray(v0::DataType::DT_DOUBLE,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
        },
        {
            "complex64",
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
            testing::CreateArray(v0::DataType::DT_COMPLEX64,
                                 testing::CreateArrayShape({}),
                                 {std::complex<float>(1.0, 1.0)})
                .value(),
        },
        {
            "complex128",
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
            testing::CreateArray(v0::DataType::DT_COMPLEX128,
                                 testing::CreateArrayShape({}),
                                 {std::complex<double>(1.0, 1.0)})
                .value(),
        },
        // {
        //     "string",
        //     tensorflow::test::AsScalar<tensorflow::tstring>("a"),
        //     testing::CreateArray(v0::DataType::DT_STRING,
        //                          testing::CreateArrayShape({}), {"a"})
        //         .value(),
        // },
        {
            "array",
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
            testing::CreateArray(v0::DataType::DT_INT32,
                                 testing::CreateArrayShape({2, 3}),
                                 {1, 2, 3, 4, 5, 6})
                .value(),
        },
    }),
    [](const ::testing::TestParamInfo<ArrayFromTensorTest::ParamType>& info) {
      return info.param.test_name;
    });

// TEST(ArrayFromTensorTest, TestReturnsArray_Bool) {
//   const tensorflow::Tensor tensor = tensorflow::test::AsScalar(true);
//   const v0::Array expected_pb =
//       testing::CreateArray(v0::DataType::DT_BOOL,
//       testing::CreateArrayShape({}),
//                            {true})
//           .value();

//   const v0::Array& actual_pb = TFF_ASSERT_OK(ArrayFromTensor(tensor));

//   EXPECT_THAT(actual_pb, testing::EqualsProto(expected_pb));
// }

TEST(ArrayContentFromTensorTest, TestReturnsArrayContent_Bool) {
  const tensorflow::Tensor tensor = tensorflow::test::AsScalar(true);
  v0::Array expected_array_pb;
  expected_array_pb.set_dtype(v0::DataType::DT_BOOL);
  *expected_array_pb.mutable_shape() = testing::CreateArrayShape({});
  *expected_array_pb.mutable_content() = "\001";

  const v0::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayContentFromTensor(tensor));

  EXPECT_THAT(actual_array_pb, testing::EqualsProto(expected_array_pb));
}

TEST(TensorFromArrayContentTest, TestReturnsTensor_Bool) {
  v0::Array array_pb;
  array_pb.set_dtype(v0::DataType::DT_BOOL);
  *array_pb.mutable_shape() = testing::CreateArrayShape({});
  *array_pb.mutable_content() = "\001";
  const tensorflow::Tensor expected_tensor = tensorflow::test::AsScalar(true);

  const tensorflow::Tensor& actual_tensor =
      TFF_ASSERT_OK(TensorFromArrayContent(array_pb));

  tensorflow::test::ExpectEqual(actual_tensor, expected_tensor);
}

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

// TEST(TensorFromArrayTest, TestReturns) {
//   v0::Array array_pb;
//   array_pb.set_dtype(v0::DataType::DT_INT32);
//   *array_pb.mutable_shape() = testing::CreateArrayShape({2, 3});
//   constexpr char content[] =
//       "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000"
//       "\000\000\006\000\000\000";
//   array_pb.mutable_content()->assign(content, sizeof(content) - 1);
//   // v0::Array array_pb = TFF_ASSERT_OK(testing::CreateArrayContent(
//   //     v0::DataType::DT_INT32, testing::CreateArrayShape({2, 3}),
//   //
//   "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000"
//   //     "\000\000\006\000\000\000"));

//   const tensorflow::Tensor& actual_tensor =
//       TFF_ASSERT_OK(TensorFromArray(array_pb));

//   tensorflow::Tensor tensor = tensorflow::test::AsTensor<int32_t>(
//       {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3}));
//   tensorflow::TensorProto tensor_pb;
//   tensor.AsProtoTensorContent(&tensor_pb);
//   tensorflow::Tensor expected_tensor(tensor.dtype());
//   EXPECT_TRUE(expected_tensor.FromProto(tensor_pb));

//   tensorflow::test::ExpectEqual(actual_tensor, expected_tensor);

//   tensorflow::TensorProto a_pb;
//   tensor.AsProtoField(&a_pb);
//   LOG(ERROR) << "--- a_pb";
//   LOG(ERROR) << a_pb;

//   tensorflow::TensorProto b_pb;
//   tensor.AsProtoTensorContent(&b_pb);
//   LOG(ERROR) << "--- b_pb";
//   LOG(ERROR) << b_pb;
//   EXPECT_THAT(a_pb, testing::EqualsProto(b_pb));

//   // tensorflow::TensorProto actual_pb;
//   // actual_tensor.AsProtoTensorContent(&actual_pb);
//   // EXPECT_THAT(actual_pb, testing::EqualsProto(tensor_pb));
// }

}  // namespace
}  // namespace tensorflow_federated
