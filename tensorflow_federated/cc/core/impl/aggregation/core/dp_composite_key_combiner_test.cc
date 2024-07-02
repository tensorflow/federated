/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using testing::internal::MakePredicateFormatterFromMatcher;

// Due to randomness in our L0 bounding algorithm, perform repeated tests.
constexpr int NUM_REPETITIONS = 25;

// The first batch of tests evaluate success of nontrivial L0 bounding, where
// "nontrivial" = the number of keys given as input exceeds the L0 bound.

TEST(DPCompositeKeyCombinerTest, AccumulateTwiceAndOutput_L0BoundIs1) {
  // For l0_bound_ = 1, the outcome of calling accumulate twice should be the
  // union of two singleton sets.

  // We will call Accumulate on Alice's data, then on Bob's
  std::initializer_list<string_view> alice_column1 = {"ripe", "old"};
  std::initializer_list<string_view> alice_column2 = {"tomato", "apple"};
  std::initializer_list<string_view> bob_column1 = {"ripe", "ripe"};
  std::initializer_list<string_view> bob_column2 = {"tomato", "apple"};

  //  Note that "ripe apple" cannot appear first because it only appears in the
  //  second call to Accumulate. Similarly, "old apple" cannot appear second.
  std::initializer_list<string_view> option0[] = {{"ripe"}, {"tomato"}};
  std::initializer_list<string_view> option1[] = {{"ripe", "ripe"},
                                                  {"tomato", "apple"}};
  std::initializer_list<string_view> option2[] = {{"old", "ripe"},
                                                  {"apple", "apple"}};
  std::initializer_list<string_view> option3[] = {{"old", "ripe"},
                                                  {"apple", "tomato"}};
  std::vector<std::initializer_list<string_view>*> options = {option0, option1,
                                                              option2, option3};

  for (int r = 0; r < NUM_REPETITIONS; r++) {
    DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_STRING, DT_STRING},
                                    1);

    Tensor alice_t1 = Tensor::Create(DT_STRING, {2},
                                     CreateTestData<string_view>(alice_column1))
                          .value();
    Tensor alice_t2 = Tensor::Create(DT_STRING, {2},
                                     CreateTestData<string_view>(alice_column2))
                          .value();
    StatusOr<Tensor> alice_result =
        combiner.Accumulate(InputTensorList({&alice_t1, &alice_t2}));
    ASSERT_OK(alice_result);

    Tensor bob_t1 =
        Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>(bob_column1))
            .value();
    Tensor bob_t2 =
        Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>(bob_column2))
            .value();
    StatusOr<Tensor> bob_result =
        combiner.Accumulate(InputTensorList({&bob_t1, &bob_t2}));
    ASSERT_OK(bob_result);

    OutputTensorList output = combiner.GetOutputKeys();
    EXPECT_THAT(output.size(), Eq(2));

    bool found_match;
    for (auto option : options) {
      found_match = true;
      for (int i = 0; i < 2; i++) {
        auto matcher = IsTensor<string_view>(
            {static_cast<int64_t>(option[i].size())}, option[i]);
        auto callable_matcher = MakePredicateFormatterFromMatcher(matcher);
        found_match = found_match && callable_matcher("output[i]", output[i]);
      }
      if (found_match) {
        break;
      }
    }
    EXPECT_TRUE(found_match);
  }
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndOutput_L0BoundIs1) {
  // Set up the data to feed into Tensor::Create and then Accumulate
  std::initializer_list<float> column1 = {1.1, 1.2, 1.3};
  std::initializer_list<string_view> column2 = {"abc", "de", ""};
  std::initializer_list<string_view> column3 = {"fghi", "jklmn", "o"};

  // We transform the initializer_lists into vectors to enable indexing
  std::vector<float> column1_vec(column1);
  std::vector<string_view> column2_vec(column2);
  std::vector<string_view> column3_vec(column3);

  // When selecting 1 key (row) out of 3, there are 3 possible ordinal
  // assignments (zero and negative 1)
  std::initializer_list<int64_t> possible_results[] = {
      {0, -1, -1}, {-1, 0, -1}, {-1, -1, 0}};

  for (int r = 0; r < NUM_REPETITIONS; r++) {
    DPCompositeKeyCombiner combiner(
        std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING}, 1);
    Tensor t1 =
        Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>(column1)).value();
    Tensor t2 =
        Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>(column2))
            .value();
    Tensor t3 =
        Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>(column3))
            .value();

    // Loop through the space of possible combinations: one of them should match
    StatusOr<Tensor> result =
        combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
    ASSERT_OK(result);

    bool valid_ordinals = false;
    for (auto possible_result : possible_results) {
      auto matcher = IsTensor<int64_t>({3}, possible_result);
      auto callable_matcher = MakePredicateFormatterFromMatcher(matcher);
      if (callable_matcher("result.value()", result.value())) {
        valid_ordinals = true;
        break;
      }
    }
    EXPECT_TRUE(valid_ordinals);

    // DPCompositeKeyCombiner should only have record of one composite key.
    // Loop through the columns: one of the combinations should match the record
    OutputTensorList output = combiner.GetOutputKeys();
    EXPECT_THAT(output.size(), Eq(3));

    float o0 = output[0].AsScalar<float>();
    string_view o1 = output[1].AsScalar<string_view>();
    string_view o2 = output[2].AsScalar<string_view>();

    bool valid_keys = false;
    for (int i = 0; i < 3; ++i) {
      if (o0 == column1_vec[i] && o1 == column2_vec[i] &&
          o2 == column3_vec[i]) {
        valid_keys = true;
        break;
      }
    }
    EXPECT_TRUE(valid_keys);
  }
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndOutput_L0BoundIs2) {
  std::initializer_list<float> column1 = {1.1, 1.2, 1.3};
  std::initializer_list<string_view> column2 = {"abc", "de", ""};
  std::initializer_list<string_view> column3 = {"fghi", "jklmn", "o"};

  std::vector<float> column1_vec(column1);
  std::vector<string_view> column2_vec(column2);
  std::vector<string_view> column3_vec(column3);

  // When selecting 2 keys (rows) out of 3, there are 6 possible ordinal
  // assignments because order matters
  std::initializer_list<int64_t> possible_results[] = {
      {0, 1, -1}, {1, 0, -1}, {0, -1, 1}, {1, -1, 0}, {-1, 0, 1}, {-1, 1, 0}};
  for (int r = 0; r < NUM_REPETITIONS; r++) {
    DPCompositeKeyCombiner combiner(
        std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING}, 2);
    Tensor t1 =
        Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>(column1)).value();
    Tensor t2 =
        Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>(column2))
            .value();
    Tensor t3 =
        Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>(column3))
            .value();

    StatusOr<Tensor> result =
        combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
    ASSERT_OK(result);

    bool valid_ordinals = false;
    for (auto possible_result : possible_results) {
      auto matcher = IsTensor<int64_t>({3}, possible_result);
      auto callable_matcher = MakePredicateFormatterFromMatcher(matcher);
      if (callable_matcher("result.value()", result.value())) {
        valid_ordinals = true;
        break;
      }
    }
    EXPECT_TRUE(valid_ordinals);

    // DPCompositeKeyCombiner should only have record of two composite keys.
    // Loop through the columns: one of the combinations should match record
    OutputTensorList output = combiner.GetOutputKeys();
    EXPECT_THAT(output.size(), Eq(3));

    auto o0 = output[0].AsSpan<float>();
    auto o1 = output[1].AsSpan<string_view>();
    auto o2 = output[2].AsSpan<string_view>();

    // original_row[r] = row of input that is in the r-th row of the output
    int original_row[] = {-1, -1};
    for (int row_of_output = 0; row_of_output < 2; row_of_output++) {
      for (int i = 0; i < 3; ++i) {
        if (o0[row_of_output] == column1_vec[i] &&
            o1[row_of_output] == column2_vec[i] &&
            o2[row_of_output] == column3_vec[i]) {
          original_row[row_of_output] = i;
          break;
        }
      }
    }
    EXPECT_TRUE(original_row[0] != original_row[1] && original_row[0] >= 0 &&
                original_row[1] >= 0);
  }
}

// When an l0_bound is not provided, DPCompositeKeyCombiner's behavior should be
// exactly the same as CompositeKeyCombiner's behavior.
// Hence, the tests below are duplicated from CompositeKeyCombiner.

TEST(DPCompositeKeyCombinerTest, EmptyInput_Invalid) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(DPCompositeKeyCombinerTest, InputWithWrongShapeTensor_Invalid) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData<int32_t>({1, 2, 3, 4}))
          .value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(DPCompositeKeyCombinerTest, InputWithTooFewTensorsInvalid) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(DPCompositeKeyCombinerTest, InputWithTooManyTensors_Invalid) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({4, 5, 6})).value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(DPCompositeKeyCombinerTest, InputWithWrongTypes_Invalid) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(DPCompositeKeyCombinerTest, OutputBeforeAccumulateOutputsEmptyTensor) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(1));
  EXPECT_THAT(output[0], IsTensor<float>({0}, {}));
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndOutput_SingleElement) {
  DPCompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1.3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({1}, {0}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(1));
  EXPECT_THAT(output[0], IsTensor<float>({1}, {1.3}));
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndOutput_NumericTypes) {
  DPCompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  Tensor t3 =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({4, 5, 6})).value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({3}, {0, 1, 2}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({3}, {1.1, 1.2, 1.3}));
  EXPECT_THAT(output[1], IsTensor<int32_t>({3}, {1, 2, 3}));
  EXPECT_THAT(output[2], IsTensor<int64_t>({3}, {4, 5, 6}));
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndOutput_StringTypes) {
  DPCompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 = Tensor::Create(DT_STRING, {3},
                             CreateTestData<string_view>({"abc", "de", ""}))
                  .value();
  Tensor t3 =
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>({"fghi", "jklmn", "o"}))
          .value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({3}, {0, 1, 2}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({3}, {1.1, 1.2, 1.3}));
  EXPECT_THAT(output[1], IsTensor<string_view>({3}, {"abc", "de", ""}));
  EXPECT_THAT(output[2], IsTensor<string_view>({3}, {"fghi", "jklmn", "o"}));
}

TEST(DPCompositeKeyCombinerTest,
     StringTypes_SameCompositeKeysResultInSameOrdinalsAcrossAccumulateCalls) {
  DPCompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1.1, 1.2, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_STRING, {4},
                     CreateTestData<string_view>({"abc", "de", "de", ""}))
          .value();
  Tensor t3 = Tensor::Create(
                  DT_STRING, {4},
                  CreateTestData<string_view>({"fghi", "jklmn", "jklmn", "o"}))
                  .value();
  StatusOr<Tensor> result1 =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result1);
  EXPECT_THAT(result1.value(), IsTensor<int64_t>({4}, {0, 1, 1, 2}));

  // Across different calls to Accumulate, tensors can have different shape.
  Tensor t4 = Tensor::Create(DT_FLOAT, {5},
                             CreateTestData<float>({1.3, 1.4, 1.1, 1.2, 1.1}))
                  .value();
  Tensor t5 = Tensor::Create(
                  DT_STRING, {5},
                  CreateTestData<string_view>({"", "abc", "abc", "de", "abc"}))
                  .value();
  Tensor t6 =
      Tensor::Create(
          DT_STRING, {5},
          CreateTestData<string_view>({"o", "pqrs", "fghi", "jklmn", "fghi"}))
          .value();
  StatusOr<Tensor> result2 =
      combiner.Accumulate(InputTensorList({&t4, &t5, &t6}));
  ASSERT_OK(result2);
  EXPECT_THAT(result2.value(), IsTensor<int64_t>({5}, {2, 3, 0, 1, 0}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({4}, {1.1, 1.2, 1.3, 1.4}));
  EXPECT_THAT(output[1], IsTensor<string_view>({4}, {"abc", "de", "", "abc"}));
  EXPECT_THAT(output[2],
              IsTensor<string_view>({4}, {"fghi", "jklmn", "o", "pqrs"}));
}

// Test that the ordinal returned by GetOrdinalFromDomainTensors is consistent
// with GetOutputKeys.
TEST(DPCompositeKeyCombinerTest, AccumulateAndGetOrdinal_NumericTypes) {
  for (int i = 0; i < NUM_REPETITIONS; i++) {
    // Make a combiner that makes <= 2 composite keys per Accumulate call
    DPCompositeKeyCombiner combiner(
        std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64}, 2);

    // Create tensors that we will use to call Accumulate.
    Tensor t1 =
        Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
            .value();
    Tensor t2 =
        Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3}))
            .value();
    Tensor t3 =
        Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({4, 5, 6}))
            .value();
    StatusOr<Tensor> result =
        combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));

    // Create 3 tensors which describe the domain of each key.
    // An extra tensor has an unused value. This is to show that GetOrdinal can
    // operate on a slice of intrinsic.parameters (ignore DP parameters).
    OutputTensorList domain_tensor_list;
    domain_tensor_list.push_back(
        Tensor::Create(DT_STRING, {1},
                       CreateTestData<string_view>({"to be skipped"}))
            .value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_FLOAT, {2}, CreateTestData<float>({0, 1.2})).value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_INT32, {1}, CreateTestData<int32_t>({2})).value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({5})).value());

    Tensor* ptr = &(domain_tensor_list[1]);
    TensorSpan domain_tensors(ptr, 3);

    // (0, 2, 5) was never Accumulated
    EXPECT_EQ(combiner.GetOrdinal(domain_tensors, {0, 0, 0}), kNoOrdinal);

    // (1.2, 2, 5) was either Accumulated or dropped (-1).
    int64_t ordinal = combiner.GetOrdinal(domain_tensors, {1, 0, 0});
    OutputTensorList output = combiner.GetOutputKeys();
    if (ordinal == kNoOrdinal) {
      // (1.2, 2, 5) should not be present in the output
      bool match_zero = (abs(1.2 - output[0].AsSpan<float>()[0]) < 1e-6) &&
                        (2 == output[1].AsSpan<int32_t>()[0]) &&
                        (5 == output[2].AsSpan<int64_t>()[0]);
      bool match_one = (abs(1.2 - output[0].AsSpan<float>()[1]) < 1e-6) &&
                       (2 == output[1].AsSpan<int32_t>()[1]) &&
                       (5 == output[2].AsSpan<int64_t>()[1]);
      EXPECT_FALSE(match_zero || match_one);
    } else {
      // (1.2, 2, 5) should be at the row indexed by ordinal
      EXPECT_TRUE(abs(1.2 - output[0].AsSpan<float>()[ordinal]) < 1e-6);
      EXPECT_EQ(2, output[1].AsSpan<int32_t>()[ordinal]);
      EXPECT_EQ(5, output[2].AsSpan<int64_t>()[ordinal]);
    }
  }
}

TEST(DPCompositeKeyCombinerTest, AccumulateAndGetOrdinal_StringTypes) {
  for (int i = 0; i < NUM_REPETITIONS; i++) {
    DPCompositeKeyCombiner combiner(
        std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING}, 2);
    Tensor t1 =
        Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
            .value();
    Tensor t2 = Tensor::Create(DT_STRING, {3},
                               CreateTestData<string_view>({"abc", "de", ""}))
                    .value();
    Tensor t3 =
        Tensor::Create(DT_STRING, {3},
                       CreateTestData<string_view>({"fghi", "jklmn", "o"}))
            .value();
    StatusOr<Tensor> result =
        combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));

    // Create 3 tensors which describe the domain of each key.
    // An extra tensor has an unused value. This is to show that GetOrdinal can
    // operate on a slice of intrinsic.parameters (ignore DP parameters).
    OutputTensorList domain_tensor_list;
    domain_tensor_list.push_back(
        Tensor::Create(DT_STRING, {1},
                       CreateTestData<string_view>({"to be skipped"}))
            .value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_FLOAT, {2}, CreateTestData<float>({0, 1.2})).value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"de"}))
            .value());
    domain_tensor_list.push_back(
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"jklmn"}))
            .value());

    Tensor* ptr = &(domain_tensor_list[1]);
    TensorSpan domain_tensors(ptr, 3);

    // (0, "de", "jklmn") was never Accumulated.
    EXPECT_EQ(combiner.GetOrdinal(domain_tensors, {0, 0, 0}), kNoOrdinal);

    // (1.2, "de", "jklmn") was either Accumulated or dropped.
    int64_t ordinal = combiner.GetOrdinal(domain_tensors, {1, 0, 0});
    OutputTensorList output = combiner.GetOutputKeys();
    if (ordinal == kNoOrdinal) {
      // (1.2, "de", "jklmn") should not be present in the output
      bool match_zero = (abs(1.2 - output[0].AsSpan<float>()[0]) < 1e-6) &&
                        ("de" == output[1].AsSpan<string_view>()[0]) &&
                        ("jklmn" == output[2].AsSpan<string_view>()[0]);
      bool match_one = (abs(1.2 - output[0].AsSpan<float>()[1]) < 1e-6) &&
                       ("de" == output[1].AsSpan<string_view>()[1]) &&
                       ("jklmn" == output[2].AsSpan<string_view>()[1]);
      EXPECT_FALSE(match_zero || match_one);
    } else {
      // (1.2, "de", "jklmn") should be at the row indexed by ordinal
      EXPECT_TRUE(abs(1.2 - output[0].AsSpan<float>()[ordinal]) < 1e-6);
      EXPECT_EQ("de", output[1].AsSpan<string_view>()[ordinal]);
      EXPECT_EQ("jklmn", output[2].AsSpan<string_view>()[ordinal]);
    }
  }
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
