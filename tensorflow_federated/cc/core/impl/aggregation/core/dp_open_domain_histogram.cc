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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/partition-selection.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_string_data.h"

namespace tensorflow_federated {
namespace aggregation {

using ::differential_privacy::NumericalMechanism;
using ::differential_privacy::sign;

namespace internal {
using ::differential_privacy::GaussianMechanism;
using ::differential_privacy::GaussianPartitionSelection;
using ::differential_privacy::LaplaceMechanism;
using ::differential_privacy::SafeAdd;

// Struct to contain the components of the noise and threshold algorithm:
// - A pointer to a NumericalMechanism object which introduces DP noise for one
//   summation that satisfies replacement DP. The distribution will either be
//   Laplace or Gaussian, whichever has less variance for the same DP parameters
// - A threshold below which noisy sums will be erased. The thresholding step
//   consumes some or all of the delta that a customer provides.
// - Also holds a boolean to indicate which noise is used.
template <typename OutputType>
struct NoiseAndThresholdBundle {
  std::unique_ptr<NumericalMechanism> mechanism;
  OutputType threshold;
  bool use_laplace;
};

// Derive NoiseAndThresholdBundle from privacy parameters and clipping norms.
template <typename OutputType>
StatusOr<NoiseAndThresholdBundle<OutputType>> SetupNoiseAndThreshold(
    double epsilon, double delta, int64_t l0_bound, OutputType linfinity_bound,
    double l1_bound, double l2_bound) {
  // The following constraints on DP parameters should be caught beforehand in
  // factory code.
  TFF_CHECK(epsilon > 0 && delta > 0 && l0_bound > 0 && linfinity_bound > 0)
      << "epsilon, delta, l0_bound, and linfinity_bound must be greater than 0";
  TFF_CHECK(delta < 1) << "delta must be less than 1";

  // For Gaussian noise, the following parameter determines how much of delta is
  // consumed for thresholding. Currently set to 1/2 of delta, but this could be
  // optimized down the line.
  constexpr double kFractionForThresholding = 0.5;
  double delta_for_thresholding = delta * kFractionForThresholding;
  double delta_for_noising = delta - delta_for_thresholding;

  // Compute L1 sensitivity from the L0 and Linfinity bounds.
  // We target replacement DP, which means L1 sensitivity is twice the maximum
  // L1 norm of any contribution. The maximum L1 norm of any contribution can
  // be derived from l0_bound and linfinity_bound (or l1_bound if provided).
  double l1_sensitivity = 2.0 * l0_bound * linfinity_bound;
  // If an L1 bound was given and it is tighter than the above, use it.
  if (l1_bound > 0 && 2.0 * l1_bound < l1_sensitivity) {
    l1_sensitivity = 2.0 * l1_bound;
  }

  // Repeat for L2 sensitivity. To derive the expression, consider two
  // neighboring user inputs (1, 1, 1, 0, 0, 0, 0) and (0, 0, 0, 0, 1, 1, 1)
  // and fix linfinity_bound = 1 & l0_bound = 3. The L2 distance between these
  // vectors---and therefore the L2 sensitivity of the sum of vectors---is
  // sqrt(6 = 2 * l0_bound * linfinity_bound)
  double l2_sensitivity = sqrt(2.0 * l0_bound) * linfinity_bound;
  if (l2_bound > 0 && 2.0 * l2_bound < l2_sensitivity) {
    l2_sensitivity = 2.0 * l2_bound;
  }

  NoiseAndThresholdBundle<OutputType> output;

  // Pick the mechanism that will add noise with smaller standard deviation.
  TFF_CHECK(epsilon > 0) << "epsilon must be greater than 0";
  double laplace_scale = (1.0 / epsilon) * l1_sensitivity;
  double laplace_stdev = sqrt(2) * laplace_scale;
  double gaussian_stdev = GaussianMechanism::CalculateStddev(
      epsilon, delta_for_noising, l2_sensitivity);

  if (laplace_stdev < gaussian_stdev) {
    // If we are going to use Laplace noise,
    // 1. record that fact
    output.use_laplace = true;

    // 2. use our parameters to create an object that will add that noise.
    LaplaceMechanism::Builder laplace_builder;
    laplace_builder.SetL1Sensitivity(l1_sensitivity).SetEpsilon(epsilon);
    TFF_ASSIGN_OR_RETURN(output.mechanism, laplace_builder.Build());

    // 3. Calculate the threshold which we will impose on noisy sums.
    // Note that l0_sensitivity = 2 * l0_bound because we target replacement DP.
    TFF_ASSIGN_OR_RETURN(
        double library_threshold,
        CalculateLaplaceThreshold<OutputType>(epsilon, delta, 2 * l0_bound,
                                              linfinity_bound, l1_sensitivity));
    // Use ceil to err on the side of caution:
    // if noisy_val is an integer less than (double) library_threshold,
    // a cast of library_threshold may make them appear equal
    if (std::is_integral<OutputType>::value) {
      library_threshold = ceil(library_threshold);
    }
    output.threshold = static_cast<OutputType>(library_threshold);

    return output;
  }

  // If we are going to use Gaussian noise,
  // 1. record that fact
  output.use_laplace = false;

  // 2. use our parameters to create an object that will add that noise.
  GaussianMechanism::Builder gaussian_builder;
  gaussian_builder.SetStandardDeviation(gaussian_stdev);
  TFF_ASSIGN_OR_RETURN(output.mechanism, gaussian_builder.Build());

  // 3. Calculate the threshold which we will impose on noisy sums. We use
  // GaussianPartitionSelection::CalculateThresholdFromStddev. It assumes that
  // linfinity_bound = 1 but the only role linfinity_bound plays is as an
  // additive offset. So we can simply shift the number it produces to compute
  // the threshold.
  TFF_ASSIGN_OR_RETURN(
      double library_threshold,
      GaussianPartitionSelection::CalculateThresholdFromStddev(
          gaussian_stdev, delta_for_thresholding, 2 * l0_bound));
  // Use ceil to err on the side of caution:
  // if noisy_val is an integer less than (double) library_threshold,
  // a cast of library_threshold may make them appear equal
  if (std::is_integral<OutputType>::value) {
    library_threshold = ceil(library_threshold);
  }

  output.threshold =
      SafeAdd<OutputType>(linfinity_bound - 1,
                          static_cast<OutputType>(library_threshold))
          .value;

  return output;
}

// Noise is added to each value stored in a column tensor. If the noised value
// falls below a given threshold, then the index of that value is removed from a
// set of survivors.
// NoiseAndThreshold will be called by DPOpenDomainHistogram::Report on multiple
// tensors. Upon completion, Report will copy a value at index i in an output
// tensor of NoiseAndThreshold if i is in the set of survivors.
// NB: It is possible to write NoiseAndThreshold to cull values that lie below
// a threshold, but the i-th item of column 1 might not correspond to the i-th
// item of column 2. So we defer the culling step until we know all indices of
// the survivors.
// References: The document Delta_For_Thresholding.pdf found in
// https://github.com/google/differential-privacy/blob/main/common_docs/ has a
// proof for the case where inputs are positive; our use of sign() generalizes
// the analysis to the non-positive case.
template <typename OutputType>
StatusOr<Tensor> NoiseAndThreshold(
    double epsilon, double delta, int64_t l0_bound, OutputType linfinity_bound,
    double l1_bound, double l2_bound, const Tensor& column_tensor,
    absl::flat_hash_set<size_t>& survivor_indices,
    std::vector<bool>& laplace_was_used) {
  TFF_ASSIGN_OR_RETURN(
      auto bundle, SetupNoiseAndThreshold(epsilon, delta, l0_bound,
                                          linfinity_bound, l1_bound, l2_bound));
  laplace_was_used.push_back(bundle.use_laplace);

  OutputType threshold = bundle.threshold;

  auto column_span = column_tensor.AsSpan<OutputType>();
  auto noisy_values = std::make_unique<MutableVectorData<OutputType>>();
  noisy_values->reserve(column_span.size());

  // For every value in the column,
  for (size_t i = 0; i < column_span.size(); i++) {
    OutputType value = column_span[i];

    // Add noise and store noisy value
    OutputType noisy_value = (bundle.mechanism)->AddNoise(value);
    noisy_values->push_back(noisy_value);

    // If threshold is not crossed, index does not belong in survivor_indices
    OutputType sign_of_value = sign<OutputType>(value);
    if (sign_of_value * noisy_value < threshold) {
      survivor_indices.erase(i);
    }
  }
  return Tensor::Create(internal::TypeTraits<OutputType>::kDataType,
                        {static_cast<int64_t>(column_span.size())},
                        std::move(noisy_values));
}

// Given a column tensor, copy elements whose indices belong to a set of
// survivors to a new tensor.
template <typename OutputType>
StatusOr<Tensor> CopyOnlySurvivors(
    const Tensor& column_tensor,
    const absl::flat_hash_set<size_t>& survivor_indices) {
  auto column_span = column_tensor.AsSpan<OutputType>();
  auto dest = std::make_unique<MutableVectorData<OutputType>>();
  dest->reserve(survivor_indices.size());
  for (size_t i = 0; i < column_span.size(); i++) {
    if (!survivor_indices.contains(i)) {
      continue;
    }
    dest->push_back(column_span[i]);
  }
  return Tensor::Create(internal::TypeTraits<OutputType>::kDataType,
                        {static_cast<int64_t>(survivor_indices.size())},
                        std::move(dest));
}
// Specialization for string_view. First make a vector of surviving std::strings
// from the string_views and survivor_indices, then create a Tensor by moving
// the contents to an intermediate VectorStringData object.
template <>
StatusOr<Tensor> CopyOnlySurvivors<string_view>(
    const Tensor& column_tensor,
    const absl::flat_hash_set<size_t>& survivor_indices) {
  auto column_span = column_tensor.AsSpan<string_view>();
  std::vector<std::string> strings_for_output;
  strings_for_output.reserve(survivor_indices.size());
  for (size_t i = 0; i < column_span.size(); i++) {
    if (!survivor_indices.contains(i)) {
      continue;
    }
    strings_for_output.push_back(std::string(column_span[i]));
  }

  return Tensor::Create(
      DT_STRING, {static_cast<int64_t>(survivor_indices.size())},
      std::make_unique<VectorStringData>(std::move(strings_for_output)));
}
}  // namespace internal

DPOpenDomainHistogram::DPOpenDomainHistogram(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
    int num_inputs)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      l0_bound_(l0_bound) {}

std::unique_ptr<DPCompositeKeyCombiner>
DPOpenDomainHistogram::CreateDPKeyCombiner(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound) {
  // If there are no input keys, support a columnar aggregation that aggregates
  // all the values in each column and produces a single output value per
  // column. This would be equivalent to having identical key values for all
  // rows.
  if (input_key_specs.empty()) {
    return nullptr;
  }

  // Otherwise create a DP-ready key combiner
  return std::make_unique<DPCompositeKeyCombiner>(
      GroupByAggregator::CreateKeyTypes(input_key_specs.size(), input_key_specs,
                                        *output_key_specs),
      l0_bound);
}

StatusOr<Tensor> DPOpenDomainHistogram::CreateOrdinalsByGroupingKeysForMerge(
    const InputTensorList& inputs) {
  if (num_keys_per_input() > 0) {
    InputTensorList keys(num_keys_per_input());
    for (int i = 0; i < num_keys_per_input(); ++i) {
      keys[i] = inputs[i];
    }
    // Call the version of Accumulate that has no L0 bounding
    return key_combiner()->CompositeKeyCombiner::Accumulate(std::move(keys));
  }
  // If there are no keys, we should aggregate all elements in a column into one
  // element, as if there were an imaginary key column with identical values for
  // all rows.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(inputs[0]->num_elements());
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType,
                        inputs[0]->shape(), std::move(ordinals));
}

StatusOr<OutputTensorList> DPOpenDomainHistogram::Report() && {
  TFF_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPOpenDomainHistogram::Report: the report goal isn't met";
  }
  // Compute the noiseless aggregate.
  OutputTensorList noiseless_aggregate = std::move(*this).TakeOutputs();

  // We skip noise addition if epsilon is too large to be meaningful
  if (epsilon_per_agg_ > kEpsilonThreshold) {
    return noiseless_aggregate;
  }

  size_t num_aggregations = intrinsics().size();

  // TakeOutputs only includes a key if its name is in output_key_spec
  size_t num_output_keys = 0;
  for (int i = 0; i < output_key_specs().size(); ++i) {
    if (output_key_specs()[i].name().empty()) continue;
    num_output_keys++;
  }

  // Create a set of indices, one for each composite key in the histogram.
  absl::flat_hash_set<size_t> survivor_indices;
  for (size_t i = 0; i < noiseless_aggregate[0].num_elements(); i++) {
    survivor_indices.insert(i);
  }

  // For each aggregation, run NoiseAndThreshold to populate a list of Tensors
  OutputTensorList noisy_values;
  noisy_values.reserve(num_aggregations);
  for (int j = 0; j < num_aggregations; ++j) {
    const auto& inner_parameters = intrinsics()[j].parameters;
    const Tensor& linfinity_tensor = inner_parameters[kLinfinityIndex];
    double l1_bound = inner_parameters[kL1Index].CastToScalar<double>();
    double l2_bound = inner_parameters[kL2Index].CastToScalar<double>();
    size_t column = num_output_keys + j;
    StatusOr<Tensor> tensor;
    NUMERICAL_ONLY_DTYPE_CASES(
        noiseless_aggregate[column].dtype(), OutputType,
        TFF_ASSIGN_OR_RETURN(
            tensor, internal::NoiseAndThreshold<OutputType>(
                        epsilon_per_agg_, delta_per_agg_, l0_bound_,
                        linfinity_tensor.CastToScalar<OutputType>(), l1_bound,
                        l2_bound, noiseless_aggregate[column], survivor_indices,
                        laplace_was_used_)));
    noisy_values.push_back(std::move(tensor.value()));
  }

  // When there are no grouping keys, aggregation will be scalar. Hence, the
  // sole "group" does not need to be dropped for DP (because it exists whether
  // or not a given client contributed data)
  if (num_keys_per_input() == 0) {
    survivor_indices.insert(0);
  }

  // Produce a new list of tensors containing only the survivors of
  // thresholding
  OutputTensorList final_histogram;
  final_histogram.reserve(noiseless_aggregate.size());
  for (size_t j = 0; j < noiseless_aggregate.size(); j++) {
    // First batch of Tensors are for keys, second are for the values
    const Tensor& column_tensor = (j < num_output_keys)
                                      ? noiseless_aggregate[j]
                                      : noisy_values[j - num_output_keys];
    StatusOr<Tensor> tensor;
    DTYPE_CASES(
        column_tensor.dtype(), OutputType,
        TFF_ASSIGN_OR_RETURN(tensor, internal::CopyOnlySurvivors<OutputType>(
                                         column_tensor, survivor_indices)));
    final_histogram.push_back(std::move(tensor.value()));
  }
  return final_histogram;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
