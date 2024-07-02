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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_factory.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"

namespace tensorflow_federated {
namespace aggregation {

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::Create(
    const Intrinsic& intrinsic) const {
  return CreateInternal(intrinsic, nullptr);
}

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::Deserialize(
    const Intrinsic& intrinsic, std::string serialized_state) const {
  GroupByAggregatorState aggregator_state;
  if (!aggregator_state.ParseFromString(serialized_state)) {
    return TFF_STATUS(INVALID_ARGUMENT) << "DPGroupByFactory::Deserialize: "
                                           "Failed to parse serialized state.";
  }
  return CreateInternal(intrinsic, &aggregator_state);
}

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::CreateInternal(
    const Intrinsic& intrinsic,
    const GroupByAggregatorState* aggregator_state) const {
  // Check if the intrinsic is well-formed.
  TFF_RETURN_IF_ERROR(GroupByFactory::CheckIntrinsic(intrinsic, kDPGroupByUri));

  // DPGroupByAggregator expects parameters
  constexpr int64_t kEpsilonIndex = 0;
  constexpr int64_t kDeltaIndex = 1;
  constexpr int64_t kL0Index = 2;
  constexpr int kNumOpenDomainParameters = 3;

  // 0. Ensure that the parameters list has a chance of being well-formed.
  if (intrinsic.parameters.size() < kNumOpenDomainParameters) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Expected at least " << kNumOpenDomainParameters
           << " parameters"
              " but got "
           << intrinsic.parameters.size() << " of them.";
  }
  bool open_domain = intrinsic.parameters.size() == kNumOpenDomainParameters;
  int64_t num_keys = intrinsic.inputs.size();

  // 1. For the closed-domain case, ensure that we have the specification for
  // the domain of each of the keys.
  if (!open_domain) {
    // The number of domain tensors should be one more than the number of
    // grouping keys. The extra one contains the names of the keys (as strings).
    std::vector<DataType> expected_types(num_keys);
    for (auto i = 0; i < num_keys; i++) {
      expected_types[i] = intrinsic.inputs[i].dtype();
    }
    int64_t num_domain_tensors =
        intrinsic.parameters.size() - kNumOpenDomainParameters;
    if (num_keys + 1 != num_domain_tensors) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Expected " << num_keys + 1
             << " domain tensors but got " << num_domain_tensors << " of them.";
    }

    // The type of the first domain tensor should be string.
    DataType domain_type =
        intrinsic.parameters[kNumOpenDomainParameters].dtype();
    if (domain_type != DT_STRING) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: First domain tensor should have string type "
             << "but got " << DataType_Name(domain_type);
    }
    // Any other tensor should match corresponding grouping key.
    for (int i = 0; i < num_keys; i++) {
      domain_type =
          intrinsic.parameters[kNumOpenDomainParameters + i + 1].dtype();
      if (domain_type != expected_types[i]) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Domain tensor for key " << i
               << " should have type " << DataType_Name(expected_types[i])
               << " but got " << DataType_Name(domain_type) << " instead.";
      }
    }
  }

  // 2. For any DP histogram, ensure that we have required DP parameters.
  // Epsilon must be a positive number
  if (internal::GetTypeKind(intrinsic.parameters[kEpsilonIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be numerical.";
  }
  double epsilon = intrinsic.parameters[kEpsilonIndex].CastToScalar<double>();
  if (epsilon <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be positive.";
  }
  double epsilon_per_agg = epsilon / intrinsic.nested_intrinsics.size();

  // Delta must be a number
  if (internal::GetTypeKind(intrinsic.parameters[kDeltaIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Delta must be numerical.";
  }
  double delta = intrinsic.parameters[kDeltaIndex].CastToScalar<double>();
  if (open_domain) {
    // If open-domain, delta must lie between 0 and 1.
    if (delta <= 0 || delta >= 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: For open-domain DP histograms, delta must "
                "lie between 0 and 1.";
    }
  } else {
    // Else, delta must be less than 1. A non-positive delta requires a positive
    // L1 bound.
    if (delta >= 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: For closed-domain DP histograms, delta must "
                "be less than 1.";
    }
  }
  double delta_per_agg = delta / intrinsic.nested_intrinsics.size();

  // L0 bound must be a number
  if (internal::GetTypeKind(intrinsic.parameters[kL0Index].dtype()) !=
      internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: L0 bound must be numerical.";
  }
  // If open-domain, L0 bound must be positive
  int64_t l0_bound = intrinsic.parameters[kL0Index].CastToScalar<int64_t>();
  if (open_domain && l0_bound <= 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: For open-domain DP histograms, L0 bound must "
              "be positive.";
  }
  // If no keys are given, scalar aggregation will occur. There is exactly one
  // "group" in that case.
  if (intrinsic.inputs.empty()) {
    l0_bound = 1;
  }

  // 3. For any DP histogram, ensure that each inner aggregation is
  // well-specified.
  // Currently, we only support nested sums.
  // The following check will be updated when this changes.
  for (const auto& intrinsic : intrinsic.nested_intrinsics) {
    if (intrinsic.uri != kDPSumUri) {
      return TFF_STATUS(UNIMPLEMENTED) << "DPGroupByFactory: Currently, only "
                                          "nested DP sums are supported.";
    }

    // Verify presence of all norm bounds
    if (intrinsic.parameters.size() != kNumDPSumParameters) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupByFactory: Linfinity, L1, and L2 bounds are expected.";
    }

    // Verify that the norm bounds are in numerical Tensors
    for (const auto& parameter_tensor : intrinsic.parameters) {
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Norm bounds must be stored in"
                  " numerical Tensors.";
      }
    }

    const auto& linfinity_tensor = intrinsic.parameters[kLinfinityIndex];
    const double l1 = intrinsic.parameters[kL1Index].CastToScalar<double>();
    const double l2 = intrinsic.parameters[kL2Index].CastToScalar<double>();

    // Check if nested intrinsic provides a positive Linfinity bound
    bool has_linfinity_bound = false;
    DataType linfinity_dtype = linfinity_tensor.dtype();
    NUMERICAL_ONLY_DTYPE_CASES(
        linfinity_dtype, InputType,
        has_linfinity_bound = linfinity_tensor.CastToScalar<InputType>() > 0);
    if (open_domain) {
      if (!has_linfinity_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: For open-domain DP histograms, each "
                  "nested "
                  "intrinsic must provide a positive Linfinity bound.";
      }
    } else {
      // Either L1 is positive, or L2 is positive, or both Linfinity is positive
      // and L0 is positive.
      bool has_l1_bound = l1 > 0;
      bool has_l2_bound = l2 > 0;
      if ((!has_linfinity_bound || l0_bound <= 0) && !has_l1_bound &&
          !has_l2_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either an L1 bound, an L2 bound, or both Linfinity and L0 "
                  "bounds.";
      }
      // If delta is 0, we will employ the Laplace mechanism (Gaussian requires
      // a positive delta). But the Laplace mechanism requires a positive L1
      // bound.
      // If query author did not provide any delta (which will be indicated with
      // -1), we again have to use the Laplace mechanism.
      if (delta <= 0 && !has_l1_bound) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupByFactory: Closed-domain DP histograms require "
                  "either a positive delta or an L1 bound.";
      }
    }
  }

  // Create nested aggregators.
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  TFF_ASSIGN_OR_RETURN(nested_aggregators, GroupByFactory::CreateAggregators(
                                               intrinsic, aggregator_state));

  // Create the DP key combiner, and only populate the key combiner with state
  // if there are keys.
  auto key_combiner = DPOpenDomainHistogram::CreateDPKeyCombiner(
      intrinsic.inputs, &intrinsic.outputs, l0_bound);
  if (aggregator_state != nullptr && key_combiner != nullptr) {
    TFF_RETURN_IF_ERROR(GroupByFactory::PopulateKeyCombinerFromState(
        *key_combiner, *aggregator_state));
  }

  int num_inputs = aggregator_state ? aggregator_state->num_inputs() : 0;

  if (open_domain) {
    // Use new rather than make_unique here because the factory function that
    // uses a non-public constructor can't use std::make_unique, and we don't
    // want to add a dependency on absl::WrapUnique.
    return std::unique_ptr<DPOpenDomainHistogram>(new DPOpenDomainHistogram(
        intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
        std::move(key_combiner), std::move(nested_aggregators), epsilon_per_agg,
        delta_per_agg, l0_bound, num_inputs));
  }

  // Closed-domain case. We create a data structure containing the domain
  // spec out of intrinsic.parameters
  const Tensor* ptr = &(intrinsic.parameters[kNumOpenDomainParameters + 1]);
  TensorSpan domain_tensors(ptr, num_keys);

  return std::unique_ptr<DPClosedDomainHistogram>(new DPClosedDomainHistogram(
      intrinsic.inputs, &intrinsic.outputs, &(intrinsic.nested_intrinsics),
      std::move(key_combiner), std::move(nested_aggregators), epsilon_per_agg,
      delta_per_agg, l0_bound, domain_tensors, num_inputs));
}

REGISTER_AGGREGATOR_FACTORY(std::string(kDPGroupByUri), DPGroupByFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
