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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

namespace tensorflow_federated {
namespace aggregation {

DPClosedDomainHistogram::DPClosedDomainHistogram(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
    TensorSpan domain_tensors, int num_inputs)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      l0_bound_(l0_bound),
      domain_tensors_(domain_tensors) {}

StatusOr<Tensor> DPClosedDomainHistogram::CreateOrdinalsByGroupingKeysForMerge(
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

StatusOr<OutputTensorList> DPClosedDomainHistogram::Report() && {
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

  // Future work will add noise to the aggregate.
  return noiseless_aggregate;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
