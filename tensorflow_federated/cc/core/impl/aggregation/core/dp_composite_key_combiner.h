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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_COMPOSITE_KEY_COMBINER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_COMPOSITE_KEY_COMBINER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// Indicate that a composite key is not mapped to a valid ordinal.
constexpr int64_t kNoOrdinal = -1;
// Default l0_bound_ value.
constexpr int64_t kDefaultL0Bound = -1;
// Shorthand for a span of Tensors
using TensorSpan = absl::Span<const Tensor>;

// Child class of CompositeKeyCombiner that enforces contribution bounding
// inside Accumulate: ensures that the number of unique composite keys that are
// mapped to non-negative ordinals is bounded by l0_bound_. Classes that use the
// outcome of Accumulate should interpret -1 as "skip".
// This class is not thread safe.
class DPCompositeKeyCombiner : public CompositeKeyCombiner {
 public:
  // DPCompositeKeyCombiner is not copyable or moveable.
  DPCompositeKeyCombiner(const DPCompositeKeyCombiner&) = delete;
  DPCompositeKeyCombiner& operator=(const DPCompositeKeyCombiner&) = delete;
  DPCompositeKeyCombiner(DPCompositeKeyCombiner&&) = delete;
  DPCompositeKeyCombiner& operator=(DPCompositeKeyCombiner&&) = delete;

  // Creates a CompositeKeyCombiner if inputs are valid or crashes otherwise.
  explicit DPCompositeKeyCombiner(const std::vector<DataType>& dtypes,
                                  int64_t l0_bound = kDefaultL0Bound);

  // If an l0_bound_ was not given or number of contributions is <= l0_bound_,
  // call parent's Accumulate. Otherwise, call AccumulateWithBound which will
  // ensure there are <= l0_bound_ contributions.
  StatusOr<Tensor> Accumulate(const InputTensorList& tensors) override;

  // AccumulateWithBound will first create the set of unique composite keys in
  // the input, in parallel with a vector of composite keys. Then it samples a
  // subset of l0_bound_ composite keys ("survivors") from the whole set.
  // Finally, it loops through the input again to map composite keys (stored in
  // its vector) to ordinals. Composite keys that are not survivors map to -1.
  // It is the responsibility of the calling code to not use them as indices;
  // -1 simply indicates that a row of data should be skipped in an inner
  // aggregation
  StatusOr<Tensor> AccumulateWithBound(const InputTensorList& tensors,
                                       TensorShape& shape, size_t num_elements);

  // Given a span of tensors (where each describes a key's domain) and indices
  // to tensor values, make a CompositeKey out of the data at those indices and
  // then retrieve the ordinal associated with that composite key (or
  // kNoOrdinal).
  // This function will be used within the Report() method of the closed-domain
  // DP histogram aggregation core.
  int64_t GetOrdinal(TensorSpan domain_tensors,
                     const absl::FixedArray<size_t>& indices);

 private:
  const int64_t l0_bound_;
  absl::BitGen bitgen_;

  // Given a span of tensors (where each describes a key's domain) and indices
  // to tensor values, make a CompositeKey out of the data at those indices.
  CompositeKey MakeCompositeKeyFromDomainTensors(
      const TensorSpan& domain_tensors,
      const absl::FixedArray<size_t>& indices);

  // Friend class that supports the operations done in
  // DPCompositeKeyCombiner::AccumulateWithBound.
  // Populates a map associating ordinals that are only meaningful in one call
  // of Accumulate with ordinals that are meaningful across calls to Accumulate.
  friend class LocalToGlobalInserter;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_COMPOSITE_KEY_COMBINER_H_
