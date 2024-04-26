/*
 * Copyright 2020 Google LLC
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

// Reusable matchers for unit testing. Useful for matching against protocol
// buffers without having to manually set up and call message differencers.
//   * EqualsProto(msg) -> matches against a message type
//   * EqualsProto(s) -> matches against a message specified by text proto
//
// The default match does precise equality comparison, but there are also
// decorator functions provided which can be used to modify the comparison logic
// that the EqualsProto matcher provides. They can be used by passing the result
// of EqualsProto into the matcher functions. For example if you use
// IgnoringRepeatedFieldOrdering(EqualsProto(s)) you will get matcher that
// matches against the text proto s, but treating repeated fields as sets
// instead of as lists.
//
// The decorator functions currently provided are:
//   * IgnoringRepeatedFieldOrdering -> treat repeated fields as sets
//
// The matching is based on what is provided by MessageDifferencer and so if new
// modifiers are needed they can be added with more decorators in the future.

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_PROTOBUF_MATCHERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_PROTOBUF_MATCHERS_H_

#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace tensorflow_federated {
namespace testing {
namespace internal_proto {

// Base implementation of a matcher for comparing a value against a protobuf
// message. This provide the comparison logic, but leaves the logic for
// constructing the actual protobuf value to be compared against as well as
// printing the expected value to the subclass. This allows for matchers that
// store different underlying data (e.g. strings vs actual messages).
class ProtoMatcherBase {
 public:
  virtual ~ProtoMatcherBase() = default;

  // Base matcher, that matches against any protobuf message type.
  bool MatchAndExplain(const google::protobuf::Message &arg,
                       ::testing::MatchResultListener *listener) const {
    if (!arg.IsInitialized()) {
      *listener << "which isn't fully initialized";
      return false;
    }

    ExpectedProtoPtr expected = CreateExpectedProto(arg, listener);
    if (expected == nullptr) return false;

    // Protobufs of different types cannot be compared. So first we check if
    // they are comparable, then we compare them.
    const bool comparable = arg.GetDescriptor() == expected->GetDescriptor();
    auto differencer = MakeDifferencer();
    const bool match = comparable && differencer->Compare(arg, *expected);

    // Explaining the match result is expensive.  We don't want to waste time
    // calculating an explanation if the listener isn't interested.
    if (listener->IsInterested()) {
      if (!comparable) {
        *listener << DescribeTypeMismatch(*expected, arg);
      } else if (!match) {
        *listener << DescribeDiff(*expected, arg);
      }
    }
    return match;
  }

  // Special overload of the matcher that also allows us to match against
  // pointers to protobufs. Written as a somewhat complex template that can
  // match against any pointer-like object (e.g. smart pointers).
  template <
      typename T,
      std::enable_if_t<
          // Check that the type can be dereferenced to get a Message.
          std::is_base_of<
              google::protobuf::Message,
              std::remove_reference_t<decltype(*std::declval<T>())>>::value &&
              // Check that the type can be compared against null.
              std::is_same<decltype(std::declval<T>() == nullptr), bool>::value,
          int> = 0>
  bool MatchAndExplain(const T &arg,
                       ::testing::MatchResultListener *listener) const {
    if (arg == nullptr) {
      *listener << "which is null";
      return false;
    }
    return MatchAndExplain(*arg, listener);
  }

  void DescribeTo(std::ostream *os) const {
    *os << "is fully initialized and ";
    DescribeRelationToExpectedProto(os);
  }

  void DescribeNegationTo(std::ostream *os) const {
    *os << "is not fully initialized or not ";
    DescribeRelationToExpectedProto(os);
  }

  // Mutators which can modify the proto field comparison properties. These
  // functions should not be utilized directly by users of the matchers. Instead
  // use the matcher functions defined in the public namespace.
  void SetIgnoreRepeatedFieldOrdering() {
    repeated_field_comparison_ = google::protobuf::util::MessageDifferencer::AS_SET;
  }

 protected:
  // A unique pointer type used by the CreateExpectedProto function. Instead of
  // the default deleter it requires the creators to construct a custom delete
  // function. This allows different subclasses to customize the delete policy
  // for example by having deletion be a no-op if the returned object is reused
  // across calls.
  using ExpectedProtoPtr =
      std::unique_ptr<const google::protobuf::Message, void (*)(const google::protobuf::Message *)>;

 private:
  // Print the expected protocol buffer value.
  virtual void PrintExpected(std::ostream *os) const = 0;

  // Returns the expected value as a protobuf object; if the object cannot be
  // created then this will explain why to 'listener' and return null.
  virtual ExpectedProtoPtr CreateExpectedProto(
      const google::protobuf::Message &arg,
      ::testing::MatchResultListener *listener) const = 0;

  // Describes the expected relation between the actual protobuf and the
  // expected one.
  void DescribeRelationToExpectedProto(std::ostream *os) const {
    *os << "equal ";
    if (repeated_field_comparison_ ==
        google::protobuf::util::MessageDifferencer::AS_SET) {
      *os << "(ignoring repeated field ordering) ";
    }
    *os << "to ";
    PrintExpected(os);
  }

  // Create a message differencer for use in comparisons.
  std::unique_ptr<google::protobuf::util::MessageDifferencer> MakeDifferencer() const {
    auto differencer = std::make_unique<google::protobuf::util::MessageDifferencer>();
    differencer->set_repeated_field_comparison(repeated_field_comparison_);
    return differencer;
  }

  // Provides a string describing a type mismatch between the two given protos.
  // Assumes that the types actually mismatch.
  std::string DescribeTypeMismatch(const google::protobuf::Message &expected,
                                   const google::protobuf::Message &actual) const {
    return "whose type should be " + expected.GetTypeName() +
           " but actually is " + actual.GetTypeName();
  }

  // Provides a string describing the difference between the two given protos.
  // Assumes that the types are actually different.
  std::string DescribeDiff(const google::protobuf::Message &expected,
                           const google::protobuf::Message &actual) const {
    auto differencer = MakeDifferencer();
    std::string diff;
    differencer->ReportDifferencesToString(&diff);

    // We must put 'expected' as the first argument here, as Compare() reports
    // the diff in terms of how the protobuf changes from the first argument to
    // the second argument.
    differencer->Compare(expected, actual);

    // Removes the trailing '\n' in the diff to make the output look nicer.
    if (diff.length() > 0 && *(diff.end() - 1) == '\n') {
      diff.erase(diff.end() - 1);
    }
    return "with the difference:\n" + diff;
  }

  // Flags to control the matching behavior.
  google::protobuf::util::MessageDifferencer::RepeatedFieldComparison
      repeated_field_comparison_ = google::protobuf::util::MessageDifferencer::AS_LIST;
};

// An implementation of the base matcher that matches against an actual protocol
// buffer message.
class ProtoMatcher : public ProtoMatcherBase {
 public:
  explicit ProtoMatcher(const google::protobuf::Message &expected)
      : expected_(CloneMessage(expected)) {}

 private:
  void PrintExpected(std::ostream *os) const override {
    *os << expected_->GetTypeName();
    std::string expected_string;
    if (google::protobuf::TextFormat::PrintToString(*expected_, &expected_string)) {
      *os << " -> {" << expected_string << "}";
    } else {
      *os << " (unknown value)";
    }
  }

  ExpectedProtoPtr CreateExpectedProto(
      const google::protobuf::Message &,
      ::testing::MatchResultListener *) const override {
    // Here we use a no-op deleter, because the returned object is a pointer to
    // the underlying stored message, which lives as long as the matcher does.
    return ExpectedProtoPtr(expected_.get(), [](const google::protobuf::Message *) {});
  }

  // Helper function that creates a copy of the given Message.
  static std::shared_ptr<const google::protobuf::Message> CloneMessage(
      const google::protobuf::Message &msg) {
    auto cloned = absl::WrapUnique(msg.New());
    cloned->CopyFrom(msg);
    return cloned;
  }

  // Store the actual underlying protocol buffer. We need to store the object on
  // the heap because we don't know the actual underlying type, only that it
  // supports the Message interface. We also need to use a shared pointer
  // because the matcher needs to be copyable. Since the underlying stored
  // message is const and never modified it is safe to have multiple copies of a
  // matcher pointing at the same underlying object.
  const std::shared_ptr<const google::protobuf::Message> expected_;
};

// An implementation of a proto matcher that stores the to-be-matched protobuf
// as a (text format) string.
class ProtoStringMatcher : public ProtoMatcherBase {
 public:
  explicit ProtoStringMatcher(std::string_view expected)
      : expected_(expected) {}

 private:
  void PrintExpected(std::ostream *os) const override {
    *os << "{" << expected_ << "}";
  }

  // Parses the expected string as a protobuf of the same type as arg and
  // returns the parsed protobuf (or null when the parse fails). Since a new
  // protobuf is constructed on each call the returned pointer actually takes
  // ownership and has a real deleter.
  ExpectedProtoPtr CreateExpectedProto(
      const google::protobuf::Message &arg,
      ::testing::MatchResultListener *listener) const override {
    // Define a collector that just accumulates a list of error message strings.
    class TextErrorCollector : public google::protobuf::io::ErrorCollector {
     public:
      const std::vector<std::string> &errors() const { return errors_; }

     private:
        void AddError(int line, int column,
                      const std::string& message) override {
        errors_.push_back(
            absl::StrFormat("line %d, column %d: %s", line, column, message));
      }
        void AddWarning(int line, int column,
                        const std::string& message) override {
        errors_.push_back(
            absl::StrFormat("line %d, column %d: %s", line, column, message));
      }
      std::vector<std::string> errors_;
    };
    TextErrorCollector collector;
    google::protobuf::TextFormat::Parser parser;
    parser.RecordErrorsTo(&collector);

    // Try the actual parse. If it fails, log all the recorded errors.
    auto expected_proto = absl::WrapUnique(arg.New());
    if (parser.ParseFromString(expected_, expected_proto.get())) {
      return ExpectedProtoPtr(expected_proto.release(),
                              [](const google::protobuf::Message *msg) { delete msg; });
    } else {
      if (listener->IsInterested()) {
        *listener << "where ";
        PrintExpected(listener->stream());
        *listener << " doesn't parse as a " << arg.GetTypeName() << ":\n"
                  << absl::StrJoin(collector.errors(), "\n");
      }
      return ExpectedProtoPtr(nullptr, [](const google::protobuf::Message *) {});
    }
  }

  const std::string expected_;
};

}  // namespace internal_proto

// Creates EqualsProto matches. They can be given either an actual Message
// object or a text proto.
inline ::testing::PolymorphicMatcher<internal_proto::ProtoMatcher> EqualsProto(
    const google::protobuf::Message &x) {
  return ::testing::MakePolymorphicMatcher(internal_proto::ProtoMatcher(x));
}
inline ::testing::PolymorphicMatcher<internal_proto::ProtoStringMatcher>
EqualsProto(std::string_view x) {
  return ::testing::MakePolymorphicMatcher(
      internal_proto::ProtoStringMatcher(x));
}

namespace proto {

// Decorators that take EqualsProto matchers and apply modifiers to them.
template <typename InnerMatcher>
inline InnerMatcher IgnoringRepeatedFieldOrdering(InnerMatcher inner_matcher) {
  inner_matcher.mutable_impl().SetIgnoreRepeatedFieldOrdering();
  return inner_matcher;
}

// Defines a helper matcher wrapper that simplifies the common pattern of
// ElementsAre(EqualsProto(...), EqualsProto(...), ...). Each argument provided
// to the template will be expanded into the corresponding EqualsProto argument.
//
// In other words, the code ElementsAreProtos(a, b, c) will be expanded into
// the equivalent ElementsAre(EqualsProto(a), EqualsProto(b), EqualsProto(c)).
template <typename... Args>
auto ElementsAreProtos(Args... args) {
  return ::testing::ElementsAre(EqualsProto(args)...);
}

// Defines an unordered equivalent to ElementsAreProtos.
template <typename... Args>
auto UnorderedElementsAreProtos(Args... args) {
  return ::testing::UnorderedElementsAre(EqualsProto(args)...);
}

}  // namespace proto
}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_PROTOBUF_MATCHERS_H_
