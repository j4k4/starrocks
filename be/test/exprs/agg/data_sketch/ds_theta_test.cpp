// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "exprs/agg/data_sketch/ds_theta.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "column/array_column.h"
#include "column/binary_column.h"
#include "column/column_builder.h"
#include "exprs/agg/base_aggregate_test.h"
#include "column/fixed_length_column.h"
#include "column/nullable_column.h"
#include "column/vectorized_fwd.h"
#include "exprs/agg/aggregate_factory.h"
#include "exprs/agg/aggregate_state_allocator.h"
#include "runtime/mem_pool.h"
#include "runtime/time_types.h"
#include "types/bitmap_value.h"
#include "util/slice.h"
#include "util/thrift_util.h"
#include "util/unaligned_access.h"

namespace starrocks {

class DataSketchsThetaTest : public testing::Test {
public:
    DataSketchsThetaTest() = default;

    void SetUp() override {
        utils = new FunctionUtils();
        ctx = utils->get_fn_ctx();
        _allocator = std::make_unique<CountingAllocatorWithHook>();
        tls_agg_state_allocator = _allocator.get();
    }
    void TearDown() override {
        delete utils;
        tls_agg_state_allocator = nullptr;
        _allocator.reset();
    }

private:
    FunctionUtils* utils{};
    FunctionContext* ctx{};
    std::unique_ptr<CountingAllocatorWithHook> _allocator;
};

TEST_F(DataSketchsThetaTest, TestSerializeDeserialize1) {
    int64_t memory_usage = 0;
    DataSketchesTheta theta(Slice(), &memory_usage);
    for (int i = 0; i < 100; i++) {
        theta.update(i);
    }
    uint8_t dst[1024];
    size_t size = theta.serialize(dst);
    ASSERT_EQ(size, theta.serialize_size());
    ASSERT_EQ(theta.estimate_cardinality(), 100);

    DataSketchesTheta theta2(Slice(dst, size), &memory_usage);
    ASSERT_EQ(theta2.serialize_size(), size);
    ASSERT_EQ(theta2.estimate_cardinality(), 100);
}

TEST_F(DataSketchsThetaTest, TestSerializeDeserialize2) {
    int64_t memory_usage = 0;
    DataSketchesTheta theta1(&memory_usage);
    DataSketchesTheta theta2(&memory_usage);
    for (int i = 0; i < 100; i++) {
        theta1.update(i);
        theta2.update(i);
    }
    DataSketchesTheta theta3(&memory_usage);
    theta3.merge(theta1);
    theta3.merge(theta2);

    uint8_t dst[1024];
    size_t size = theta3.serialize(dst);
    ASSERT_EQ(size, theta3.serialize_size());
    ASSERT_EQ(theta3.estimate_cardinality(), 100);

    // deserialize
    {
        DataSketchesTheta theta4(&memory_usage);
        theta4.deserialize(Slice(dst, size));
        ASSERT_EQ(theta4.serialize_size(), size);
        ASSERT_EQ(theta4.estimate_cardinality(), 100);
    }
    {
        DataSketchesTheta theta4(Slice(dst, size), &memory_usage);
        ASSERT_EQ(theta4.serialize_size(), size);
        ASSERT_EQ(theta4.estimate_cardinality(), 100);
    }
}

// The serialized payload must start with the 2-byte StarRocks header
// [0xFE][lg_k] so that lg_k survives serialize -> deserialize round-trips.
TEST_F(DataSketchsThetaTest, HeaderRoundTripPreservesLgK) {
    int64_t memory_usage = 0;
    // Deliberately not DEFAULT_THETA_LOG_K: a value equal to the default would let this pass
    // even if deserialize ignored the header entirely.
    constexpr uint8_t kCustomLgK = 16;
    DataSketchesTheta theta(kCustomLgK, &memory_usage);
    for (int i = 0; i < 500; ++i) {
        theta.update(i);
    }

    uint8_t dst[8192];
    size_t size = theta.serialize(dst);
    ASSERT_GE(size, DataSketchesTheta::kHeaderSize);
    EXPECT_EQ(dst[0], DataSketchesTheta::kMagicByte);
    EXPECT_EQ(dst[1], kCustomLgK);

    DataSketchesTheta restored(&memory_usage);
    ASSERT_TRUE(restored.deserialize(Slice(dst, size)));
    EXPECT_EQ(restored.get_lg_config_k(), kCustomLgK);
    // 500 distinct values is far below k, so the sketch is in exact mode and the estimate is
    // integral regardless of lg_k.
    EXPECT_EQ(restored.estimate_cardinality(), 500);
}

// Header-less payloads written by older StarRocks versions must still
// deserialize. Compact-theta's first byte is pre_longs in {1,2,3}, which can
// never collide with the magic byte 0xFE.
TEST_F(DataSketchsThetaTest, LegacyBytesWithoutHeaderDeserialize) {
    int64_t memory_usage = 0;
    DataSketchesTheta source(&memory_usage);
    for (int i = 0; i < 100; ++i) {
        source.update(i);
    }
    uint8_t framed[1024];
    size_t framed_size = source.serialize(framed);
    ASSERT_GT(framed_size, DataSketchesTheta::kHeaderSize);

    // Drop the 2-byte StarRocks header to mimic a pre-magic-byte payload.
    const uint8_t* legacy = framed + DataSketchesTheta::kHeaderSize;
    size_t legacy_size = framed_size - DataSketchesTheta::kHeaderSize;
    ASSERT_NE(legacy[0], DataSketchesTheta::kMagicByte);

    DataSketchesTheta restored(&memory_usage);
    ASSERT_TRUE(restored.deserialize(Slice(legacy, legacy_size)));
    EXPECT_EQ(restored.estimate_cardinality(), 100);
    // Header-less payloads carry no lg_k, so deserialization keeps the resolved default. This is
    // the case that makes theta_sketch_default_lg_k load-bearing rather than cosmetic.
    EXPECT_EQ(restored.get_lg_config_k(), default_theta_lg_k());
}

// During merge() in the aggregate path, an empty state adopts the lg_k carried
// in the wire format of the incoming sketch. This keeps the configured
// precision intact across shuffle boundaries.
TEST_F(DataSketchsThetaTest, DeserializeAdoptsIncomingLgK) {
    int64_t memory_usage = 0;
    // Must differ from DEFAULT_THETA_LOG_K for the assertion below to mean anything.
    constexpr uint8_t kIncomingLgK = 16;
    DataSketchesTheta incoming(kIncomingLgK, &memory_usage);
    for (int i = 0; i < 100; ++i) {
        incoming.update(i);
    }

    uint8_t buf[2048];
    size_t size = incoming.serialize(buf);

    DataSketchesTheta parsed(Slice(buf, size), &memory_usage);
    EXPECT_EQ(parsed.get_lg_config_k(), kIncomingLgK);

    // Mirror what ThetaSketchAggregateFunction::merge() does when the state is
    // still empty: construct using the incoming sketch's lg_k, then merge.
    DataSketchesTheta state(parsed.get_lg_config_k(), &memory_usage);
    state.merge(parsed);
    EXPECT_EQ(state.get_lg_config_k(), kIncomingLgK);
    EXPECT_EQ(state.estimate_cardinality(), 100);
}

// A corrupted or malicious lg_k byte must not poison the state: we keep the
// default and still parse the body (the theta-union update will fail safely
// only on truly malformed bytes).
TEST_F(DataSketchsThetaTest, OutOfRangeLgKInHeaderFallsBackToDefault) {
    int64_t memory_usage = 0;
    DataSketchesTheta source(&memory_usage);
    for (int i = 0; i < 50; ++i) {
        source.update(i);
    }
    uint8_t framed[1024];
    size_t size = source.serialize(framed);

    // 0xFF is outside [MIN_LG_K=5, MAX_LG_K=26].
    framed[1] = 0xFF;

    DataSketchesTheta restored(&memory_usage);
    ASSERT_TRUE(restored.deserialize(Slice(framed, size)));
    EXPECT_EQ(restored.get_lg_config_k(), default_theta_lg_k());
    EXPECT_EQ(restored.estimate_cardinality(), 50);
}

// A state that never saw update() or merge() serializes to just the header.
// Round-tripping that 2-byte payload must not throw and must keep the
// estimate at zero.
TEST_F(DataSketchsThetaTest, EmptyStateSerializeRoundTrip) {
    int64_t memory_usage = 0;
    DataSketchesTheta empty(&memory_usage);
    EXPECT_EQ(empty.estimate_cardinality(), 0);

    uint8_t dst[16];
    size_t size = empty.serialize(dst);
    EXPECT_EQ(size, DataSketchesTheta::kHeaderSize);
    EXPECT_EQ(dst[0], DataSketchesTheta::kMagicByte);

    DataSketchesTheta restored(&memory_usage);
    ASSERT_TRUE(restored.deserialize(Slice(dst, size)));
    EXPECT_EQ(restored.estimate_cardinality(), 0);
}

// estimate_cardinality() rounds the double estimate instead of letting the implicit
// conversion to int64_t truncate it. Asserted as a property against llround of the
// sketch's own estimate rather than against hardcoded numbers, so the test survives a
// datasketches upgrade that shifts the estimates.
//
// lg_k is pinned explicitly rather than left to default_theta_lg_k(), so the cardinalities below
// stay above k and the sketches stay in estimation mode whatever theta_sketch_default_lg_k is set
// to. Estimation mode is what gives the estimate a fractional part at all; an exact-mode sketch is
// integral, and rounds and truncates identically.
TEST_F(DataSketchsThetaTest, EstimateCardinalityRoundsRatherThanTruncates) {
    constexpr uint8_t kLgK = 12;
    bool saw_a_fraction_that_rounds_up = false;

    for (int distinct_count : {10000, 25000, 50000, 100000, 250000}) {
        int64_t memory_usage = 0;
        DataSketchesTheta theta(kLgK, &memory_usage);
        for (int i = 0; i < distinct_count; ++i) {
            theta.update(i);
        }

        const double raw = theta.get_sketch_update()->get_estimate();
        ASSERT_NE(raw, std::trunc(raw)) << "estimation mode should not produce an integral estimate";
        EXPECT_EQ(theta.estimate_cardinality(), std::llround(raw)) << "at " << distinct_count;

        if (raw - std::floor(raw) >= 0.5) {
            saw_a_fraction_that_rounds_up = true;
            // the case truncation would have got wrong
            EXPECT_NE(theta.estimate_cardinality(), static_cast<int64_t>(raw)) << "at " << distinct_count;
        }
    }

    // Without this the loop above would still pass if every estimate happened to round down,
    // which would make it blind to a regression back to truncation.
    ASSERT_TRUE(saw_a_fraction_that_rounds_up)
            << "no sampled cardinality produced a fraction >= 0.5, so nothing here distinguishes "
               "rounding from truncation; pick different cardinalities";
}

// default_theta_lg_k() resolves the BE config, so it must always land inside the range the sketch
// builder accepts. A value outside it would otherwise surface as a throw from the builder on the
// first update rather than as a configuration error.
TEST_F(DataSketchsThetaTest, DefaultLgKResolvesInsideTheAcceptedRange) {
    const uint8_t resolved = default_theta_lg_k();
    EXPECT_GE(resolved, datasketches::theta_constants::MIN_LG_K);
    EXPECT_LE(resolved, datasketches::theta_constants::MAX_LG_K);

    // Unless be.conf says otherwise, the resolved value is the library default. Tests run with
    // stock config, so this pins that the config plumbing does not silently substitute something
    // else.
    EXPECT_EQ(resolved, DEFAULT_THETA_LOG_K);

    // A default-constructed sketch picks the resolved value up, which is the property the union
    // path over header-less sketches depends on.
    int64_t memory_usage = 0;
    DataSketchesTheta theta(&memory_usage);
    EXPECT_EQ(theta.get_lg_config_k(), resolved);
}

// Merging sketches of differing lg_k must reach the same state whichever order they arrive in,
// because under a distributed shuffle the arrival order is not reproducible. The observable
// consequence is the lg_k the state serializes with, so that is what this asserts.
//
// This exercises ThetaSketchAggregateFunction::merge rather than DataSketchesTheta, because the
// promotion rule lives in the aggregate.
TEST_F(DataSketchsThetaTest, MergeReachesTheSameLgKWhicheverOrderSketchesArrive) {
    int64_t memory_usage = 0;
    constexpr uint8_t kCoarseLgK = 12;
    constexpr uint8_t kFineLgK = 16;

    uint8_t coarse[8192];
    uint8_t fine[8192];
    size_t coarse_size = 0;
    size_t fine_size = 0;
    {
        DataSketchesTheta coarse_sketch(kCoarseLgK, &memory_usage);
        DataSketchesTheta fine_sketch(kFineLgK, &memory_usage);
        for (int i = 0; i < 300; ++i) {
            coarse_sketch.update(i);
            fine_sketch.update(i + 150);
        }
        coarse_size = coarse_sketch.serialize(coarse);
        fine_size = fine_sketch.serialize(fine);
    }
    ASSERT_EQ(coarse[1], kCoarseLgK);
    ASSERT_EQ(fine[1], kFineLgK);

    // Merges the two payloads in the given order and reports the lg_k byte the resulting state
    // serializes with.
    auto merged_lg_k = [](const uint8_t* first, size_t first_size, const uint8_t* second,
                          size_t second_size) -> uint8_t {
        std::vector<TypeDescriptor> arg_types = {TypeDescriptor::from_logical_type(TYPE_VARCHAR)};
        auto return_type = TypeDescriptor::from_logical_type(TYPE_BIGINT);
        std::unique_ptr<FunctionContext> ctx(
                FunctionContext::create_test_context(std::move(arg_types), return_type));

        const AggregateFunction* func =
                get_aggregate_function("ds_theta_count_distinct", TYPE_VARCHAR, TYPE_BIGINT, false);
        auto state = ManagedAggrState::create(ctx.get(), func);

        auto input = BinaryColumn::create();
        input->append(Slice(first, first_size));
        input->append(Slice(second, second_size));

        func->merge(ctx.get(), input.get(), state->state(), 0);
        func->merge(ctx.get(), input.get(), state->state(), 1);

        MutableColumnPtr out = BinaryColumn::create();
        func->serialize_to_column(ctx.get(), state->state(), out.get());

        Slice serialized = down_cast<BinaryColumn*>(out.get())->get_slice(0);
        EXPECT_GE(serialized.size, DataSketchesTheta::kHeaderSize);
        EXPECT_EQ(static_cast<uint8_t>(serialized.data[0]), DataSketchesTheta::kMagicByte);

        return static_cast<uint8_t>(serialized.data[1]);
    };

    EXPECT_EQ(merged_lg_k(fine, fine_size, coarse, coarse_size), kFineLgK);
    // The order that first-arrival adoption got wrong: the coarse sketch would have pinned the
    // state at lg_k 12 and the finer sketch would have been merged into it.
    EXPECT_EQ(merged_lg_k(coarse, coarse_size, fine, fine_size), kFineLgK);
}

// The union path is a separate branch of estimate_cardinality() and has to round too.
TEST_F(DataSketchsThetaTest, EstimateCardinalityRoundsOnTheUnionPath) {
    constexpr uint8_t kLgK = 12;
    int64_t memory_usage = 0;
    DataSketchesTheta left(kLgK, &memory_usage);
    DataSketchesTheta right(kLgK, &memory_usage);
    for (int i = 0; i < 60000; ++i) {
        left.update(i);
        right.update(i + 30000);
    }

    DataSketchesTheta merged(kLgK, &memory_usage);
    merged.merge(left);
    merged.merge(right);

    const double raw = merged.get_sketch_union()->get_result().get_estimate();
    ASSERT_NE(raw, std::trunc(raw));
    EXPECT_EQ(merged.estimate_cardinality(), std::llround(raw));
}
} // namespace starrocks
