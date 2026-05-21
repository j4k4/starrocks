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

#include <glog/logging.h>

#include <cmath>
#include <memory>
#include <string>

#include "common/config.h"
#include "util/slice.h"

namespace starrocks {

uint8_t default_theta_lg_k() {
    // Resolved once: the config is not mutable, and a per-construction read would repeat the
    // range check and the warning on the hot path.
    static const uint8_t resolved = []() -> uint8_t {
        const int32_t configured = config::theta_sketch_default_lg_k;
        if (configured < datasketches::theta_constants::MIN_LG_K ||
            configured > datasketches::theta_constants::MAX_LG_K) {
            LOG(WARNING) << "theta_sketch_default_lg_k=" << configured << " is outside ["
                         << static_cast<int>(datasketches::theta_constants::MIN_LG_K) << ", "
                         << static_cast<int>(datasketches::theta_constants::MAX_LG_K) << "], falling back to "
                         << static_cast<int>(DEFAULT_THETA_LOG_K);
            return DEFAULT_THETA_LOG_K;
        }
        return static_cast<uint8_t>(configured);
    }();
    return resolved;
}

DataSketchesTheta::DataSketchesTheta(const Slice& src, int64_t* memory_usage) : _memory_usage(memory_usage) {
    if (!deserialize(src)) {
        DLOG(INFO) << "Failed to init DataSketchesTheta from slice, will be reset to 0.";
    }
}

void DataSketchesTheta::update(uint64_t hash_value) {
    auto* sk_update = get_sketch_update();
    sk_update->update(hash_value);
}

void DataSketchesTheta::merge(const DataSketchesTheta& other) {
    auto* sk_union = get_sketch_union();
    if (_sketch_update) {
        sk_union->update(*_sketch_update);
        _sketch_update.reset();
    }
    if (other._sketch_update != nullptr) {
        sk_union->update(*other._sketch_update);
    }
    if (other._sketch_union != nullptr) {
        sk_union->update(other._sketch_union->get_result());
    }
}

size_t DataSketchesTheta::serialize(uint8_t* dst) const {
    dst[0] = kMagicByte;
    dst[1] = _lg_k;
    if (_sketch_update) {
        auto bytes = _sketch_update->compact().serialize();
        std::copy(bytes.begin(), bytes.end(), dst + kHeaderSize);
        return kHeaderSize + bytes.size();
    } else if (_sketch_union) {
        auto bytes = _sketch_union->get_result().serialize();
        std::copy(bytes.begin(), bytes.end(), dst + kHeaderSize);
        return kHeaderSize + bytes.size();
    } else {
        theta_compact_sketch::vector_bytes bytes;
        std::copy(bytes.begin(), bytes.end(), dst + kHeaderSize);
        return kHeaderSize + bytes.size();
    }
}

uint64_t DataSketchesTheta::serialize_size() const {
    if (_sketch_update) {
        auto bytes = _sketch_update->compact().serialize();
        return kHeaderSize + bytes.size();
    } else if (_sketch_union) {
        auto bytes = _sketch_union->get_result().serialize();
        return kHeaderSize + bytes.size();
    } else {
        theta_compact_sketch::vector_bytes bytes;
        return kHeaderSize + bytes.size();
    }
}

bool DataSketchesTheta::deserialize(const Slice& slice) {
    if (slice.empty()) {
        return true;
    }
    const auto* data = reinterpret_cast<const uint8_t*>(slice.get_data());
    size_t size = slice.get_size();
    // StarRocks wraps the compact theta sketch with a 2-byte header
    // [magic=0xFE][lg_k] so that lg_k survives serialize/deserialize.
    // Legacy data (raw compact theta bytes) is detected by the absence
    // of the magic byte; compact-theta's first byte is pre_longs (1, 2, or 3).
    if (size >= 2 && data[0] == kMagicByte) {
        uint8_t lg_k = data[1];
        if (lg_k >= datasketches::theta_constants::MIN_LG_K && lg_k <= datasketches::theta_constants::MAX_LG_K) {
            _lg_k = lg_k;
        }
        data += 2;
        size -= 2;
    }
    // Header-only payloads (e.g. a state that has never seen update/merge) carry
    // no compact-sketch bytes; treat them as empty.
    if (size == 0) {
        return true;
    }
    try {
        auto sk = wrapped_compact_theta_sketch::wrap(data, size);
        get_sketch_union()->update(sk);
    } catch (std::logic_error& e) {
        DLOG(INFO) << "DataSketchesTheta deserialize error with exception:" << e.what();
        return false;
    }
    return true;
}

int64_t DataSketchesTheta::estimate_cardinality() const {
    // Round rather than truncate.
    //
    // get_estimate() returns a double and the return type here is int64_t, so an implicit
    // conversion would truncate toward zero. Truncation loses up to a whole distinct entry and
    // always in the same direction, which shows up as a systematic off-by-one against any
    // consumer that rounds: a sketch estimating 167161.72 answers 167161 while the caller that
    // rounds the same sketch answers 167162. Measured over real KPI slices in estimation mode,
    // truncation and rounding disagreed on two of three, because the disagreement happens
    // whenever the fractional part reaches 0.5 rather than in some rare corner.
    //
    // This deliberately differs from DataSketchesHll::estimate_cardinality, which truncates.
    // The theta family matches HLL on *return type* — BIGINT, so ds_theta_estimate and
    // ds_hll_estimate stay interchangeable in a query — but not on this, because theta sketches
    // here are compared against estimates produced elsewhere from the same sketch bytes and a
    // consistent rounding rule is what makes those comparable. Do not "restore parity" by
    // dropping the llround; change HLL instead if the two really have to agree.
    //
    // Rounding to an integer also makes the choice of estimator arithmetic irrelevant: the
    // reciprocal form (retained * (2^63 / theta)) and the division form (retained / (theta /
    // 2^63)) differ by at most one ULP, which cannot change a rounded result except for a value
    // sitting within one ULP of exactly .5.
    if (_sketch_update != nullptr) {
        return std::llround(_sketch_update->get_estimate());
    } else if (_sketch_union != nullptr) {
        return std::llround(_sketch_union->get_result().get_estimate());
    } else {
        return 0;
    }
}

} // namespace starrocks
