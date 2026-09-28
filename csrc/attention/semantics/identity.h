// Author: Shicheng Wen

#pragma once

#include <cstdint>

namespace xattn {
namespace ops {
namespace attention {
namespace semantics {

enum class DirectionKind : std::uint8_t {
  kForward,
  kBackward,
};

enum class FormulaKind : std::uint8_t {
  kStandard,
  kSoftDelta,
};

enum class BoundaryKind : std::uint8_t {
  kInclusive,
  kStrictPast,
  kInclusiveStrictPastPair,
};

enum class PositionKind : std::uint8_t {
  kGlobal,
  kReset,
  kGlobalOrResetBundle,
};

enum class PartitionKind : std::uint8_t {
  kNone,
  kSegmentIndex,
  kVarlen,
  kDenseOrSegmentBundle,
};

enum class RouteKind : std::uint8_t {
  kDense,
  kSegment,
  kVarlen,
  kDenseOrSegmentBundle,
};

struct ForwardDirection {
  static constexpr DirectionKind kKind = DirectionKind::kForward;
};

struct BackwardDirection {
  static constexpr DirectionKind kKind = DirectionKind::kBackward;
};

struct StandardFormula {
  static constexpr FormulaKind kKind = FormulaKind::kStandard;
};

struct SoftDeltaFormula {
  static constexpr FormulaKind kKind = FormulaKind::kSoftDelta;
};

struct InclusiveBoundary {
  static constexpr BoundaryKind kKind = BoundaryKind::kInclusive;
};

struct StrictPastBoundary {
  static constexpr BoundaryKind kKind = BoundaryKind::kStrictPast;
};

struct InclusiveStrictPastBoundaryPair {
  static constexpr BoundaryKind kKind =
      BoundaryKind::kInclusiveStrictPastPair;
  using PrimaryBoundary = InclusiveBoundary;
  using CorrectionBoundary = StrictPastBoundary;
};

struct GlobalPosition {
  static constexpr PositionKind kKind = PositionKind::kGlobal;
};

struct ResetPosition {
  static constexpr PositionKind kKind = PositionKind::kReset;
};

struct GlobalOrResetPositionBundle {
  static constexpr PositionKind kKind =
      PositionKind::kGlobalOrResetBundle;
};

struct NoPartition {
  static constexpr PartitionKind kKind = PartitionKind::kNone;
};

struct SegmentIndexPartition {
  static constexpr PartitionKind kKind = PartitionKind::kSegmentIndex;
};

struct VarlenPartition {
  static constexpr PartitionKind kKind = PartitionKind::kVarlen;
};

struct DenseOrSegmentPartitionBundle {
  static constexpr PartitionKind kKind =
      PartitionKind::kDenseOrSegmentBundle;
};

struct DenseRoute {
  static constexpr RouteKind kKind = RouteKind::kDense;
};

struct SegmentRoute {
  static constexpr RouteKind kKind = RouteKind::kSegment;
};

struct VarlenRoute {
  static constexpr RouteKind kKind = RouteKind::kVarlen;
};

struct DenseOrSegmentRouteBundle {
  static constexpr RouteKind kKind = RouteKind::kDenseOrSegmentBundle;
};

struct RuntimeHeadMapping {};
struct RuntimeBackwardDeterminism {};
struct NoDeterminism {};

template <
    typename Direction, typename Formula, typename Visibility,
    typename Boundary, typename Position, typename Partition,
    typename Route, typename HeadMapping, typename Determinism>
struct AttentionSemanticIdentity {
  using DirectionPolicy = Direction;
  using FormulaPolicy = Formula;
  using VisibilityPolicy = Visibility;
  using BoundaryPolicy = Boundary;
  using PositionPolicy = Position;
  using PartitionPolicy = Partition;
  using RoutePolicy = Route;
  using HeadMappingPolicy = HeadMapping;
  using DeterminismPolicy = Determinism;
};

}  // namespace semantics
}  // namespace attention
}  // namespace ops
}  // namespace xattn
