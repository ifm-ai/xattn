#pragma once

#include "softdelta/fwd_tile_scheduler.h"

namespace xattn {
namespace ops {

class FlashSoftDeltaSingleCTAChunkOrderedTileScheduler {
 public:
  using SharedStorage = int;
  static constexpr bool HasMBlockRange = false;
  static constexpr bool RequiresProducerWarp1 = false;

  struct Params {
    int total_tiles;
    int num_blocks;
    int num_heads;
    int grid_size;
    int phase_count;
    int remaining_blocks;
    int full_rows;
    int residual_phases;
    int short_column_tiles;
    bool use_chunk_order;
    cutlass::FastDivmod batch_tiles_divmod;
    cutlass::FastDivmod head_divmod;
    cutlass::FastDivmod short_rows_divmod;
    cutlass::FastDivmod long_rows_divmod;
  };

  static Params to_underlying_arguments(
      flash::TileSchedulerArguments const& args,
      int attention_chunk,
      int num_sm) {
    assert(args.qhead_per_khead % 2 == 0);
    assert(attention_chunk > 0);
    const int total_tiles =
        args.num_blocks * args.num_head * args.num_batch;
    const int grid_size = total_tiles < num_sm ? total_tiles : num_sm;
    const int phase_count = (attention_chunk + 63) / 64;
    const bool use_chunk_order =
        phase_count > 2 && args.num_blocks > 2 * phase_count;
    const int remaining_blocks = use_chunk_order
        ? args.num_blocks - phase_count
        : 0;
    const int full_rows = use_chunk_order
        ? remaining_blocks / phase_count
        : 1;
    const int residual_phases = use_chunk_order
        ? remaining_blocks % phase_count
        : 0;
    const int short_column_tiles =
        (phase_count - residual_phases) * full_rows;
    return {
        total_tiles,
        args.num_blocks,
        args.num_head,
        grid_size,
        phase_count,
        remaining_blocks,
        full_rows,
        residual_phases,
        short_column_tiles,
        use_chunk_order,
        cutlass::FastDivmod(args.num_blocks * args.num_head),
        cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(full_rows),
        cutlass::FastDivmod(full_rows + 1)};
  }

  static dim3 get_grid_shape(Params const& params, int) {
    return {static_cast<uint32_t>(params.grid_size)};
  }

  struct WorkTileInfo {
    int tile_idx;
    int wave;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_tiles;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      int ordered_tile = tile_idx;
      if (params.use_chunk_order) {
        const int wave_begin = wave * params.grid_size;
        const int remaining_tiles = params.total_tiles - wave_begin;
        const int wave_size = remaining_tiles < params.grid_size
            ? remaining_tiles
            : params.grid_size;
        const int rotation =
            ((wave >> 1) * (2 * params.num_heads)) % wave_size;
        int wave_tile = tile_idx - wave_begin + rotation;
        if (wave_tile >= wave_size) {
          wave_tile -= wave_size;
        }
        ordered_tile = wave_begin + wave_tile;
      }

      int batch_tile;
      const int batch = params.batch_tiles_divmod.divmod(
          batch_tile, ordered_tile);
      int head;
      const int block_order = params.head_divmod.divmod(
          head, batch_tile);
      int block;
      if (!params.use_chunk_order) {
        block = params.num_blocks - 1 - block_order;
      } else if (block_order < params.remaining_blocks) {
        int row_rank;
        int phase_rank;
        int row;
        int phase;
        if (block_order < params.short_column_tiles) {
          phase_rank = params.short_rows_divmod.divmod(
              row_rank, block_order);
          phase = params.phase_count - 1 - phase_rank;
          row = params.full_rows - 1 - row_rank;
        } else {
          phase_rank = params.long_rows_divmod.divmod(
              row_rank,
              block_order - params.short_column_tiles);
          phase = params.residual_phases - 1 - phase_rank;
          row = params.full_rows - row_rank;
        }
        block = params.phase_count + row * params.phase_count + phase;
      } else {
        block = params.phase_count - 1 -
            (block_order - params.remaining_blocks);
      }
      return {block, 2 * head, batch, 0};
    }
  };

  CUTLASS_DEVICE
  explicit FlashSoftDeltaSingleCTAChunkOrderedTileScheduler(
      SharedStorage*) {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const&) const {
    return {int(blockIdx.x), 0};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const&, WorkTileInfo&) const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE
  WorkTileInfo get_next_work(
      Params const&, WorkTileInfo const& current_work) const {
    const int block_count = int(gridDim.x);
    const int block_idx = int(blockIdx.x);
    const int wave = current_work.wave + 1;
    const int wave_block_idx = wave & 1
        ? block_count - 1 - block_idx
        : block_idx;
    return {
        wave * block_count + wave_block_idx,
        wave};
  }
};

}  // namespace ops
}  // namespace xattn
