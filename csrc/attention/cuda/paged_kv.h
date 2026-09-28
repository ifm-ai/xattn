/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "attention/cuda/detail/cute_compat.h"

namespace flash {

using namespace cute;

template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
    // KV_Same_Iter=false retains previous-iteration V pointers.
    // LoadsPerRow_LB: K-direction load lower bound per row.

    static constexpr bool SameHeadDim = (kHeadDim == kHeadDimV);
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);

    // Paged K/V cp.async path.
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    // 128-byte row partition.
    static_assert(kHeadDimGCD % LoadsPerRow_LB == 0, "Headdim and HeaddimV must be a multiple of LoadsPerRow_LB");
    static constexpr int kBytePerRow = kHeadDimGCD / LoadsPerRow_LB * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // Row loaders stay within one warp.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load
    using GmemTiledCopyKVStore = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;

    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
    using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));
    using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));
    using TensortVpV = decltype(make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{}));

    // Row-lane page-pointer partition.
    static_assert(CUTE_STATIC_V(size<1>(TensortKcK{})) == CUTE_STATIC_V(size<1>(TensortVcV{})));
    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));
    using TensorKVPtr = decltype(make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{}));

    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    cutlass::FastDivmod const &page_size_divmod;
    cutlass::FastDivmod const &blockN_per_page_size_divmod;
    int const thread_idx;
    int const seqlen_k;
    int const leftpad_k;
    int const* const ptr_page_table;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
    TensorPageTable mPageTable;
    TensorKV mK_paged, mV_paged;
    TensortKpK tKpK;
    TensortVpV tVpV;
    TensorPageOffset tPrPageOffset;
    TensorKVPtr tPrVPtr;
    int bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;  // Only used for TMA

    CUTLASS_DEVICE
    PagedKVManager(int const* const ptr_page_table_,
                   ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
                   Element* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
                   Element* const ptr_V, int const headdim_v, StrideKV const &stride_V,
                   cutlass::FastDivmod const &page_size_divmod,
                   cutlass::FastDivmod const &blockN_per_page_size_divmod,
                   int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k,
                   int bidb_kv_idx
                   )
        : page_size_divmod(page_size_divmod)
        , blockN_per_page_size_divmod(blockN_per_page_size_divmod)
        , thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , leftpad_k(leftpad_k)
        , ptr_page_table(ptr_page_table_)
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
        , bidb_kv_idx(bidb_kv_idx)
        , bidb_kv_idx_prev(bidb_kv_idx)

    {
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
        mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
        auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
        mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh, _);
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        #pragma unroll
        for (int k = 0; k < size<1>(tVpV_); ++k) { tVpV_(_0{}, k) = get<1>(tVcV(_0{}, _0{}, k)) < get<1>(shape_V); }
        tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
    };

    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table(const int n_block) {
        // Warp-local uncoalesced page-table loads.
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            // Compile-time row upper bound.
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_offset};
        }
        if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
    };

    template <bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table_TMA(const int n_block) {
        // Requires page_size % kBlockN == 0 and leftpad_k == 0.
        if (ptr_page_table) {
            bidb_kv_idx = mPageTable[blockN_per_page_size_divmod.divmod(n_block_idx, n_block)];
        } else {
            n_block_idx = n_block;
        }
        if constexpr (First_iter && !KV_Same_Iter) {
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
        }
    };

    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_K_TMA() {
        return {n_block_idx, bidb_kv_idx};
    };

    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_V_TMA() {
        if constexpr (KV_Same_Iter) {
            return {n_block_idx, bidb_kv_idx};
        } else {
            cute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
            return indices;
        }
    };

    CUTLASS_DEVICE
    TensorKVPtr compute_K_ptr() {
        Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
        }
        return tPrKPtr;
    };

    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
        }
    };

    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        Tensor tPrKPtr = compute_K_ptr();

        // Compile-time thread-zero indices.
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        // Thread-zero-relative row limit.
        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
            ? seqlen_k - n_block * kBlockN
            : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load = EvenN
                ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
                : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
                }
            }
        }
    };

    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // Compile-time thread-zero indices.
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);

        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // cp.async zero fill with final-row SMEM bounds.
            if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tVcV(_0{}, m, _0{})) < kBlockN) {
                bool const should_load = !Seqlenk_mask || get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
                Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
                Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tVsV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tVpV(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };

    template <typename TensorK>
    CUTLASS_DEVICE
    void store_K(const int n_block, TensorK &&tKrK) {
        Tensor tPrKPtr = compute_K_ptr();
        // Load-compatible thread-zero partition.
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        // Thread-zero-relative row limit.
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tKrK); ++m) {
            bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKrK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tKpK(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tKrK(_, m, k), mK_paged_cur_copy(_, ki));
                    }
                }
            }
        }
    };

    template <typename TensorV>
    CUTLASS_DEVICE
    void store_V(const int n_block, TensorV &&tVrV) {
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // Compile-time thread-zero indices.
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVrV); ++m) {
            bool const should_load = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
            Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tVrV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tVpV(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tVrV(_, m, k), mV_paged_cur_copy(_, ki));
                    }
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };


};

} // namespace flash
