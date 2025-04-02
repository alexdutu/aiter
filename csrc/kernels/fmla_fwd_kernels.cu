// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>

// =====================================================================================================================
// Definitions and helper structures
//

template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN_,
          int32_t kNumWarps_>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kSizeD                  = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                 = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps               = kNumWarps_;
    static constexpr int32_t kNumThreads             = kNumWarps * ck_tile::get_warp_size();
    static constexpr int32_t kNumWarpsSoftmax        = 4;
    static constexpr int32_t kNumThreadsSoftmax      = kNumWarpsSoftmax * ck_tile::get_warp_size();
    static constexpr int32_t kNumWarpsCombine        = 2;
    static constexpr int32_t kNumThreadsCombine      = kNumWarpsCombine * ck_tile::get_warp_size();
    static constexpr int32_t kBlockM                 = kBlockM_;
    static constexpr int32_t kBlockN                 = kBlockN_;
    static constexpr int32_t kFixedOverheadNumBlocks = 5;
    static constexpr int32_t kMaxBatchSize           = 4096;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 64, 4>;

template <typename Traits, typename scalar_t, typename acc_t>
struct FlashMlaKernelPolicy
{
private:
    // Returns count of warps which don't contain any idle thread.
    template <int32_t NumWarps, int32_t M, int32_t N>
    CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
    {
        static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);
        constexpr int32_t ElemPerThread = (M * N) / (NumWarps * ck_tile::get_warp_size());
        if constexpr(0 < ElemPerThread)
        {
            return NumWarps;
        }
        else
        {
            return GetMaxNumWarpsForTile<NumWarps / 2, M, N>();
        }
    }

    // Returns vector size for given warp count for handing the specified matrix.
    template <int32_t NumWarps, int32_t M, int32_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
    {
        constexpr int32_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N>();
        constexpr int32_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());
        constexpr int32_t MaxNPerThread = 16 / sizeof(DataType);
        return ck_tile::min(MaxNPerThread, ElemPerThread);
    }

    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
    {
        constexpr int32_t kVectorN     = GetVectorSizeForTile<Traits::kNumWarpsCombine, 1, Traits::kSizeDV, DataType>();
        constexpr int32_t kThrPerWarpN = ck_tile::get_warp_size();
        constexpr int32_t kNumWarpN    = Traits::kNumWarpsCombine;

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,    // no replicate
                ck_tile::tuple<ck_tile::sequence<1>,
                               ck_tile::sequence<kNumWarpN, kThrPerWarpN, kVectorN>>,
                ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2>>,
                ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 2>>{});
    }

public:
    CK_TILE_DEVICE static auto MakeOaccuTileWindow(
        void* p_output_accum,
        const int32_t hsidx,
        const int32_t size_hs,
        const int32_t split_offset,
        const int32_t num_splits)
    {
        const int32_t offset_oaccum = split_offset * size_hs * Traits::kSizeDV;

        // Shape of tensor: [size_hs * num_splits, Traits::kSizeDV]
        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<acc_t*>(p_output_accum) + offset_oaccum,
                ck_tile::make_tuple(size_hs * num_splits, Traits::kSizeDV), // lengths
                ck_tile::make_tuple(Traits::kSizeDV, 1),                    // strides
                ck_tile::number<Traits::kSizeDV>{},                         // last dim alignment
                ck_tile::number<1>{});                                      // last dim stride

        // Each thread group handles tile whose shape is [1, Traits::kSizeDV]
        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {hsidx * Traits::kSizeDV, 0});                          // origin
        
        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<acc_t>());
    }

    CK_TILE_DEVICE static auto MakeOutputTileWindow(
        void* p_output,
        const int32_t offset_b,
        const int32_t offset_s,
        const int32_t offset_h)
    {
        scalar_t* p_out = reinterpret_cast<scalar_t*>(p_output) + offset_b + offset_s + offset_h;

        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_out,
                ck_tile::make_tuple(1, Traits::kSizeDV),    // lengths
                ck_tile::make_tuple(Traits::kSizeDV, 1),    // strides
                ck_tile::number<Traits::kSizeDV>{},         // last dim alignment
                ck_tile::number<1>{});                      // last dim stride
        
        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {0, 0});                                                // origin

        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<scalar_t>());
    }
};

union TileSchedulerMetaData
{
    struct Core
    {
        int32_t begin_batch_idx;
        int32_t begin_seqlen_idx;
        int32_t end_batch_idx;
        int32_t end_seqlen_idx;
        int32_t begin_n_split_idx;
    };
    uint32_t data[8];
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);

struct FlashMlaFwdParams
{
    int32_t* __restrict__ p_seqlens_k;
    int32_t* __restrict__ p_block_table;
    int32_t* __restrict__ p_tile_scheduler_metadata;
    int32_t* __restrict__ p_num_splits;
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;
    int32_t size_s;
    int32_t size_h;
    int32_t hq_hk_ratio;
    int32_t num_groups;
    int32_t num_cu_parts;
    int64_t block_table_batch_stride;
    int32_t page_block_size;
    float   scale_softmax;
    float   scale_softmax_log2;
    bool    is_causal;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;     // stride in batch of query
    index_t stride_s_q;     //    ... in sequence ...
    index_t stride_h_q;     //    ... in head ...
    index_t stride_b_k;     // stride in batch of key
    index_t stride_s_k;     //    ... in sequence ...
    index_t stride_h_k;     //    ... in head ...
    index_t stride_b_v;     // stride in batch of value
    index_t stride_s_v;     //    ... in sequence ...
    index_t stride_h_v;     //    ... in head ...
    index_t stride_b_o;     // stride in batch of output
    index_t stride_s_o;     //    ... in sequence ...
    index_t stride_h_o;     //    ... in head ...
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Traits>
__global__ void kn_get_mla_metadata(
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_num_splits,
    const int32_t* p_seqlens_k,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    __shared__ int lds_num_blocks[Traits::kMaxBatchSize];
    __shared__ int lds_num_splits[Traits::kMaxBatchSize];

    int32_t sum_blocks = 0;
    for (int32_t i = threadIdx.x; i < batch_size; i += ck_tile::get_warp_size())
    {
        const int32_t num_blocks = ck_tile::integer_divide_ceil(p_seqlens_k[i], Traits::kBlockN);
        sum_blocks += num_blocks + Traits::kFixedOverheadNumBlocks;
        lds_num_blocks[i] = num_blocks;
    }

    for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
    {
        sum_blocks += __shfl_xor(sum_blocks, offset);
    }

    if (threadIdx.x == 0)
    {
        // expected payload handled by each cu part.
        const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, num_cu_parts) +
                                Traits::kFixedOverheadNumBlocks;

        int32_t curr_batch = 0;         // batch ID of the batch which is under review
        int32_t curr_block = 0;         // #blocks handled by previous cu part(s)
        int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch
        int32_t cum_num_splits = 0;

        lds_num_splits[0] = 0;
        for (int32_t i = 0; i < num_cu_parts; ++i)
        {
            TileSchedulerMetaData::Core metadata;
            metadata.begin_batch_idx   = curr_batch;
            metadata.begin_seqlen_idx  = curr_block * Traits::kBlockN;
            metadata.begin_n_split_idx = curr_n_split_idx;

            int remain_payload = payload;
            while (curr_batch < batch_size)
            {
                const int32_t num_blocks = lds_num_blocks[curr_batch];
                const int32_t curr_remain_blocks = num_blocks - curr_block;

                // If current cu part is able to handle this batch of seqences
                if (remain_payload >= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks))
                {
                    cum_num_splits += (curr_n_split_idx + 1);
                    lds_num_splits[curr_batch + 1] = cum_num_splits;
                    remain_payload -= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks);
                    ++curr_batch;
                    curr_block = 0;
                    curr_n_split_idx = 0;
                }
                else
                {
                    if (remain_payload > Traits::kFixedOverheadNumBlocks)
                    {
                        curr_block += (remain_payload - Traits::kFixedOverheadNumBlocks);
                        ++curr_n_split_idx;
                    }
                    break;
                }
            }

            metadata.end_batch_idx  = curr_block > 0 ? curr_batch : (curr_batch - 1);
            metadata.end_seqlen_idx = curr_block > 0 ? curr_block * Traits::kBlockN : p_seqlens_k[curr_batch - 1];
            *reinterpret_cast<TileSchedulerMetaData::Core*>(
                p_tile_scheduler_metadata + i * TileSchedulerMetaDataSizeInDw) = metadata;
        }
    }

    for (int32_t i = threadIdx.x; i < batch_size; i += ck_tile::get_warp_size())
    {
        p_num_splits[i] = lds_num_splits[i];
    }
}

template <typename Traits, int32_t kMaxSplits, typename scalar_t>
__global__ void kn_fmla_fwd_splictkv_combine(
    const FlashMlaFwdParams params)
{
    using Policy  = FlashMlaKernelPolicy<Traits, scalar_t, float>;
    using index_t = int64_t;

    __shared__ float lds_lse_scale[kMaxSplits];

    const int32_t bidx = blockIdx.z;

    const int32_t split_offset = params.p_num_splits[bidx];
    const int32_t num_splits   = params.p_num_splits[bidx + 1] - split_offset;
    assert(num_splits <= kMaxSplits);

    if (num_splits > 1)
    {
        const int32_t lane_id          = ck_tile::get_lane_id();
        const int32_t hidx             = blockIdx.y;
        const int32_t sidx             = blockIdx.x;
        const int32_t hsidx            = hidx * params.size_s + sidx;
        const int32_t size_hs          = params.size_h * params.size_s;
        const index_t offset_lse_accum = split_offset * size_hs + hsidx;
        const index_t offset_lse       = bidx * size_hs + hsidx;

        if (ck_tile::get_warp_id() == 0)
        {
            const float* p_lse_accum = reinterpret_cast<float*>(params.p_softmax_lseaccum) + offset_lse_accum;
            float* p_lse             = reinterpret_cast<float*>(params.p_softmax_lse) + offset_lse;

            constexpr int32_t kNumLsePerThr = ck_tile::integer_divide_ceil(kMaxSplits, ck_tile::get_warp_size());
            float local_lse[kNumLsePerThr];

            // Load thread local LSE and get local max LSE
            float max_lse = -INFINITY;
            #pragma unroll
            for (int32_t i = 0; i < kNumLsePerThr; ++i)
            {
                const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
                const float lse = (split_idx < num_splits) ? p_lse_accum[split_idx * size_hs] : -INFINITY;
                local_lse[i] = lse;
                max_lse = ck_tile::max(max_lse, lse);
            }

            // Get global max LSE
            #pragma unroll
            for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
            {
                max_lse = ck_tile::max(max_lse, __shfl_xor(max_lse, offset));
            }

            // Get sum of LSE
            float sum_lse = 0.f;
            #pragma unroll
            for (int32_t i = 0; i < kNumLsePerThr; ++i)
            {
                sum_lse += expf(local_lse[i] - max_lse);
            }
            #pragma unroll
            for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
            {
                sum_lse += __shfl_xor(sum_lse, offset);
            }

            // Get global LSE
            float global_lse = ((sum_lse == 0.f) || (sum_lse != sum_lse)) ? INFINITY : (logf(sum_lse) + max_lse);
            if (lane_id == 0)
            {
                *p_lse = global_lse;
            }

            // Write LSE to LDS
            #pragma unroll
            for (int32_t i = 0; i < kNumLsePerThr; ++i)
            {
                const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
                if (split_idx < num_splits)
                {
                    lds_lse_scale[split_idx] = expf(local_lse[i] - global_lse);
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        ck_tile::block_sync_lds();

        static_assert(Traits::kSizeDV % Traits::kNumThreadsCombine == 0);

        auto oaccu_window =
            Policy::MakeOaccuTileWindow(params.p_output_accum, hsidx, size_hs, split_offset, num_splits);

        auto reg_out = ck_tile::make_static_distributed_tensor<float>(
            decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
        ck_tile::set_tile(reg_out, 0.f);

        for (int32_t split_idx = 0; split_idx < num_splits; ++split_idx)
        {
            const float lse_scale = lds_lse_scale[split_idx];
            auto oaccu = ck_tile::load_tile(oaccu_window);
            ck_tile::sweep_tile(oaccu, [&](auto idx) {
                reg_out(idx) = lse_scale * oaccu(idx);
            });
            ck_tile::move_tile_window(oaccu_window, {size_hs * Traits::kSizeDV, 0});
        }

        auto dram_out = Policy::MakeOutputTileWindow(params.p_output,
                                                     bidx * params.stride_b_o,
                                                     hidx * params.stride_h_o,
                                                     sidx * params.stride_s_o);
        ck_tile::store_tile(dram_out, ck_tile::cast_tile<scalar_t>(reg_out));
    }
}

// =====================================================================================================================
// Dispatches
//

template <typename Traits>
void dispatch_get_mla_metadata(
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_num_splits,
    const int32_t* p_cache_seqlens,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint32_t grid  = 1;
    const uint32_t block = ck_tile::get_warp_size();

    kn_get_mla_metadata<Traits><<<grid, block, 0, stream>>>(
        p_tile_scheduler_metadata,
        p_num_splits,
        p_cache_seqlens,
        batch_size,
        num_cu_parts);
}

template <typename Traits, typename scalar_t>
void dispatch_fmla_fwd_splictkv_combine(
    const FlashMlaFwdParams& params)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid  = dim3(params.size_s, params.size_h, params.size_b);
    const dim3 block = dim3(Traits::kNumThreadsCombine);

    if (params.num_cu_parts <= 1) return;

    if (params.num_cu_parts <= 32)
    {
        kn_fmla_fwd_splictkv_combine<Traits, 32, scalar_t><<<grid, block, 0, stream>>>(params);
    }
    else if (params.num_cu_parts <= 64)
    {
        kn_fmla_fwd_splictkv_combine<Traits, 64, scalar_t><<<grid, block, 0, stream>>>(params);
    }
    else if (params.num_cu_parts <= 96)
    {
        kn_fmla_fwd_splictkv_combine<Traits, 96, scalar_t><<<grid, block, 0, stream>>>(params);
    }
    else if (params.num_cu_parts <= 128)
    {
        kn_fmla_fwd_splictkv_combine<Traits, 128, scalar_t><<<grid, block, 0, stream>>>(params);
    }
    else
    {
        // TORCH_CHECK(false, "fmla_fwd_splictkv_combine cannot support the specified num_cu_parts ",
        //                    toString(params.num_cu_parts), ".");
        assert(false);
    }
}

#define DISPATCH_TYPES(TYPE, NAME, ...)                 \
    switch ((TYPE))                                     \
    {                                                   \
        case at::ScalarType::BFloat16:                  \
        {                                               \
            using scalar_t = ck_tile::bf16_t;           \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        case at::ScalarType::Half:                      \
        {                                               \
            using scalar_t = ck_tile::half_t;           \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        default:                                        \
            TORCH_CHECK(false, NAME " does't support ", \
                        at::toString((TYPE)), ".");     \
    }

// =====================================================================================================================
// Interfaces
//

std::vector<torch::Tensor> get_mla_metadata(
    const torch::Tensor& cache_seqlens,
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k)
{
    using Traits = FlashMlaKernelTraitsInstance;

    const torch::TensorOptions tensor_options = cache_seqlens.options();
    const int32_t batch_size = cache_seqlens.size(0);
    assert(batch_size <= Traits::kMaxBatchSize);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
    const int32_t cu_count = dev_prop.multiProcessorCount;
    const int32_t cu_parts = cu_count / num_heads_k /
                             ck_tile::integer_divide_ceil(num_heads_per_head_k, Traits::kBlockM);

    if (cu_parts > 1)
    {
        auto tile_scheduler_metadata = torch::empty({cu_parts, TileSchedulerMetaDataSizeInDw}, tensor_options);
        auto num_splits = torch::empty({batch_size + 1}, tensor_options);

        dispatch_get_mla_metadata<Traits>(
            tile_scheduler_metadata.data_ptr<int32_t>(),
            num_splits.data_ptr<int32_t>(),
            cache_seqlens.data_ptr<int32_t>(),
            batch_size,
            cu_parts);

        return {tile_scheduler_metadata, num_splits};
    }
    else
    {
        // For case of prefill, use regular solution rather than warp group solution.
        // Allocate specific size of metadata for hint of this perpose.
        auto tile_scheduler_metadata = torch::full({1, TileSchedulerMetaDataSizeInDw}, 0, tensor_options);
        auto num_splits = torch::full({1+1}, 0, tensor_options);

        return {tile_scheduler_metadata, num_splits};
    }
}

std::vector<torch::Tensor> flash_mla_fwd_with_kvcache_impl(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const int32_t        head_size_v,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const float          softmax_scale,
    const bool           is_causal,
    const torch::Tensor& tile_scheduler_metadata,
    const torch::Tensor& num_splits)
{
    using Traits = FlashMlaKernelTraitsInstance;

    FlashMlaFwdParams params = {};
    
    DISPATCH_TYPES(
        query.scalar_type(),
        "fmla_fwd",
        [&](){
            dispatch_fmla_fwd_splictkv_combine<Traits, scalar_t>(params);
        }(););
}
