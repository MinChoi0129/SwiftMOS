#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "atomics.cuh"

template<typename scalar_t>
inline scalar_t* DATA_PTR(const at::Tensor& t) {
    return t.data_ptr<scalar_t>();
}

// maxpool
namespace maxpool {
    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputComputeIdx(real* pcds_ind_data, int64_t* voxel_max_idx_data,
        int64_t BS, int64_t /*C*/, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / N;
            int64_t n = i - bs * N;

            int64_t index_ind = i * D;
            int64_t index_voxel = bs * voxel_out_stride[0]; // c=0

            int flag = 1;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) {
                    index_voxel += ind_tmp * voxel_out_stride[2 + d];
                } else {
                    flag = 0;
                }
            }
            if (flag == 1) voxel_max_idx_data[i] = index_voxel;
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputInit(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_max_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_max_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                voxel_out_data[index_voxel0 + c * voxel_out_stride[1]] = pcds_feat_data[i];
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputKernel(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_max_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_max_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                atomMax(&voxel_out_data[index_voxel0 + c * voxel_out_stride[1]], pcds_feat_data[i]);
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateBackwardKernel(real* pcds_feat_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data, int64_t* voxel_max_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_max_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                int64_t index_voxel = index_voxel0 + c * voxel_out_stride[1];
                if (voxel_out_data[index_voxel] == pcds_feat_data[i]) {
                    grad_pcds_feat_data[i] = grad_voxel_out_data[index_voxel];
                }
            }
        }
    }
}

// minpool
namespace minpool {
    template<typename real>
    __global__ void VoxelMinPoolUpdateOutputComputeIdx(real* pcds_ind_data, int64_t* voxel_min_idx_data,
        int64_t BS, int64_t /*C*/, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / N;
            int64_t n = i - bs * N;

            int64_t index_ind = i * D;
            int64_t index_voxel = bs * voxel_out_stride[0]; // c=0

            int flag = 1;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) {
                    index_voxel += ind_tmp * voxel_out_stride[2 + d];
                } else {
                    flag = 0;
                }
            }
            if (flag == 1) voxel_min_idx_data[i] = index_voxel;
        }
    }

    template<typename real>
    __global__ void VoxelMinPoolUpdateOutputInit(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_min_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_min_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                voxel_out_data[index_voxel0 + c * voxel_out_stride[1]] = pcds_feat_data[i];
            }
        }
    }

    template<typename real>
    __global__ void VoxelMinPoolUpdateOutputKernel(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_min_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_min_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                atomMin(&voxel_out_data[index_voxel0 + c * voxel_out_stride[1]], pcds_feat_data[i]);
            }
        }
    }

    template<typename real>
    __global__ void VoxelMinPoolUpdateBackwardKernel(real* pcds_feat_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data, int64_t* voxel_min_idx_data,
        int64_t BS, int64_t C, int64_t N, int64_t /*D*/, int64_t loop,
        int64_t* /*voxel_out_size*/, int64_t* voxel_out_stride, int64_t* /*output_size*/)
    {
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i += blockDim.x * gridDim.x) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_voxel0 = voxel_min_idx_data[bs * N + n];
            if (index_voxel0 >= 0) {
                int64_t index_voxel = index_voxel0 + c * voxel_out_stride[1];
                if (voxel_out_data[index_voxel] == pcds_feat_data[i]) {
                    grad_pcds_feat_data[i] = grad_voxel_out_data[index_voxel];
                }
            }
        }
    }
}

void voxel_maxpooling_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
    at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C  = pcds_feat.size(1);
    int64_t N  = pcds_feat.size(2);
    int64_t D  = pcds_ind.size(2);

    int64_t loop1 = BS * N;
    int64_t loop2 = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateOutput", [&] {
        auto* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        auto* pcds_ind_data  = DATA_PTR<scalar_t>(pcds_ind);
        auto* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        auto* voxel_max_idx_data = DATA_PTR<int64_t>(voxel_max_idx);

        maxpool::VoxelMaxPoolUpdateOutputComputeIdx<scalar_t><<<BLOCKS(loop1), THREADS>>>(
            pcds_ind_data, voxel_max_idx_data, BS, C, N, D, loop1,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size), DATA_PTR<float>(scale_rate));

        maxpool::VoxelMaxPoolUpdateOutputInit<scalar_t><<<BLOCKS(loop2), THREADS>>>(
            pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size));

        maxpool::VoxelMaxPoolUpdateOutputKernel<scalar_t><<<BLOCKS(loop2), THREADS>>>(
            pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size));
    });
}

void voxel_maxpooling_cuda_backward(at::Tensor pcds_feat, at::Tensor /*pcds_ind*/, at::Tensor voxel_out, at::Tensor voxel_max_idx,
    at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor /*scale_rate*/)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C  = pcds_feat.size(1);
    int64_t N  = pcds_feat.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMaxPoolBackward", [&] {
        auto* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        auto* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        auto* voxel_max_idx_data = DATA_PTR<int64_t>(voxel_max_idx);
        auto* grad_pcds_feat_data = DATA_PTR<scalar_t>(grad_pcds_feat);
        auto* grad_voxel_out_data = DATA_PTR<scalar_t>(grad_voxel_out);

        maxpool::VoxelMaxPoolUpdateBackwardKernel<scalar_t><<<BLOCKS(loop), THREADS>>>(
            pcds_feat_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data, voxel_max_idx_data,
            BS, C, N, 0, loop, DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));
    });
}

void voxel_minpooling_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_min_idx,
    at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C  = pcds_feat.size(1);
    int64_t N  = pcds_feat.size(2);
    int64_t D  = pcds_ind.size(2);

    int64_t loop1 = BS * N;
    int64_t loop2 = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMinPoolUpdateOutput", [&] {
        auto* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        auto* pcds_ind_data  = DATA_PTR<scalar_t>(pcds_ind);
        auto* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        auto* voxel_min_idx_data = DATA_PTR<int64_t>(voxel_min_idx);

        minpool::VoxelMinPoolUpdateOutputComputeIdx<scalar_t><<<BLOCKS(loop1), THREADS>>>(
            pcds_ind_data, voxel_min_idx_data, BS, C, N, D, loop1,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size), DATA_PTR<float>(scale_rate));

        minpool::VoxelMinPoolUpdateOutputInit<scalar_t><<<BLOCKS(loop2), THREADS>>>(
            pcds_feat_data, voxel_out_data, voxel_min_idx_data, BS, C, N, D, loop2,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size));

        minpool::VoxelMinPoolUpdateOutputKernel<scalar_t><<<BLOCKS(loop2), THREADS>>>(
            pcds_feat_data, voxel_out_data, voxel_min_idx_data, BS, C, N, D, loop2,
            DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride),
            DATA_PTR<int64_t>(output_size));
    });
}

void voxel_minpooling_cuda_backward(at::Tensor pcds_feat, at::Tensor /*pcds_ind*/, at::Tensor voxel_out, at::Tensor voxel_min_idx,
    at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor /*scale_rate*/)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C  = pcds_feat.size(1);
    int64_t N  = pcds_feat.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMinPoolBackward", [&] {
        auto* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        auto* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        auto* voxel_min_idx_data = DATA_PTR<int64_t>(voxel_min_idx);
        auto* grad_pcds_feat_data = DATA_PTR<scalar_t>(grad_pcds_feat);
        auto* grad_voxel_out_data = DATA_PTR<scalar_t>(grad_voxel_out);

        minpool::VoxelMinPoolUpdateBackwardKernel<scalar_t><<<BLOCKS(loop), THREADS>>>(
            pcds_feat_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data, voxel_min_idx_data,
            BS, C, N, 0, loop, DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));
    });
}
