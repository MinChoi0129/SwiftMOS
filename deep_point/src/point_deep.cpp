#include <torch/extension.h>
#include <vector>
#include <cmath>

namespace maxpool {
    template<typename real>
    void VoxelMaxPoolUpdateOutputInit(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag) voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
        }
    }

    template<typename real>
    void VoxelMaxPoolUpdateOutputKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag && voxel_out_data[index_voxel] < pcds_feat_data[index_pcds]) {
                voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
            }
        }
    }

    template<typename real>
    void VoxelMaxPoolUpdateBackwardKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag && voxel_out_data[index_voxel] == pcds_feat_data[index_pcds]) {
                grad_pcds_feat_data[index_pcds] = grad_voxel_out_data[index_voxel];
            }
        }
    }
}

namespace minpool {
    template<typename real>
    void VoxelMinPoolUpdateOutputInit(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag) voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
        }
    }

    template<typename real>
    void VoxelMinPoolUpdateOutputKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag && voxel_out_data[index_voxel] > pcds_feat_data[index_pcds]) {
                voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
            }
        }
    }

    template<typename real>
    void VoxelMinPoolUpdateBackwardKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data,
        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for (int64_t i = 0; i < loop; i++) {
            int64_t bs = i / (C * N);
            int64_t index_res = i - bs * C * N;
            int64_t c = index_res / N;
            int64_t n = index_res - c * N;

            int64_t index_pcds = i;
            int64_t index_ind  = bs * N * D + n * D;
            int64_t index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1];

            bool flag = true;
            for (int64_t d = 0; d < D; d++) {
                int64_t ind_tmp = static_cast<int64_t>(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if ((ind_tmp >= 0) && (ind_tmp < output_size[d])) index_voxel += ind_tmp * voxel_out_stride[2 + d];
                else { flag = false; }
            }
            if (flag && voxel_out_data[index_voxel] == pcds_feat_data[index_pcds]) {
                grad_pcds_feat_data[index_pcds] = grad_voxel_out_data[index_voxel];
            }
        }
    }
}

void voxel_maxpooling_cpu_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor /*voxel_max_idx*/,
    at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0), C = pcds_feat.size(1), N = pcds_feat.size(2), D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMaxPoolCPU", [&] {
        auto* pcds_feat_data = pcds_feat.data_ptr<scalar_t>();
        auto* pcds_ind_data  = pcds_ind.data_ptr<scalar_t>();
        auto* voxel_out_data = voxel_out.data_ptr<scalar_t>();

        maxpool::VoxelMaxPoolUpdateOutputInit<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
            voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());

        maxpool::VoxelMaxPoolUpdateOutputKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
            voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());
    });
}

void voxel_maxpooling_cpu_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor /*voxel_max_idx*/,
    at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0), C = pcds_feat.size(1), N = pcds_feat.size(2), D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMaxPoolCPUBackward", [&] {
        auto* pcds_feat_data = pcds_feat.data_ptr<scalar_t>();
        auto* pcds_ind_data  = pcds_ind.data_ptr<scalar_t>();
        auto* voxel_out_data = voxel_out.data_ptr<scalar_t>();
        auto* grad_pcds_feat_data = grad_pcds_feat.data_ptr<scalar_t>();
        auto* grad_voxel_out_data = grad_voxel_out.data_ptr<scalar_t>();

        maxpool::VoxelMaxPoolUpdateBackwardKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data,
            BS, C, N, D, loop, voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());
    });
}

void voxel_minpooling_cpu_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor /*voxel_min_idx*/,
    at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0), C = pcds_feat.size(1), N = pcds_feat.size(2), D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMinPoolCPU", [&] {
        auto* pcds_feat_data = pcds_feat.data_ptr<scalar_t>();
        auto* pcds_ind_data  = pcds_ind.data_ptr<scalar_t>();
        auto* voxel_out_data = voxel_out.data_ptr<scalar_t>();

        minpool::VoxelMinPoolUpdateOutputInit<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
            voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());

        minpool::VoxelMinPoolUpdateOutputKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
            voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());
    });
}

void voxel_minpooling_cpu_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor /*voxel_min_idx*/,
    at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0), C = pcds_feat.size(1), N = pcds_feat.size(2), D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES(pcds_feat.scalar_type(), "VoxelMinPoolCPUBackward", [&] {
        auto* pcds_feat_data = pcds_feat.data_ptr<scalar_t>();
        auto* pcds_ind_data  = pcds_ind.data_ptr<scalar_t>();
        auto* voxel_out_data = voxel_out.data_ptr<scalar_t>();
        auto* grad_pcds_feat_data = grad_pcds_feat.data_ptr<scalar_t>();
        auto* grad_voxel_out_data = grad_voxel_out.data_ptr<scalar_t>();

        minpool::VoxelMinPoolUpdateBackwardKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data,
            BS, C, N, D, loop, voxel_out_size.data_ptr<int64_t>(), voxel_out_stride.data_ptr<int64_t>(), output_size.data_ptr<int64_t>(), scale_rate.data_ptr<float>());
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_maxpooling_cpu_forward",  &voxel_maxpooling_cpu_forward,  "maxpooling forward (CPU)");
  m.def("voxel_maxpooling_cpu_backward", &voxel_maxpooling_cpu_backward, "maxpooling backward (CPU)");
  m.def("voxel_minpooling_cpu_forward",  &voxel_minpooling_cpu_forward,  "minpooling forward (CPU)");
  m.def("voxel_minpooling_cpu_backward", &voxel_minpooling_cpu_backward, "minpooling backward (CPU)");
}
