/****************************************************************************************
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
****************************************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <algorithm>

#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

// Triton Backend API
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_output_responder.h>
#include <triton/core/tritonserver.h>
#include <triton/common/triton_json.h>

// MIGraphX C API
#include <migraphx/migraphx.h>

// ROCm / HIP Runtime API
#include <hip/hip_runtime.h>

#define CHECK_MIGRAPHX(cmd)                                                           \
  do {                                                                                \
    migraphx_status status = (cmd);                                                   \
    if (status != migraphx_status_success) {                                          \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "MIGraphX C API 调用失败"); \
    }                                                                                 \
  } while (0)

#define CHECK_HIP(cmd)                                                                \
  do {                                                                                \
    hipError_t err = (cmd);                                                           \
    if (err != hipSuccess) {                                                          \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, \
          (std::string("HIP Error: ") + hipGetErrorString(err)).c_str());               \
    }                                                                                 \
  } while (0)

// -----------------------------------------------------------------------------
// 数据类型双向转换辅助函数
// -----------------------------------------------------------------------------
static migraphx_shape_datatype_t TritonToMIGraphXType(TRITONSERVER_DataType dtype) {
  switch (dtype) {
    case TRITONSERVER_TYPE_FP32:  return migraphx_shape_float_type;
    case TRITONSERVER_TYPE_FP16:  return migraphx_shape_half_type;
    case TRITONSERVER_TYPE_INT32: return migraphx_shape_int32_type;
    case TRITONSERVER_TYPE_INT64: return migraphx_shape_int64_type;
    case TRITONSERVER_TYPE_INT8:  return migraphx_shape_int8_type;
    case TRITONSERVER_TYPE_UINT8: return migraphx_shape_uint8_type;
    case TRITONSERVER_TYPE_BOOL:  return migraphx_shape_bool_type;
    case TRITONSERVER_TYPE_FP64:  return migraphx_shape_double_type;
    default:                      return migraphx_shape_float_type;
  }
}

static TRITONSERVER_DataType MIGraphXToTritonType(migraphx_shape_datatype_t dtype) {
  switch (dtype) {
    case migraphx_shape_float_type:  return TRITONSERVER_TYPE_FP32;
    case migraphx_shape_half_type:   return TRITONSERVER_TYPE_FP16;
    case migraphx_shape_int32_type:  return TRITONSERVER_TYPE_INT32;
    case migraphx_shape_int64_type:  return TRITONSERVER_TYPE_INT64;
    case migraphx_shape_int8_type:   return TRITONSERVER_TYPE_INT8;
    case migraphx_shape_uint8_type:  return TRITONSERVER_TYPE_UINT8;
    case migraphx_shape_bool_type:   return TRITONSERVER_TYPE_BOOL;
    case migraphx_shape_double_type: return TRITONSERVER_TYPE_FP64;
    default:                         return TRITONSERVER_TYPE_FP32;
  }
}

enum class ModelFormat {
  UNKNOWN,
  ONNX
};

struct ModelPathInfo {
  std::string path;
  ModelFormat format;
};

static bool PathExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

ModelPathInfo DetectModelPathAndFormat(TRITONBACKEND_ModelInstance* instance) {
  TRITONBACKEND_Model* model = nullptr;
  TRITONBACKEND_ModelInstanceModel(instance, &model);

  uint64_t model_version = 0;
  TRITONBACKEND_ModelVersion(model, &model_version);

  const char* repo_path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  TRITONBACKEND_ModelRepository(model, &artifact_type, &repo_path);

  std::string base_dir = std::string(repo_path) + "/" + std::to_string(model_version);
  std::string onnx_file = base_dir + "/model.onnx";
  if (PathExists(onnx_file)) {
    return {onnx_file, ModelFormat::ONNX};
  }

  return {"", ModelFormat::UNKNOWN};
}

// -----------------------------------------------------------------------------
// 常驻输入/输出显存资源结构体
// -----------------------------------------------------------------------------
struct PersistentDeviceBuffer {
  void* device_ptr = nullptr;
  size_t allocated_bytes = 0;

  void EnsureCapacity(size_t required_bytes) {
    if (required_bytes <= allocated_bytes) {
      return; // 容量充足，直接复用已有显存，不触发 hipMalloc
    }

    if (device_ptr != nullptr) {
      CHECK_HIP(hipFree(device_ptr));
      device_ptr = nullptr;
    }

    // 预留 1.5 倍冗余空间，避免输入维度小幅度抖动频繁触发分配
    size_t new_capacity = std::max(allocated_bytes + (allocated_bytes >> 1), required_bytes);
    CHECK_HIP(hipMalloc(&device_ptr, new_capacity));
    allocated_bytes = new_capacity;
  }

  void Release() {
    if (device_ptr != nullptr) {
      CHECK_HIP(hipFree(device_ptr));
      device_ptr = nullptr;
      allocated_bytes = 0;
    }
  }
};

struct OutputBufferResource {
  void* device_ptr = nullptr;
  size_t byte_size = 0;
  migraphx_shape_t shape = nullptr;
};

class ModelInstanceState {
 public:
  ModelInstanceState(TRITONBACKEND_ModelInstance* triton_model_instance);
  ~ModelInstanceState();

  void ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  void InitMIGraphX();

  TRITONBACKEND_ModelInstance* triton_model_instance_;
  migraphx_program_t program_;
  int32_t device_id_ = 0;
  uint64_t max_batch_size_ = 0;  // 从 model config 读取

  // 常驻输入显存池: 输入名称 -> 显存资源 (Runtime 阶段复用，避免 hipMalloc)
  std::unordered_map<std::string, PersistentDeviceBuffer> input_buffers_;

  // 常驻输出显存表: 参数名 -> 显存资源
  std::unordered_map<std::string, OutputBufferResource> preallocated_outputs_;
};

ModelInstanceState::ModelInstanceState(TRITONBACKEND_ModelInstance* triton_model_instance)
    : triton_model_instance_(triton_model_instance), program_(nullptr) {
  InitMIGraphX();
}

ModelInstanceState::~ModelInstanceState() {
  CHECK_HIP(hipSetDevice(device_id_));

  // 释放持久化输入缓冲池
  for (auto& kv : input_buffers_) {
    kv.second.Release();
  }
  input_buffers_.clear();

  // 释放持久化输出缓冲
  for (auto& kv : preallocated_outputs_) {
    if (kv.second.device_ptr != nullptr) {
      CHECK_HIP(hipFree(kv.second.device_ptr));
      kv.second.device_ptr = nullptr;
    }
    if (kv.second.shape != nullptr) {
      migraphx_shape_destroy(kv.second.shape);
      kv.second.shape = nullptr;
    }
  }
  preallocated_outputs_.clear();

  if (program_ != nullptr) {
    migraphx_program_destroy(program_);
    program_ = nullptr;
  }
}

void ModelInstanceState::InitMIGraphX() {
  TRITONSERVER_InstanceGroupKind kind;
  TRITONBACKEND_ModelInstanceKind(triton_model_instance_, &kind);
  TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance_, &device_id_);
  CHECK_HIP(hipSetDevice(device_id_));
  // 读取模型配置的 max_batch_size: 区分 batching (max_batch_size>0) 与
  // non-batching (max_batch_size==0, 此时请求 dims 已含完整 shape)。
  {
    TRITONBACKEND_Model* triton_model = nullptr;
    TRITONBACKEND_ModelInstanceModel(triton_model_instance_, &triton_model);
    if (triton_model != nullptr) {
      TRITONSERVER_Message* config_message = nullptr;
      if (TRITONBACKEND_ModelConfig(triton_model, 1, &config_message) == nullptr &&
          config_message != nullptr) {
        const char* config_json = nullptr;
        size_t config_json_size = 0;
        if (TRITONSERVER_MessageSerializeToJson(
                config_message, &config_json, &config_json_size) == nullptr) {
          triton::common::TritonJson::Value cfg;
          if (cfg.Parse(config_json, config_json_size) == nullptr) {
            int64_t mb = 0;
            if (cfg.MemberAsInt("max_batch_size", &mb) == nullptr && mb > 0) {
              max_batch_size_ = static_cast<uint64_t>(mb);
            }
          }
        }
      }
    }
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        (std::string("[BACKEND] max_batch_size=") + std::to_string(max_batch_size_) +
         (max_batch_size_ > 0 ? " (batching)" : " (non-batching)")).c_str());
  }


  ModelPathInfo model_info = DetectModelPathAndFormat(triton_model_instance_);
  if (model_info.format == ModelFormat::UNKNOWN) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to find supported model (model.onnx)");
    return;
  }

  migraphx_onnx_options_t onnx_opts;
  CHECK_MIGRAPHX(migraphx_onnx_options_create(&onnx_opts));

  migraphx_program_t raw_prog = nullptr;
  CHECK_MIGRAPHX(migraphx_parse_onnx(&raw_prog, model_info.path.c_str(), onnx_opts));

  migraphx_compile_options_t comp_opts;
  CHECK_MIGRAPHX(migraphx_compile_options_create(&comp_opts));

  migraphx_target_t target;
  CHECK_MIGRAPHX(migraphx_target_create(&target, "gpu"));

  CHECK_MIGRAPHX(migraphx_program_compile(raw_prog, target, comp_opts));
  program_ = raw_prog;

  migraphx_compile_options_destroy(comp_opts);
  migraphx_onnx_options_destroy(onnx_opts);
  migraphx_target_destroy(target);

  // 探查编译图的所有参数，预先为所有输出分配固定 Device 显存
  migraphx_program_parameter_shapes_t param_shapes;
  CHECK_MIGRAPHX(migraphx_program_get_parameter_shapes(&param_shapes, program_));

  size_t param_count = 0;
  CHECK_MIGRAPHX(migraphx_program_parameter_shapes_size(&param_count, param_shapes));

  std::vector<const char*> param_names(param_count);
  CHECK_MIGRAPHX(migraphx_program_parameter_shapes_names(param_names.data(), param_shapes));

  for (size_t p = 0; p < param_count; ++p) {
    std::string param_name(param_names[p]);

    // 筛选输出节点（包含 output 关键字或内部前缀 #）
    if (param_name.find("output") != std::string::npos || param_name[0] == '#') {
      const_migraphx_shape_t p_shape;
      CHECK_MIGRAPHX(migraphx_program_parameter_shapes_get(&p_shape, param_shapes, param_names[p]));

      size_t byte_size = 0;
      CHECK_MIGRAPHX(migraphx_shape_bytes(&byte_size, p_shape));

      void* d_ptr = nullptr;
      CHECK_HIP(hipMalloc(&d_ptr, byte_size));

      OutputBufferResource res;
      res.device_ptr = d_ptr;
      res.byte_size = byte_size;

      migraphx_shape_datatype_t dtype;
      CHECK_MIGRAPHX(migraphx_shape_type(&dtype, p_shape));
      const size_t* lens = nullptr;
      size_t out_size = 0;
      CHECK_MIGRAPHX(migraphx_shape_lengths(&lens, &out_size, p_shape));
      size_t ndim = 0;
      CHECK_MIGRAPHX(migraphx_shape_ndim(&ndim, p_shape));

      CHECK_MIGRAPHX(migraphx_shape_create(&res.shape, dtype, const_cast<size_t*>(lens), ndim));

      preallocated_outputs_[param_name] = res;
    }
  }

  migraphx_program_parameter_shapes_destroy(param_shapes);
}

void ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count) {

  CHECK_HIP(hipSetDevice(device_id_));

  std::vector<TRITONBACKEND_Request*> valid_requests;
  valid_requests.reserve(request_count);

  // 仅 batching 模式 (max_batch_size>0) 下, Triton 才会在请求 dims 的第 0 维
  // 前置 batch 维; non-batching 模式请求 dims 即完整 shape, 不含 batch。
  const bool batching = (max_batch_size_ > 0);
  uint64_t total_batch_size = 0;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    if (request == nullptr) {
      continue;
    }
    valid_requests.push_back(request);
    if (batching) {
      // batching: 从请求第一个输入取其 batch (dims[0])
      TRITONBACKEND_Input* first_in = nullptr;
      TRITONBACKEND_RequestInputByIndex(request, 0, &first_in);
      if (first_in != nullptr) {
        const int64_t* req_dims = nullptr;
        uint32_t req_dims_count = 0;
        TRITONBACKEND_InputProperties(
            first_in, nullptr, nullptr, &req_dims, &req_dims_count, nullptr, nullptr);
        total_batch_size += (req_dims_count > 0) ? req_dims[0] : 1;
      } else {
        total_batch_size += 1;
      }
    } else {
      // non-batching: 每个请求视为 1 个样本, 不解释 dims[0] 为 batch
      total_batch_size += 1;
    }
  }

  if (valid_requests.empty()) {
    return;
  }

  // 为每个 Request 创建对应的 TRITONBACKEND_Response
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(valid_requests.size());
  for (auto* req : valid_requests) {
    TRITONBACKEND_Response* resp = nullptr;
    TRITONBACKEND_ResponseNew(&resp, req);
    responses.push_back(resp);
  }

  TRITONBACKEND_MemoryManager* memory_manager = nullptr;
  TRITONBACKEND_Model* triton_model = nullptr;
  TRITONBACKEND_ModelInstanceModel(triton_model_instance_, &triton_model);
  TRITONBACKEND_Backend* triton_backend = nullptr;
  TRITONBACKEND_ModelBackend(triton_model, &triton_backend);
  TRITONBACKEND_BackendMemoryManager(triton_backend, &memory_manager);

  migraphx_program_parameters_t prog_params;
  CHECK_MIGRAPHX(migraphx_program_parameters_create(&prog_params));

  std::vector<migraphx_shape_t> dynamic_input_shapes;
  std::vector<migraphx_argument_t> dynamic_args;

  // --------------------------------------------------------------------------
  // 1. 遍历输入（直接复用持久化显存池，消灭运行时 hipMalloc）
  // --------------------------------------------------------------------------
  uint32_t input_count = 0;
  TRITONBACKEND_RequestInputCount(valid_requests[0], &input_count);

  for (uint32_t i = 0; i < input_count; ++i) {
    TRITONBACKEND_Input* in = nullptr;
    TRITONBACKEND_RequestInputByIndex(valid_requests[0], i, &in);

    const char* in_name_cstr = nullptr;
    TRITONSERVER_DataType in_dtype;
    const int64_t* in_dims = nullptr;
    uint32_t in_dims_count = 0;
    uint64_t per_req_byte_size = 0;
    uint32_t dummy_buf_count = 0;

    TRITONBACKEND_InputProperties(
        in, &in_name_cstr, &in_dtype, &in_dims, &in_dims_count, &per_req_byte_size, &dummy_buf_count);

    std::string tensor_name(in_name_cstr);

    // 统计当前 Batch 中所有请求该张量的实际总字节数
    size_t total_tensor_bytes = 0;
    for (auto* req : valid_requests) {
      TRITONBACKEND_Input* req_in = nullptr;
      TRITONBACKEND_RequestInput(req, tensor_name.c_str(), &req_in);
      uint64_t req_bytes = 0;
      TRITONBACKEND_InputProperties(req_in, nullptr, nullptr, nullptr, nullptr, &req_bytes, nullptr);
      total_tensor_bytes += req_bytes;
    }

    // 从常驻池中直接复用 Device 显存；容量足够时直接跳过分配
    PersistentDeviceBuffer& persistent_buf = input_buffers_[tensor_name];
    persistent_buf.EnsureCapacity(total_tensor_bytes);
    void* gpu_input_buffer = persistent_buf.device_ptr;

    // 逐个遍历 Request 并拷贝其 buffer 分片至连续的 GPU 显存
    size_t current_offset = 0;
    for (auto* req : valid_requests) {
      TRITONBACKEND_Input* req_in = nullptr;
      TRITONBACKEND_RequestInput(req, tensor_name.c_str(), &req_in);

      uint32_t buf_count = 0;
      TRITONBACKEND_InputProperties(req_in, nullptr, nullptr, nullptr, nullptr, nullptr, &buf_count);

      for (uint32_t b = 0; b < buf_count; ++b) {
        const void* src_buffer = nullptr;
        uint64_t src_byte_size = 0;
        TRITONSERVER_MemoryType src_mem_type;
        int64_t src_mem_id = 0;

        TRITONBACKEND_InputBuffer(
            req_in, b, &src_buffer, &src_byte_size, &src_mem_type, &src_mem_id);

        char* dst_ptr = reinterpret_cast<char*>(gpu_input_buffer) + current_offset;

        if (src_mem_type == TRITONSERVER_MEMORY_GPU) {
          CHECK_HIP(hipMemcpyDtoD(dst_ptr, const_cast<void*>(src_buffer), src_byte_size));
        } else {
          CHECK_HIP(hipMemcpyHtoD(dst_ptr, const_cast<void*>(src_buffer), src_byte_size));
        }
        current_offset += src_byte_size;
      }
    }

    // 构造 Batch Shape 并包装为 MIGraphX 参数
    std::vector<size_t> in_lens;
    in_lens.reserve(in_dims_count);
    for (uint32_t d = 0; d < in_dims_count; ++d) {
      if (d == 0 && batching) {
        // batching: 用请求累计 batch 覆盖 dims[0]
        in_lens.push_back(static_cast<size_t>(total_batch_size));
      } else {
        // non-batching: dims 已是完整 shape, 原样使用
        in_lens.push_back(static_cast<size_t>(in_dims[d]));
      }
    }

    migraphx_shape_t in_shape;
    CHECK_MIGRAPHX(migraphx_shape_create(
        &in_shape, TritonToMIGraphXType(in_dtype), in_lens.data(), in_lens.size()));
    dynamic_input_shapes.push_back(in_shape);

    migraphx_argument_t in_arg;
    CHECK_MIGRAPHX(migraphx_argument_create(&in_arg, in_shape, gpu_input_buffer));
    dynamic_args.push_back(in_arg);

    CHECK_MIGRAPHX(migraphx_program_parameters_add(prog_params, tensor_name.c_str(), in_arg));
  }

  // --------------------------------------------------------------------------
  // 2. 绑定常驻输出 Device 显存
  // --------------------------------------------------------------------------
  for (const auto& kv : preallocated_outputs_) {
    const std::string& out_param_name = kv.first;
    const OutputBufferResource& res = kv.second;

    migraphx_argument_t out_arg;
    CHECK_MIGRAPHX(migraphx_argument_create(&out_arg, res.shape, res.device_ptr));
    dynamic_args.push_back(out_arg);

    CHECK_MIGRAPHX(migraphx_program_parameters_add(prog_params, out_param_name.c_str(), out_arg));
  }

  // --------------------------------------------------------------------------
  // 3. 执行 GPU 推理并同步
  // --------------------------------------------------------------------------
  migraphx_arguments_t out_args;
  CHECK_MIGRAPHX(migraphx_program_run(&out_args, program_, prog_params));

  // --------------------------------------------------------------------------
  // 4. 将输出通过 BackendOutputResponder 切片写回 Response
  // --------------------------------------------------------------------------
  std::unique_ptr<triton::backend::BackendOutputResponder> responder(
      new triton::backend::BackendOutputResponder(
          valid_requests.data(), valid_requests.size(), &responses,
          static_cast<int>(max_batch_size_) /* max_batch_size */,
          memory_manager, false, nullptr /* stream */));

  std::vector<std::string> requested_output_names;
  uint32_t output_req_count = 0;
  TRITONBACKEND_RequestOutputCount(valid_requests[0], &output_req_count);

  for (uint32_t o = 0; o < output_req_count; ++o) {
    const char* out_name = nullptr;
    TRITONBACKEND_RequestOutputName(valid_requests[0], o, &out_name);
    requested_output_names.emplace_back(out_name);
  }

  size_t num_outputs = 0;
  CHECK_MIGRAPHX(migraphx_arguments_size(&num_outputs, out_args));

  for (size_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
    const_migraphx_argument_t out_arg;
    CHECK_MIGRAPHX(migraphx_arguments_get(&out_arg, out_args, out_idx));

    char* out_device_data = nullptr;
    CHECK_MIGRAPHX(migraphx_argument_buffer(&out_device_data, out_arg));

    const_migraphx_shape_t out_shape;
    CHECK_MIGRAPHX(migraphx_argument_shape(&out_shape, out_arg));

    migraphx_shape_datatype_t mgx_dtype;
    CHECK_MIGRAPHX(migraphx_shape_type(&mgx_dtype, out_shape));
    TRITONSERVER_DataType triton_out_dtype = MIGraphXToTritonType(mgx_dtype);

    const size_t* shape_lens = nullptr;
    size_t out_size = 0;
    CHECK_MIGRAPHX(migraphx_shape_lengths(&shape_lens, &out_size, out_shape));
    size_t shape_ndim = 0;
    CHECK_MIGRAPHX(migraphx_shape_ndim(&shape_ndim, out_shape));

    std::vector<int64_t> output_shape;
    output_shape.reserve(shape_ndim);
    for (size_t d = 0; d < shape_ndim; ++d) {
      output_shape.push_back(static_cast<int64_t>(shape_lens[d]));
    }

    std::string current_output_name = (out_idx < requested_output_names.size())
                                          ? requested_output_names[out_idx]
                                          : ("output_" + std::to_string(out_idx));

    responder->ProcessTensor(
        current_output_name, triton_out_dtype, output_shape,
        out_device_data, TRITONSERVER_MEMORY_GPU, device_id_);
  }

  // 确保数据彻底完成拷贝
  CHECK_HIP(hipStreamSynchronize(hipStreamDefault));

  // --------------------------------------------------------------------------
  // 5. 发送响应与释放请求
  // --------------------------------------------------------------------------
  for (size_t i = 0; i < responses.size(); ++i) {
    if (responses[i] != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[i], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(
            valid_requests[i], TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release request");
  }

  // --------------------------------------------------------------------------
  // 6. 统一释放句柄（不再释放显存，显存保留在 input_buffers_ 中供后续推理复用）
  // --------------------------------------------------------------------------
  for (auto s : dynamic_input_shapes) {
    migraphx_shape_destroy(s);
  }
  for (auto a : dynamic_args) {
    migraphx_argument_destroy(a);
  }
  migraphx_program_parameters_destroy(prog_params);
  migraphx_arguments_destroy(out_args);
}

extern "C" {

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state = nullptr;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

  instance_state->ProcessRequests(requests, request_count);
  return nullptr;
}

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  auto instance_state = new ModelInstanceState(instance);
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));
  return nullptr;
}

TRITONBACKEND_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  if (vstate != nullptr) {
    delete reinterpret_cast<ModelInstanceState*>(vstate);
  }
  return nullptr;
}

} // extern "C"
