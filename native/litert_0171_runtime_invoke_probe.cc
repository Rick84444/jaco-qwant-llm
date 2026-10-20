// Genuine LiteRT 0.17.1 C API execution of an upstream public .tflite file.
// Isolated Linux test. Never imports Gemma, modifies models, or touches Android.
#include <cstdio>
#include <cstring>
#include <vector>

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

namespace {
struct Handles {
  LiteRtEnvironment environment = nullptr;
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  std::vector<LiteRtTensorBuffer> inputs;
  std::vector<LiteRtTensorBuffer> outputs;

  ~Handles() {
    for (auto b : inputs) if (b) LiteRtDestroyTensorBuffer(b);
    for (auto b : outputs) if (b) LiteRtDestroyTensorBuffer(b);
    if (compiled) LiteRtDestroyCompiledModel(compiled);
    if (options) LiteRtDestroyOptions(options);
    if (model) LiteRtDestroyModel(model);
    if (environment) LiteRtDestroyEnvironment(environment);
  }
};

bool Check(LiteRtStatus status, const char* stage) {
  if (status == kLiteRtStatusOk) return true;
  std::fprintf(stderr, "JACO_0171_INVOKE: failed=%s status=%d\n",
               stage, static_cast<int>(status));
  return false;
}

bool Allocate(Handles& h, LiteRtSignature signature, bool for_input) {
  LiteRtParamIndex count = 0;
  if (!Check(for_input ? LiteRtGetNumSignatureInputs(signature, &count) :
                         LiteRtGetNumSignatureOutputs(signature, &count),
             "signature_count")) return false;
  if (count <= 0 || count > 8) {
    std::fprintf(stderr, "JACO_0171_INVOKE: unsupported_input_output_count=%d\n",
                 static_cast<int>(count));
    return false;
  }
  std::vector<LiteRtTensorBuffer>& buffers = for_input ? h.inputs : h.outputs;
  buffers.resize(static_cast<size_t>(count), nullptr);
  for (LiteRtParamIndex index = 0; index < count; ++index) {
    LiteRtTensor tensor = nullptr;
    if (!Check(for_input ?
                 LiteRtGetSignatureInputTensorByIndex(signature, index, &tensor) :
                 LiteRtGetSignatureOutputTensorByIndex(signature, index, &tensor),
               "get_signature_tensor")) return false;
    LiteRtRankedTensorType shape = {};
    if (!Check(LiteRtGetRankedTensorType(tensor, &shape), "tensor_type"))
      return false;
    LiteRtTensorBufferRequirements requirements = nullptr;
    if (!Check(for_input ?
                  LiteRtGetCompiledModelInputBufferRequirements(
                      h.compiled, 0, index, &requirements) :
                  LiteRtGetCompiledModelOutputBufferRequirements(
                      h.compiled, 0, index, &requirements),
               "requirements")) return false;
    if (!Check(LiteRtCreateManagedTensorBufferFromRequirements(
                   h.environment, &shape, requirements,
                   &buffers[static_cast<size_t>(index)]),
               "allocate_tensor_buffer")) return false;
    if (for_input) {
      size_t bytes = 0;
      if (!Check(LiteRtGetTensorBufferRequirementsBufferSize(
                     requirements, &bytes), "buffer_size") ||
          bytes == 0 || bytes > 64 * 1024 * 1024) return false;
      void* address = nullptr;
      if (!Check(LiteRtLockTensorBuffer(
                     buffers[static_cast<size_t>(index)], &address,
                     kLiteRtTensorBufferLockModeWrite), "lock_input")) return false;
      if (!address) return false;
      std::memset(address, 0, bytes);
      if (!Check(LiteRtUnlockTensorBuffer(buffers[static_cast<size_t>(index)]),
                 "unlock_input")) return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fputs("Usage: jaco_litert_0171_invoke <public-model.tflite>\n", stderr);
    return 2;
  }
  Handles h;
  if (!Check(LiteRtCreateEnvironment(0, nullptr, &h.environment),
             "environment") ||
      !Check(LiteRtCreateModelFromFile(h.environment, argv[1], &h.model),
             "model") ||
      !Check(LiteRtCreateOptions(&h.options), "options") ||
      !Check(LiteRtSetOptionsHardwareAccelerators(
                 h.options, kLiteRtHwAcceleratorCpu), "cpu") ||
      !Check(LiteRtCreateCompiledModel(h.environment, h.model, h.options,
                                      &h.compiled), "compile"))
    return 1;

  LiteRtParamIndex n_signatures = 0;
  LiteRtSignature signature = nullptr;
  if (!Check(LiteRtGetNumModelSignatures(h.model, &n_signatures),
             "signatures") ||
      n_signatures < 1 ||
      !Check(LiteRtGetModelSignature(h.model, 0, &signature), "signature") ||
      !Allocate(h, signature, true) ||
      !Allocate(h, signature, false))
    return 1;

  if (!Check(LiteRtRunCompiledModel(h.compiled, 0,
                                   h.inputs.size(), h.inputs.data(),
                                   h.outputs.size(), h.outputs.data()),
             "run"))
    return 1;
  std::printf("JACO_0171_INVOKE: run_ok=1 inputs=%zu outputs=%zu "
              "gemma_executed=0 native_acceleration_claim=0\n",
              h.inputs.size(), h.outputs.size());
  return 0;
}
