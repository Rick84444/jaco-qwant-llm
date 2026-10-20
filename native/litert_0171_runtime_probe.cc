// Minimal, isolated native LiteRT v0.17.1 interpreter-construction probe.
//
// Links to the exact pinned, ACFQ-patched LiteRT build. Loads a public test
// model and attempts CPU model compilation. This DOES NOT execute model
// inference, prove Gemma decoding, or benchmark a replacement kernel.
// Never load a user model or touch a running Termux LiteRT-LM instance.
#include <cstdio>
#include <cstdlib>

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"

int main(int argc, char** argv) {
  if (argc != 2 || argv[1] == nullptr) {
    std::fprintf(stderr, "Usage: jaco_litert_0171_probe <public-test-model.tflite>\n");
    return 2;
  }

  LiteRtEnvironment environment = nullptr;
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtCompiledModel compiled_model = nullptr;

  auto fail = [](const char* phase, LiteRtStatus status) {
    std::fprintf(stderr, "JACO_0171_PROBE: FAILED phase=%s code=%d\n",
                 phase, static_cast<int>(status));
  };

  int exit_code = 1;
  LiteRtStatus status = LiteRtCreateEnvironment(0, nullptr, &environment);
  if (status != kLiteRtStatusOk) {
    fail("create_environment", status);
    goto cleanup;
  }
  status = LiteRtCreateModelFromFile(environment, argv[1], &model);
  if (status != kLiteRtStatusOk) {
    fail("create_model", status);
    goto cleanup;
  }
  status = LiteRtCreateOptions(&options);
  if (status != kLiteRtStatusOk) {
    fail("create_options", status);
    goto cleanup;
  }
  status = LiteRtSetOptionsHardwareAccelerators(options,
                                                 kLiteRtHwAcceleratorCpu);
  if (status != kLiteRtStatusOk) {
    fail("set_cpu", status);
    goto cleanup;
  }
  status = LiteRtCreateCompiledModel(environment, model, options,
                                     &compiled_model);
  if (status != kLiteRtStatusOk) {
    fail("compile_cpu_model", status);
    goto cleanup;
  }
  std::puts("JACO_0171_PROBE: CPU model compiled; no inference executed");
  exit_code = 0;

cleanup:
  if (compiled_model) LiteRtDestroyCompiledModel(compiled_model);
  if (options) LiteRtDestroyOptions(options);
  if (model) LiteRtDestroyModel(model);
  if (environment) LiteRtDestroyEnvironment(environment);
  return exit_code;
}
