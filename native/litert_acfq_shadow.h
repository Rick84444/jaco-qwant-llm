#ifndef JACO_LITERT_ACFQ_SHADOW_H_
#define JACO_LITERT_ACFQ_SHADOW_H_

// Experimental, opt-in, read-only-inference shadow of the built-in TFLite
// FullyConnected callback. This executes our native INT8xINT4 GEMV after
// the original op, and does NOT replace its output or claim acceleration.
// Built into an isolated LiteRT source tree, never a live binary injection.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include "tflite/core/c/common.h"
#include "tflite/core/kernels/register.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "litert/runtime/jaco_acfq_native/gemma_int4_kernel.h"

namespace jaco_acfq {

inline const TfLiteTensor* Tensor(const TfLiteContext* ctx, int idx) {
  if (!ctx || idx < 0 || idx >= ctx->tensors_size || !ctx->tensors) return nullptr;
  return &ctx->tensors[idx];
}
inline bool Elements(const TfLiteTensor* t, int64_t* count) {
  if (!t || !t->dims) return false;
  int64_t total=1;
  for(int i=0;i<t->dims->size;++i) {
    int value=t->dims->data[i];
    if(value<=0 || total>INT64_MAX/value) return false;
    total*=value;
  }
  *count=total;
  return true;
}

inline TfLiteStatus ShadowFullyConnected(TfLiteContext* ctx, TfLiteNode* node) {
  const TfLiteRegistration* builtin = tflite::ops::builtin::Register_FULLY_CONNECTED();
  if(!builtin || !builtin->invoke) return kTfLiteError;
  // Original TFLite logic always owns the actual inference output.
  TfLiteStatus result=builtin->invoke(ctx,node);
  if(result!=kTfLiteOk) return result;

  static std::atomic<uint64_t> attempted{0};
  static std::atomic<uint64_t> eligible{0};
  static std::atomic<uint64_t> native_calls{0};
  // Bounded default overhead: only the first eligible operation in-process.
  ++attempted;
  if(!node || !node->inputs || node->inputs->size<2 || !node->outputs ||
     node->outputs->size!=1) return result;
  const TfLiteTensor* input=Tensor(ctx,node->inputs->data[0]);
  const TfLiteTensor* weights=Tensor(ctx,node->inputs->data[1]);
  const TfLiteTensor* output=Tensor(ctx,node->outputs->data[0]);
  if(!input || !weights || !output || input->type!=kTfLiteInt8 ||
     weights->type!=kTfLiteInt4 ||
     !(output->type==kTfLiteInt8 || output->type==kTfLiteFloat32) ||
     !input->data.int8 || !weights->data.raw_const || !output->data.raw_const)
    return result;

  int64_t in_elements=0,out_elements=0;
  if(!Elements(input,&in_elements) || !Elements(output,&out_elements) ||
     !weights->dims || weights->dims->size!=2) return result;
  const int n=weights->dims->data[0],k=weights->dims->data[1];
  if(k!=1536 || (n!=256 && n!=2048) || in_elements!=k ||
     out_elements!=n || input->params.scale<=0 ||
     !std::isfinite(input->params.scale)) return result;
  const size_t bytes=jaco_gemma_int4_required_bytes(k,n);
  if(!bytes || weights->bytes<bytes ||
     weights->quantization.type!=kTfLiteAffineQuantization ||
     !weights->quantization.params) return result;
  const auto* q=static_cast<const TfLiteAffineQuantization*>(
      weights->quantization.params);
  if(!q->scale || q->scale->size!=n || !q->zero_point ||
     (q->zero_point->size!=n && q->zero_point->size!=1))
    return result;
  const uint64_t seq=eligible.fetch_add(1);
  if(seq>0) return result;

  std::vector<int32_t> wzp(static_cast<size_t>(n),0);
  for(int j=0;j<n;++j) wzp[static_cast<size_t>(j)] =
      q->zero_point->data[q->zero_point->size==1 ? 0 : j];
  std::vector<float> computed(static_cast<size_t>(n),0.0f);
  uint64_t products=0,blocks=0;
  JacoGemmaInt4Params p={};
  p.input=reinterpret_cast<const int8_t*>(input->data.int8);
  p.input_len=static_cast<size_t>(k);
  p.input_scale=input->params.scale;
  p.input_zero_point=input->params.zero_point;
  p.weights=reinterpret_cast<const uint8_t*>(weights->data.raw_const);
  p.weights_len=weights->bytes;
  p.k=k;p.n=n;
  p.weight_scales=q->scale->data;
  p.weight_scales_len=static_cast<size_t>(n);
  p.weight_zero_points=wzp.data();
  p.weight_zero_points_len=static_cast<size_t>(n);
  // Bias is not included in the shadow operation. This diagnostic measures
  // kernel invocation, not final op-level numerical equivalence.
  p.output=computed.data();
  p.output_len=static_cast<size_t>(n);
  p.executed_products=&products;
  p.neon_blocks=&blocks;
  const int code=jaco_gemma_int4_gemv(&p);
  if(code==JACO_GEMMA_INT4_OK) {
    ++native_calls;
    std::fprintf(stderr,
        "JACO_ACFQ_SHADOW: native_invocations=%llu N=%d K=%d "
        "products=%llu neon_blocks=%llu original_output_preserved=1 "
        "bias_not_compared=1\n",
        static_cast<unsigned long long>(native_calls.load()),n,k,
        static_cast<unsigned long long>(products),
        static_cast<unsigned long long>(blocks));
  } else {
    std::fprintf(stderr,
        "JACO_ACFQ_SHADOW: candidate_rejected=%d N=%d K=%d "
        "original_output_preserved=1\n",code,n,k);
  }
  return result;
}

inline TfLiteRegistration* ShadowRegistration() {
  static TfLiteRegistration reg=[] {
    TfLiteRegistration r=*tflite::ops::builtin::Register_FULLY_CONNECTED();
    r.invoke=&ShadowFullyConnected;
    return r;
  }();
  return &reg;
}

}  // namespace jaco_acfq
#endif  // JACO_LITERT_ACFQ_SHADOW_H_
