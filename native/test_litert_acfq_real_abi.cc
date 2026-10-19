// Real pinned TFLite struct layouts; only original operator callback is stubbed.
// NOT a linked LiteRT interpreter or a real Gemma execution.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "litert/runtime/jaco_acfq_shadow.h"
namespace {
int original_calls = 0;
TfLiteStatus OriginalInvoke(TfLiteContext* ctx, TfLiteNode* node) {
  ++original_calls;
  auto& out = ctx->tensors[node->outputs->data[0]];
  for (int i=0; i<256; ++i) out.data.int8[i] = 17;
  return kTfLiteOk;
}
TfLiteIntArray* IntArray(int n) {
  auto* a=static_cast<TfLiteIntArray*>(std::calloc(1,sizeof(TfLiteIntArray)+
                                  static_cast<size_t>(n)*sizeof(int)));
  assert(a); a->size=n; return a;
}
TfLiteFloatArray* FloatArray(int n) {
  auto* a=static_cast<TfLiteFloatArray*>(std::calloc(1,sizeof(TfLiteFloatArray)+
                                  static_cast<size_t>(n)*sizeof(float)));
  assert(a); a->size=n; return a;
}
}
namespace tflite { namespace ops { namespace builtin {
TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration original{};
  original.invoke = &OriginalInvoke;
  return &original;
}
}}}
int main() {
  constexpr int K=1536,N=256;
  std::vector<int8_t> input(K,3),output(N,0);
  std::vector<uint8_t> packed(K*N/2,0x11);
  auto* in_shape=IntArray(3);
  auto* w_shape=IntArray(2);
  auto* out_shape=IntArray(3);
  in_shape->data[0]=1;in_shape->data[1]=1;in_shape->data[2]=K;
  w_shape->data[0]=N;w_shape->data[1]=K;
  out_shape->data[0]=1;out_shape->data[1]=1;out_shape->data[2]=N;
  auto* scales=FloatArray(N);
  auto* zeros=IntArray(N);
  for(int i=0;i<N;++i){scales->data[i]=.0125f;zeros->data[i]=0;}
  TfLiteAffineQuantization q{scales,zeros,0};
  TfLiteTensor tensors[3]{};
  tensors[0].type=kTfLiteInt8;
  tensors[0].data.int8=input.data();
  tensors[0].bytes=K;
  tensors[0].dims=in_shape;
  tensors[0].params={.01f,0};
  tensors[1].type=kTfLiteInt4;
  tensors[1].data.raw_const=reinterpret_cast<const char*>(packed.data());
  tensors[1].bytes=packed.size();
  tensors[1].dims=w_shape;
  tensors[1].quantization={kTfLiteAffineQuantization,&q};
  tensors[2].type=kTfLiteInt8;
  tensors[2].data.int8=output.data();
  tensors[2].bytes=N;
  tensors[2].dims=out_shape;
  auto* indices=IntArray(2);
  auto* results=IntArray(1);
  indices->data[0]=0;indices->data[1]=1;results->data[0]=2;
  TfLiteNode node{};node.inputs=indices;node.outputs=results;
  TfLiteContext ctx{};ctx.tensors_size=3;ctx.tensors=tensors;
  auto* registration=jaco_acfq::ShadowRegistration();
  assert(registration && registration->invoke);
  assert(registration->invoke(&ctx,&node)==kTfLiteOk);
  assert(original_calls==1);
  for(int i=0;i<N;++i)assert(output[i]==17);
  std::puts("JACO real-header ABI PASS: authentic TFLite structs; original output preserved");
  std::free(in_shape);std::free(w_shape);std::free(out_shape);
  std::free(scales);std::free(zeros);std::free(indices);std::free(results);
  return 0;
}
