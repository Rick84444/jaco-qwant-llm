// Standalone shadow registration ABI smoke test with SMALL MOCK types.
// It verifies dispatch and output preservation, NOT actual LiteRT ABI.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "litert_acfq_shadow.h"

namespace {
int original_calls=0;
TfLiteStatus OriginalInvoke(TfLiteContext* ctx,TfLiteNode* node) {
  ++original_calls;
  TfLiteTensor& out=ctx->tensors[node->outputs->data[0]];
  for(int i=0;i<256;++i)out.data.int8[i]=17;
  return kTfLiteOk;
}
}  // namespace

namespace tflite {namespace ops {namespace builtin {
TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration original={nullptr,nullptr,nullptr,&OriginalInvoke};
  return &original;
}
}}}

int main() {
  constexpr int K=1536,N=256;
  std::vector<int8_t> inputs(K,3),outputs(N,0);
  std::vector<uint8_t> packed(K*N/2,0x11);
  auto* input_shape=new TfLiteIntArray{};
  auto* weight_shape=new TfLiteIntArray{};
  auto* output_shape=new TfLiteIntArray{};
  input_shape->size=3; input_shape->data[0]=1;input_shape->data[1]=1;
  input_shape->data[2]=K;
  weight_shape->size=2;weight_shape->data[0]=N;weight_shape->data[1]=K;
  output_shape->size=3;output_shape->data[0]=1;output_shape->data[1]=1;
  output_shape->data[2]=N;
  auto* scales=new TfLiteFloatArray{};
  auto* zeroes=new TfLiteIntArray{};
  scales->size=N;zeroes->size=N;
  for(int i=0;i<N;++i){scales->data[i]=.0125f;zeroes->data[i]=0;}
  TfLiteAffineQuantization affine={scales,zeroes,0};
  TfLiteTensor tensors[3]={};
  tensors[0].type=kTfLiteInt8;
  tensors[0].data.int8=inputs.data();
  tensors[0].bytes=K;tensors[0].dims=input_shape;
  tensors[0].params={.01f,0};
  tensors[1].type=kTfLiteInt4;
  tensors[1].data.raw_const=reinterpret_cast<const char*>(packed.data());
  tensors[1].bytes=packed.size();
  tensors[1].dims=weight_shape;
  tensors[1].quantization={kTfLiteAffineQuantization,&affine};
  tensors[2].type=kTfLiteInt8;
  tensors[2].data.int8=outputs.data();
  tensors[2].bytes=N;tensors[2].dims=output_shape;
  auto* node_inputs=new TfLiteIntArray{};
  auto* node_outputs=new TfLiteIntArray{};
  node_inputs->size=2;node_inputs->data[0]=0;node_inputs->data[1]=1;
  node_outputs->size=1;node_outputs->data[0]=2;
  TfLiteNode node={node_inputs,node_outputs};
  TfLiteContext ctx={3,tensors};
  auto* wrapped=jaco_acfq::ShadowRegistration();
  assert(wrapped && wrapped->invoke);
  assert(wrapped->invoke(&ctx,&node)==kTfLiteOk);
  assert(original_calls==1);
  for(int i=0;i<N;++i)assert(outputs[i]==17);
  std::puts("JACO shadow smoke PASS: original output unchanged");
  delete input_shape;delete weight_shape;delete output_shape;
  delete scales;delete zeroes;delete node_inputs;delete node_outputs;
  return 0;
}
