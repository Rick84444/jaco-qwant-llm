#ifndef JACO_TEST_TFLITE_COMMON_H
#define JACO_TEST_TFLITE_COMMON_H
#include <cstddef>
#include <cstdint>
enum TfLiteStatus{kTfLiteOk=0,kTfLiteError=1};
enum TfLiteType{kTfLiteNoType=0,kTfLiteFloat32=1,kTfLiteInt8=9,kTfLiteInt4=17};
enum TfLiteQuantizationType{kTfLiteNoQuantization=0,kTfLiteAffineQuantization=1};
struct TfLiteIntArray {int size;int data[2048];};
struct TfLiteFloatArray {int size;float data[2048];};
struct TfLiteAffineQuantization {TfLiteFloatArray* scale;TfLiteIntArray* zero_point;int quantized_dimension;};
struct TfLiteQuantizationParams{float scale;int zero_point;};
struct TfLiteQuantization {TfLiteQuantizationType type;void* params;};
union TfLitePtrUnion{int8_t* int8;const char* raw_const;};
struct TfLiteTensor{
  TfLiteType type;size_t bytes;TfLitePtrUnion data;TfLiteQuantization quantization;
  TfLiteQuantizationParams params;TfLiteIntArray* dims;
};
struct TfLiteContext{int tensors_size;TfLiteTensor* tensors;};
struct TfLiteNode{TfLiteIntArray* inputs;TfLiteIntArray* outputs;};
struct TfLiteRegistration{
  void* (*init)(TfLiteContext*,const char*,size_t);
  void (*free)(TfLiteContext*,void*);
  TfLiteStatus (*prepare)(TfLiteContext*,TfLiteNode*);
  TfLiteStatus (*invoke)(TfLiteContext*,TfLiteNode*);
};
#endif
