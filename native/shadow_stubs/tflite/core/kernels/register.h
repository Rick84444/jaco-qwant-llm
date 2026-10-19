#ifndef JACO_TEST_TFLITE_REGISTER_H
#define JACO_TEST_TFLITE_REGISTER_H
#include "tflite/core/c/common.h"
namespace tflite {namespace ops {namespace builtin {
TfLiteRegistration* Register_FULLY_CONNECTED();
}}}
#endif
