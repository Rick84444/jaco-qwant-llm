# Real Gemma decoder quantization study

Read-only additional analysis of the installed `.litertlm` model's
**decode** subgraph(s). Inspect all FullyConnected operators, count input x
weight dtype combinations, and sample TFLite quantization metadata:
`scale`, `zero_point`, `quantized_dimension`, and `details_type`.

```sh
cd ~/jaco-qwant-llm
python3 native/gemma_decode_fc.py \
 --model "$HOME/.litert-lm/models/gemma4-e2b/model.litertlm"
python3 -m unittest discover -s native -p 'test_*.py'
```

Outputs `bench/gemma-decode-fc.json`. Only metadata are parsed; raw weights
are never copied, model files are not modified. The input format reported for
real Gemma is INT8 and many decoding weights are INT4; the existing ACFQ
native C ABI takes FP32 activations and its own group-of-32 packed Q4/Q8
weights with **input-channel-major [K,N]** layout, whereas a LiteRT
FullyConnected weight tensor declares shape **[N,K]**. Mere shape/dtype
matches are not enough to infer compatible memory packing or scales.

Do not identify any operator as integrated until an isolated LiteRT native
build instruments **real decode execution** through ACFQ and proves numerical
fidelity and comparative whole-request speed.
