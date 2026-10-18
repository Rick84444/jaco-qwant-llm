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

## Large-scale quantization regression (2026-10-18)

Real Gemma failed when a quantization vector exceeded the original generic
250,000-element FlatBuffer structural limit. That limit is appropriate for
enumerating operator/tensor tables, not large scalar metadata arrays. The
updated `vector_sample` reads only the vector header and **at most three
values**, while bounds-checking the *entire* declared byte range against
the current TFLite section. It also exposes TFLite's
`BlockwiseQuantization` / `MultiAxisQuantization` detail table when present:
scale-tensor index, zero-point-tensor index, and block size. These are
indices/metadata, not dequantized weights. The installed Gemma tensor format
and native ACFQ integration remain unverified until real output is checked.
