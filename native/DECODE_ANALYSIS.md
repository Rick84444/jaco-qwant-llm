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

## Local Gemma 4 E2B decoder evidence (observed 2026-10-18)

The operator histogram below was **measured on the user's local model** with
`native/gemma_decode_fc.py` at repository commit
`940c3285a7d706854c3c6dbc6f476066eccfe568`.
It comes from the supplied Termux terminal output, not a committed binary
model or an independently reproduced remote run. The source report's local
file checksum was not included.

| Decode FC operand types | Observed operators |
| --- | ---: |
| INT8 activations × INT4 weights | 145 |
| INT8 activations × INT8 weights | 71 |
| INT8 activations × INT2 weights | 60 |
| FLOAT32 activations × INT2 weights | 1 |
| **Total** | **277** |

The sampled weight tensors include `[2048,1536]` INT4 at operator 12,
`[256,1536]` INT4 at operator 15, and `[262144,1536]` INT2 at operator
2064. The displayed samples show `quantized_dimension=0`, one scale and
zero point per output channel, and zero-point **previews** of 0. These are
observations of selected operators only; the full quantization arrays were
not inspected. There is no evidence that all zero-points are zero.

Machine-readable transcript-derived evidence is at
[`evidence/gemma4-e2b-decode-observed.json`](../evidence/gemma4-e2b-decode-observed.json).
Compare it against a real on-device report:

```sh
python3 native/decode_evidence.py --report bench/gemma-decode-fc.json
python3 -m unittest discover -s native -p 'test_*.py'
```

The comparison validates the total number of decoder FC operators and the
dtype histogram, not packed weight layout or any installed LiteRT ABI.

### Senior engineering assessment

* **Recommended first candidate:** INT8 × INT4 decode FullyConnected
  (`145 / 277` operators), subject to workload profiling.
* **Important constraint:** LiteRT weight tensors are `[N,K]` and use
  per-output-channel quantization, whereas our experimental ACFQ kernel
  expects `[K,N]` input-channel-major rows, FP32 input, and separate FP16
  scales for every group of 32 output weights.
* **Outstanding evidence:** native kernel registration, INT4 packing,
  activation scales/zero points, bias/requantization, memory bandwidth, and
  verified ACFQ calls during real Gemma token generation.
* **No measured speed gain or Gemma integration** should be inferred from
  the histogram, and no changes are made to the installed model or server.
