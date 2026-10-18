# Native Gemma/ACFQ integration — evidence requirements

ACFQ now has an independently callable C ABI, with F32 and Q4/Q8
numerical tests. The Q4/Q8 formats tested are **our own row-major input
channel layout**, not proven compatible with Gemma 4 E2B LiteRT weights.

Run local tests:
```sh
python3 -m unittest discover -s native -p 'test_*.py'
```

Inventory the real installed artifacts without modification:
```sh
python3 native/gemma_install_probe.py \
 --model "$HOME/.litert-lm/models/gemma4-e2b/model.litertlm" \
 --library "$PREFIX/lib/python3.13/site-packages/litert_lm/liblitert-lm.so"
python3 -m json.tool bench/gemma-install-probe.json
```

Artifact hashes do not prove model graph formats or source/binary parity.
NEXT: use a supported LiteRT model graph parser to identify actual
MatMul/FullyConnected tensor formats, then build an isolated compiled
backend with a traceable **actual Gemma inference invocation** of ACFQ,
and paired numerical/quality/performance comparisons. We must not inject
a kernel by monkeypatching Python, replace the installed liblitert-lm.so,
or claim the 3131 proxy is ACFQ execution. Ports 9379 and 3131 unchanged.
