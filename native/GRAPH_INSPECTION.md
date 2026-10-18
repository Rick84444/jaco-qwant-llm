# Read-only Gemma 4 model-graph inventory

`inspect_gemma_graph.py` maps the documented LiteRT-LM container header
and its TFLiteModel sections. It parses the embedded TFLite FlatBuffer
operator code table, subgraphs, and candidate FC/BATCH_MATMUL/CUSTOM
input tensor shapes and dtypes **without loading tensor weights into RAM**
or changing the model.

Source schemas:
- LiteRT-LM pinned `schema/core/litertlm_header_schema.fbs`
- LiteRT-LM pinned `schema/core/litertlm_read.cc`
- TFLite v3 `schema.fbs`

```sh
python3 native/inspect_gemma_graph.py \
  --model "$HOME/.litert-lm/models/gemma4-e2b/model.litertlm"
python3 - <<'PY'
import json
r=json.load(open("bench/gemma-graph.json"))
for s in r["sections"]:
    print(s["index"],s["kind"],s["bytes"])
    if "graph" in s:
        print(s["graph"].get("status"))
        for g in s["graph"].get("subgraphs",[]):
            print(g["name"],g["operator_histogram"])
            print("candidate",g["candidate_operators"][:3])
PY
```

The inspection is intentionally read-only and conservative. The presence
of `FULLY_CONNECTED`, `BATCH_MATMUL` or `CUSTOM` operator **does not**
prove ACFQ is registerable at that location. Tensor shapes and dtypes
alone cannot demonstrate packed weight compatibility, native execution,
or numerical performance. Unsupported model variants remain clearly
flagged; do not modify installed LiteRT, ports or model files.
