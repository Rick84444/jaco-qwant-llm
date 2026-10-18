#!/usr/bin/env python3
"""Inspect real Gemma decode FullyConnected tensor and quantization metadata.

Read-only FlatBuffer inspection. No operator execution, weight repacking or
LiteRT modification. Uses inspect_gemma_graph's bounded FlatBuffer reader.
"""
import argparse
import collections
import json
import mmap
import struct
from pathlib import Path

from inspect_gemma_graph import FB, FormatError, TYPES, inspect_file

def vector_sample(fb, table, field, fmt, limit=3):
    """Bounded preview of potentially very large quantization arrays.

    FB.vector has a 250k-element structural limit for tables and indices.
    Weight scale/zero-point vectors may legitimately exceed that number.
    Read their length directly, verify the ENTIRE byte range lies in this
    TFLite section, then decode only a fixed-size preview.
    """
    if not 0 <= limit <= 16:
        raise ValueError("preview limit must be between 0 and 16")
    offset=fb.reference(table,field)
    if offset is None:
        return {"count":0,"preview":[]}
    count=fb.u32(offset)
    start=offset+4
    element_size=struct.calcsize(fmt)
    fb.check(start,count*element_size)
    return {"count":count,
            "preview":[fb.read(fmt,start+element_size*i)
                       for i in range(min(count,limit))]}


QUANTIZATION_DETAILS={0:"NONE",1:"CustomQuantization",
                      2:"BlockwiseQuantization",3:"MultiAxisQuantization"}


def quantization_details(fb,quant):
    """Describe special TFLite quantization, without decoding any weights."""
    dtype=fb.scalar(quant,4,"<B")
    result={"type":QUANTIZATION_DETAILS.get(dtype,f"UNKNOWN_{dtype}"),
            "type_id":dtype}
    if dtype == 0:
        return result
    detail=fb.reference(quant,5)
    if detail is None:
        raise FormatError("quantization details_type is set but details are absent")
    if dtype in (2,3):
        result["scale_tensor_index"]=fb.scalar(detail,0,"<i")
        result["zero_point_tensor_index"]=fb.scalar(detail,1,"<i")
        result["block_size"]=fb.scalar(detail,2,"<i")
        if dtype == 3:
            result["quantized_dimensions"]=fb.values(detail,3,12)
    # Unknown/custom details are explicitly not interpreted as standard scales.
    return result

def tensor_detail(fb,tensors,index):
    if index<0:return {"index":index,"optional":True}
    if tensors is None or index>=tensors[1]:raise FormatError("tensor index outside subgraph")
    tab=fb.element_table(tensors,index)
    dtype=fb.scalar(tab,1,"<b")
    result={"index":index,"shape":fb.values(tab,0,12),
            "dtype":TYPES.get(dtype,"TYPE_"+str(dtype)),
            "buffer_index":fb.scalar(tab,2,"<I")}
    quant=fb.reference(tab,4)
    if quant is not None:
        result["quantization"]={
            "scales":vector_sample(fb,quant,2,"<f"),
            "zero_points":vector_sample(fb,quant,3,"<q"),
            "quantized_dimension":fb.scalar(quant,6,"<i"),
            "details_type":fb.scalar(quant,4,"<B"),
            "details":quantization_details(fb,quant)}
    else:result["quantization"]=None
    return result

def compatibility(a,w):
    if a.get("dtype")=="FLOAT32" and w.get("dtype")=="FLOAT32":
        return "fp32-candidate-needs-weight-transposition-and-runtime-hook"
    return "unsupported-direct-ACFQ-ABI"
 
def inspect_decode(path):
    inventory=inspect_file(path)
    evidence=[]
    with path.open("rb") as file:
        with mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mm:
            for sec in inventory["sections"]:
                if sec["kind"]!="TFLiteModel":continue
                begin,end=sec["begin_offset"],sec["end_offset"]
                fb=FB(mm,begin,end)
                if bytes(mm[begin+4:begin+8])!=b"TFL3":continue
                model=fb.root()
                opcode_vector=fb.vector(model,1)
                opcodes=[]
                if opcode_vector is not None:
                    for i in range(opcode_vector[1]):
                        tab=fb.element_table(opcode_vector,i)
                        code=fb.scalar(tab,3,"<i",None)
                        if code is None:code=fb.scalar(tab,0,"<b")
                        opcodes.append(code)
                subgraphs=fb.vector(model,2)
                if subgraphs is None:continue
                for si in range(subgraphs[1]):
                    sg=fb.element_table(subgraphs,si)
                    if fb.string(sg,4)!="decode":continue
                    tensors=fb.vector(sg,0)
                    operators=fb.vector(sg,3)
                    counts=collections.Counter()
                    samples=[]
                    count=0
                    if operators:
                        for oi in range(operators[1]):
                            op=fb.element_table(operators,oi)
                            codeidx=fb.scalar(op,0,"<I")
                            if codeidx>=len(opcodes):raise FormatError("invalid operator code index")
                            if opcodes[codeidx]!=9:continue
                            inputs=fb.values(op,1,16)
                            if len(inputs)<2:raise FormatError("FullyConnected has <2 inputs")
                            activation=tensor_detail(fb,tensors,inputs[0])
                            weight=tensor_detail(fb,tensors,inputs[1])
                            label=f"{activation.get('dtype')} x {weight.get('dtype')}"
                            counts[label]+=1
                            count+=1
                            if len(samples)<12 and counts[label]<=3:
                                samples.append({"operator_index":oi,
                                    "activation":activation,"weight":weight,
                                    "direct_acfq_compatibility":compatibility(activation,weight)})
                    evidence.append({"section":sec["index"],"subgraph_index":si,
                        "operator_count":operators[1] if operators else 0,
                        "fully_connected_count":count,
                        "dtype_pairs":dict(counts.most_common()),
                        "representative_operators":samples})
    return {
        "schema":1,"model":str(path),"decode_subgraphs":evidence,
        "all_decode_fc_examined":sum(e["fully_connected_count"] for e in evidence),
        "direct_acfq_compatibility_verified":False,
        "weight_packing_verified":False,
        "actual_gemma_acfq_execution":False,
        "required_native_changes":[
          "Trace real LiteRT INT8 x INT4 FullyConnected backend and per-tensor/per-channel quantization semantics",
          "Implement compatible INT8-input and INT4-weight kernel or explicit mathematically verified conversion",
          "Resolve weight [N,K] versus ACFQ channel-major [K,N] and actual scale/zero-point handling",
          "Register/integrate at a supported native backend boundary in isolated build",
          "Prove real Gemma invocation, numerical fidelity, quality and net decoding-speed improvement"],
        "note":"Quantization metadata is descriptive; a tensor dtype alone never proves packed-weight ABI compatibility."
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("bench/gemma-decode-fc.json"))
    args=parser.parse_args()
    result=inspect_decode(args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(f"Decode subgraphs: {len(result['decode_subgraphs'])}; FullyConnected: {result['all_decode_fc_examined']}; ACFQ not integrated.")
    for graph in result["decode_subgraphs"]:
        print(f"  section {graph['section']}: {graph['dtype_pairs']}")
        for item in graph["representative_operators"]:
            q=item["weight"].get("quantization") or {}
            print(f"    op {item['operator_index']}: weights={item['weight']['dtype']} {item['weight']['shape']}, quant={q}")
if __name__=="__main__":main()
