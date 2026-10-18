#!/usr/bin/env python3
"""Read-only .litertlm section and embedded TFLite operator inventory.

Uses only Python's standard library. Does NOT read/dequantize model weights,
register any operator or run inference. Based on LiteRT-LM header schema and
TFLite FlatBuffer schema; unsupported layouts are reported, never guessed.
"""
import argparse
import collections
import json
import mmap
import os
import struct
from pathlib import Path

SECTION_TYPES = {
    0: "NONE", 1: "GenericBinaryData", 2: "Deprecated",
    3: "TFLiteModel", 4: "SP_Tokenizer", 5: "LlmMetadataProto",
    6: "HF_Tokenizer_Zlib", 7: "TFLiteWeights",
    8: "EmbeddingMetadataProto", 9: "ExecutorMetadataProto",
    10: "Text2SpeechMetadataProto", 11: "Speech2TextMetadataProto",
    12: "Text2ImageMetadataProto",
}
TYPES={0:"FLOAT32",1:"FLOAT16",2:"INT32",3:"UINT8",9:"INT8",
       17:"INT4",18:"BFLOAT16",19:"INT2",20:"UINT4"}
OPS={0:"ADD",6:"DEQUANTIZE",9:"FULLY_CONNECTED",18:"MUL",
     22:"RESHAPE",32:"CUSTOM",126:"BATCH_MATMUL"}
CANDIDATES={9,126}
LIMIT=250000

class FormatError(ValueError):
    pass

class FB:
    def __init__(self, data, lo, hi):
        self.data,self.lo,self.hi=data,lo,hi
    def check(self,p,n):
        if p < self.lo or n < 0 or p+n > self.hi:
            raise FormatError("FlatBuffer reference outside section")
    def read(self,fmt,p):
        n=struct.calcsize(fmt);self.check(p,n)
        return struct.unpack_from(fmt,self.data,p)[0]
    def u32(self,p):return self.read("<I",p)
    def i32(self,p):return self.read("<i",p)
    def u16(self,p):return self.read("<H",p)
    def u8(self,p):return self.read("<B",p)
    def root(self):
        return self.indirect(self.lo)
    def indirect(self,p):
        result=p+self.u32(p);self.check(result,4)
        return result
    def field(self,table,index):
        self.check(table,4)
        vt=table-self.i32(table)
        self.check(vt,4)
        length=self.u16(vt)
        if length<4 or length>4096:raise FormatError("invalid vtable length")
        self.check(vt,length)
        off=vt+4+2*index
        if off+2>vt+length:return None
        delta=self.u16(off)
        if not delta:return None
        location=table+delta
        self.check(location,1)
        return location
    def scalar(self,table,index,fmt,default=0):
        p=self.field(table,index)
        return default if p is None else self.read(fmt,p)
    def reference(self,table,index):
        p=self.field(table,index)
        return None if p is None else self.indirect(p)
    def vector(self,table,index):
        ref=self.reference(table,index)
        if ref is None:return None
        count=self.u32(ref)
        if count>LIMIT:raise FormatError("vector exceeds safe limit")
        return ref+4,count
    def element_table(self,vector,idx):
        begin,n=vector
        if not 0<=idx<n:raise FormatError("vector index out of bounds")
        return self.indirect(begin+4*idx)
    def values(self,table,index,limit=32):
        vector=self.vector(table,index)
        if vector is None:return []
        begin,n=vector
        if n>LIMIT:raise FormatError("oversized value vector")
        self.check(begin,n*4)
        return [self.i32(begin+4*i) for i in range(min(n,limit))]
    def string(self,table,index,maxlen=160):
        pos=self.reference(table,index)
        if pos is None:return None
        n=self.u32(pos)
        if n>maxlen:return "<long string>"
        self.check(pos+4,n)
        return bytes(self.data[pos+4:pos+4+n]).decode("utf-8","replace")

def inspect_tflite(mm,start,end,max_candidates=12):
    fb=FB(mm,start,end)
    fb.check(start,8)
    if bytes(mm[start+4:start+8])!=b"TFL3":
        return {"status":"unsupported-tflite-identifier","identifier_hex":bytes(mm[start+4:start+8]).hex()}
    model=fb.root()
    schema=fb.scalar(model,0,"<I")
    codes=[]
    cv=fb.vector(model,1)
    if cv:
        for i in range(min(cv[1],2048)):
            table=fb.element_table(cv,i)
            # TFLite v3a+: index 3 contains extended builtin_code.
            code=fb.scalar(table,3,"<i",None)
            if code is None:code=fb.scalar(table,0,"<b",0)
            custom=fb.string(table,1) if code==32 else None
            codes.append((code,custom))
    subgraphs=[]
    sv=fb.vector(model,2)
    if sv:
        for si in range(min(sv[1],16)):
            sg=fb.element_table(sv,si)
            tensors=fb.vector(sg,0)
            operators=fb.vector(sg,3)
            opcounts=collections.Counter()
            candidates=[]
            if operators:
                for oi in range(operators[1]):
                    op=fb.element_table(operators,oi)
                    code_index=fb.scalar(op,0,"<I")
                    if code_index>=len(codes):
                        raise FormatError("operator references invalid opcode")
                    code,custom=codes[code_index]
                    name=("CUSTOM:"+custom) if code==32 and custom else OPS.get(code,"OP_"+str(code))
                    opcounts[name]+=1
                    if (code in CANDIDATES or code==32) and len(candidates)<max_candidates:
                        operands=[]
                        for tensor_index in fb.values(op,1,16):
                            if tensor_index < 0:
                                operands.append({"tensor_index":tensor_index})
                                continue
                            if tensors is None or tensor_index>=tensors[1]:
                                raise FormatError("operator references invalid tensor")
                            t=fb.element_table(tensors,tensor_index)
                            dtype=fb.scalar(t,1,"<b")
                            operands.append({"tensor_index":tensor_index,
                               "name":fb.string(t,3),
                               "shape":fb.values(t,0,12),
                               "dtype":TYPES.get(dtype,"TYPE_"+str(dtype)),
                               "buffer_index":fb.scalar(t,2,"<I")})
                        candidates.append({"operator_index":oi,"operator":name,"inputs":operands})
            subgraphs.append({"index":si,"name":fb.string(sg,4),
                "tensor_count":tensors[1] if tensors else 0,
                "operator_count":operators[1] if operators else 0,
                "operator_histogram":dict(opcounts.most_common()),
                "candidate_operators":candidates})
    return {"status":"tflite-schema-inventory","schema_version":schema,
            "opcode_count":len(codes),"subgraphs":subgraphs}

def inspect_file(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    with path.open("rb") as file:
        size=os.fstat(file.fileno()).st_size
        if size<32:raise FormatError("not a LiteRT-LM container: too short")
        with mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mm:
            if bytes(mm[:8])!=b"LITERTLM":
                raise FormatError("wrong LiteRT-LM magic")
            major,minor,patch=struct.unpack_from("<III",mm,8)
            header_end=struct.unpack_from("<Q",mm,24)[0]
            if header_end<=32 or header_end>size or header_end>64*1024*1024:
                raise FormatError("invalid or excessive LiteRT-LM header")
            fb=FB(mm,32,header_end)
            root=fb.root()
            section_meta=fb.reference(root,1)
            if section_meta is None:raise FormatError("no section metadata")
            objects=fb.vector(section_meta,0)
            if objects is None:raise FormatError("no section list")
            if objects[1]>1024:raise FormatError("too many sections")
            sections=[]
            for i in range(objects[1]):
                obj=fb.element_table(objects,i)
                begin=fb.scalar(obj,1,"<Q")
                end=fb.scalar(obj,2,"<Q")
                kind=fb.scalar(obj,3,"<B")
                if begin< header_end or end<=begin or end>size:
                    raise FormatError("invalid section range")
                result={"index":i,"kind":SECTION_TYPES.get(kind,str(kind)),
                        "begin_offset":begin,"end_offset":end,
                        "bytes":end-begin}
                if kind==3:
                    try:result["graph"]=inspect_tflite(mm,begin,end)
                    except FormatError as exc:result["graph"]={"status":"parse-error","reason":str(exc)}
                sections.append(result)
    return {"schema":1,"path":str(path),"bytes":size,
            "container_version":f"{major}.{minor}.{patch}",
            "sections":sections,
            "acfq_compatible_layout_verified":False,
            "acfq_inference_integrated":False,
            "interpretation":"Read-only metadata and graph operator enumeration; no weight-layout or call-path proof"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("bench/gemma-graph.json"))
    args=parser.parse_args()
    result=inspect_file(args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(f"Sections: {len(result['sections'])}. Saved {args.output}. ACFQ inference NOT integrated.")
    for s in result['sections']:
        graph=s.get("graph")
        n=len(graph.get("subgraphs",[])) if graph else 0
        print(f"  #{s['index']}: {s['kind']} ({s['bytes']} bytes), subgraphs={n}")
if __name__=="__main__":main()
