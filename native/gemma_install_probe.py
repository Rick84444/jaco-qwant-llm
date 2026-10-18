#!/usr/bin/env python3
"""Read-only artifact identification; no model format claims."""
import argparse,hashlib,importlib.metadata,json
from pathlib import Path
def identify(path):
 if not path.is_file():return {'path':str(path),'present':False}
 digest=hashlib.sha256()
 with path.open('rb') as f:
  head=f.read(32);digest.update(head)
  for part in iter(lambda:f.read(1048576),b''):digest.update(part)
 return {'path':str(path),'present':True,'size':path.stat().st_size,'sha256':digest.hexdigest(),'header_hex':head.hex()}
def report(model,library):
 try:version=importlib.metadata.version('litert-lm')
 except importlib.metadata.PackageNotFoundError:version=None
 return {'model':identify(model),'library':identify(library),'python_package_version':version,
         'weight_layout_verified':False,'actual_gemma_acfq_calls':0,
         'inference_integrated':False,'interpretation':'Artifact inventory; not a graph parser or integration proof'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path)
 p.add_argument('--library',required=True,type=Path)
 p.add_argument('--output',type=Path,default=Path('bench/gemma-install-probe.json'))
 a=p.parse_args();result=report(a.model,a.library);a.output.parent.mkdir(exist_ok=True,parents=True)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(f'Wrote {a.output}; weight layout and ACFQ integration remain unverified')
if __name__=='__main__':main()
