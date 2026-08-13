import os, glob, json, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
JAC="/Users/ayush/.local/bin/jac"
SRC=os.path.expanduser("~/repos/jac-data-gen/data/jac_outputs")
files=sorted(glob.glob(SRC+"/*.jac"))
OUT="/tmp/nitin_triage/compile_results.jsonl"
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["file"])
        except Exception: pass
todo=[f for f in files if os.path.basename(f) not in done]
print("total",len(files),"todo",len(todo),flush=True)
import threading
lock=threading.Lock()
fh=open(OUT,"a")
n=[0]
def work(f):
    b=os.path.basename(f)
    d=tempfile.mkdtemp(prefix="jc_", dir="/tmp/nitin_triage/work")
    try:
        p=os.path.join(d,"snippet.jac")
        shutil.copyfile(f,p)
        try:
            r=subprocess.run([JAC,"run","snippet.jac"],capture_output=True,text=True,timeout=60,cwd=d)
            rc,err,out=r.returncode,r.stderr,r.stdout
        except subprocess.TimeoutExpired:
            rc,err,out=124,"TIMEOUT 60s",""
        except Exception as e:
            rc,err,out=125,str(e),""
    finally:
        shutil.rmtree(d,ignore_errors=True)
    rec={"file":b,"rc":rc,"stderr":err[-1200:],"stdout":out[:200]}
    with lock:
        fh.write(json.dumps(rec)+"\n"); fh.flush()
        n[0]+=1
        if n[0]%250==0: print("done",n[0],flush=True)
with ThreadPoolExecutor(max_workers=12) as ex:
    list(ex.map(work, todo))
fh.close()
print("ALL DONE",flush=True)
