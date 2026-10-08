#!/usr/bin/env python3
"""Report runtime readiness without downloading model weights."""
import argparse, importlib.metadata as md, json, platform, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSIONS=["torch","unsloth","transformers","datasets","accelerate","peft","bitsandbytes","cityflow","laya","numpy","pandas","matplotlib","wandb","tqdm"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--strict",action="store_true"); p.add_argument("--model-path",default="checkpoints/traffic-qwen-v1-dual-choice"); p.add_argument("--jevlight-dir",default=".deps/JevLight"); p.add_argument("--traffic-file",default="data/Jinan/3_4/anon_3_4_jinan_real.json"); a=p.parse_args()
    out={"python":sys.version,"platform":platform.platform(),"versions":{},"cuda_available":False,"gpu":None,"gpu_memory_total_bytes":None}
    for name in VERSIONS:
        try: out["versions"][name]=md.version(name)
        except md.PackageNotFoundError: out["versions"][name]=None
    try:
        import torch
        out.update(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"]=torch.cuda.get_device_name(0); out["gpu_memory_total_bytes"]=torch.cuda.get_device_properties(0).total_memory
        import unsloth
        from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
        out["unsloth_decision_api"]={"FastDecisionModel":True,"DecisionTrainer":True,"bf16_supported":is_bfloat16_supported()}
        import cityflow
        out["cityflow_import"]="ok"
    except Exception as exc: out["import_error"]=f"{type(exc).__name__}: {exc}"
    model=ROOT/a.model_path; jev=ROOT/a.jevlight_dir
    out["local_adapter_present"]=model.is_dir() and (model/"adapter_model.safetensors").is_file() and (model/"joint_head.safetensors").is_file()
    out["jevlight_checkout_present"]=(jev/".git").exists(); out["traffic_file_present"]=(jev/a.traffic_file).is_file()
    out["base_model_id"]="unsloth/Qwen3.5-0.8B"; out["base_model_local_cache_present"]=False
    try:
        from huggingface_hub import try_to_load_from_cache
        cached=try_to_load_from_cache(out["base_model_id"],"config.json")
        out["base_model_local_cache_present"]=isinstance(cached,str) and Path(cached).is_file()
    except Exception: pass
    print(json.dumps(out,indent=2,default=str))
    if a.strict and (out.get("import_error") or not out["cuda_available"] or not out["jevlight_checkout_present"] or not out["traffic_file_present"]): raise SystemExit(1)
if __name__=="__main__": main()
