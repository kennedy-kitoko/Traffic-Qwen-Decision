"""Evaluate Traffic-Qwen V1 Dual-Choice with the JevLight OneLine CityFlow path."""
from __future__ import annotations
import argparse, json, os, statistics, time
from pathlib import Path

DATASETS={"jinan":("3_4","Jinan"),"hangzhou":("4_4","Hangzhou"),"newyork_28x7":("28_7","NewYork")}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=DATASETS,default="jinan")
    p.add_argument("--traffic_file",default="anon_3_4_jinan_real.json")
    p.add_argument("--run_counts",type=int,default=3600)
    p.add_argument("--model_path",default="models/traffic-qwen-v1-dual-choice")
    p.add_argument("--model_revision",default=None)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=3407)
    mode=p.add_mutually_exclusive_group(); mode.add_argument("--guardrail",action="store_true"); mode.add_argument("--no_guardrail",action="store_true")
    fallback=p.add_mutually_exclusive_group(); fallback.add_argument("--fallback",action="store_true"); fallback.add_argument("--no_fallback",action="store_true")
    p.add_argument("--min_confidence",type=float,default=0.0)
    p.add_argument("--proj_name",default="Traffic-Qwen-V1-Dual-Choice")
    p.add_argument("--output_dir",default="")
    p.add_argument("--skip_benchmark_update",action="store_true",
                   help="Keep benchmark JSON files in the JevLight checkout untouched")
    return p.parse_args()

def summarize_trace(trace_path, result, cfg, runtime):
    import numpy as np
    from collections import Counter
    raw=json.loads(Path(trace_path).read_text())
    decisions=[]; phase_counts=Counter(); duration_counts=Counter(); combined=Counter(); switches=0; retained=0; opportunities=0
    per_intersection_phase=Counter(); max_nonservice=0
    confidences=[]; pconf=[]; dconf=[]; latencies=[]; fallbacks=0; corrections=0
    queue_series=[]; waiting_series=[]; max_nonservice=0
    for iid, records in enumerate(raw):
        last_phase=None; served_times={i:[] for i in range(4)}
        for r in records:
            if r.get("phase_probabilities") is None: continue
            item={**r,"intersection_index":iid}; decisions.append(item)
            pp=r.get("phase_probabilities") or {}; dp=r.get("duration_probabilities") or {}
            phase=int(r["applied_phase"])-1 if r.get("applied_phase") else int(str(r["predicted_phase"]).split("_")[-1])
            duration=int(r["applied_duration"] or r["predicted_duration"])
            phase_counts[phase]+=1; duration_counts[duration]+=1; combined[f"phase_{phase}_{duration}"]+=1
            per_intersection_phase[f"{records[0].get('intersection_id', iid)}|phase_{phase}"]+=1
            served_times[phase].append(int(r.get("simulation_time",0) or 0))
            if last_phase is not None:
                opportunities+=1
                if phase==last_phase: retained+=1
                else: switches+=1
            last_phase=phase
            if r.get("fallback_used"): fallbacks+=1
            if r.get("guardrail_corrected"): corrections+=1
            if r.get("phase_confidence") is not None: pconf.append(float(r["phase_confidence"]))
            if r.get("duration_confidence") is not None: dconf.append(float(r["duration_confidence"]))
            if r.get("latency_ms") is not None: latencies.append(float(r["latency_ms"]))
            if r.get("decision_confidence") is not None: confidences.append(float(r["decision_confidence"]))
        # Approximate non-service intervals from successive decision timestamps for each phase.
        horizon=int(cfg["RUN_COUNTS"])
        for times in served_times.values():
            gaps=[times[0]] if times else [horizon]
            gaps += [b-a for a,b in zip(times,times[1:])]
            if times: gaps.append(max(0,horizon-times[-1]))
            max_nonservice=max(max_nonservice,max(gaps,default=0))
    # Aggregate aligned pre-decision trace observations across intersections per 5 s tick.
    ticks=min((len(x) for x in raw),default=0)
    for tick in range(ticks):
        q=0.0; wait_sum=0.0
        for inter in raw:
            st=inter[tick].get("state",{})
            if not isinstance(st,dict): continue
            for movement in st.values():
                if not isinstance(movement,dict): continue
                mq=float(movement.get("queue_len",0) or 0); q+=mq
                wait_sum+=mq*float(movement.get("avg_wait_time",0) or 0)
        queue_series.append(q)
        waiting_series.append(wait_sum/q if q>0 else 0.0)
    lat=np.asarray(latencies,dtype=float)
    metric={"reward":result["test_reward_over"],"avg_queue":result["test_avg_queue_len_over"],
            "avg_waiting_time_seconds":result["test_avg_waiting_time_over"],"avg_travel_time_seconds":result["test_avg_travel_time_over"],
            "horizon_seconds":cfg["RUN_COUNTS"],"model_decisions":len(decisions),"request_fallbacks":fallbacks,
            "guardrail_corrections":corrections,"runtime_seconds":runtime,
            "confidence_min":min(confidences) if confidences else min(pconf,default=None),
            "confidence_mean":statistics.mean(confidences) if confidences else statistics.mean(pconf) if pconf else None,
            "confidence_max":max(confidences) if confidences else max(pconf,default=None),
            "phase_confidence_mean":statistics.mean(pconf) if pconf else None,
            "duration_confidence_mean":statistics.mean(dconf) if dconf else None,
            "latency_mean_ms":float(lat.mean()) if len(lat) else None,"latency_p50_ms":float(np.percentile(lat,50)) if len(lat) else None,
            "latency_p95_ms":float(np.percentile(lat,95)) if len(lat) else None,"latency_max_ms":float(lat.max()) if len(lat) else None,
            "GPU_peak_memory_GiB":None,"phase_switches":switches,
            "current_phase_retention":retained/opportunities if opportunities else None,
            "maximum_non_service_time_seconds":max_nonservice,"phase_distribution":{str(k):v for k,v in sorted(phase_counts.items())},
            "intersection_phase_distribution":dict(per_intersection_phase),
            "duration_distribution_seconds":{str(k):v for k,v in sorted(duration_counts.items())},
            "combined_action_distribution":dict(combined),"guardrail_enabled":cfg["AGENT_CONF"]["TRAFFIC_QWEN_GUARDRAIL"],
            "fallback_enabled":cfg["AGENT_CONF"]["TRAFFIC_QWEN_FALLBACK"],
            "queue_series_mean_by_sample":queue_series,"waiting_series_mean_proxy_by_sample":waiting_series,
            "maximum_observed_queue_from_trace":max(queue_series,default=None)}
    # Decision traces include one record for every inference; preserve raw OneLine trace separately.
    out=Path(cfg["OUTPUT_DIR"]); out.mkdir(parents=True,exist_ok=True)
    with (out/"decision_trace.jsonl").open("w",encoding="utf-8") as f:
        for r in decisions: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    metric["trace_path"]=str(trace_path)
    return metric

def main():
    args=parse_args()
    # Keep Unsloth/Torch ahead of CityFlow extension imports on this CUDA build.
    os.environ.update(WANDB_MODE="offline",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    from models.traffic_qwen_agent import TrafficQwenAgent
    import random
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    warm=TrafficQwenAgent.warmup(args.model_path,args.device,args.seed,args.model_revision)
    # CityFlowEnv draws its Engine seed from NumPy global state during reset.
    # Reset it after model loading so paired arms get the exact same Engine seed.
    np.random.seed(args.seed)
    cityflow_seed=int(np.random.RandomState(args.seed).randint(0,100))
    print("Traffic-Qwen warmup completed",flush=True)
    roadnet,template=DATASETS[args.dataset]; rows,cols=map(int,roadnet.split("_")); n=rows*cols
    data_path=os.path.join("data",template,roadnet); traffic_path=os.path.join(data_path,args.traffic_file); roadnet_file=f"roadnet_{roadnet}.json"
    for path in (traffic_path,os.path.join(data_path,roadnet_file)):
        if not os.path.exists(path): raise FileNotFoundError(path)
    timestamp=time.strftime("%Y%m%d_%H%M%S")
    output=Path(args.output_dir or f"records/traffic_qwen_v1/{timestamp}_{args.run_counts}s")
    if output.exists(): raise FileExistsError(output)
    from utils import config
    from utils.my_utils import merge
    from utils.oneline import OneLine
    action_duration={i:s for i,s in enumerate((15,20,25,30,35,40))}
    traffic_conf={"MODEL_NAME":"TrafficQwen","MODEL":"TrafficQwen","PROJECT_NAME":args.proj_name,
        "RUN_COUNTS":args.run_counts,"NUM_ROW":rows,"NUM_COL":cols,"NUM_AGENTS":n,"NUM_INTERSECTIONS":n,
        "TRAFFIC_FILE":args.traffic_file,"ROADNET_FILE":roadnet_file,"ACTION_PATTERN":"set","MIN_ACTION_TIME":10,
        "MIN_ACTION_TIME1":5,"DECISION_INTERVAL":5,"YELLOW_TIME":5,"DURATION_EXCLUDES_YELLOW":True,
        "ACTION_DURATION":action_duration,"LIST_STATE_FEATURE":["cur_phase","time_this_phase","traffic_movement_pressure_queue"],
        "DIC_REWARD_INFO":{"queue_length":-0.25},"LIST_MODEL_NEED_TO_UPDATE":[],"NUM_PHASES":4,
        "AGENT_CONF":{"TRAFFIC_QWEN_GUARDRAIL":bool(args.guardrail),"TRAFFIC_QWEN_FALLBACK":bool(args.fallback)},
        "LABEL":f"seed_{args.seed}_guardrail_{bool(args.guardrail)}_fallback_{bool(args.fallback)}"}
    agent_conf={"TRAFFIC_QWEN_MODEL_PATH":args.model_path,"TRAFFIC_QWEN_DEVICE":args.device,
        "TRAFFIC_QWEN_MODEL_REVISION":args.model_revision,
        "TRAFFIC_QWEN_GUARDRAIL":bool(args.guardrail),"TRAFFIC_QWEN_FALLBACK":bool(args.fallback),
        "TRAFFIC_QWEN_MIN_CONFIDENCE":args.min_confidence,"SEED":args.seed}
    paths={"PATH_TO_MODEL":f"model/traffic_qwen_runtime/{timestamp}_{args.run_counts}s",
           "PATH_TO_WORK_DIRECTORY":str(output),"PATH_TO_DATA":data_path}
    cfg={"RUN_COUNTS":args.run_counts,"AGENT_CONF":agent_conf,"OUTPUT_DIR":str(output)}
    env_artifact=(output.parent / "environment_manifests" if args.output_dir
                  else Path("artifacts/traffic_qwen/v1_dual_choice/evaluation"))
    env_artifact.mkdir(parents=True,exist_ok=True)
    wandb_dir=env_artifact/"wandb"/output.name
    wandb_dir.mkdir(parents=True,exist_ok=True)
    os.environ["WANDB_DIR"]=str(wandb_dir)
    (env_artifact/f"environment_{timestamp}_{args.run_counts}s_guardrail_{bool(args.guardrail)}.json").write_text(json.dumps({
        "python":__import__("sys").version,"seed":args.seed,"torch":__import__("torch").__version__,
        "cuda":__import__("torch").version.cuda,"gpu":__import__("torch").cuda.get_device_name(0),
        "model":args.model_path,"device":args.device,"run_counts":args.run_counts,
        "model_revision":args.model_revision,
        "seed":args.seed,"cityflow_engine_seed":cityflow_seed,
        "guardrail":args.guardrail,"fallback":args.fallback,"warmup":warm},indent=2,ensure_ascii=False)+"\n")
    runner=OneLine(dic_agent_conf=agent_conf,dic_traffic_env_conf=merge(config.dic_traffic_env_conf,traffic_conf),
        dic_path=merge(config.DIC_PATH,paths),roadnet=f"{template}-{roadnet}",trafficflow=args.traffic_file.rsplit(".",1)[0])
    started=time.perf_counter(); result=runner.train(round=0); runtime=time.perf_counter()-started
    metric=summarize_trace(output/"state_action.json",result,cfg,runtime)
    import torch
    metric["GPU_peak_memory_GiB"]=float(torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else None
    metric.update({"controller":"Traffic-Qwen-V1-Dual-Choice","dataset":f"{template}/{roadnet}/{args.traffic_file}",
                   "seed":args.seed,"model_path":args.model_path,"warmup":warm})
    (output/"metric_timeseries.json").write_text(json.dumps({
        "time_seconds":list(range(0,(len(metric["queue_series_mean_by_sample"]))*5,5)),
        "queue_total_from_trace":metric["queue_series_mean_by_sample"],
        "waiting_time_mean_proxy_seconds":metric["waiting_series_mean_proxy_by_sample"]},separators=(",",":"))+"\n")
    (output/"traffic_qwen_metrics.json").write_text(json.dumps(metric,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(metric,indent=2,ensure_ascii=False),flush=True)
    if (not args.skip_benchmark_update and args.run_counts==3600 and args.dataset=="jinan"
            and args.traffic_file=="anon_3_4_jinan_real.json"):
        benchmark_path=Path("results/benchmark_jinan.json"); original=json.loads(benchmark_path.read_text())
        expanded=json.loads(json.dumps(original)); output_benchmark=Path("results/benchmark_jinan_with_traffic_qwen_v1.json")
        if output_benchmark.exists():
            previous=json.loads(output_benchmark.read_text())
            retained=[r for r in previous["metrics"]["controllers"] if r["name"].startswith("Traffic-Qwen-V1-Dual-Choice")]
        else: retained=[]
        this_name="Traffic-Qwen-V1-Dual-Choice" if not args.guardrail else "Traffic-Qwen-V1-Dual-Choice-guardrail"
        retained=[r for r in retained if r["name"]!=this_name]
        expanded["metrics"]["controllers"].extend(retained)
        compact={k:v for k,v in metric.items() if k not in {"queue_series_mean_by_sample","waiting_series_mean_proxy_by_sample"}}
        expanded["metrics"]["controllers"].append({"name":this_name,**compact})
        output_benchmark.write_text(json.dumps(expanded,indent=2)+"\n")

if __name__=="__main__": main()
