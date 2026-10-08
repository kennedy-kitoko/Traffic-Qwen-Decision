"""Validation-temperature calibration and held-out offline evaluation."""
from __future__ import annotations
import json, math, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

PHASES = [f"phase_{i}" for i in range(4)]
DURATIONS = ["15", "20", "25", "30", "35", "40"]

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def predict_rows(model, tokenizer, rows):
    from unsloth import FastDecisionModel
    result=[]
    for i, row in enumerate(rows):
        prediction=FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
        entry={"gold_phase":row["gold"]["phase"]["choice"], "gold_duration":row["gold"]["duration"]["choice"],
               "phase_probabilities":prediction["phase"]["probabilities"],
               "duration_probabilities":prediction["duration"]["probabilities"],
               "intersection_id":row.get("metadata",{}).get("intersection_id",row["state"].get("intersection_id")),
               "simulation_time_seconds":row.get("metadata",{}).get("simulation_time_seconds",row["state"].get("simulation_time_seconds")),
               "queue_total":sum(float(v.get("queue_vehicles",0) or 0) for v in row["state"].get("movement_lanes",{}).values())}
        result.append(entry)
        if (i+1)%25==0: print(f"predicted {i+1}/{len(rows)}", flush=True)
    return result

def temperature(p, t):
    x=np.log(np.clip(np.asarray(p,dtype=float),1e-12,1.0))/t
    x-=x.max()
    e=np.exp(x)
    return e/e.sum()

def fit_temperature(pred, task, labels, classes):
    best=(float("inf"),1.0)
    for t in np.exp(np.linspace(math.log(.05),math.log(20),401)):
        nll=0.0
        for row,label in zip(pred,labels):
            p=row[f"{task}_probabilities"]
            probs=temperature([p[c] for c in classes],t)
            nll-=math.log(max(probs[classes.index(label)],1e-12))
        nll/=len(labels)
        if nll<best[0]: best=(nll,float(t))
    return best[1]

def metrics(pred, task, labels, classes, t=1.0):
    probs=np.asarray([temperature([r[f"{task}_probabilities"][c] for c in classes],t) for r in pred])
    gold=np.asarray([classes.index(x) for x in labels])
    top=probs.argmax(axis=1)
    cm=np.zeros((len(classes),len(classes)),dtype=int)
    for y,p in zip(gold,top): cm[y,p]+=1
    accuracy=float(np.mean(gold==top))
    supports=cm.sum(axis=1)
    recalls=np.divide(np.diag(cm),supports,out=np.zeros(len(classes),dtype=float),where=supports>0)
    precisions=np.divide(np.diag(cm),cm.sum(axis=0),out=np.zeros(len(classes),dtype=float),where=cm.sum(axis=0)>0)
    f1=np.divide(2*precisions*recalls,precisions+recalls,out=np.zeros(len(classes)),where=(precisions+recalls)>0)
    one=np.eye(len(classes))[gold]
    brier=float(np.mean(np.sum((probs-one)**2,axis=1)))
    nll=float(-np.mean(np.log(np.clip(probs[np.arange(len(gold)),gold],1e-12,1))))
    conf=probs.max(axis=1); correct=(top==gold)
    ece=0.0; bins=[]
    for lo in np.linspace(0,1,11)[:-1]:
        hi=lo+.1
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any():
            c=float(conf[mask].mean()); a=float(correct[mask].mean()); ece+=mask.mean()*abs(c-a)
            bins.append({"lower":float(lo),"upper":float(hi),"count":int(mask.sum()),"confidence":c,"accuracy":a})
    out={"count":len(labels),"accuracy":accuracy,"balanced_accuracy":float(np.mean(recalls)),"macro_f1":float(np.mean(f1)),
         "confusion_matrix":cm.tolist(),"classes":classes,"brier_score":brier,"ece":float(ece),"log_loss":nll,
         "mean_confidence_correct":float(conf[correct].mean()) if correct.any() else None,
         "mean_confidence_incorrect":float(conf[~correct].mean()) if (~correct).any() else None,"calibration_bins":bins}
    if task=="phase":
        out["top2_accuracy"]=float(np.mean([gold[i] in np.argsort(probs[i])[-2:] for i in range(len(gold))]))
    else:
        nums=np.asarray([int(x) for x in labels]); predicted=np.asarray([int(classes[i]) for i in top])
        out["accuracy_within_5_seconds"]=float(np.mean(np.abs(nums-predicted)<=5))
        out["mae_seconds"]=float(np.mean(np.abs(nums-predicted)))
    return out, probs, top

def main():
    import torch, unsloth
    from unsloth import FastDecisionModel, is_bfloat16_supported
    from pathlib import Path
    parser=__import__("argparse").ArgumentParser()
    parser.add_argument("--config",default="configs/traffic_qwen_v1_dual_choice.json")
    parser.add_argument("--output-dir",default="results/reports/offline_evaluation")
    parser.add_argument("--model-path",default=None)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.model_path: config["output_dir"]=args.model_path
    output=Path(args.output_dir)
    figdir=output/"figures_offline"; figdir.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",WANDB_MODE="offline")
    model,tokenizer=FastDecisionModel.from_pretrained(config["output_dir"],max_seq_length=2048,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,load_in_4bit=True,
        use_gradient_checkpointing="unsloth",local_files_only=True)
    rows={name:read_rows(config[f"{name}_data"]) for name in ("validation","test")}
    start=time.perf_counter(); pred={k:predict_rows(model,tokenizer,v) for k,v in rows.items()}; infer_seconds=time.perf_counter()-start
    labels={k:{task:[r["gold"][task]["choice"] for r in v] for task in ("phase","duration")} for k,v in rows.items()}
    temps={task:fit_temperature(pred["validation"],task,labels["validation"][task],PHASES if task=="phase" else DURATIONS) for task in ("phase","duration")}
    evaluated={}
    for task,classes in (("phase",PHASES),("duration",DURATIONS)):
        before,_,_=metrics(pred["test"],task,labels["test"][task],classes,1.0)
        after,probs,tops=metrics(pred["test"],task,labels["test"][task],classes,temps[task])
        evaluated[task]={"temperature_validation_only":temps[task],"before_calibration":before,"after_calibration":after}
        for i,row in enumerate(pred["test"]):
            row[f"calibrated_{task}_confidence"]=float(probs[i].max())
            row[f"predicted_{task}"]=classes[tops[i]]
    phase_pred=[r["predicted_phase"] for r in pred["test"]]
    duration_pred=[r["predicted_duration"] for r in pred["test"]]
    combined=Counter()
    for i in range(len(rows["test"])):
        p_ok=phase_pred[i]==labels["test"]["phase"][i]; d_ok=duration_pred[i]==labels["test"]["duration"][i]
        combined["both_exact" if p_ok and d_ok else "phase_only" if p_ok else "duration_only" if d_ok else "neither"]+=1
    per_intersection={}
    for iid in sorted({r["intersection_id"] for r in pred["test"]}):
        ix=[i for i,r in enumerate(pred["test"]) if r["intersection_id"]==iid]
        per_intersection[iid]={"count":len(ix),"phase_accuracy":float(np.mean([phase_pred[i]==labels["test"]["phase"][i] for i in ix])),
                               "duration_accuracy":float(np.mean([duration_pred[i]==labels["test"]["duration"][i] for i in ix]))}
    congestion={}
    for name,lo,hi in (("low_0_5",0,5),("medium_5_20",5,20),("high_20_plus",20,float("inf"))):
        ix=[i for i,r in enumerate(pred["test"]) if lo<=r["queue_total"]<hi]
        congestion[name]={"count":len(ix),"phase_accuracy":float(np.mean([phase_pred[i]==labels["test"]["phase"][i] for i in ix])) if ix else None,
                          "duration_accuracy":float(np.mean([duration_pred[i]==labels["test"]["duration"][i] for i in ix])) if ix else None}
    result={"status":"completed","model":"Traffic-Qwen-V1-Dual-Choice","checkpoint":config["output_dir"],
            "test_is_heldout_time_segment_same_episode":True,"calibration_split":"validation only",
            "rows":{"validation":len(rows["validation"]),"test":len(rows["test"])},"temperature_fit_seconds":infer_seconds,
            "tasks":evaluated,"combined_exact_outcomes":dict(combined),"per_intersection":per_intersection,"congestion_bins":congestion,
            "test_predictions":pred["test"]}
    (output/"offline_evaluation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    import matplotlib.pyplot as plt
    for task,classes in (("phase",PHASES),("duration",DURATIONS)):
        before=evaluated[task]["before_calibration"]; after=evaluated[task]["after_calibration"]
        cm=np.asarray(after["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(len(classes)),classes,rotation=45,ha="right"); ax.set_yticks(range(len(classes)),classes); ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figdir/f"{task}_confusion.png"); plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,5)); bins=after["calibration_bins"]; ax.plot([0,1],[0,1],"--",color="gray"); ax.plot([x["confidence"] for x in bins],[x["accuracy"] for x in bins],marker="o"); ax.set(xlim=(0,1),ylim=(0,1),xlabel="Confidence",ylabel="Accuracy",title=f"{task} calibrated reliability"); fig.tight_layout(); fig.savefig(figdir/f"{task}_calibration.png"); plt.close(fig)
    (output/"offline_evaluation.md").write_text(
        "# Traffic-Qwen V1 Dual-Choice offline evaluation\n\n"
        "Labels imitate guarded JevLight actions; this is not zero-shot or RL. Calibration temperatures were fitted on validation only. The held-out test is a chronological segment from the same episode, so it does not establish unseen-episode generalization.\n\n"
        f"Validation rows: {len(rows['validation'])}; test rows: {len(rows['test'])}.\n\n"
        f"Phase temperature: {temps['phase']:.4g}; duration temperature: {temps['duration']:.4g}.\n\n"
        f"Phase test: accuracy {evaluated['phase']['after_calibration']['accuracy']:.3f}, balanced accuracy {evaluated['phase']['after_calibration']['balanced_accuracy']:.3f}, macro-F1 {evaluated['phase']['after_calibration']['macro_f1']:.3f}, top-2 {evaluated['phase']['after_calibration']['top2_accuracy']:.3f}, ECE {evaluated['phase']['after_calibration']['ece']:.3f}, Brier {evaluated['phase']['after_calibration']['brier_score']:.3f}, NLL {evaluated['phase']['after_calibration']['log_loss']:.3f}.\n\n"
        f"Duration test: exact accuracy {evaluated['duration']['after_calibration']['accuracy']:.3f}, within 5s {evaluated['duration']['after_calibration']['accuracy_within_5_seconds']:.3f}, MAE {evaluated['duration']['after_calibration']['mae_seconds']:.2f}s, macro-F1 {evaluated['duration']['after_calibration']['macro_f1']:.3f}, ECE {evaluated['duration']['after_calibration']['ece']:.3f}, Brier {evaluated['duration']['after_calibration']['brier_score']:.3f}, NLL {evaluated['duration']['after_calibration']['log_loss']:.3f}.\n\n"
        f"Combined exact outcomes: `{dict(combined)}`. See JSON for before/after calibration, per-intersection and congestion metrics.\n",encoding="utf-8")
    print(json.dumps({"phase":evaluated["phase"]["after_calibration"],"duration":evaluated["duration"]["after_calibration"],"combined":dict(combined),"temperatures":temps},indent=2))

if __name__=="__main__": main()
