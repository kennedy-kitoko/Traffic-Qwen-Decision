"""Create the offline report/figures from saved test predictions, with no model calls."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path("results/reports")
def main():
    result=json.loads((ROOT/"offline_evaluation.json").read_text()); figs=Path("results/figures/offline");figs.mkdir(exist_ok=True)
    for task in ("phase","duration"):
        entry=result["tasks"][task]; metrics=entry["after_calibration"]; classes=metrics["classes"]
        cm=np.asarray(metrics["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues");ax.set_xticks(range(len(classes)),classes,rotation=45,ha="right");ax.set_yticks(range(len(classes)),classes);ax.set_xlabel("Predicted");ax.set_ylabel("Gold");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figs/f"{task}_confusion.png");plt.close(fig)
        bins=metrics["calibration_bins"]
        fig,ax=plt.subplots(figsize=(5,5));ax.plot([0,1],[0,1],"--",color="gray",label="ideal");ax.plot([x["confidence"] for x in bins],[x["accuracy"] for x in bins],marker="o",label="calibrated");ax.set(xlim=(0,1),ylim=(0,1),xlabel="Confidence",ylabel="Accuracy",title=f"{task} reliability");ax.legend();fig.tight_layout();fig.savefig(figs/f"{task}_calibration.png");plt.close(fig)
    preds=result["test_predictions"]
    for field,filename,title in (("predicted_phase","predicted_phase_distribution.png","Predicted phase counts"),("predicted_duration","predicted_duration_distribution.png","Predicted duration counts")):
        vals=[p[field] for p in preds]; counts={x:vals.count(x) for x in dict.fromkeys(vals)}
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(counts),list(counts.values()));ax.set(title=title,ylabel="Test decisions");fig.tight_layout();fig.savefig(figs/filename);plt.close(fig)
    p=result["tasks"]["phase"];d=result["tasks"]["duration"]
    result_md=("# Traffic-Qwen V1 Dual-Choice offline evaluation\n\n"
      "Labels imitate guarded JevLight actions; this is not zero-shot or RL. Temperatures were fitted on validation only. The held-out test is a chronological segment from the same episode, so it does not establish unseen-episode generalization.\n\n"
      f"Validation rows: {result['rows']['validation']}; test rows: {result['rows']['test']}.\n\n"
      f"Phase temperature: {p['temperature_validation_only']:.4g} (validation NLL {p['before_calibration']['log_loss']:.4f} → {p['after_calibration']['log_loss']:.4f} test NLL).\n\n"
      f"Phase test: accuracy {p['after_calibration']['accuracy']:.3f}, balanced accuracy {p['after_calibration']['balanced_accuracy']:.3f}, macro-F1 {p['after_calibration']['macro_f1']:.3f}, top-2 {p['after_calibration']['top2_accuracy']:.3f}, ECE {p['after_calibration']['ece']:.3f}, Brier {p['after_calibration']['brier_score']:.3f}.\n\n"
      f"Duration temperature: {d['temperature_validation_only']:.4g}. Duration test: exact accuracy {d['after_calibration']['accuracy']:.3f}, within 5s {d['after_calibration']['accuracy_within_5_seconds']:.3f}, MAE {d['after_calibration']['mae_seconds']:.2f}s, macro-F1 {d['after_calibration']['macro_f1']:.3f}, ECE {d['after_calibration']['ece']:.3f}, Brier {d['after_calibration']['brier_score']:.3f}, NLL {d['after_calibration']['log_loss']:.3f}.\n\n"
      f"Combined exact outcomes: `{result['combined_exact_outcomes']}`. The duration accuracy is inflated by the strongly imbalanced label distribution; macro-F1 and per-class confusion must be read alongside it. See JSON for per-intersection and congestion metrics.\n")
    (ROOT/"offline_evaluation.md").write_text(result_md,encoding="utf-8")
    print(result_md)
if __name__=="__main__": main()
