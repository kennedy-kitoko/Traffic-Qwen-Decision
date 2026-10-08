#!/usr/bin/env python3
"""Rebuild a comparison CSV from the preserved benchmark and measured Qwen summaries."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=json.loads((ROOT/"results/baselines/benchmark_jinan.json").read_text())
    qwen=[json.loads((ROOT/"results/runs/traffic_qwen_no_guardrail.json").read_text()),
          json.loads((ROOT/"results/runs/traffic_qwen_guardrail.json").read_text())]
    rows=[]
    for item in base["metrics"]["controllers"]:
        rows.append({"Controller":item["name"],"Avg queue":item.get("avg_queue"),
          "Avg waiting time":item.get("avg_waiting_time_seconds"),"Avg travel time":item.get("avg_travel_time_seconds"),
          "Decisions":item.get("model_decisions",item.get("api_decisions")),"Fallbacks":item.get("request_fallbacks"),
          "Guardrail corrections":item.get("guardrail_corrections"),"Runtime":item.get("runtime_seconds")})
    for item in qwen:
        rows.append({"Controller":item["controller"]+("-guardrail" if item["guardrail_enabled"] else ""),
          "Avg queue":item["avg_queue"],"Avg waiting time":item["avg_waiting_time_seconds"],
          "Avg travel time":item["avg_travel_time_seconds"],"Decisions":item["model_decisions"],
          "Fallbacks":item["request_fallbacks"],"Guardrail corrections":item["guardrail_corrections"],
          "Runtime":item["runtime_seconds"]})
    out=ROOT/"results/rebuilt_comparison.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} measured rows to {out}")
if __name__=="__main__": main()