import json
import random
import concurrent.futures
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import cfg
from src.utils import abs_path, print_info, print_good, print_warn, print_error, save_txt
from src.mining.llm import ChatSession, extract_code_block
from src.mining.rules import MinedRule

class BlindEvaluator:
    def __init__(self):
        self.conf = cfg.mining_blind_eval
        self.sys_prompt_path = abs_path("src/mining/prompts.txt")
        with open(self.sys_prompt_path, "r", encoding="utf-8") as f:
            self.sys_prompt = f.read()
            
        # Validation
        valid_algs = ["subgroup", "rulefit", "arm"]
        valid_targets = ["binary", "logits", "binned_proba"]
        
        for alg in self.conf["algorithms"]:
            if alg not in valid_algs:
                raise ValueError(f"Invalid algorithm in config: {alg}")
        for tgt in self.conf["targets"]:
            if tgt not in valid_targets:
                raise ValueError(f"Invalid target in config: {tgt}")
        for model in self.conf["models"]:
            if model not in cfg.llm["eval_models"]:
                raise ValueError(f"Model {model} not found in cfg.llm['eval_models']")

    def build_group_text(self, rules: List[MinedRule]) -> str:
        if not rules:
            return "No rules found."
        return "\n".join([f"- {str(r)}" for r in rules])

    def run_eval_task(self, alg: str, model: str, round_idx: int, groups_data: Dict[str, List[MinedRule]]) -> Dict[str, Any]:
        """
        Single task for LLM evaluation.
        """
        targets = list(groups_data.keys())
        rng = random.Random(cfg.seed + hash(alg) + hash(model) + round_idx)
        
        # Shuffle targets and map to A, B, C...
        shuffled_targets = list(targets)
        rng.shuffle(shuffled_targets)
        
        mapping = {chr(65 + i): tgt for i, tgt in enumerate(shuffled_targets)}
        reverse_mapping = {tgt: chr(65 + i) for i, tgt in enumerate(shuffled_targets)}
        
        user_prompt_lines = [
            f"Please evaluate the following {len(targets)} groups of rules mined using different target definitions on the same dataset.",
            "Each group represents rules for the SAME underlying task but with different labels or target transformations.",
            ""
        ]
        
        for label, tgt in mapping.items():
            user_prompt_lines.append(f"### Group {label}")
            user_prompt_lines.append(self.build_group_text(groups_data[tgt]))
            user_prompt_lines.append("")
            
        user_prompt = "\n".join(user_prompt_lines)
        
        try:
            session = ChatSession(model=model, mode="one-shot", sys_prompt=self.sys_prompt)
            response_raw = session.ask(user_prompt)
            json_str = extract_code_block(response_raw)
            
            # Basic cleanup of JSON if it has trailing/leading text outside code blocks
            # (though extract_code_block should handle typical cases)
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
                
            result = json.loads(json_str)
            
            # Validate result
            scores = result.get("scores", {})
            reasons = result.get("brief_reason", {})
            
            final_scores = {}
            final_reasons = {}
            for label, tgt in mapping.items():
                s = scores.get(label)
                if not isinstance(s, (int, float)):
                    raise ValueError(f"Score for {label} is not a number: {s}")
                final_scores[tgt] = float(s)
                final_reasons[tgt] = reasons.get(label, "No reason provided.")
                
            return {
                "algorithm": alg,
                "model": model,
                "round": round_idx,
                "scores": final_scores,
                "reasons": final_reasons,
                "status": "success"
            }
        except Exception as e:
            print_error(f"Error in LLM task ({alg}, {model}, round {round_idx}): {e}")
            return {
                "algorithm": alg,
                "model": model,
                "round": round_idx,
                "error": str(e),
                "status": "error"
            }

    def evaluate(self, mined_data: Dict[str, Dict[str, List[MinedRule]]]) -> str:
        """
        Runs the full evaluation.
        mined_data[alg][target] = List[MinedRule]
        """
        tasks = []
        for alg in self.conf["algorithms"]:
            if alg not in mined_data:
                print_warn(f"Algorithm {alg} not found in mined data, skipping.")
                continue
            
            groups_data = {tgt: mined_data[alg][tgt] for tgt in self.conf["targets"] if tgt in mined_data[alg]}
            if not groups_data:
                continue
                
            for model in self.conf["models"]:
                for r in range(self.conf["rounds_per_model"]):
                    tasks.append((alg, model, r, groups_data))
        
        print_info(f"Starting {len(tasks)} parallel LLM evaluation tasks...")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tasks) if tasks else 1)) as executor:
            future_to_task = {executor.submit(self.run_eval_task, *task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                results.append(future.result())
                
        return self.generate_report(results)

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        successes = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        
        report = []
        report.append(f"algorithms: {json.dumps(self.conf['algorithms'])}")
        report.append(f"targets: {json.dumps(self.conf['targets'])}")
        report.append(f"models: {json.dumps(self.conf['models'])}")
        report.append(f"rounds_per_model: {self.conf['rounds_per_model']}")
        report.append("\n" + "="*70)
        
        # Aggregate Overall
        rows = []
        for r in successes:
            for tgt, score in r["scores"].items():
                rows.append({
                    "algorithm": r["algorithm"],
                    "target": tgt,
                    "model": r["model"],
                    "score": score
                })
        
        df = pd.DataFrame(rows)
        
        report.append("\nOverall (aggregated across models and rounds):")
        if not df.empty:
            overall = df.groupby(["algorithm", "target"])["score"].agg(["mean", "std", "count"]).reset_index()
            report.append(overall.to_string(index=False))
        else:
            report.append("No successful evaluations.")
            
        report.append("\n" + "="*70)
        report.append("\nBy model:")
        if not df.empty:
            by_model = df.groupby(["model", "algorithm", "target"])["score"].agg(["mean", "std", "count"]).reset_index()
            report.append(by_model.to_string(index=False))
        else:
            report.append("No successful evaluations.")
            
        if errors:
            report.append("\n" + "="*70)
            report.append("\nErrors:")
            for e in errors:
                report.append(f"  - [{e['algorithm']} | {e['model']} | Round {e['round']}]: {e['error']}")
                
        final_report = "\n".join(report)
        out_path = abs_path(cfg.paths["output"]) / "mining" / "llm_blind_eval_report.txt"
        save_txt(final_report, out_path)
        print_good(f"LLM Blind Evaluation report saved to {out_path}")
        return final_report