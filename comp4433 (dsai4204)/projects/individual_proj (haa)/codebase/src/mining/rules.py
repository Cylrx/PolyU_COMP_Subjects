from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class MinedRule:
    text: str
    score: float
    label: Optional[Union[str, int]] = None

    def __str__(self):
        return f"{self.text} => [{self.label}]"

def stratified_top_k(
    rules: List[MinedRule], 
    top_k: int, 
    label_order: List[Union[str, int]]
) -> List[MinedRule]:
    """
    Selects top_k rules using stratified sampling based on labels.
    If rules cannot be evenly divided, the highest score rules among remaining candidates are picked.
    """
    if not rules:
        return []
    
    # Group rules by label
    by_label: Dict[Any, List[MinedRule]] = {}
    for r in rules:
        by_label.setdefault(r.label, []).append(r)
    
    # Sort each group by score
    for lbl in by_label:
        by_label[lbl].sort(key=lambda x: x.score, reverse=True)
    
    selected = []
    num_labels = len(label_order)
    per_label = top_k // num_labels if num_labels > 0 else 0
    
    # 1. Take per_label from each group
    for lbl in label_order:
        if lbl in by_label:
            group = by_label[lbl]
            take = min(len(group), per_label)
            selected.extend(group[:take])
            by_label[lbl] = group[take:] # Update remaining
    
    # 2. Fill remaining from candidates sorted by score
    remaining_candidates = []
    for lbl in by_label:
        remaining_candidates.extend(by_label[lbl])
    
    remaining_candidates.sort(key=lambda x: x.score, reverse=True)
    needed = top_k - len(selected)
    if needed > 0:
        selected.extend(remaining_candidates[:needed])
        
    return selected

def from_subgroup(df: pd.DataFrame, target: str, top_k: int, bin_names: List[str]) -> List[MinedRule]:
    """
    Converts subgroup discovery results to MinedRules.
    """
    rules = []
    if df.empty:
        return []
    
    for _, row in df.iterrows():
        desc = str(row["description"])
        # pysubgroup format is often 'attr1==val1 AND attr2==val2'
        # we keep it as is.
        score = row["quality"]
        
        label = None
        if target == "binned_proba":
            label = row["class"] # 'cat' dtype in run.py adds 'class'
        elif target == "binary":
            val = row["class"]
            label = "mid to high" if val == 1 else "low to mid"
        elif target == "logits":
            label = "higher risk score"
            
        rules.append(MinedRule(text=desc, score=score, label=label))
    
    # Since run.py might have already done top_k per class, we re-apply stratification for consistency
    label_order = []
    if target == "binned_proba":
        label_order = bin_names
    elif target == "binary":
        label_order = ["low to mid", "mid to high"]
    else:
        label_order = ["higher risk score"]
        
    return stratified_top_k(rules, top_k, label_order)

def from_arm(df: pd.DataFrame, target: str, top_k: int, bin_names: List[str], metric: str = "lift") -> List[MinedRule]:
    """
    Converts association rule mining results to MinedRules.
    """
    rules = []
    if df.empty:
        return []
    
    # Targeted rules in run.py have 'antecedents' and 'consequents'
    # Format: antecedents => [label]
    target_col = "output" # Hardcoded in dataset cfg usually, but we check consequents
    
    for _, row in df.iterrows():
        ante = str(row["antecedents"]).replace(", ", " AND ")
        cons = str(row["consequents"])
        score = row[metric]
        
        label = None
        if target == "binned_proba":
            # cons looks like 'output=high'
            for b in bin_names:
                if f"={b}" in cons:
                    label = b
                    break
        elif target == "binary":
            if "=1" in cons:
                label = "mid to high"
            elif "=0" in cons:
                label = "low to mid"
        elif target == "logits":
            label = "higher risk score"
            
        if label:
            rules.append(MinedRule(text=f"[{ante}]", score=score, label=label))
            
    label_order = bin_names if target == "binned_proba" else (["low to mid", "mid to high"] if target == "binary" else ["higher risk score"])
    return stratified_top_k(rules, top_k, label_order)

def from_rulefit(df: pd.DataFrame, target: str, top_k: int, bin_names: List[str]) -> List[MinedRule]:
    """
    Converts RuleFit results to MinedRules.
    """
    rules = []
    if df.empty:
        return []
        
    for _, row in df.iterrows():
        desc = str(row["rule"])
        score = row["importance"]
        coef = row["coef"]
        
        label = None
        if target == "binned_proba":
            label = row["class"]
        elif target == "binary":
            # For binary in run.py, if dtype=='bin', it returns one DF.
            # If dtype=='cat', it returns concatenated.
            if "class" in df.columns:
                val = row["class"]
                label = "mid to high" if val == 1 else "low to mid"
            else:
                # Infer from coef if single binary fit
                label = "mid to high" if coef > 0 else "low to mid"
        elif target == "logits":
            label = "higher risk score"
            
        rules.append(MinedRule(text=f"[{desc}]", score=score, label=label))
        
    label_order = bin_names if target == "binned_proba" else (["low to mid", "mid to high"] if target == "binary" else ["higher risk score"])
    return stratified_top_k(rules, top_k, label_order)

