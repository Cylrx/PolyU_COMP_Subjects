from src.config import cfg
from src.statistics import stats
from src.utils import abs_path, print_error, save_txt, print_good, ensure_dir, print_info

from typing import List, Dict, Any, Optional
import numpy as np
import pysubgroup as ps
import pandas as pd
from pathlib import Path

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from imodels import RuleFitClassifier, RuleFitRegressor
from src.mining.rules import from_subgroup, from_arm, from_rulefit
from src.mining.blind_eval import BlindEvaluator

class ReportBuffer:
    def __init__(self, title: str, config_summary: Dict[str, Any]):
        self.lines = [f"{'='*30} {title} {'='*30}", ""]
        self.lines.append("Configuration:")
        for k, v in config_summary.items():
            self.lines.append(f"  {k}: {v}")
        self.lines.append("\n" + "="*70 + "\n")
    
    def add_section(self, section_name: str):
        self.lines.append(f"\n{'#'*10} Target: {section_name} {'#'*10}")
    
    def add_block(self, title: str, content: str):
        self.lines.append(f"\n[{title}]")
        self.lines.append(content)
            
    def render(self) -> str:
        return "\n".join(self.lines)

def subgroup_discovery(data: pd.DataFrame, dtype: str, target_key: str) -> pd.DataFrame:
    """
    Executes subgroup discovery using pysubgroup.
    Returns a combined dataframe of results for all categories if cat or bin, else a single df.
    """
    target_col = cfg.dataset["label"]
    conf = cfg.subgroup_discovery
    
    search_space = ps.create_selectors(data, ignore=[target_col])

    def execute_sd(ps_target, qf) -> pd.DataFrame:
        task = ps.SubgroupDiscoveryTask(
            data, 
            ps_target, 
            search_space, 
            result_set_size=conf["top_k"] * 2, # Take more to allow for stratification/filtering
            depth=conf["depth"], 
            qf=qf
        )
        result = ps.DFS().execute(task)
        df = result.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["quality", "description"])
        
        if "description" not in df.columns and "subgroup" in df.columns:
            df["description"] = df["subgroup"].astype(str)
        
        cols = ["quality", "description"]
        return df[[c for c in cols if c in df.columns]]

    if dtype == "bin":
        # For binary, we want both classes (0 and 1) for the blind eval/fairness
        results = []
        for val in [0, 1]:
            res = execute_sd(ps.BinaryTarget(target_col, val), ps.WRAccQF())
            res.insert(0, "class", val)
            results.append(res)
        return pd.concat(results) if results else pd.DataFrame()
    elif dtype == "num":
        return execute_sd(ps.NumericTarget(target_col), ps.StandardQFNumeric(a=1.0))
    elif dtype == "cat":
        results = []
        categories = sorted(data[target_col].unique().tolist()) 
        for cat in categories:
            res = execute_sd(ps.BinaryTarget(target_col, cat), ps.WRAccQF())
            res.insert(0, "class", cat)
            results.append(res)
        return pd.concat(results) if results else pd.DataFrame()
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def association_rule_mining(data: pd.DataFrame, dtype: str, target_key: str) -> pd.DataFrame:
    """
    Implements targeted association rule mining.
    """
    conf = cfg.association_rules
    target_col = cfg.dataset["label"]
    features_cfg = cfg.dataset["features"]
    
    df_mining = data.copy()
    
    # 1. Discretize features
    transactions_df = pd.DataFrame()
    for col, ftype in features_cfg.items():
        if ftype == "cat":
            transactions_df[col] = df_mining[col].apply(lambda x: f"{col}={x}")
        else:
            try:
                bins = pd.qcut(df_mining[col], q=conf.get("num_bins", 5), duplicates="drop")
                transactions_df[col] = bins.apply(lambda x: f"{col}={x}")
            except Exception:
                bins = pd.cut(df_mining[col], bins=conf.get("num_bins", 5))
                transactions_df[col] = bins.apply(lambda x: f"{col}={x}")

    # 2. Target item
    if dtype == "num":
        n_bins = cfg.mining_targets["num_bins"]
        bin_names = cfg.mining_targets["bin_names"]
        target_series = pd.cut(df_mining[target_col], bins=n_bins, labels=bin_names)
        transactions_df[target_col] = target_series.apply(lambda x: f"{target_col}={x}")
    else:
        transactions_df[target_col] = df_mining[target_col].apply(lambda x: f"{target_col}={x}")

    # 3. Transaction Encoding
    records = transactions_df.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # 4. Mine Rules
    frequent_itemsets = apriori(df_onehot, min_support=conf["min_support"], use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=conf["min_threshold"])
    if rules.empty:
        return pd.DataFrame()

    # 5. Targeted Filtering
    rules["is_target"] = rules["consequents"].apply(lambda x: any(str(i).startswith(f"{target_col}=") for i in x))
    targeted_rules = rules[rules["is_target"]].copy()
    
    if targeted_rules.empty:
        return pd.DataFrame()

    # Sort and take more than top_k for stratification
    targeted_rules = targeted_rules.sort_values(by=conf["metric"], ascending=False).head(conf["top_k"] * 2)
    
    targeted_rules["antecedents"] = targeted_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    targeted_rules["consequents"] = targeted_rules["consequents"].apply(lambda x: ", ".join(list(x)))
    
    cols = ["support", "confidence", "lift", "antecedents", "consequents"]
    return targeted_rules[cols]

def tree_based_rule_mining(data: pd.DataFrame, dtype: str, target_key: str) -> pd.DataFrame:
    """
    Implements RuleFit mining.
    """
    conf = cfg.rulefit
    target_col = cfg.dataset["label"]
    features_cfg = cfg.dataset["features"]
    
    X_raw = data.drop(columns=[target_col])
    y = data[target_col]
    
    X = pd.get_dummies(X_raw, columns=[c for c, t in features_cfg.items() if t == "cat"], drop_first=True)
    feature_names = X.columns.tolist()
    
    def fit_and_get_rules(X_fit, y_fit, is_classification: bool) -> pd.DataFrame:
        model = RuleFitClassifier(
            tree_size=conf["tree_size"], 
            max_rules=conf["max_rules"],
            random_state=cfg.seed
        ) if is_classification else RuleFitRegressor(
            tree_size=conf["tree_size"], 
            max_rules=conf["max_rules"],
            random_state=cfg.seed
        )
        
        model.fit(X_fit.values, y_fit.values, feature_names=feature_names)
        rules_df = model._get_rules()
        rules_df = rules_df[rules_df["coef"] != 0].copy()
        
        if not conf.get("include_linear", False):
            rules_df = rules_df[rules_df["type"] == "rule"]
            
        rules_df = rules_df.sort_values(by="importance", ascending=False).head(conf["top_k"] * 2)
        return rules_df[["importance", "coef", "support", "rule"]]

    if dtype == "bin":
        # If binary, we can either fit once or fit for both classes.
        # RuleFitRegressor for logits (num) or RuleFitClassifier for binary.
        # For blind eval fairness, if target is binary, we might want to fit for 0 and 1 separately
        # if we want to ensure class balance in rules.
        results = []
        for val in [0, 1]:
            y_bin = (y == val).astype(int)
            res = fit_and_get_rules(X, y_bin, is_classification=True)
            res.insert(0, "class", val)
            results.append(res)
        return pd.concat(results) if results else pd.DataFrame()
    elif dtype == "num":
        return fit_and_get_rules(X, y, is_classification=False)
    elif dtype == "cat":
        results = []
        categories = sorted(y.unique().tolist())
        for cat in categories:
            y_bin = (y == cat).astype(int)
            res = fit_and_get_rules(X, y_bin, is_classification=True)
            res.insert(0, "class", cat)
            results.append(res)
        return pd.concat(results) if results else pd.DataFrame()
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def target_transform(data: pd.DataFrame, proba: pd.Series, target: str):
    logits = np.log(proba) - np.log1p(-proba)
    target_col = cfg.dataset["label"]
    
    df_new = data.copy()
    match target:
        case "binary": return df_new, "bin"
        case "logits": 
            df_new[target_col] = logits
            return df_new, "num"
        case "binned_proba": 
            n_bins = cfg.mining_targets["num_bins"]
            bin_names = cfg.mining_targets["bin_names"]
            df_new[target_col] = pd.cut(logits, bins=n_bins, labels=bin_names)
            return df_new, "cat"
        case _: raise ValueError(f"Target {target} not supported")

def main(): 
    data_path = abs_path(cfg.paths['data'])
    proba_path = abs_path(cfg.paths['proba'])
    
    data = pd.read_csv(data_path)
    proba_df = pd.read_csv(proba_path)
    p_y1_mean = proba_df["p_y1_mean"]
    
    targets: List[str] = cfg.mining_targets["targets"]
    stats.init(data)

    sd_report = ReportBuffer("Subgroup Discovery", cfg.subgroup_discovery)
    ar_report = ReportBuffer("Association Rule Mining", cfg.association_rules)
    rf_report = ReportBuffer("RuleFit", cfg.rulefit)

    # Storage for blind evaluation
    mined_data: Dict[str, Dict[str, List[Any]]] = {
        "subgroup": {},
        "arm": {},
        "rulefit": {}
    }

    bin_names = cfg.mining_targets["bin_names"]
    top_k = cfg.subgroup_discovery["top_k"] # Assume same top_k for all for fairness

    for target_key in targets: 
        print(f"\nProcessing target: {target_key}...")
        curr_data, dtype = target_transform(data, p_y1_mean, target_key)

        # Subgroup Discovery
        sd_res = subgroup_discovery(curr_data, dtype, target_key)
        sd_rules = from_subgroup(sd_res, target_key, top_k, bin_names)
        mined_data["subgroup"][target_key] = sd_rules
        sd_report.add_section(target_key)
        sd_report.add_block("Results", "\n".join([str(r) for r in sd_rules]) if sd_rules else "No subgroups found.")

        # Association Rules
        ar_res = association_rule_mining(curr_data, dtype, target_key)
        ar_rules = from_arm(ar_res, target_key, top_k, bin_names)
        mined_data["arm"][target_key] = ar_rules
        ar_report.add_section(target_key)
        ar_report.add_block("Results", "\n".join([str(r) for r in ar_rules]) if ar_rules else "No rules found.")

        # RuleFit
        rf_res = tree_based_rule_mining(curr_data, dtype, target_key)
        rf_rules = from_rulefit(rf_res, target_key, top_k, bin_names)
        mined_data["rulefit"][target_key] = rf_rules
        rf_report.add_section(target_key)
        rf_report.add_block("Results", "\n".join([str(r) for r in rf_rules]) if rf_rules else "No rules found.")

    # Save Mining Reports
    out_dir = abs_path(cfg.paths["output"]) / "mining"
    ensure_dir(out_dir)
    save_txt(sd_report.render(), out_dir / "pysubgroup_report.txt")
    save_txt(ar_report.render(), out_dir / "association_rules_report.txt")
    save_txt(rf_report.render(), out_dir / "rulefit_report.txt")
    
    # Run LLM Blind Evaluation
    print_info("Starting LLM Blind Evaluation...")
    evaluator = BlindEvaluator()
    evaluator.evaluate(mined_data)
    
    print_good(f"All mining reports and blind evaluation saved to {out_dir}")

if __name__ == "__main__":
    main()
