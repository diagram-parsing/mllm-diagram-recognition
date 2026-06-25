import os
import json
import math
import re
from collections import Counter

import jellyfish
import numpy as np
from scipy.optimize import linear_sum_assignment

NO_MATCH_COST = 1_000_000.0
NODE_METRICS = ("node_name", "node_name_class")
EDGE_METRICS = (
    "edge_endpoint",
    "edge_endpoint_class",
    "edge_endpoint_class_label",
)
METRIC_DISPLAY_NAMES = {
    "node_name": "Node F1 (name)",
    "node_name_class": "Node F1 (name+class)",
    "edge_endpoint": "Edge F1 (endpoints)",
    "edge_endpoint_class": "Edge F1 (endpoints+class)",
    "edge_endpoint_class_label": "Edge F1 (endpoints+class+label)",
}


# TODO: Compare to old normalization procedure

#  old code: "null", "None", "none" as strings, new code only catches Python None
_NULL_STRINGS = {"null", "none", "n/a", "na", ""}
def normalize_string(s):
    if s is None:
        return ""
    cleaned = re.sub(r"[_\s]+", "", str(s).strip()).lower()
    if cleaned in _NULL_STRINGS:
        return ""
    return cleaned


_THINK_END_RE = re.compile(r"</think>|<\|end_of_thinking\|>|</thinking>", re.IGNORECASE)

def strip_thinking_text(content):
    """Remove model reasoning that precedes the actual answer.

    Handles both a full <think>...</think> block and the common case where
    only a closing token is emitted: everything up to and including the last
    think-end token is discarded.
    """
    # Prefer removing a well-formed block first
    block = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    if block != content:
        content = block

    # Fallback: drop everything up to and including the final end-think token
    matches = list(_THINK_END_RE.finditer(content))
    if matches:
        content = content[matches[-1].end():]

    return content



def node_text(node):
    return normalize_string(node.get("text"))


def node_class(node):
    return normalize_string(node.get("class"))


def edge_class(edge):
    return normalize_string(edge.get("class"))


def edge_label(edge):
    return normalize_string(edge.get("label", edge.get("text")))


def build_id_to_text(nodes):
    id2text = {}
    for n in nodes:
        id2text[n.get("id")] = node_text(n)
    return id2text


def normalized_levenshtein(a, b):
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    dist = jellyfish.levenshtein_distance(a, b)
    return dist / max_len


def node_cost(gt_node, pred_node, gt_id2text, pred_id2text):
    gt_text = gt_id2text.get(gt_node.get("id"), "")
    pr_text = pred_id2text.get(pred_node.get("id"), "")

    if (not gt_text or not pr_text) and node_class(gt_node) != node_class(pred_node):
        return NO_MATCH_COST

    return normalized_levenshtein(gt_text, pr_text)


def node_cost_with_class(gt_node, pred_node, gt_id2text, pred_id2text):
    if node_class(gt_node) != node_class(pred_node):
        return NO_MATCH_COST
    return node_cost(gt_node, pred_node, gt_id2text, pred_id2text)


def match_hungarian(
    gt_items,
    pred_items,
    cost_fn,
    threshold,
    gt_id2text,
    pred_id2text,
    return_pairs=False
):
    if len(gt_items) == 0 or len(pred_items) == 0:
        result = (0, len(gt_items), len(pred_items))
        return (*result, []) if return_pairs else result

    cost_matrix = np.zeros((len(gt_items), len(pred_items)))

    for i, gt in enumerate(gt_items):
        for j, pr in enumerate(pred_items):
            cost_matrix[i, j] = cost_fn(gt, pr, gt_id2text, pred_id2text)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = 0
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches += 1
            pairs.append((gt_items[r], pred_items[c]))

    result = (matches, len(gt_items), len(pred_items))
    return (*result, pairs) if return_pairs else result


def build_pred_to_gt_node_map(node_pairs):
    pred_to_gt = {}
    for gt_node, pred_node in node_pairs:
        gt_id = gt_node.get("id")
        pred_id = pred_node.get("id")
        if gt_id is not None and pred_id is not None:
            pred_to_gt[pred_id] = gt_id
    return pred_to_gt


def edge_signature(edge, pred_to_gt=None, require_class=False, require_label=False):
    tail = edge.get("tail")
    head = edge.get("head")

    if pred_to_gt is not None:
        tail = pred_to_gt.get(tail)
        head = pred_to_gt.get(head)
        if tail is None or head is None:
            return None

    signature = [tail, head]
    if require_class:
        signature.append(edge_class(edge))
    if require_label:
        signature.append(edge_label(edge))
    return tuple(signature)


def count_edge_matches(gt_edges, pred_edges, pred_to_gt, require_class=False, require_label=False):
    gt_counter = Counter(
        edge_signature(e, require_class=require_class, require_label=require_label)
        for e in gt_edges
    )
    pred_counter = Counter(
        sig
        for sig in (
            edge_signature(
                e,
                pred_to_gt=pred_to_gt,
                require_class=require_class,
                require_label=require_label
            )
            for e in pred_edges
        )
        if sig is not None
    )

    matches = sum(
        min(count, pred_counter.get(signature, 0))
        for signature, count in gt_counter.items()
    )
    return matches, len(gt_edges), len(pred_edges)

# matching logic for when ground truth nodes are given to the model and we can directly match based on edge texts
def edge_signature_text_based(edge, require_class=False, require_label=False):
    tail = normalize_string(edge.get("tail", ""))
    head = normalize_string(edge.get("head", ""))

    signature = [tail, head]
    if require_class:
        signature.append(edge_class(edge))
    if require_label:
        signature.append(edge_label(edge))
    return tuple(signature)


def count_edge_matches_text_based(gt_edges, pred_edges, require_class=False, require_label=False):
    gt_counter = Counter(
        edge_signature_text_based(e, require_class=require_class, require_label=require_label)
        for e in gt_edges
    )
    pred_counter = Counter(
        edge_signature_text_based(e, require_class=require_class, require_label=require_label)
        for e in pred_edges
    )
    matches = sum(
        min(count, pred_counter.get(sig, 0))
        for sig, count in gt_counter.items()
    )
    return matches, len(gt_edges), len(pred_edges)

def build_nx_graph(data):
    import networkx as nx

    graph = nx.DiGraph()
    for index, node in enumerate(data.get("nodes", [])):
        node_id = node.get("id", f"__missing_node_id_{index}")
        graph.add_node(node_id, text=node_text(node), class_=node_class(node))

    for index, edge in enumerate(data.get("relations", [])):
        tail = edge.get("tail", f"__missing_tail_{index}")
        head = edge.get("head", f"__missing_head_{index}")
        graph.add_edge(tail, head, class_=edge_class(edge), label=edge_label(edge))

    return graph


def compute_graph_edit_distance(gt_data, pred_data, timeout=1.0):
    import networkx as nx

    gt_graph = build_nx_graph(gt_data)
    pred_graph = build_nx_graph(pred_data)

    def node_subst_cost(attrs_1, attrs_2):
        text_cost = normalized_levenshtein(attrs_1.get("text", ""), attrs_2.get("text", ""))
        class_cost = 0.0 if attrs_1.get("class_") == attrs_2.get("class_") else 1.0
        return text_cost + class_cost

    def edge_subst_cost(attrs_1, attrs_2):
        class_cost = 0.0 if attrs_1.get("class_") == attrs_2.get("class_") else 1.0
        label_cost = 0.0 if attrs_1.get("label") == attrs_2.get("label") else 1.0
        return class_cost + label_cost

    return nx.graph_edit_distance(
        gt_graph,
        pred_graph,
        node_subst_cost=node_subst_cost,
        node_del_cost=lambda attrs: 1.0,
        node_ins_cost=lambda attrs: 1.0,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=lambda attrs: 1.0,
        edge_ins_cost=lambda attrs: 1.0,
        timeout=timeout,
    )


def compute_f1(matches, total_gt, total_pred):
    precision = matches / total_pred if total_pred > 0 else 0.0
    recall = matches / total_gt if total_gt > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def compute_mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def run_sort_key(run_name):
    match = re.fullmatch(r"run(\d+)", run_name)
    if match:
        return (0, int(match.group(1)))
    return (1, run_name)


def initialize_run_total():
    stats = {
        "valid_extractions": 0,
        "missing_predictions": 0,
        "graph_edit_distance_sum": 0.0,
        "graph_edit_distance_count": 0,
        "graph_edit_distance_timeouts": 0,
    }
    for metric in NODE_METRICS + EDGE_METRICS:
        stats[f"{metric}_matches"] = 0
        stats[f"{metric}_gt"] = 0
        stats[f"{metric}_pred"] = 0
    return stats


def add_metric_counts(run_totals, run_name, metric_counts):
    if run_name not in run_totals:
        run_totals[run_name] = initialize_run_total()

    for metric, (matches, total_gt, total_pred) in metric_counts.items():
        run_totals[run_name][f"{metric}_matches"] += matches
        run_totals[run_name][f"{metric}_gt"] += total_gt
        run_totals[run_name][f"{metric}_pred"] += total_pred


def clean_filename(name):
    name = os.path.basename(name)
    name = re.sub(r"_run\d+", "", name)
    name = re.sub(r"_temp=.*", "", name)
    name = re.sub(r"\.txt$", "", name)
    name = re.sub(r"\.json$", "", name)
    return name


def load_json_or_txt(path, strip_thinking=False):
    with open(path, "r") as f:
        content = f.read()

    if strip_thinking:
        content = strip_thinking_text(content)

    if "```" in content:
        match = re.search(r"```json(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1)

    return json.loads(content)


def extract_run_from_filename(filename):
    """
    Extract run name from filename if present.
    E.g. "abc_run1_temp=0.1.json" -> "run1_temp=0.1"
    """
    base = os.path.basename(filename)
    match = re.search(r'(_run\d+(?:_temp=[\d.]+)?)', base)
    if match:
        return match.group(1)[1:]  # remove leading underscore
    return "default"


def extract_runs(data, filename):
    runs = []
    run_from_filename = extract_run_from_filename(filename)

    # Case 2: Single graph dict
    if isinstance(data, dict) and "nodes" in data and "relations" in data:

        if run_from_filename != "default":
            runs.append((run_from_filename, data))
            return runs

        runs.append(("default", data))
        return runs

    # Relations Only
    if isinstance(data, dict) and "relations" in data and "nodes" not in data:
      runs.append((run_from_filename, data))
      return runs

    # Case 3: Dict of runs
    if isinstance(data, dict):
      for key, val in data.items():
          if isinstance(val, dict) and ("nodes" in val or "relations" in val):
              runs.append((key, val))
      if runs:  # only return if we actually found something
          return runs

    # Case 3: List of graph dicts
    if isinstance(data, list) and "nodes" in data and "relations" in data:
        if run_from_filename != "default":
            runs.append((run_from_filename, data))
            return runs
        runs.append(("default", data))
        return runs

    # Case 4: List of dicts
    if isinstance(data, list):
        if isinstance(data[0], dict) and "nodes" in data[0] and "relations" in data[0]:
            run_name = run_from_filename if run_from_filename != "default" else "default"
            runs.append((run_name, data[0]))
            return runs
        elif isinstance(data[0], dict):
            for i, val in enumerate(data):
                if isinstance(val, dict) and "nodes" in val and "relations" in val:
                    runs.append((i, val))
            return runs

    return runs


def evaluate(
    model_path,
    dataset=None,
    label_dir=None,
    threshold=0.2,
    output_csv=None,
    compute_ged=True,
    ged_timeout=1.0,
    expected_runs=None,
    strip_thinking=False
):
    results = []
    run_totals = {}
    evaluated_pairs = set()
    gt_data_by_file = {}
    observed_run_names = set()

    model_files = []
    ged_available = False

    if compute_ged:
        try:
            import networkx
            ged_available = True
        except ImportError:
            print("[INFO] Graph edit distance skipped: install `networkx` to enable it.")


    if dataset is not None:
        label_dir = os.path.join("labels", dataset, "test")
        print(f"Using label directory: {label_dir}")
    elif label_dir is None:
        raise ValueError("You must provide either `dataset` or `label_dir`")

    label_files = sorted(
        f for f in os.listdir(label_dir)
        if os.path.isfile(os.path.join(label_dir, f))
    )
    total_label_files = len(label_files)
    print(f'Total label files: {total_label_files}')
    if os.path.isdir(model_path):
        print("Loading Extractions from Directory...")
        for f in os.listdir(model_path):
            if f.endswith(".json") or f.endswith(".txt"):
                model_files.append(os.path.join(model_path, f))
    else:
        print("Loading Extractions from aggregated Json-File")
        model_files = [model_path]


    count_1 = 0
    count_2 = 0
    count_3 = 0

    for mf in model_files:
        try:
            data = load_json_or_txt(mf, strip_thinking)
        except Exception as e:
            print(f"[INFO] Failed reading the following file {mf}: {e}")
            continue

        if isinstance(data, dict) and not ("nodes" in data):
            #if count_1 == 0:
            #    print("Found run key in JSON file.")
            items = data.items()
        else:
            #if count_1 == 0:
                #print("Nodes key at top level detected.")
            items = [(os.path.basename(mf), data)]

        count_1 += 1 # simple counter for print debugs

        for filename, file_data in items:
            base_name = clean_filename(filename) + ".json"
            label_path = os.path.join(label_dir, base_name)

            # How many are skipped?
            if not os.path.exists(label_path):
                count_3 += 1
                continue

            try:
                with open(label_path, "r") as f:
                    gt_data = json.load(f)
            except Exception as e:
                print(f"Error loading label file {label_path}: {e}")
                continue

            gt_nodes = gt_data.get("nodes", [])
            gt_edges = gt_data.get("relations", [])
            gt_data_by_file[base_name] = gt_data

            runs = extract_runs(file_data, filename)
            for run_name, run_data in runs:
                if count_2 == 0:
                    print(f'Evaluation run_name: {run_name} (default if file had no run)')
                observed_run_names.add(run_name)
                evaluated_pairs.add((base_name, run_name))
                pred_nodes = run_data.get("nodes", [])
                pred_edges = run_data.get("relations", [])

                gt_id2text = build_id_to_text(gt_nodes)
                pr_id2text = build_id_to_text(pred_nodes)


                # logic for handling relations only output
                nodes_provided = len(pred_nodes) > 0

                if nodes_provided:
                  nm, ngt, npred, node_pairs = match_hungarian(
                      gt_nodes, pred_nodes,
                      node_cost,
                      threshold,
                      gt_id2text,
                      pr_id2text,
                      return_pairs=True
                  )
                  nm_class, _, _ = match_hungarian(
                      gt_nodes,
                      pred_nodes,
                      node_cost_with_class,
                      threshold,
                      gt_id2text,
                      pr_id2text
                  )
                  pred_to_gt_nodes = build_pred_to_gt_node_map(node_pairs)

                  em_endpoint, egt, epred = count_edge_matches(
                      gt_edges,
                      pred_edges,
                      pred_to_gt_nodes
                  )
                  em_class, _, _ = count_edge_matches(
                      gt_edges,
                      pred_edges,
                      pred_to_gt_nodes,
                      require_class=True
                  )
                  em_label, _, _ = count_edge_matches(
                      gt_edges,
                      pred_edges,
                      pred_to_gt_nodes,
                      require_class=True,
                      require_label=True
                  )

                else:
                  # Ablation: GT nodes were provided to the model, no node prediction needed.
                  ngt = len(gt_nodes)
                  nm, npred, nm_class = ngt, ngt, ngt

                  em_endpoint, egt, epred = count_edge_matches_text_based(gt_edges, pred_edges)
                  em_class, _, _ = count_edge_matches_text_based(gt_edges, pred_edges, require_class=True)
                  em_label, _, _ = count_edge_matches_text_based(gt_edges, pred_edges, require_class=True, require_label=True)

                n_prec, n_rec, n_f1 = compute_f1(nm, ngt, npred)
                n_class_prec, n_class_rec, n_class_f1 = compute_f1(nm_class, ngt, npred)
                e_prec, e_rec, e_f1 = compute_f1(em_endpoint, egt, epred)
                e_class_prec, e_class_rec, e_class_f1 = compute_f1(em_class, egt, epred)
                e_label_prec, e_label_rec, e_label_f1 = compute_f1(em_label, egt, epred)

                ged = None
                if ged_available and nodes_provided:
                    try:
                        ged = compute_graph_edit_distance(gt_data, run_data, timeout=ged_timeout)
                    except Exception as e:
                        print(f"[INFO] Graph edit distance failed for {base_name} ({run_name}): {e}")

                results.append({
                    "file": base_name,
                    "run": run_name,
                    "node_name_f1": n_f1,
                    "node_name_class_f1": n_class_f1,
                    "edge_endpoint_f1": e_f1,
                    "edge_endpoint_class_f1": e_class_f1,
                    "edge_endpoint_class_label_f1": e_label_f1,
                    "graph_edit_distance": ged,
                    # Backward-compatible aliases for the old metric names.
                    "node_f1": n_f1,
                    "edge_f1": e_f1,
                    "missing_prediction": False,
                })

                metric_counts = {
                    "node_name": (nm, ngt, npred),
                    "node_name_class": (nm_class, ngt, npred),
                    "edge_endpoint": (em_endpoint, egt, epred),
                    "edge_endpoint_class": (em_class, egt, epred),
                    "edge_endpoint_class_label": (em_label, egt, epred),
                }
                add_metric_counts(run_totals, run_name, metric_counts)

                if ged is None and ged_available and nodes_provided:
                    run_totals[run_name]["graph_edit_distance_timeouts"] += 1
                elif ged is not None:
                    run_totals[run_name]["graph_edit_distance_sum"] += ged
                    run_totals[run_name]["graph_edit_distance_count"] += 1

                run_totals[run_name]["valid_extractions"] += 1
            count_2 += 1

    if expected_runs is not None:
        expected_run_names = sorted(expected_runs, key=run_sort_key)
    elif observed_run_names:
        expected_run_names = sorted(observed_run_names, key=run_sort_key)
    else:
        expected_run_names = ["default"]

    for label_file in label_files:
        for run_name in expected_run_names:
            if (label_file, run_name) in evaluated_pairs:
                continue

            if label_file in gt_data_by_file:
                gt_data = gt_data_by_file[label_file]
            else:
                label_path = os.path.join(label_dir, label_file)
                try:
                    with open(label_path, "r") as f:
                        gt_data = json.load(f)
                except Exception as e:
                    print(f"Error loading label file {label_path}: {e}")
                    continue
                gt_data_by_file[label_file] = gt_data

            gt_nodes = gt_data.get("nodes", [])
            gt_edges = gt_data.get("relations", [])
            ngt = len(gt_nodes)
            egt = len(gt_edges)

            metric_counts = {
                "node_name": (0, ngt, 0),
                "node_name_class": (0, ngt, 0),
                "edge_endpoint": (0, egt, 0),
                "edge_endpoint_class": (0, egt, 0),
                "edge_endpoint_class_label": (0, egt, 0),
            }
            add_metric_counts(run_totals, run_name, metric_counts)
            run_totals[run_name]["missing_predictions"] += 1

            results.append({
                "file": label_file,
                "run": run_name,
                "node_name_f1": 0.0,
                "node_name_class_f1": 0.0,
                "edge_endpoint_f1": 0.0,
                "edge_endpoint_class_f1": 0.0,
                "edge_endpoint_class_label_f1": 0.0,
                "graph_edit_distance": None,
                "node_f1": 0.0,
                "edge_f1": 0.0,
                "missing_prediction": True,
            })

    run_metrics = {}

    for run_name, stats in run_totals.items():
        run_metrics[run_name] = {}
        for metric in NODE_METRICS + EDGE_METRICS:
            precision, recall, f1 = compute_f1(
                stats[f"{metric}_matches"],
                stats[f"{metric}_gt"],
                stats[f"{metric}_pred"]
            )
            run_metrics[run_name][f"{metric}_f1"] = f1
            run_metrics[run_name][f"{metric}_precision"] = precision
            run_metrics[run_name][f"{metric}_recall"] = recall

        run_metrics[run_name]["node_f1"] = run_metrics[run_name]["node_name_f1"]
        run_metrics[run_name]["node_precision"] = run_metrics[run_name]["node_name_precision"]
        run_metrics[run_name]["node_recall"] = run_metrics[run_name]["node_name_recall"]
        run_metrics[run_name]["edge_f1"] = run_metrics[run_name]["edge_endpoint_f1"]
        run_metrics[run_name]["edge_precision"] = run_metrics[run_name]["edge_endpoint_precision"]
        run_metrics[run_name]["edge_recall"] = run_metrics[run_name]["edge_endpoint_recall"]

        ged_count = stats["graph_edit_distance_count"]
        if ged_count > 0:
            run_metrics[run_name]["graph_edit_distance_mean"] = (
                stats["graph_edit_distance_sum"] / ged_count
            )
        else:
            run_metrics[run_name]["graph_edit_distance_mean"] = None

        print(f"[{run_name}] Valid extractions evaluated: {run_totals[run_name]['valid_extractions']}")
        if run_totals[run_name]["missing_predictions"] > 0:
            print(
                f"[{run_name}] Missing predictions scored as empty: "
                f"{run_totals[run_name]['missing_predictions']}"
            )
        print(
            f"[{run_name}] Node F1 (name): "
            f"{round(run_metrics[run_name]['node_name_f1'], 4)} "
            f"(P={round(run_metrics[run_name]['node_name_precision'], 4)}, "
            f"R={round(run_metrics[run_name]['node_name_recall'], 4)})"
        )
        print(
            f"[{run_name}] Node F1 (name+class): "
            f"{round(run_metrics[run_name]['node_name_class_f1'], 4)} "
            f"(P={round(run_metrics[run_name]['node_name_class_precision'], 4)}, "
            f"R={round(run_metrics[run_name]['node_name_class_recall'], 4)})"
        )
        print(
            f"[{run_name}] Edge F1 (endpoints): "
            f"{round(run_metrics[run_name]['edge_endpoint_f1'], 4)} "
            f"(P={round(run_metrics[run_name]['edge_endpoint_precision'], 4)}, "
            f"R={round(run_metrics[run_name]['edge_endpoint_recall'], 4)})"
        )
        print(
            f"[{run_name}] Edge F1 (endpoints+class): "
            f"{round(run_metrics[run_name]['edge_endpoint_class_f1'], 4)} "
            f"(P={round(run_metrics[run_name]['edge_endpoint_class_precision'], 4)}, "
            f"R={round(run_metrics[run_name]['edge_endpoint_class_recall'], 4)})"
        )
        print(
            f"[{run_name}] Edge F1 (endpoints+class+label): "
            f"{round(run_metrics[run_name]['edge_endpoint_class_label_f1'], 4)} "
            f"(P={round(run_metrics[run_name]['edge_endpoint_class_label_precision'], 4)}, "
            f"R={round(run_metrics[run_name]['edge_endpoint_class_label_recall'], 4)})"
        )
        if run_metrics[run_name]["graph_edit_distance_mean"] is not None:
            print(
                f"[{run_name}] Graph edit distance (mean): "
                f"{round(run_metrics[run_name]['graph_edit_distance_mean'], 4)}"
            )
        if stats["graph_edit_distance_timeouts"] > 0:
            print(
                f"[{run_name}] Graph edit distance unavailable/timed out: "
                f"{stats['graph_edit_distance_timeouts']}"
            )

    aggregate_metrics = {}
    if run_metrics:
        print(f"\nAggregate over {len(run_metrics)} runs:")
        for metric in NODE_METRICS + EDGE_METRICS:
            values = [
                metrics[f"{metric}_f1"]
                for metrics in run_metrics.values()
            ]
            mean, std = compute_mean_std(values)
            aggregate_metrics[f"{metric}_f1_mean"] = mean
            aggregate_metrics[f"{metric}_f1_std"] = std
            print(
                f"[all-runs] {METRIC_DISPLAY_NAMES[metric]}: "
                f"mean={round(mean, 4)}, std={round(std, 4)}"
            )

        ged_values = [
            metrics["graph_edit_distance_mean"]
            for metrics in run_metrics.values()
            if metrics["graph_edit_distance_mean"] is not None
        ]
        if ged_values:
            ged_mean, ged_std = compute_mean_std(ged_values)
            aggregate_metrics["graph_edit_distance_mean"] = ged_mean
            aggregate_metrics["graph_edit_distance_std"] = ged_std
            print(
                "[all-runs] Graph edit distance: "
                f"mean={round(ged_mean, 4)}, std={round(ged_std, 4)}"
            )

    '''
    if output_csv and len(results) > 0:
        import csv
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    '''
    if output_csv and run_metrics:
        import csv

        metric_keys = []
        for metric in NODE_METRICS + EDGE_METRICS:
            metric_keys.extend([
                f"{metric}_f1",
                f"{metric}_precision",
                f"{metric}_recall",
            ])

        fieldnames = (
            ["run", "valid_extractions", "missing_predictions"]
            + metric_keys
            + ["graph_edit_distance_mean"]
        )

        ordered_runs = sorted(run_metrics.keys(), key=run_sort_key)

        def _round(value):
            return round(value, 6) if isinstance(value, (int, float)) else value

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Per-run rows
            for run_name in ordered_runs:
                metrics = run_metrics[run_name]
                stats = run_totals.get(run_name, {})
                row = {
                    "run": run_name,
                    "valid_extractions": stats.get("valid_extractions", 0),
                    "missing_predictions": stats.get("missing_predictions", 0),
                    "graph_edit_distance_mean": _round(metrics.get("graph_edit_distance_mean")),
                }
                for key in metric_keys:
                    row[key] = _round(metrics.get(key))
                writer.writerow(row)

            # Aggregate rows across all runs
            mean_row = {"run": "ALL_RUNS_MEAN", "valid_extractions": "", "missing_predictions": ""}
            std_row = {"run": "ALL_RUNS_STD", "valid_extractions": "", "missing_predictions": ""}

            for key in metric_keys:
                values = [
                    run_metrics[r][key]
                    for r in ordered_runs
                    if run_metrics[r].get(key) is not None
                ]
                mean, std = compute_mean_std(values)
                mean_row[key] = _round(mean)
                std_row[key] = _round(std)

            ged_values = [
                run_metrics[r]["graph_edit_distance_mean"]
                for r in ordered_runs
                if run_metrics[r].get("graph_edit_distance_mean") is not None
            ]
            ged_mean, ged_std = compute_mean_std(ged_values)
            mean_row["graph_edit_distance_mean"] = _round(ged_mean)
            std_row["graph_edit_distance_mean"] = _round(ged_std)

            writer.writerow(mean_row)
            writer.writerow(std_row)

    evaluated_files = [r["file"] for r in results]
    print(f"\nUnique files evaluated: {len(set(evaluated_files))}")
    print(f"Total evaluation events: {len(evaluated_files)}")
    if len(evaluated_files) != len(set(evaluated_files)):
        from collections import Counter
        dupes = [f for f, c in Counter(evaluated_files).items() if c > 1]
        print(f"WARNING: {len(dupes)} files evaluated more than once: {dupes}")
    label_files = set(label_files)
    covered = set(evaluated_files)
    missing = label_files - covered
    print(f"Label files not covered: {len(missing)}")
    print(f"Skipped ground truth files: {count_3}")
    if missing:
        print(f"  Missing files: {sorted(missing)}")

    return {
        "per_run": run_metrics,
        "aggregate": aggregate_metrics,
        "details": results
    }
