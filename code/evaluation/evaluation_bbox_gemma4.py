import os
import json
import math
import re
from collections import Counter

import jellyfish
import numpy as np
from scipy.optimize import linear_sum_assignment

BBOX_IOU_THRESHOLD = 0.5
NO_MATCH_COST = 1_000_000.0

_THINK_END_RE = re.compile(r"</think>|<\|end_of_thinking\|>|</thinking>", re.IGNORECASE)


def run_sort_key(run_name):
    match = re.fullmatch(r"run(\d+)", run_name)
    if match:
        return (0, int(match.group(1)))
    return (1, run_name)


def compute_f1(matches, total_gt, total_pred):
    precision = matches / total_pred if total_pred > 0 else 0.0
    recall = matches / total_gt if total_gt > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


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
            for key, val in data.items():
                if isinstance(val, dict) and "nodes" in val and "relations" in val:
                    runs.append((key, val))
                    return runs



    return runs


def compute_mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def bbox_iou(box_a, box_b):
    """IoU of two [x1, y1, x2, y2] boxes. Returns 0.0 if either is missing/degenerate."""
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _unwrap_prediction(data):
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
    return data


def _edge_tail(edge):
    """GT tail endpoint. For predictions, source/target are reversed, so use target."""
    if "tail" in edge:
        return edge["tail"]
    return edge.get("target")


def _edge_head(edge):
    """GT head endpoint. For predictions, source/target are reversed, so use source."""
    if "head" in edge:
        return edge["head"]
    return edge.get("source")


def _norm_class(value):
    if value is None:
        return None

    value = str(value).strip().lower()

    aliases = {
        "arrow_uni": "unidirectional",
        "arrow": "unidirectional",
        "sequenceflow": "sequenceflow",
        "node": "construct",
    }

    return aliases.get(value, value)


def _class_matches(gt_obj, pred_obj):
    gt_class = _norm_class(gt_obj.get("class"))
    pred_class = _norm_class(pred_obj.get("class"))

    if gt_class is None:
        return True

    return pred_class == gt_class


def _hungarian_count(valid_mask, score_matrix):
    if not valid_mask.any():
        return 0

    cost = np.where(valid_mask, 1.0 - score_matrix, NO_MATCH_COST)
    row_ind, col_ind = linear_sum_assignment(cost)

    return int(sum(1 for r, c in zip(row_ind, col_ind) if valid_mask[r, c]))


def match_nodes_by_iou(gt_nodes, pred_nodes, iou_threshold=BBOX_IOU_THRESHOLD):
    """
    Hungarian one-to-one assignment on 1 - IoU between node bboxes.

    Returns:
        bbox_matches: number of pairs with IoU >= threshold
        bbox_class_matches: number of pairs with IoU >= threshold and matching class
        pred_to_gt_id: dict mapping pred node id -> matched gt node id
    """
    if not gt_nodes or not pred_nodes:
        return 0, 0, {}

    n_gt, n_pred = len(gt_nodes), len(pred_nodes)
    iou_matrix = np.zeros((n_gt, n_pred))
    class_ok = np.zeros((n_gt, n_pred), dtype=bool)

    for i, gt in enumerate(gt_nodes):
        for j, pr in enumerate(pred_nodes):
            iou_matrix[i, j] = bbox_iou(gt.get("bbox"), pr.get("bbox"))
            class_ok[i, j] = _class_matches(gt, pr)

    iou_mask = iou_matrix >= iou_threshold
    class_mask = iou_mask & class_ok

    bbox_matches = _hungarian_count(iou_mask, iou_matrix)
    bbox_class_matches = _hungarian_count(class_mask, iou_matrix)

    pred_to_gt_id = {}
    if iou_mask.any():
        cost = np.where(iou_mask, 1.0 - iou_matrix, NO_MATCH_COST)
        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if iou_mask[r, c]:
                gt_id = gt_nodes[r].get("id")
                pred_id = pred_nodes[c].get("id")
                if gt_id is not None and pred_id is not None:
                    pred_to_gt_id[pred_id] = gt_id

    return bbox_matches, bbox_class_matches, pred_to_gt_id


def match_edges_by_iou(gt_edges, pred_edges, pred_to_gt_id, iou_threshold=BBOX_IOU_THRESHOLD):
    """
    Hungarian one-to-one assignments on edge bbox IoU.

    Prediction source/target are treated as reversed:
        pred.target -> GT tail
        pred.source -> GT head

    Returns:
        edge_matches: bbox IoU >= threshold
        edge_class_matches: bbox IoU >= threshold and matching class
        edge_endpoints_matches: bbox IoU >= threshold and matching endpoints
        edge_class_endpoints_matches: bbox IoU >= threshold, matching class, and matching endpoints
    """
    if not gt_edges or not pred_edges:
        return 0, 0, 0, 0

    n_gt, n_pred = len(gt_edges), len(pred_edges)
    iou_matrix = np.zeros((n_gt, n_pred))
    class_ok = np.zeros((n_gt, n_pred), dtype=bool)
    endpoints_ok = np.zeros((n_gt, n_pred), dtype=bool)

    for i, gt in enumerate(gt_edges):
        gt_tail = _edge_tail(gt)
        gt_head = _edge_head(gt)
        gt_bbox = gt.get("bbox")

        for j, pr in enumerate(pred_edges):
            iou_matrix[i, j] = bbox_iou(gt_bbox, pr.get("bbox"))
            class_ok[i, j] = _class_matches(gt, pr)

            pr_tail_raw = _edge_tail(pr)
            pr_head_raw = _edge_head(pr)

            pr_tail = pred_to_gt_id.get(pr_tail_raw) if pr_tail_raw is not None else None
            pr_head = pred_to_gt_id.get(pr_head_raw) if pr_head_raw is not None else None

            ok = True
            if gt_tail is not None and pr_tail != gt_tail:
                ok = False
            if gt_head is not None and pr_head != gt_head:
                ok = False

            endpoints_ok[i, j] = ok

    iou_mask = iou_matrix >= iou_threshold

    edge_matches = _hungarian_count(iou_mask, iou_matrix)
    edge_class_matches = _hungarian_count(iou_mask & class_ok, iou_matrix)
    edge_endpoints_matches = _hungarian_count(iou_mask & endpoints_ok, iou_matrix)
    edge_class_endpoints_matches = _hungarian_count(iou_mask & class_ok & endpoints_ok, iou_matrix)

    return edge_matches, edge_class_matches, edge_endpoints_matches, edge_class_endpoints_matches


def evaluate_bbox(
    model_path,
    dataset=None,
    label_dir=None,
    iou_threshold=BBOX_IOU_THRESHOLD,
    output_csv=None,
    expected_runs=None,
    strip_thinking=False,
):
    """
    Bbox-based evaluator.

    Node metrics:
      - node_bbox: bbox IoU >= iou_threshold.
      - node_bbox_class: bbox IoU >= iou_threshold AND class matches.

    Edge metrics:
      - edge_bbox: bbox IoU >= iou_threshold.
      - edge_bbox_class: bbox IoU >= iou_threshold AND class matches.
      - edge_bbox_endpoints: bbox IoU >= iou_threshold AND each non-null GT
        endpoint matches its pred counterpart through the node correspondence.
      - edge_bbox_class_endpoints: bbox IoU >= iou_threshold AND class matches
        AND endpoints match.

    Reuses helpers from the existing module: compute_f1, compute_mean_std,
    run_sort_key, load_json_or_txt, clean_filename, extract_runs.
    """
    results = []
    run_totals = {}
    evaluated_pairs = set()
    gt_data_by_file = {}
    observed_run_names = set()

    if dataset is not None:
        label_dir = os.path.join("labels_bbox", dataset, "test")
        print(f"Using label directory: {label_dir}")
    elif label_dir is None:
        raise ValueError("You must provide either `dataset` or `label_dir`")

    label_files = sorted(
        f for f in os.listdir(label_dir)
        if os.path.isfile(os.path.join(label_dir, f))
    )
    print(f"Total label files: {len(label_files)}")

    if os.path.isdir(model_path):
        print("Loading extractions from directory...")
        model_files = [
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if f.endswith(".json") or f.endswith(".txt")
        ]
    else:
        print("Loading extractions from aggregated JSON file")
        model_files = [model_path]

    def _init_totals():
        return {
            "valid_extractions": 0,
            "missing_predictions": 0,
            "node_matches": 0,
            "node_class_matches": 0,
            "node_gt": 0,
            "node_pred": 0,
            "edge_bbox_matches": 0,
            "edge_class_matches": 0,
            "edge_endpoints_matches": 0,
            "edge_class_endpoints_matches": 0,
            "edge_gt": 0,
            "edge_pred": 0,
        }

    def _add(run_name, nm, ncm, ngt, npred, em_bbox, em_class, em_endpoints, em_class_endpoints, egt, epred):
        if run_name not in run_totals:
            run_totals[run_name] = _init_totals()
        t = run_totals[run_name]
        t["node_matches"] += nm
        t["node_class_matches"] += ncm
        t["node_gt"] += ngt
        t["node_pred"] += npred
        t["edge_bbox_matches"] += em_bbox
        t["edge_class_matches"] += em_class
        t["edge_endpoints_matches"] += em_endpoints
        t["edge_class_endpoints_matches"] += em_class_endpoints
        t["edge_gt"] += egt
        t["edge_pred"] += epred

    skipped_no_label = 0

    for mf in model_files:
        try:
            data = load_json_or_txt(mf, strip_thinking)
        except Exception as e:
            print(f"[INFO] Failed reading {mf}: {e}")
            continue

        if isinstance(data, list):
            data = _unwrap_prediction(data)

        if isinstance(data, dict) and "nodes" not in data:
            items = data.items()
        else:
            items = [(os.path.basename(mf), data)]

        for filename, file_data in items:
            base_name = clean_filename(filename) + ".json"
            label_path = os.path.join(label_dir, base_name)
            if not os.path.exists(label_path):
                skipped_no_label += 1
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

            for run_name, run_data in extract_runs(file_data, filename):
                run_data = _unwrap_prediction(run_data)

                observed_run_names.add(run_name)
                evaluated_pairs.add((base_name, run_name))

                pred_nodes = run_data.get("nodes", [])
                pred_edges = run_data.get("relations", [])

                nm, ncm, pred_to_gt_id = match_nodes_by_iou(
                    gt_nodes, pred_nodes, iou_threshold=iou_threshold
                )
                em_bbox, em_class, em_endpoints, em_class_endpoints = match_edges_by_iou(
                    gt_edges, pred_edges, pred_to_gt_id, iou_threshold=iou_threshold
                )

                ngt, npred = len(gt_nodes), len(pred_nodes)
                egt, epred = len(gt_edges), len(pred_edges)

                n_prec, n_rec, n_f1 = compute_f1(nm, ngt, npred)
                nc_prec, nc_rec, nc_f1 = compute_f1(ncm, ngt, npred)
                eb_prec, eb_rec, eb_f1 = compute_f1(em_bbox, egt, epred)
                ec_prec, ec_rec, ec_f1 = compute_f1(em_class, egt, epred)
                ee_prec, ee_rec, ee_f1 = compute_f1(em_endpoints, egt, epred)
                ece_prec, ece_rec, ece_f1 = compute_f1(em_class_endpoints, egt, epred)

                results.append({
                    "file": base_name,
                    "run": run_name,
                    "node_bbox_f1": n_f1,
                    "node_bbox_precision": n_prec,
                    "node_bbox_recall": n_rec,
                    "node_bbox_class_f1": nc_f1,
                    "node_bbox_class_precision": nc_prec,
                    "node_bbox_class_recall": nc_rec,
                    "edge_bbox_f1": eb_f1,
                    "edge_bbox_precision": eb_prec,
                    "edge_bbox_recall": eb_rec,
                    "edge_bbox_class_f1": ec_f1,
                    "edge_bbox_class_precision": ec_prec,
                    "edge_bbox_class_recall": ec_rec,
                    "edge_bbox_endpoints_f1": ee_f1,
                    "edge_bbox_endpoints_precision": ee_prec,
                    "edge_bbox_endpoints_recall": ee_rec,
                    "edge_bbox_class_endpoints_f1": ece_f1,
                    "edge_bbox_class_endpoints_precision": ece_prec,
                    "edge_bbox_class_endpoints_recall": ece_rec,
                    "missing_prediction": False,
                })
                _add(run_name, nm, ncm, ngt, npred, em_bbox, em_class, em_endpoints, em_class_endpoints, egt, epred)
                run_totals[run_name]["valid_extractions"] += 1

    # Score label files without a prediction as zero
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
                try:
                    with open(os.path.join(label_dir, label_file), "r") as f:
                        gt_data = json.load(f)
                except Exception as e:
                    print(f"Error loading label file {label_file}: {e}")
                    continue
                gt_data_by_file[label_file] = gt_data

            ngt = len(gt_data.get("nodes", []))
            egt = len(gt_data.get("relations", []))
            _add(run_name, 0, 0, ngt, 0, 0, 0, 0, 0, egt, 0)
            run_totals.setdefault(run_name, _init_totals())["missing_predictions"] += 1

            results.append({
                "file": label_file,
                "run": run_name,
                "node_bbox_f1": 0.0,
                "node_bbox_precision": 0.0,
                "node_bbox_recall": 0.0,
                "node_bbox_class_f1": 0.0,
                "node_bbox_class_precision": 0.0,
                "node_bbox_class_recall": 0.0,
                "edge_bbox_f1": 0.0,
                "edge_bbox_precision": 0.0,
                "edge_bbox_recall": 0.0,
                "edge_bbox_class_f1": 0.0,
                "edge_bbox_class_precision": 0.0,
                "edge_bbox_class_recall": 0.0,
                "edge_bbox_endpoints_f1": 0.0,
                "edge_bbox_endpoints_precision": 0.0,
                "edge_bbox_endpoints_recall": 0.0,
                "edge_bbox_class_endpoints_f1": 0.0,
                "edge_bbox_class_endpoints_precision": 0.0,
                "edge_bbox_class_endpoints_recall": 0.0,
                "missing_prediction": True,
            })

    # Per-run aggregation
    run_metrics = {}
    for run_name, stats in run_totals.items():
        n_prec, n_rec, n_f1 = compute_f1(
            stats["node_matches"], stats["node_gt"], stats["node_pred"]
        )
        nc_prec, nc_rec, nc_f1 = compute_f1(
            stats["node_class_matches"], stats["node_gt"], stats["node_pred"]
        )
        eb_prec, eb_rec, eb_f1 = compute_f1(
            stats["edge_bbox_matches"], stats["edge_gt"], stats["edge_pred"]
        )
        ec_prec, ec_rec, ec_f1 = compute_f1(
            stats["edge_class_matches"], stats["edge_gt"], stats["edge_pred"]
        )
        ee_prec, ee_rec, ee_f1 = compute_f1(
            stats["edge_endpoints_matches"], stats["edge_gt"], stats["edge_pred"]
        )
        ece_prec, ece_rec, ece_f1 = compute_f1(
            stats["edge_class_endpoints_matches"], stats["edge_gt"], stats["edge_pred"]
        )
        run_metrics[run_name] = {
            "node_bbox_precision": n_prec,
            "node_bbox_recall": n_rec,
            "node_bbox_f1": n_f1,
            "node_bbox_class_precision": nc_prec,
            "node_bbox_class_recall": nc_rec,
            "node_bbox_class_f1": nc_f1,
            "edge_bbox_precision": eb_prec,
            "edge_bbox_recall": eb_rec,
            "edge_bbox_f1": eb_f1,
            "edge_bbox_class_precision": ec_prec,
            "edge_bbox_class_recall": ec_rec,
            "edge_bbox_class_f1": ec_f1,
            "edge_bbox_endpoints_precision": ee_prec,
            "edge_bbox_endpoints_recall": ee_rec,
            "edge_bbox_endpoints_f1": ee_f1,
            "edge_bbox_class_endpoints_precision": ece_prec,
            "edge_bbox_class_endpoints_recall": ece_rec,
            "edge_bbox_class_endpoints_f1": ece_f1,
        }
        print(f"[{run_name}] Valid extractions: {stats['valid_extractions']}")
        if stats["missing_predictions"] > 0:
            print(f"[{run_name}] Missing predictions scored as empty: {stats['missing_predictions']}")
        print(f"[{run_name}] Node bbox F1: {round(n_f1, 4)} (P={round(n_prec, 4)}, R={round(n_rec, 4)})")
        print(f"[{run_name}] Node bbox + class F1: {round(nc_f1, 4)} (P={round(nc_prec, 4)}, R={round(nc_rec, 4)})")
        print(f"[{run_name}] Edge bbox F1 (IoU only): {round(eb_f1, 4)} (P={round(eb_prec, 4)}, R={round(eb_rec, 4)})")
        print(f"[{run_name}] Edge bbox + class F1: {round(ec_f1, 4)} (P={round(ec_prec, 4)}, R={round(ec_rec, 4)})")
        print(f"[{run_name}] Edge bbox F1 (IoU + endpoints): {round(ee_f1, 4)} (P={round(ee_prec, 4)}, R={round(ee_rec, 4)})")
        print(f"[{run_name}] Edge bbox + class F1 (IoU + endpoints): {round(ece_f1, 4)} (P={round(ece_prec, 4)}, R={round(ece_rec, 4)})")

    # Across-run aggregation
    aggregate_metrics = {}
    if run_metrics:
        print(f"\nAggregate over {len(run_metrics)} runs:")
        for key, label in (
            ("node_bbox_f1", "Node bbox F1"),
            ("node_bbox_class_f1", "Node bbox + class F1"),
            ("edge_bbox_f1", "Edge bbox F1 (IoU only)"),
            ("edge_bbox_class_f1", "Edge bbox + class F1"),
            ("edge_bbox_endpoints_f1", "Edge bbox F1 (IoU + endpoints)"),
            ("edge_bbox_class_endpoints_f1", "Edge bbox + class F1 (IoU + endpoints)"),
        ):
            values = [m[key] for m in run_metrics.values()]
            mean, std = compute_mean_std(values)
            aggregate_metrics[f"{key}_mean"] = mean
            aggregate_metrics[f"{key}_std"] = std
            print(f"[all-runs] {label}: mean={round(mean, 4)}, std={round(std, 4)}")

    # CSV
    if output_csv and run_metrics:
        import csv
        metric_keys = [
            "node_bbox_f1",
            "node_bbox_precision",
            "node_bbox_recall",
            "node_bbox_class_f1",
            "node_bbox_class_precision",
            "node_bbox_class_recall",
            "edge_bbox_f1",
            "edge_bbox_precision",
            "edge_bbox_recall",
            "edge_bbox_class_f1",
            "edge_bbox_class_precision",
            "edge_bbox_class_recall",
            "edge_bbox_endpoints_f1",
            "edge_bbox_endpoints_precision",
            "edge_bbox_endpoints_recall",
            "edge_bbox_class_endpoints_f1",
            "edge_bbox_class_endpoints_precision",
            "edge_bbox_class_endpoints_recall",
        ]
        fieldnames = ["run", "valid_extractions", "missing_predictions"] + metric_keys
        ordered_runs = sorted(run_metrics.keys(), key=run_sort_key)

        def _round(v):
            return round(v, 6) if isinstance(v, (int, float)) else v

        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for run_name in ordered_runs:
                m = run_metrics[run_name]
                stats = run_totals.get(run_name, {})
                row = {
                    "run": run_name,
                    "valid_extractions": stats.get("valid_extractions", 0),
                    "missing_predictions": stats.get("missing_predictions", 0),
                }
                for k in metric_keys:
                    row[k] = _round(m.get(k))
                w.writerow(row)

            mean_row = {"run": "ALL_RUNS_MEAN", "valid_extractions": "", "missing_predictions": ""}
            std_row = {"run": "ALL_RUNS_STD", "valid_extractions": "", "missing_predictions": ""}
            for k in metric_keys:
                values = [run_metrics[r][k] for r in ordered_runs if run_metrics[r].get(k) is not None]
                mean, std = compute_mean_std(values)
                mean_row[k] = _round(mean)
                std_row[k] = _round(std)
            w.writerow(mean_row)
            w.writerow(std_row)

    print(f"\nUnique files evaluated: {len(set(r['file'] for r in results))}")
    print(f"Total evaluation events: {len(results)}")
    print(f"Skipped (no matching label): {skipped_no_label}")

    return {
        "per_run": run_metrics,
        "aggregate": aggregate_metrics,
        "details": results,
    }
