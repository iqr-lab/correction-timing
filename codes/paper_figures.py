import pickle
import argparse
# import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from matplotlib.patches import Patch
import pandas as pd
import os, pickle, numpy as np
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon, ttest_1samp
from scipy.stats import t
import os, re, pickle, math, glob
import pandas as pd
from typing import Iterable, Optional, Union, List, Tuple, Dict
import seaborn as sns
import textwrap

# analyzing and plotting data for the paper

# F1 and correction MAE
def plot_box_and_whisker_all(): # metric plot
    # metric_name = 'Correction MAE'
    metric_name = 'F1 Score'
    timing_ratios = [0.7, 0.8, 0.9, 1.0]
    # timing_ratios = [0.7, 0.8, 0.9]

    all_feature_values = []
    boltzmann_values = []

    for timing_ratio in timing_ratios:
        run_list_all = []
        run_list_bol = []
        for i in range(200):
            # Load all_feature
            with open(f"./results/evaluation_com/"+str(int(100*timing_ratio))+'_60_'+str(int(i))+".pkl", 'rb') as f:
                run_list = pickle.load(f)
            run_list_all.append(run_list)

            with open(f"./results/evaluation_bol/"+str(int(100*timing_ratio))+'_60_'+str(int(i))+".pkl", 'rb') as f:
                run_list = pickle.load(f)
            run_list_bol.append(run_list)

        all_feature_metric = [run[metric_name] for run in run_list_all]
        all_feature_values.append(all_feature_metric)

        boltzmann_metric = [run[metric_name] for run in run_list_bol]
        boltzmann_values.append(boltzmann_metric)

    # Prepare x positions
    x = np.arange(len(timing_ratios)) + 1
    offset = 0.15

    plt.figure(figsize=(12, 6))

    # Plot boxplots
    plt.boxplot(all_feature_values, 
                positions=x - offset, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#91bfdb', color='blue'),
                medianprops=dict(color='blue'))
    
    plt.boxplot(boltzmann_values, 
                positions=x + offset, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#ffffbf', color='green'),
                medianprops=dict(color='green'))


    # Convert timing_ratios to percent labels
    percentage_labels = [f'{int(r * 100)}%' for r in timing_ratios]

    # Labels and ticks
    plt.xticks(x, percentage_labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Timing (%)', fontsize=24)
    plt.ylabel(metric_name, fontsize=24)
    # plt.title(f'Box-and-Whisker Plot of {metric_name}: All Feature vs Boltzmann', fontsize=16)
    # plt.ylim(0, 1.1)
    plt.grid(True, axis='y')

    # Legend with box patches
    plt.legend(handles=[
        Patch(facecolor='#91bfdb', edgecolor='blue', label='Multi Feature'),
        Patch(facecolor='#ffffbf', edgecolor='green', label='Boltzmann')
    ], fontsize=18,
    loc='lower left')

    # Calculate p-values and add brackets + stars
    for idx in range(len(timing_ratios)):
        all_data = all_feature_values[idx]
        boltz_data = boltzmann_values[idx]

        u_stat, p_val = mannwhitneyu(all_data, boltz_data, alternative='two-sided')
        print(p_val)

        # Decide number of stars
        if p_val <= 0.001:
            star = '***'
        elif p_val <= 0.01:
            star = '**'
        elif p_val <= 0.05:
            star = '*'
        else:
            star = ''

        if star:
            # Get max y value to set the bracket height
            y_max = max(max(all_data), max(boltz_data))

            # Set bracket height
            height = y_max + 0.05
            h = 0.02  # height of the bracket arms

            # x-coordinates of the two boxes
            x1 = x[idx] - offset
            x2 = x[idx] + offset

            # Draw the bracket
            plt.plot([x1, x1, x2, x2], [height, height + h, height + h, height], color='black', linewidth=1.5)

            # Add the star label above
            plt.text((x1 + x2) / 2, height + h + 0.01, star, ha='center', va='bottom', fontsize=14, color='red')

    plt.tight_layout()
    plt.ylim(0, 1.1)
    
    # plt.savefig("/Users/anjiabei/Documents/research/figures/goal_infer_paper/correction_mae_all.png", dpi = 300)
    # plt.savefig("/Users/anjiabei/Documents/research/figures/goal_infer_paper/F1_all.png", dpi = 300)
    plt.show()


def plot_box_and_whisker_counting_all(): # predicted corrections vs actulal corrections
    comp = "all"
    metric_name = 'Correction Count'
    metric_name_actual = 'Actual Correction Count'
    timing_ratios = [0.7, 0.8, 0.9, 1.0]

    # Load correction count once (no need to reload every loop)
    # with open(f"./model_evaluation/correction_count_all", 'rb') as f:
    #     ct = pickle.load(f)

    all_feature_values = []
    boltzmann_values = []
    # actual_ct = []
    all_feature_values_actual = []
    boltzmann_values_actual = []

    for timing_ratio in timing_ratios:
        run_list_all = []
        run_list_bol = []
        for i in range(200):
            # Load all_feature
            with open(f"./results/evaluation/"+str(int(100*timing_ratio))+'_60_'+str(int(i))+".pkl", 'rb') as f:
                run_list = pickle.load(f)
            run_list_all.append(run_list)

            with open(f"./results/evaluation_bol/"+str(int(100*timing_ratio))+'_60_'+str(int(i))+".pkl", 'rb') as f:
                run_list = pickle.load(f)
            run_list_bol.append(run_list)
        
        all_feature_metric = [run[metric_name] for run in run_list_all]
        all_feature_values.append(all_feature_metric)
        all_feature_metric_actual = [run[metric_name_actual] for run in run_list_all]
        all_feature_values_actual.append(all_feature_metric_actual)

        boltzmann_metric = [run[metric_name] for run in run_list_bol]
        boltzmann_values.append(boltzmann_metric)
        boltzmann_metric_actual = [run[metric_name_actual] for run in run_list_bol]
        boltzmann_values_actual.append(boltzmann_metric_actual)


    all_feature_values = (np.array(all_feature_values) / np.array(all_feature_values_actual)).tolist()
    boltzmann_values = (np.array(boltzmann_values) / np.array(boltzmann_values_actual)).tolist()

    # print(all_feature_values.shape)

    # Prepare x positions
    x = np.arange(len(timing_ratios)) + 1
    offset = 0.15

    plt.figure(figsize=(12, 6))

    # Plot boxplots
    plt.boxplot(all_feature_values, 
                positions=x - offset, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#91bfdb', color='blue'),
                medianprops=dict(color='blue'))
    
    plt.boxplot(boltzmann_values, 
                positions=x + offset, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor='#ffffbf', color='green'),
                medianprops=dict(color='green'))

    # Convert timing_ratios to percent labels
    percentage_labels = [f'{int(r * 100)}%' for r in timing_ratios]

    # Labels and ticks
    plt.xticks(x, percentage_labels, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Timing (%)', fontsize=24)
    plt.ylabel("Predicted Correction Ratio", fontsize=24)
    # plt.title(f'Box-and-Whisker Plot of {metric_name}: All Feature vs Boltzmann', fontsize=16)
    # plt.ylim(0, 1.1)
    plt.grid(True, axis='y')

    # Legend with box patches
    plt.legend(handles=[
        Patch(facecolor='#91bfdb', edgecolor='blue', label='Multi Feature'),
        Patch(facecolor='#ffffbf', edgecolor='green', label='Boltzmann')
    ], fontsize=18,
    loc='lower left')
    # bbox_to_anchor=(0.8, 0.8))

    # Calculate p-values and add brackets + stars
    for idx in range(len(timing_ratios)):
        all_data = all_feature_values[idx]
        boltz_data = boltzmann_values[idx]

        u_stat, p_val = mannwhitneyu(all_data, boltz_data, alternative='two-sided')

        # Decide number of stars
        if p_val <= 0.001:
            star = '***'
        elif p_val <= 0.01:
            star = '**'
        elif p_val <= 0.05:
            star = '*'
        else:
            star = ''

        if star:
            # Get max y value to set the bracket height
            y_max = max(max(all_data), max(boltz_data))

            # Set bracket height
            height = y_max + 0.05
            h = 0.02  # height of the bracket arms

            # x-coordinates of the two boxes
            x1 = x[idx] - offset
            x2 = x[idx] + offset

            # Draw the bracket
            plt.plot([x1, x1, x2, x2], [height, height + h, height + h, height], color='black', linewidth=1.5)

            # Add the star label above
            plt.text((x1 + x2) / 2, height + h + 0.01, star, ha='center', va='bottom', fontsize=14, color='red')

    plt.tight_layout()
    plt.ylim(0, 1.2)
    plt.savefig("/Users/anjiabei/Documents/research/figures/goal_infer_paper/prediction_ratio_all.png", dpi = 300)
    plt.show()



def plot_baseline_vs_ablation_boxplots_vertical_stacked_trs( # f1 score for each ablation model + baseline for different timing ratios
    trs,
    split=60,
    repeats=range(10),
    n_features=7,
    base_dir="./results/evaluation_com",
    ablation_dir="./results/evaluation_ablation_com",
    feature_names=None,
    alpha=0.05,
    save_path=None,
    colors=None,         # [baseline_color, *feature_colors] OR just feature colors
    showfliers=False,    # set True to show outliers
    fontsize=12,
    baseline_ref="median",        # "median" or "mean" line from baseline
    refline_kwargs=None,          # e.g., dict(color="crimson", ls="--", lw=1.4, alpha=0.8)
    max_label_chars=12,           # wrap feature names per line
    max_label_lines=2,            # number of lines for wrapped labels
    xtick_fontsize=None           # fontsize for x tick labels (None -> use rc/default)
):
    """
    Stacks one subplot per timing ratio (tr), sharing the same x-axis (feature names).
    For each tr subplot:
      - x-axis: ['Baseline Model', *feature_names]
      - y-axis: F1 score
      - Each box = F1 distribution across repeats
      - Significance: Mann–Whitney U (ablation vs baseline), Holm-Bonferroni corrected (hat with stars)
      - Draw a horizontal reference line at the baseline median (default) or mean.

    Parameters
    ----------
    baseline_ref : {"median","mean"}
        Which baseline statistic to draw as a horizontal reference line.
    refline_kwargs : dict
        Matplotlib line style for the reference line (e.g., color, ls, lw, alpha).
    """

    trs = list(trs)
    repeats = list(repeats)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    # labels for x-axis (shared)
    raw_labels = ["Baseline Model"] + list(feature_names)

    # file name patterns (basename matching)
    pat_base = re.compile(r"^(\d+)_(\d+)_(\d+)\.pkl$")
    pat_abl  = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.pkl$")

    # --- helpers ---
    def holm_bonferroni(pvals):
        pvals = np.array(pvals, dtype=float)
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty_like(pvals)
        for rank, idx in enumerate(order, start=1):
            adj[idx] = min(1.0, (m - rank + 1) * pvals[idx])
        return adj

    def p_to_stars(p):
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < alpha else "ns"

    def draw_hat(ax, x1, x2, y, h, stars, text_extra=0.0):
        # hat shape + stars text; keep text clipped inside axes
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="k")
        ax.text((x1 + x2) / 2, y + h + text_extra, stars,
                ha="center", va="bottom", fontsize=fontsize-1, clip_on=True)

    def wrap_label(s: str, max_chars=12, max_lines=2):
        s = s.replace("_", " ")
        parts = textwrap.wrap(s, width=max_chars)
        if not parts:
            return s
        if len(parts) > max_lines:
            last = parts[max_lines-1]
            last = (last[:-3] + "...") if len(last) >= 3 else (last + "...")
            parts = parts[:max_lines-1] + [last]
        return "\n".join(parts)

    if refline_kwargs is None:
        refline_kwargs = dict(color="crimson", ls="--", lw=1.4, alpha=0.9)

    # --- load data ---
    base_data = {}        # tr -> list of baseline F1 over repeats
    ablation_data = {}    # (tr, feat_idx) -> list of F1 over repeats

    for fp in glob.glob(os.path.join(base_dir, "*.pkl")):
        m = pat_base.match(os.path.basename(fp))
        if not m:
            continue
        tr_b, split_b, rep_b = map(int, m.groups())
        if split_b != split or tr_b not in trs or rep_b not in repeats:
            continue
        with open(fp, "rb") as f:
            res = pickle.load(f)
        base_data.setdefault(tr_b, []).append(res["F1 Score"])

    for fp in glob.glob(os.path.join(ablation_dir, "*.pkl")):
        m = pat_abl.match(os.path.basename(fp))
        if not m:
            continue
        tr_a, split_a, rep_a, del_a = map(int, m.groups())
        if split_a != split or tr_a not in trs or rep_a not in repeats or del_a >= n_features:
            continue
        with open(fp, "rb") as f:
            res = pickle.load(f)
        ablation_data.setdefault((tr_a, del_a), []).append(res["F1 Score"])

    # --- colors ---
    if colors is None:
        cmap = plt.cm.viridis
        colors = ["#8e8e8e"] + [cmap(i / max(1, n_features - 1)) for i in range(n_features)]
    else:
        colors = list(colors)
        if len(colors) == n_features:
            colors = ["#8e8e8e"] + colors  # default baseline gray
        elif len(colors) == n_features + 1:
            pass
        else:
            raise ValueError(f"`colors` must have length {n_features} (features only) or {n_features+1} (baseline + features).")

    # --- figure layout (one row per tr), share x ---
    n_tr = len(trs)
    fig_height = max(3.0, 3.6 * n_tr)     # scale height with number of rows
    fig_width  = max(8.0, 0.8 * (n_features + 1) + 2.0)

    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(n_tr, 1, figsize=(fig_width, fig_height), sharex=True)
    if n_tr == 1:
        axes = [axes]

    # --- plot per tr ---
    for ax, tr in zip(axes, trs):
        if tr not in base_data:
            ax.set_title(f"tr={tr}% (no data)")
            ax.axis("off")
            continue

        baseline_vals = np.array(base_data[tr], dtype=float)

        # assemble boxes: baseline first, then each ablation by feature index
        all_boxes = [baseline_vals]
        for fidx in range(n_features):
            vals = np.array(ablation_data.get((tr, fidx), []), dtype=float)
            all_boxes.append(vals)

        positions = np.arange(len(all_boxes))  # 0..n_features (with 0 for baseline)

        bp = ax.boxplot(
            all_boxes,
            positions=positions,
            vert=True,
            patch_artist=True,
            widths=0.6,
            showfliers=showfliers,
            manage_ticks=False
        )

        # style
        for patch, col in zip(bp["boxes"], colors[:len(all_boxes)]):
            patch.set(facecolor=col, alpha=0.9)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.4)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=1)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=1)

        ax.set_title(f"Timing {tr}%")
        ax.set_ylabel("F1 Score")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # --- baseline reference line (median or mean) ---
        if baseline_ref == "mean":
            ref_y = float(np.nanmean(baseline_vals)) if baseline_vals.size else np.nan
            ref_label = "Baseline mean"
        else:
            ref_y = float(np.nanmedian(baseline_vals)) if baseline_vals.size else np.nan
            ref_label = "Baseline median"

        # if np.isfinite(ref_y):
        #     ax.axhline(ref_y, **refline_kwargs)
        #     ax.text(
        #         1.0, ref_y, f"  {ref_label}",
        #         transform=ax.get_yaxis_transform(which='grid'),
        #         va="center", ha="left",
        #         fontsize=max(8, fontsize-2),
        #         color=refline_kwargs.get("color", "crimson")
        #     )

        # --- significance vs baseline (index 0) ---
        p_raw, idx_pairs = [], []
        for i in range(1, len(all_boxes)):
            A, B = all_boxes[0], all_boxes[i]
            if len(A) > 0 and len(B) > 0:
                _, p = mannwhitneyu(A, B, alternative="two-sided")
            else:
                p = np.nan
            p_raw.append(p)
            idx_pairs.append((0, i))

        valid_mask = ~np.isnan(np.array(p_raw, dtype=float))
        if np.any(valid_mask):
            p_adj_valid = holm_bonferroni(np.array(p_raw, dtype=float)[valid_mask])
            p_adj = np.full(len(p_raw), np.nan, dtype=float)
            p_adj[valid_mask] = p_adj_valid
        else:
            p_adj = np.array(p_raw, dtype=float)

        # hat placement above tallest whisker (ignores hidden outliers)
        whisker_tops = [w.get_ydata()[1] for w in bp["whiskers"]]
        if whisker_tops:
            y_max = float(np.nanmax(whisker_tops))
        else:
            y_max = max(np.nanmax(b) for b in all_boxes if len(b))
        y_min = min(np.nanmin(b) for b in all_boxes if len(b))
        span = (y_max - y_min) if np.isfinite(y_max) and np.isfinite(y_min) else 1.0

        base_y = y_max + 0.04 * span        # overall offset above boxes
        h = 0.03 * span                      # hat height
        text_extra = 0.006 * span            # extra lift for the stars text

        # draw hats + keep stars inside
        stack = 0
        tallest_top = y_max
        for (x0, x1), p in zip(idx_pairs, p_adj):
            if np.isnan(p):
                continue
            stars = p_to_stars(p)
            if stars == "ns":
                continue
            y_hat_bottom = base_y + stack * (h * 1.7)
            draw_hat(ax, x0, x1, y_hat_bottom, h, stars, text_extra=text_extra)
            tallest_top = max(tallest_top, y_hat_bottom + h + text_extra)
            stack += 1

        # ensure stars stay inside the axes
        y0, y1 = ax.get_ylim()
        required_top = tallest_top * 1.03  # tiny cushion
        if required_top > y1:
            ax.set_ylim(y0, required_top)

    # shared x-axis: wrapped feature names (plus baseline)
    wrapped_labels = [wrap_label(lbl, max_chars=max_label_chars, max_lines=max_label_lines) for lbl in raw_labels]
    axes[-1].set_xticks(np.arange(len(wrapped_labels)))
    axes[-1].set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize = 9)
                            #  fontsize=(xtick_fontsize if xtick_fontsize is not None else fontsize))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




def plot_box_sig_predictors_vs_tr_stacked( # KLD plots for each traget aggregated across shapes
    trs: Iterable[float],
    shapes: Iterable[str],
    targets: Iterable[int],
    reps: Iterable[int] = range(50),
    split: int = 60,
    base: str = "./KLDs",
    method_labels: Optional[List[str]] = None,
    methods_idx: Optional[List[int]] = None,   # exactly 3
    fontsize: int = 12,
    save_path: Optional[str] = None,
    bar_spacing: float = 2.2,                  # spacing factor between sig bars
):
    """
    One figure with subplots stacked vertically: one subplot per target.
    Each subplot: x-axis = tr; at each tr, 3 side-by-side boxplots (colored by predictor).
    Each box pools ALL KLDs from `all_kls` across repeats (and shapes if multiple passed).
    Significance: Mann–Whitney U (two-sided) per tr for the 3 pairwise predictor comps, Holm-adjusted.
    """
    trs = list(trs)
    shapes = list(shapes)
    targets = list(targets)
    reps = list(reps)

    # ---------- helpers ----------
    def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(order, start=1):
            adj[idx] = min(1.0, (m - rank + 1) * pvals[idx])
        return adj

    def p_to_stars(p: float) -> str:
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    def add_sig_bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color = "k")
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")

    def load_all_for_one(tr: float, shape: str, target: int) -> List[np.ndarray]:
        """
        Returns a list [arr_pred0, arr_pred1, ...] where each arr contains ALL KLDs
        from `all_kls` across the requested repeats for this (tr, shape, target).
        Falls back to mean_kls if all_kls is missing (treats them as singletons).
        """
        accum: List[List[float]] = []  # accum[pred] = list of floats
        n_preds_local = None

        for rep in reps:
            path = os.path.join(base, f"{int(100*tr)}_{split}_shape_{shape}_target_{int(target)}_{int(rep)}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                res = pickle.load(f)

            if "all_kls" in res and res["all_kls"] is not None:
                # all_kls expected shape: list length n_preds; each entry is list of floats (per-example KLDs)
                lists = [np.asarray(x, dtype=float).ravel() for x in res["all_kls"]]
            elif "mean_kls" in res:  # fallback: treat per-rep means as singletons
                lists = [np.asarray([v], dtype=float) for v in res["mean_kls"]]
            else:
                continue

            # if "mean_kls" in res:  # fallback: treat per-rep means as singletons
            #     lists = [np.asarray([v], dtype=float) for v in res["mean_kls"]]
            # else:
            #     continue

            if n_preds_local is None:
                n_preds_local = len(lists)
                accum = [[] for _ in range(n_preds_local)]

            # if inconsistent n_preds, skip this file
            if len(lists) != n_preds_local:
                continue

            for j, arr in enumerate(lists):
                # drop NaNs
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    accum[j].extend(arr.tolist())

        if not accum:
            return []

        return [np.asarray(a, dtype=float) for a in accum]

    # ---- detect n_preds and set methods/labels
    n_preds_detected = None
    for tgt in targets:
        for tr in trs:
            for shp in shapes:
                arrs = load_all_for_one(tr, shp, tgt)
                if arrs:
                    n_preds_detected = len(arrs)
                    break
            if n_preds_detected:
                break
        if n_preds_detected:
            break
    if n_preds_detected is None:
        raise FileNotFoundError("No KLD files found or `all_kls`/`mean_kls` missing.")

    if methods_idx is None:
        methods_idx = [0, 1, 2]
    if len(methods_idx) != 3 or any(m < 0 or m >= n_preds_detected for m in methods_idx):
        raise ValueError(f"`methods_idx` must be 3 valid indices within [0,{n_preds_detected-1}].")

    default_labels = [f"Pred {j+1}" for j in range(n_preds_detected)]
    labels_all = method_labels if (method_labels and len(method_labels) == n_preds_detected) else default_labels
    labels = [labels_all[j] for j in methods_idx]

    # colors for 3 predictors (paper-friendly)
    colors = ["#fc8d59", "#ffffbf", "#99d594"]

    # ---------- figure ----------
    n_tr = len(trs)
    n_tgt = len(targets)
    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(n_tgt, 1, figsize=(max(8, 1.2 * n_tr * 3), 4.6 * n_tgt), sharex=True)
    if n_tgt == 1:
        axes = [axes]

    group_centers = np.arange(1, n_tr + 1, dtype=float)
    offset = 0.25

    # ---------- per target ----------
    for row, tgt in enumerate(targets):
        ax = axes[row]

        # Gather per-tr pooled arrays for the 3 selected predictors, combining across shapes
        per_tr_vals: Dict[float, List[np.ndarray]] = {}
        for tr in trs:
            # load for each shape, then concat per predictor across shapes
            per_shape = [load_all_for_one(tr, shp, tgt) for shp in shapes]
            per_shape = [ps for ps in per_shape if ps]   # drop empties

            if not per_shape:
                per_tr_vals[tr] = [np.array([]), np.array([]), np.array([])]
            else:
                # per_shape is a list of lists: [ [arr_p0, arr_p1, ...], [arr_p0, arr_p1, ...], ...]
                pooled = []
                for m in methods_idx:
                    pooled.append(
                        np.concatenate([ps[m] for ps in per_shape if len(ps) > m and ps[m].size], axis=0)
                    )
                per_tr_vals[tr] = pooled

        # ----- Draw colored boxplots -----
        positions, box_data, box_colors = [], [], []
        for i, tr in enumerate(trs):
            vals = per_tr_vals[tr]
            # ensure we always have 3 arrays
            if len(vals) != 3:
                vals = [np.array([]), np.array([]), np.array([])]
            posA, posB, posC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
            positions += [posA, posB, posC]
            box_data += [vals[0], vals[1], vals[2]]
            box_colors += colors

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True
        )

        for patch, c in zip(bp["boxes"], box_colors):
            patch.set(facecolor=c, alpha=0.8)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=1)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=1)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.5)

        ax.set_ylabel("KLD")
        ax.set_ylim(0, 35)
        ax.set_title(f"Target {tgt + 1}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # legend only on top subplot
        # if row == 0:
        #     handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
        #     ax.legend(handles, labels, title="Model", ncols=3, loc="upper right", frameon=False)
        # if row == 0:
        #     handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
        #     ax.legend(handles, labels, title="Model", ncols=3, loc="upper right", frameon=False)

        # ---- significance: Mann–Whitney U (independent samples) ----
        for i, tr in enumerate(trs):
            A, B, C = per_tr_vals[tr]
            groups = [A, B, C]
            if any(g.size == 0 for g in groups):
                continue

            pairs = [(0, 1), (0, 2), (1, 2)]
            p_raw = []
            for (a, b) in pairs:
                # Two-sided Mann–Whitney U (robust to diff sample sizes, nonparametric)
                U, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
                p_raw.append(p)
            p_raw = np.array(p_raw, dtype=float)
            p_adj = holm_bonferroni(p_raw)

            local_max = max(g.max() for g in groups if g.size)
            # whisker_tops = [whisker.get_ydata()[1] for whisker in bp["whiskers"]]
            # local_max = np.nanmax(whisker_tops)
            y0, y1 = ax.get_ylim()
            span = y1 - y0
            base_y = local_max + 0.03 * span
            h = 0.03 * span

            xA, xB, xC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
            stack = 0
            for (a, b), p in zip(pairs, p_adj):
                stars = p_to_stars(p)
                if stars == "ns":
                    continue
                if   (a, b) == (0, 1): x1, x2 = xA, xB
                elif (a, b) == (0, 2): x1, x2 = xA, xC
                else:                  x1, x2 = xB, xC
                add_sig_bracket(ax, x1, x2, base_y + stack * (h * bar_spacing), h, stars)
                stack += 1

    # shared x-axis
    axes[-1].set_xticks(group_centers)
    axes[-1].set_xticklabels([f"{int(100*t)}%" for t in trs])
    axes[-1].set_xlabel("Timing (%)")

    # handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
    # fig.legend(
    #     handles, labels,
    #     title="Model",
    #     ncols=3,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.05),  # position above figure
    #     frameon=False
    # )

    # # plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]

    # First, reserve vertical space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])   # <-- leaves ~8% at the top

    # Now place ONE legend for the whole figure
    fig.legend(
        handles, labels,
        title=None,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_box_sig_predictors_vs_tr_stacked_by_shape( # KLD plots for each shape aggregated across targets
    trs: Iterable[float],
    shapes: Iterable[str],
    targets: Iterable[int],
    reps: Iterable[int] = range(50),
    split: int = 60,
    base: str = "./KLDs",
    method_labels: Optional[List[str]] = None,
    methods_idx: Optional[List[int]] = None,     # exactly 3 predictor indices
    fontsize: int = 12,
    save_path: Optional[str] = None,
    bar_spacing: float = 2.2,                    # vertical spacing between sig bars
):
    """
    One figure with subplots stacked vertically: one subplot per SHAPE.
    Each subplot: x-axis = tr; at each tr, 3 side-by-side boxplots (colored by predictor).
    Each box pools ALL per-example KLDs from `all_kls` across:
      - all requested TARGETS
      - all requested REPEATS
    Significance per tr: Mann–Whitney U (two-sided) for the 3 predictor pairs, Holm-adjusted.
    """
    trs = list(trs)
    shapes = list(shapes)
    targets = list(targets)
    reps = list(reps)

    # ---------- helpers ----------
    def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(order, start=1):
            adj[idx] = min(1.0, (m - rank + 1) * pvals[idx])
        return adj

    def p_to_stars(p: float) -> str:
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    def add_sig_bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2)
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")

    def load_all_for_one(tr: float, shape: str, target: int) -> List[np.ndarray]:
        """
        Returns a list [arr_pred0, arr_pred1, ...] where each arr contains ALL KLDs
        from `all_kls` across the requested repeats for (tr, shape, target).
        Falls back to mean_kls (as singletons) if all_kls is missing.
        """
        accum: List[List[float]] = []
        n_preds_local = None

        for rep in reps:
            path = os.path.join(base, f"{int(100*tr)}_{split}_shape_{shape}_target_{int(target)}_{int(rep)}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                res = pickle.load(f)

            if "all_kls" in res and res["all_kls"] is not None:
                lists = [np.asarray(x, dtype=float).ravel() for x in res["all_kls"]]
            elif "mean_kls" in res:
                lists = [np.asarray([v], dtype=float) for v in res["mean_kls"]]
            else:
                continue

            # if "mean_kls" in res:
            #     lists = [np.asarray([v], dtype=float) for v in res["mean_kls"]]
            # else:
            #     continue

            if n_preds_local is None:
                n_preds_local = len(lists)
                accum = [[] for _ in range(n_preds_local)]
            if len(lists) != n_preds_local:
                continue

            for j, arr in enumerate(lists):
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    accum[j].extend(arr.tolist())

        if not accum:
            return []
        return [np.asarray(a, dtype=float) for a in accum]

    # ---- detect n_preds & labels
    n_preds_detected = None
    for shp in shapes:
        for tr in trs:
            for tgt in targets:
                arrs = load_all_for_one(tr, shp, tgt)
                if arrs:
                    n_preds_detected = len(arrs)
                    break
            if n_preds_detected:
                break
        if n_preds_detected:
            break
    if n_preds_detected is None:
        raise FileNotFoundError("No KLD files found or `all_kls`/`mean_kls` missing.")

    if methods_idx is None:
        methods_idx = [0, 1, 2]
    if len(methods_idx) != 3 or any(m < 0 or m >= n_preds_detected for m in methods_idx):
        raise ValueError(f"`methods_idx` must be 3 valid indices within [0,{n_preds_detected-1}].")

    default_labels = [f"Pred {j+1}" for j in range(n_preds_detected)]
    labels_all = method_labels if (method_labels and len(method_labels) == n_preds_detected) else default_labels
    labels = [labels_all[j] for j in methods_idx]

    # colors for 3 predictors (paper-friendly)
    colors = ["#fc8d59", "#ffffbf", "#99d594"]

    # ---------- figure ----------
    n_tr = len(trs)
    n_shapes = len(shapes)
    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(n_shapes, 1, figsize=(max(8, 1.2 * n_tr * 3), 4.6 * n_shapes), sharex=True)
    if n_shapes == 1:
        axes = [axes]

    group_centers = np.arange(1, n_tr + 1, dtype=float)
    offset = 0.25

    # ---------- per shape subplot ----------
    for row, shp in enumerate(shapes):
        ax = axes[row]

        # For each tr: pool across ALL targets, per predictor
        per_tr_vals: Dict[float, List[np.ndarray]] = {}
        for tr in trs:
            per_tgt = [load_all_for_one(tr, shp, tgt) for tgt in targets]
            per_tgt = [pt for pt in per_tgt if pt]  # drop empties
            if not per_tgt:
                per_tr_vals[tr] = [np.array([]), np.array([]), np.array([])]
            else:
                pooled = []
                for m in methods_idx:
                    pooled.append(
                        np.concatenate([pt[m] for pt in per_tgt if len(pt) > m and pt[m].size], axis=0)
                    )
                per_tr_vals[tr] = pooled

        # ----- boxplots -----
        positions, box_data, box_colors = [], [], []
        for i, tr in enumerate(trs):
            vals = per_tr_vals[tr]
            if len(vals) != 3:
                vals = [np.array([]), np.array([]), np.array([])]
            posA, posB, posC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
            positions += [posA, posB, posC]
            box_data += [vals[0], vals[1], vals[2]]
            box_colors += colors

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            manage_ticks=False
        )
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set(facecolor=c, alpha=0.8)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=1)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=1)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.5)

        ax.set_ylabel("KLD")
        ax.set_title(f"Shape: {shp}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # legend only on top subplot
        if row == 0:
            handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
            ax.legend(handles, labels, title="Model", ncols=3, loc="upper right", frameon=False)

        # ---- significance (Mann–Whitney U) per tr ----
        for i, tr in enumerate(trs):
            A, B, C = per_tr_vals[tr]
            groups = [A, B, C]
            if any(g.size == 0 for g in groups):
                continue

            pairs = [(0, 1), (0, 2), (1, 2)]
            p_raw = []
            for (a, b) in pairs:
                U, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
                p_raw.append(p)
            p_adj = holm_bonferroni(np.array(p_raw, dtype=float))

            local_max = max(g.max() for g in groups if g.size)
            y0, y1 = ax.get_ylim(); span = y1 - y0
            base_y = local_max + 0.03 * span
            h = 0.02 * span

            xA, xB, xC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
            stack = 0
            for (a, b), p in zip(pairs, p_adj):
                stars = p_to_stars(p)
                if stars == "ns":
                    continue
                if   (a, b) == (0, 1): x1, x2 = xA, xB
                elif (a, b) == (0, 2): x1, x2 = xA, xC
                else:                  x1, x2 = xB, xC
                add_sig_bracket(ax, x1, x2, base_y + stack * (h * bar_spacing), h, stars)
                stack += 1

    # shared x-axis
    axes[-1].set_xticks(group_centers)
    axes[-1].set_xticklabels([f"{int(100*t)}%" for t in trs])
    axes[-1].set_xlabel("Timing Ratio (tr)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()



def plot_box_sig_predictors_vs_tr_agg_targets( # KLD plots aggregated across targets and shapes
    trs: Iterable[float],
    shapes: Iterable[str],
    targets: Iterable[int],
    reps: Iterable[int] = range(50),
    split: int = 60,
    base: str = "./KLDs",
    method_labels: Optional[List[str]] = None,
    methods_idx: Optional[List[int]] = None,
    fontsize: int = 12,
    save_path: Optional[str] = None,
    bar_spacing: float = 2.2,
    ylim: Optional[tuple] = (0, 32),
):
    trs     = list(trs); shapes  = list(shapes); targets = list(targets); reps = list(reps)

    def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
        order = np.argsort(pvals); m = len(pvals)
        adj = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(order, start=1):
            adj[idx] = min(1.0, (m - rank + 1) * pvals[idx])
        return adj

    def p_to_stars(p: float) -> str:
        return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

    def add_sig_hat(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="k")
        ax.text((x1 + x2) / 2, y + h + 0.002, text, ha="center", va="bottom")

    def load_all_for(tr: float, shape: str, target: int) -> List[np.ndarray]:
        accum: List[List[float]] = []; n_preds_local = None
        for rep in reps:
            path = os.path.join(base, f"{int(100*tr)}_{split}_shape_{shape}_target_{int(target)}_{int(rep)}.pkl")
            if not os.path.exists(path): continue
            with open(path, "rb") as f:
                res = pickle.load(f)
            if "all_kls" in res and res["all_kls"] is not None:
                lists = [np.asarray(x, dtype=float).ravel() for x in res["all_kls"]]
            elif "mean_kls" in res:
                lists = [np.asarray([v], dtype=float) for v in res["mean_kls"]]
            else:
                continue
            if n_preds_local is None:
                n_preds_local = len(lists)
                accum = [[] for _ in range(n_preds_local)]
            if len(lists) != n_preds_local:  # skip inconsistent
                continue
            for j, arr in enumerate(lists):
                arr = arr[~np.isnan(arr)]
                if arr.size: accum[j].extend(arr.tolist())
        if not accum: return []
        return [np.asarray(a, dtype=float) for a in accum]

    # detect n_preds & labels
    n_preds_detected = None
    for tr in trs:
        for shp in shapes:
            for tgt in targets:
                arrs = load_all_for(tr, shp, tgt)
                if arrs:
                    n_preds_detected = len(arrs); break
            if n_preds_detected: break
        if n_preds_detected: break
    if n_preds_detected is None:
        raise FileNotFoundError("No KLD files found or `all_kls`/`mean_kls` missing.")

    if methods_idx is None:
        methods_idx = [0, 1, 2]
    if len(methods_idx) != 3 or any(m < 0 or m >= n_preds_detected for m in methods_idx):
        raise ValueError(f"`methods_idx` must be 3 valid indices within [0,{n_preds_detected-1}].")

    default_labels = [f"Pred {j+1}" for j in range(n_preds_detected)]
    labels_all = method_labels if (method_labels and len(method_labels) == n_preds_detected) else default_labels
    labels = [labels_all[j] for j in methods_idx]
    colors = ["#fc8d59", "#ffffbf", "#99d594"]

    # pooled arrays per tr
    per_tr_vals: Dict[float, List[np.ndarray]] = {}
    for tr in trs:
        chunks = []
        for shp in shapes:
            for tgt in targets:
                arrs = load_all_for(tr, shp, tgt)
                if arrs: chunks.append(arrs)
        if not chunks:
            per_tr_vals[tr] = [np.array([]), np.array([]), np.array([])]
        else:
            pooled = []
            for m in methods_idx:
                pooled.append(np.concatenate([c[m] for c in chunks if len(c) > m and c[m].size], axis=0))
            per_tr_vals[tr] = pooled

    # figure
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(trs) * 3), 6.5))

    group_centers = np.arange(1, len(trs) + 1, dtype=float)
    offset = 0.25
    positions, box_data, box_colors = [], [], []
    for i, tr in enumerate(trs):
        vals = per_tr_vals[tr]
        if len(vals) != 3: vals = [np.array([]), np.array([]), np.array([])]
        posA, posB, posC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
        positions += [posA, posB, posC]
        box_data += [vals[0], vals[1], vals[2]]
        box_colors += colors

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.22,
        patch_artist=True, manage_ticks=False, showfliers=True
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set(facecolor=c, alpha=0.8)
    for whisker in bp["whiskers"]: whisker.set(color="black", linewidth=1)
    for cap in bp["caps"]: cap.set(color="black", linewidth=1)
    for median in bp["medians"]: median.set(color="black", linewidth=1.5)

    ax.set_ylabel("KLD")
    if ylim is not None: ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{int(100*t)}%" for t in trs])
    ax.set_xlabel("Timing (%)")

    # ---- significance ----
    for i, tr in enumerate(trs):
        A, B, C = per_tr_vals[tr]
        groups = [A, B, C]
        if any(g.size == 0 for g in groups): continue
        pairs = [(0,1),(0,2),(1,2)]
        p_raw = [mannwhitneyu(groups[a], groups[b], alternative="two-sided")[1] for (a,b) in pairs]
        p_adj = holm_bonferroni(np.array(p_raw, dtype=float))

        whisker_tops = [whisker.get_ydata()[1] for whisker in bp["whiskers"]]
        # local_max = np.nanmax(whisker_tops)
        local_max = max(g.max() for g in groups if g.size)
        y0, y1 = ax.get_ylim(); span = y1 - y0
        base_y = local_max + 0.03 * span; h = 0.02 * span
        xA, xB, xC = group_centers[i] - offset, group_centers[i], group_centers[i] + offset
        stack = 0
        for (a,b), p in zip(pairs, p_adj):
            stars = p_to_stars(p)
            if stars == "ns": continue
            if   (a,b)==(0,1): x1,x2 = xA,xB
            elif (a,b)==(0,2): x1,x2 = xA,xC
            else:              x1,x2 = xB,xC
            add_sig_hat(ax, x1, x2, base_y + stack * (h * bar_spacing), h, stars)
            stack += 1

    # ======== TOP-CENTER LEGEND (figure-level) ========
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in colors]
    # reserve space at top, then add legend above axes
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        handles, labels,
        title=None,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )
    # ================================================

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




def plot_kld_mean_by_alpha(  # kld mean +- std vs alpha (weights)
    base_dir="./KLDs_alpha",
    trs=(70, 80, 90),  # aggregate over these timing ratios
    split=60,
    alphas=(10,20,30,40,50,60,70,80,90),
    shapes=("triangle","circle","square","rectangle"),
    targets=(0,1,2,3),
    reps=range(5),
    ylabel="KLD",
    title=None,
    save_path=None,
    fontsize=16,          # base font size
    labelsize=18,         # axis labels
    ticksize=12,          # tick labels
    legendsize=13,        # legend font
    titlesize=18          # title font
):
    """
    Reads ./KLDs_alpha files, stacks all KLDs across reps, shapes, targets, 
    and timing ratios (trs), and plots the mean KLD vs α aggregated over trs.
    """
    pat = re.compile(
        r"^(\d+)_(\d+)_shape_([A-Za-z0-9\-]+)_target_(\d+)_(\d+)_alpha_(\d+)\.pkl$"
    )

    alphas = list(alphas)
    trs = list(trs)
    pools = {a: [] for a in alphas}

    for fp in glob.glob(os.path.join(base_dir, "*.pkl")):
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        tr_f, split_f, shape_f, targ_f, rep_f, alpha_f = m.groups()
        tr_f, split_f, targ_f, rep_f, alpha_f = map(int, [tr_f, split_f, targ_f, rep_f, alpha_f])

        # Only include selected timing ratios
        if tr_f not in trs or split_f != split:
            continue
        if shape_f not in shapes or targ_f not in targets or rep_f not in reps or alpha_f not in alphas:
            continue

        with open(fp, "rb") as f:
            res = pickle.load(f)
        if "all_kls" not in res or res["all_kls"] is None:
            continue

        arrays = []
        for arr in res["all_kls"]:
            a = np.asarray(arr, dtype=float).ravel()
            if a.size:
                arrays.append(a[~np.isnan(a)])
        if arrays:
            stacked = np.concatenate(arrays, axis=0)
            pools[alpha_f].append(stacked)

    mean_vals, std_vals, xlabels = [], [], []
    for a in sorted(pools.keys()):
        if pools[a]:
            all_vals = np.concatenate(pools[a])
            mean_vals.append(np.mean(all_vals))
            std_vals.append(np.std(all_vals))
        else:
            mean_vals.append(np.nan)
            std_vals.append(np.nan)
        xlabels.append(a / 100)

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    print(mean_vals)

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": labelsize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
        "legend.fontsize": legendsize,
        "axes.titlesize": titlesize
    })

    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(
        xlabels, mean_vals, yerr=std_vals, fmt='-o', color='#2b83ba', markersize=10,
        ecolor='gray', elinewidth=2, linewidth=4, capsize=4, label='Aggregated Mean ± SD'
    )

    ax.set_xlabel("α")
    ax.set_ylabel(ylabel)
    # if title is None:
    #     title = f"{ylabel} vs α (aggregated over trs={trs}, split={split}%)"
    # ax.set_title(title, pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    # # for f1 and correction MAE
    # plot_box_and_whisker_all()

    # # for predicted correction ratio
    # plot_box_and_whisker_counting_all()

 
    # for feature ablation
    colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]

    
    feature_names = [
        "Optimal Pos. Align.",
        "Optimal Vel. Align.",
        "Direct Vel. Align.",
        "Distance to Goal",
        "Velocity Consistency",
        "Legibility",
        "Optimality",
    ]
    plot_baseline_vs_ablation_boxplots_vertical_stacked_trs(
        trs=[70, 80, 90, 100],
        split=60,
        repeats=range(200),
        n_features=7,
        base_dir="./results/evaluation_com",
        ablation_dir="./results/evaluation_ablation_com",
        feature_names=feature_names,
        colors = colors,
        save_path="/Users/anjiabei/Documents/research/figures/goal_infer_paper/f1_boxplots_by_tr.png"
    )


    # # for KLD aggregated across targets
    # plot_box_sig_predictors_vs_tr_stacked(
    #     trs=[0.7, 0.8, 0.9, 1],
    #     shapes=["circle","square","triangle", "rectangle"],
    #     targets=[0,1,2,3],
    #     reps=range(50),
    #     split=60,
    #     base="./KLDs",
    #     method_labels=["WHEN","WHERE","COMBINED"],  # full list; we’ll pick 3 below
    #     methods_idx=[0,1,2],           # pick exactly 3 predictors to compare
    #     # agg_across_shapes="concat",    # or "mean"
    #     fontsize=16,
    #     save_path="/Users/anjiabei/Documents/research/figures/goal_infer_paper/kld_tr_stacked.png"
    # )


    # # for KLD aggregated across shapes
    # plot_box_sig_predictors_vs_tr_stacked_by_shape(
    #     trs=[0.7, 0.8, 0.9, 1],
    #     shapes=["circle","square","triangle", "rectangle"],
    #     targets=[0,1,2,3],
    #     reps=range(50),
    #     split=60,
    #     base="./KLDs",
    #     method_labels=["WHEN","WHERE","COMBINED"],  # full list; we’ll pick 3 below
    #     methods_idx=[0,1,2],           # pick exactly 3 predictors to compare
    #     # agg_across_shapes="concat",    # or "mean"
    #     fontsize=14,
    #     save_path="/Users/anjiabei/Documents/research/figures/goal_infer_paper/kld_tr_stacked_by_shape.png"
    # )


    # # for KLD aggregated across targets and shapes
    # plot_box_sig_predictors_vs_tr_agg_targets(
    #     trs=[0.7, 0.8, 0.9, 1],
    #     shapes=["circle","square","triangle", "rectangle"],
    #     targets=[0,1,2,3],
    #     reps=range(50),
    #     split=60,
    #     base="./KLDs_leaving",
    #     method_labels=["WHEN","WHERE","COMBINED"],  # full list; we’ll pick 3 below
    #     methods_idx=[0,1,2],           # pick exactly 3 predictors to compare
    #     # agg_across_shapes="concat",    # or "mean"
    #     fontsize=18,
    #     save_path="/Users/anjiabei/Documents/research/figures/goal_infer_paper/kld_tr_aggre_target_leaving.png"
    # )



    # # for KLD mean vs alpha
    # plot_kld_mean_by_alpha(
    #     base_dir="./KLDs_alpha",
    #     trs=([80]),
    #     split=60,
    #     alphas=(10,20,30,40,50,60,70,80,90),
    #     shapes=("circle","triangle", "rectangle", "square"),
    #     ylabel="KLD",
    #     save_path="/Users/anjiabei/Documents/research/figures/goal_infer_paper/kld_mean_vs_alpha_tr80.png"
    # )