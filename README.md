# Enhancing Goal Inference via Correction Timing

This repository contains code for generating example data, extracting features, training models, and evaluating goal inference using correction timing in human–robot interaction.

---

## Example Data

We provide example data that can be used to reproduce results with the released code.

- **800 trajectories with corrections**
- **800 trajectories without corrections**
- 4 shapes × 4 targets
- 50 trajectories per (shape, target) condition
- Balanced corrected / uncorrected

Saved as:
```bash
config/example_data.pkl
```

Adding Correction Timing to Pre-Planned Trajectories
```bash
python3 rescaled_correction_timing.py
```
Output:
```bash
config/example_data_rescaled.pkl
```

## Feature Extraction

```bash
python3 extract_features.py -c {corrected} -tr {timing ratio}
```
Arguments
-c True → corrected data
-c False → uncorrected data (use -tr 1 for uncorrected)
-tr 0.n → timing ratio (0.7-1)
Output directory:
```bash
features/
```

## Informative Sample Selection

To obtain the most informative samples using entropy:
```bash
python3 cal_entropy.py -sh {shape}
```
Output:
```bash
dropping_blocks/samples/
```
These indices correspond to simulator poses used to drop blocks in real-world experiments.
They allow comparison between real-world data and multiple simulators to determine which simulator best approximates real behavior.

## Simulator Comparison

To determine which simulator best matches real-world data:
```bash
python3 pick_sim.py -sh {shape}
```

Ground Truth GMM from Best Simulator
```bash
python3 best_sim_gmm.py -sh {shape}
```
Saved in:
```bash
goal_infer_files/gmms/ground_truth/
```

## Data Splitting

Generate train(60%)/validation(10%)/test(30%) splits:
```bash
python3 split_data.py -s {split} -tr {timing ratio} -sh {shape} -tg {target} -r {rep}
```
Arguments
-tr → timing ratio (0.7–1)
-s → training fraction of the splits (in our project s = 0.6)
-tg → target index (0–3)
-sh → shape
-r → repetition index (0–9), corresponds to the splits

Run all conditions:
```bash
bash tasklist.sh
```
Saved in:
```bash
splits/
```

## WHERE Model (Release Position Inference)

Train model to infer release position from grasp (group all shapes and targets together):
```bash
python3 where_model.py -s {split = 0.6} -tr {timing ratio} -r {rep}
```
Run all conditions:
```bash
bash tasklist_where_infer.sh
```
Saved in:
```bash
goal_infer_files/where_infer/
```

## GMM for P(Where | Goal)

Train GMM models for where people release the gripper based on goals:
```bash
python3 gmm_training.py -s {split = 0.6} -tr {timing ratio} -r {rep}
```
Run all conditions:
```bash
bash tasklist_gmm.sh
```
Saved in:
```bash
goal_infer_files/gmms/leaving/
```

## WHEN Model (Transformer – All Features)

Train transformer using all features:
```bash
python3 transformer_training_all_feature.py -s {split = 0.6} -tr {timing ratio} -r {rep}
```
Run all conditions:
```bash
bash tasklist_transformer.sh
```
Saved in:
```bash
goal_infer_files/when_weights/
```

## WHEN Model (Boltzmann Feature Only)

```bash
python3 transformer_bol.py -s {split = 0.6} -tr {timing ratio} -r {r}
```
Run all conditions:
```bash
bash tasklist_transformer_bol.sh
```
Saved in:
```bash
goal_infer_files/when_weights_bol/
```

## Ablation Studies (Feature Removal)

Train transformer with one feature removed:
```bash
python3 transformer_ablation.py -s {split = 0.6} -tr {timing ratio} -r {rep} -d {removed feature index}
```
Run all conditions:
```bash
bash tasklist_transformer_ablation.sh
```
Saved in:
```bash
goal_infer_files/when_weights_ablation/
```

## Feature Extraction with Goal Sampling

Extract features for sampled goal offsets:
```bash
python3 extract_features_xysampling.py -tr {timing ratio} -x {x_offset} -y {y_offset}
```
-x → x offset that can be defined by desired sample space and resolution
-y → y offset that can be defined by desired sample space and resolution

Generate task lists:
```bash
python3 generate_tasklist.py
bash tasklist_feature_offset.sh
```
Saved in:
```bash
features/sampled/
```

## Goal Inference Evaluation (KLD)

Compute KLD between ground truth and: **PWHEN, PWHERE, PCOMBINED**
```bash
python3 goal_infer.py -s {split = 0.6} -tr {timing ratio} -sh {shape} -tg {target} -r {rep} -t {type}
```
Arguments
-t grasp → infer release from grasp and infer goal
-t release → infer goal from release

Run all conditions:
```bash
bash tasklist_goal_infer.sh
```
Saved in:
```bash
results/KLDs/
results/KLDs_leaving/
```

## Alpha Weight Sensitivity Analysis

Evaluate effect of weighting parameter α:
```bash
python3 goal_infer_alpha.py -s {split = 0.6} -tr {timing ratio} -sh {shape} -tg {target} -r {rep} -a {alpha} -t {grasp}
```
Run all conditions:
```bash
bash tasklist_goal_infer_alpha.sh
```
Saved in:
```bash
results/KLDs_alpha/
```

## Notes

Example data differs from original experimental data and has been sanitized for release.

All bash tasklists automate full condition sweeps.

Outputs are organized by model type and experiment condition.