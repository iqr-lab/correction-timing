import pickle
import random
import numpy as np
from collections import defaultdict, Counter



KEYS_TO_KEEP = [
    "shape", "target", "success", "corrected",
    "pre_pose_list", "pre_timestamp", 
    "correction_pose_list", "fake_correction", "post_pose_list", "entire_pose_list"
]

TRAJ_KEYS = [
    "pre_pose_list",
    "correction_pose_list",
    "post_pose_list",
    "entire_pose_list",
]


DEFAULT_SHAPES = ["circle", "triangle", "square", "rectangle"]
DEFAULT_TARGETS = [0, 1, 2, 3]

# get rid of unused keys
def pick_keys():

    with open("../../enhancing_goal_inference_via_correction_timing_codes_data_source/corl_data.pkl", "rb") as f:
        obj = pickle.load(f)


    KEYS_TO_KEEP = ['shape', 'target',  'success', 'corrected', 
                    'pre_pose_list', 'pre_timestamp',
                    'correction_pose_list', 'fake_correction', 'post_pose_list',  'entire_pose_list'
    ]

    data = [
        {k: d[k] for k in KEYS_TO_KEEP if k in d}
        for d in obj
    ]

    with open('../../enhancing_goal_inference_via_correction_timing_codes_data_source/data_keys.pkl', 'wb') as f:
        pickle.dump(data, f)





# ----------------------------
# Trajectory sanitization
# ----------------------------
def to_xyz_list(traj_list):
    """
    traj_list: list of arrays/lists with >=3 dims (often 6: xyzrpy)
    returns: list of python lists [x,y,z]
    """
    out = []
    for p in traj_list:
        p = np.asarray(p).reshape(-1)
        if p.size < 3:
            raise ValueError("Trajectory point has <3 dims; cannot extract xyz.")
        out.append(p[:3].astype(float).tolist())
    return out


def add_noise_xyz_list(
    xyz_list,
    noise_type="uniform",
    noise_min=0.01,
    noise_max=0.1,
    per_traj_offset=True,
):
    """
    xyz_list: list of [x,y,z]
    Adds noise in meters. If per_traj_offset=True, applies one constant xyz offset
    to the entire trajectory (often looks more realistic than pointwise jitter).
    """
    xyz = np.asarray(xyz_list, dtype=float)  # (T,3)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_list must be T x 3")

    noise_m = float(np.random.uniform(noise_min, noise_max))

    if noise_type == "uniform":
        if per_traj_offset:
            offset = np.random.uniform(-noise_m, noise_m, size=(1, 3))
            noise = np.repeat(offset, xyz.shape[0], axis=0)
        else:
            noise = np.random.uniform(-noise_m, noise_m, size=xyz.shape)
    elif noise_type == "gaussian":
        if per_traj_offset:
            offset = np.random.normal(0.0, noise_m, size=(1, 3))
            noise = np.repeat(offset, xyz.shape[0], axis=0)
        else:
            noise = np.random.normal(0.0, noise_m, size=xyz.shape)
    else:
        raise ValueError("noise_type must be 'uniform' or 'gaussian'")

    return (xyz + noise).tolist()


def sanitize_episode(
    ep,
    noise_type="uniform",
    noise_min=0.01,
    noise_max=0.1,
    per_traj_offset=True,
):
    """
    Returns sanitized COPY:
    - keeps non-trajectory keys unchanged
    - trajectory keys -> xyz only + noise
    Missing/empty traj keys are left as None or [] (unchanged structure).
    """
    # Keep only the keys you want to publish
    new_ep = {k: ep.get(k, None) for k in KEYS_TO_KEEP}

    # Only transform trajectory keys
    for k in TRAJ_KEYS:
        traj = new_ep.get(k, None)
        if not isinstance(traj, (list, tuple)) or len(traj) == 0:
            continue

        xyz_list = to_xyz_list(traj)
        new_ep[k] = add_noise_xyz_list(
            xyz_list,
            noise_type=noise_type,
            noise_min=noise_min,
            noise_max=noise_max,
            per_traj_offset=per_traj_offset,
        )

    return new_ep


# ----------------------------
# Stratified sampling
# ----------------------------
def stratified_sample(
    data,
    n_per_cell=10,
    seed=42,
    shapes=DEFAULT_SHAPES,
    targets=DEFAULT_TARGETS,
    corrected_values=(True, False),
):
    """
    Sample n_per_cell per (corrected, shape, target).
    Raises ValueError if any cell is insufficient.
    """
    random.seed(seed)

    buckets = defaultdict(list)

    for ep in data:
        c = ep.get("corrected", None)
        s = ep.get("shape", None)
        t = ep.get("target", None)

        # strict corrected bool handling
        if c is True:
            c_key = True
        elif c is False:
            c_key = False
        else:
            continue

        if not isinstance(s, str):
            continue
        s_key = s.strip().lower()

        try:
            t_key = int(t)
        except Exception:
            continue

        if c_key in corrected_values and s_key in shapes and t_key in targets:
            buckets[(c_key, s_key, t_key)].append(ep)

    sampled = []
    missing = []
    for c in corrected_values:
        for s in shapes:
            for t in targets:
                cell = buckets.get((c, s, t), [])
                if len(cell) < n_per_cell:
                    missing.append((c, s, t, len(cell)))
                else:
                    sampled.extend(random.sample(cell, n_per_cell))

    if missing:
        lines = ["Not enough episodes for some (corrected, shape, target) cells:"]
        for c, s, t, have in missing:
            lines.append(f"  corrected={c}, shape={s}, target={t}: have {have}, need {n_per_cell}")
        raise ValueError("\n".join(lines))

    random.shuffle(sampled)
    return sampled


# ----------------------------
# Combined: make example data
# ----------------------------
def make_example_data_stratified(
    in_pkl,
    out_pkl,
    n_per_cell=10,
    seed=42,
    noise_type="uniform",
    noise_min=0.01,
    noise_max=0.1,
    per_traj_offset=True,
    shapes=DEFAULT_SHAPES,
    targets=DEFAULT_TARGETS,
):
    """
    Creates example pkl with stratified sampling and sanitized trajectories.
    Total episodes = 2 * len(shapes) * len(targets) * n_per_cell
                  = 2 * 4 * 4 * n_per_cell = 32 * n_per_cell
    With n_per_cell=10 -> 320 episodes.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or (len(data) and not isinstance(data[0], dict)):
        raise TypeError("Expected the pkl to contain a list of dicts.")

    sampled = stratified_sample(
        data,
        n_per_cell=n_per_cell,
        seed=seed,
        shapes=[s.lower() for s in shapes],
        targets=targets,
        corrected_values=(True, False),
    )

    example = [
        sanitize_episode(
            ep,
            noise_type=noise_type,
            noise_min=noise_min,
            noise_max=noise_max,
            per_traj_offset=per_traj_offset,
        )
        for ep in sampled
    ]

    with open(out_pkl, "wb") as f:
        pickle.dump(example, f)

    # sanity checks
    total_expected = 2 * len(shapes) * len(targets) * n_per_cell
    print(f"Saved {len(example)} episodes -> {out_pkl} (expected {total_expected})")

    ctr = Counter((ep.get("corrected"), ep.get("shape"), int(ep.get("target"))) for ep in example)
    print("Cell count min/max:", min(ctr.values()), max(ctr.values()))
    print("Corrected totals:",
          sum(ep.get("corrected") is True for ep in example),
          sum(ep.get("corrected") is False for ep in example))

    # quick traj shape check
    e0 = example[0]
    for k in TRAJ_KEYS:
        traj = e0.get(k, None)
        if isinstance(traj, list) and len(traj) > 0:
            print(f"{k}: T={len(traj)}, dim={len(traj[0])}")
            break


# # get rid of the velocity key
# def no_vel():

#     with open("../config/example_data.pkl", "rb") as f:
#         obj = pickle.load(f)


#     KEYS_TO_KEEP = ['shape', 'target',  'success', 'corrected', 
#                     'pre_pose_list', 'pre_timestamp', 
#                     'correction_pose_list', 'fake_correction', 'post_pose_list',  'entire_pose_list'
#     ]

#     data = [
#         {k: d[k] for k in KEYS_TO_KEEP if k in d}
#         for d in obj
#     ]

#     with open('../config/example_data.pkl', 'wb') as f:
#         pickle.dump(data, f)


if __name__ == "__main__":

    # with open("../../enhancing_goal_inference_via_correction_timing_codes_data_source/corl_data.pkl", "rb") as f:
    # with open("../../enhancing_goal_inference_via_correction_timing_codes_data_source/rescaled_traj.pkl", "rb") as f: 
    # with open("../../enhancing_goal_inference_via_correction_timing_codes_data_source/data_keys.pkl", "rb") as f:
    with open("../config/example_data_rescaled.pkl", "rb") as f:
        obj = pickle.load(f)
    
    # # shuffle
    # random.seed(42)
    # random.shuffle(obj)

    # print(type(obj))
    ind = 1
    print(len(obj)) #(16x50x2)
    print(obj[ind].keys()) # only corrected traj has cor_pose_list
    print(obj[ind]['corrected'])
    print(obj[ind]['shape'])
    # print(obj[ind]["correction_pose_list"])
    # print(obj[ind]["pre_pose_list"])
    # print(obj[ind]["pre_timestamp"])
    # print(obj[ind]['entire_pose_list'])
    print(obj[ind]['post_pose_list'])
    print(obj[ind]['fake_correction'])
    # with open('../config/example_data.pkl', 'wb') as f:
    #     pickle.dump(obj, f)

    # pick_keys()
    # make_example_data('../../enhancing_goal_inference_via_correction_timing_codes_data_source/data_keys.pkl', 
    #                   '../config/example_data.pkl', n_samples=100,
    #                   noise_type="gaussian", per_traj_offset=False)
    # no_vel()

    # make_example_data_stratified(
    #     in_pkl="../../enhancing_goal_inference_via_correction_timing_codes_data_source/data_keys.pkl",
    #     out_pkl="../config/example_data.pkl",
    #     n_per_cell=50,          # -> 320 total
    #     seed=42,
    #     noise_type="gaussian",   # or "gaussian"
    #     noise_min=0.01,
    #     noise_max=0.1,
    #     per_traj_offset=False,   # offset for each waypoint
    # )