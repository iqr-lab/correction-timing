import pickle
import random
import numpy as np



KEYS_TO_KEEP = [
    "shape", "target", "success", "corrected",
    "pre_pose_list", "pre_timestamp", "pre_pose_vel",
    "correction_pose_list", "fake_correction", "post_pose_list", "entire_pose_list"
]

TRAJ_KEYS = [
    "pre_pose_list",
    "pre_pose_vel",
    "correction_pose_list",
    "post_pose_list",
    "entire_pose_list",
]

# get rid of unused keys
def pick_keys():

    with open("../source/corl_data.pkl", "rb") as f:
        obj = pickle.load(f)


    KEYS_TO_KEEP = ['shape', 'target',  'success', 'corrected', 
                    'pre_pose_list', 'pre_timestamp', 'pre_pose_vel', 
                    'correction_pose_list', 'fake_correction', 'post_pose_list',  'entire_pose_list'
    ]

    data = [
        {k: d[k] for k in KEYS_TO_KEEP if k in d}
        for d in obj
    ]

    with open('../source/data_keys.pkl', 'wb') as f:
        pickle.dump(data, f)





def to_xyz_list(traj_list):
    """
    traj_list: list of arrays/lists with >=3 dims (often 6: xyzrpy)
    returns: list of length T, each a python list [x,y,z]
    """
    out = []
    for p in traj_list:
        p = np.asarray(p).reshape(-1)
        out.append(p[:3].astype(float).tolist())
    return out

def add_noise_xyz_list(xyz_list, noise_type="uniform", noise_min=0.01, noise_max=0.1, per_traj_offset=True):
    """
    xyz_list: list of [x,y,z]
    Adds noise in meters.
    per_traj_offset=True applies a single offset to the whole trajectory (often looks realistic).
    """
    xyz = np.asarray(xyz_list, dtype=float)  # (T,3)

    # pick a noise scale per trajectory (you can make this per-episode if you prefer)
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

    xyz_noisy = xyz + noise
    return xyz_noisy.tolist()

def sanitize_episode(ep, noise_type="uniform", noise_min=0.01, noise_max=0.1, per_traj_offset=True):
    """
    Returns a sanitized COPY of one episode dict:
    - keeps all non-trajectory keys exactly the same
    - for trajectory keys: xyz only + noise
    """
    new_ep = {k: ep.get(k, None) for k in KEYS_TO_KEEP}  # keep only these keys

    for k in TRAJ_KEYS:
        traj = new_ep.get(k, None)
        if not isinstance(traj, (list, tuple)) or len(traj) == 0:
            continue
        
        # convert to xyz and add noise
        xyz_list = to_xyz_list(new_ep[k])
        new_ep[k] = add_noise_xyz_list(
            xyz_list,
            noise_type=noise_type,
            noise_min=noise_min,
            noise_max=noise_max,
            per_traj_offset=per_traj_offset
        )

    return new_ep

def make_example_data(in_pkl, out_pkl, n_samples=100, seed=42,
                      noise_type="uniform", noise_min=0.01, noise_max=0.1,
                      per_traj_offset=True):
    random.seed(seed)
    np.random.seed(seed)

    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or (len(data) and not isinstance(data[0], dict)):
        raise TypeError("Expected the pkl to contain a list of dicts.")

    if n_samples > len(data):
        raise ValueError(f"Requested {n_samples} samples but only have {len(data)} entries.")

    idxs = random.sample(range(len(data)), n_samples)
    sampled = [data[i] for i in idxs]

    example = [
        sanitize_episode(ep, noise_type=noise_type, noise_min=noise_min, noise_max=noise_max,
                         per_traj_offset=per_traj_offset)
        for ep in sampled
    ]

    with open(out_pkl, "wb") as f:
        pickle.dump(example, f)

    # quick sanity check
    print(f"Saved {len(example)} episodes -> {out_pkl}")
    e0 = example[0]
    print("Keys:", list(e0.keys()))
    for k in TRAJ_KEYS:
        if e0.get(k):
            print(k, "T=", len(e0[k]), "dim=", len(e0[k][0]))


if __name__ == "__main__":

    # with open("../source/corl_data.pkl", "rb") as f:
    # with open("../source/rescaled_traj.pkl", "rb") as f: 
    # with open("../source/data_keys.pkl", "rb") as f:
    with open("../source/example_data.pkl", "rb") as f:
        obj = pickle.load(f)

    # print(type(obj))
    print(len(obj))
    print(obj[0].keys()) # only corrected traj has cor_pose_list
    print(obj[10]['entire_pose_list'])

    # pick_keys()
    # make_example_data('../source/data_keys.pkl', '../source/example_data.pkl', n_samples=100,
    #                   noise_type="gaussian", per_traj_offset=False)