import numpy as np

# Define ranges
# q_vals = np.arange(-0.2, 0.2 + 0.01, 0.01)
# r_vals = np.arange(-0.3, 0.3 + 0.01, 0.01)

# make sample space smaller for example data
q_vals = np.arange(-0.2, 0.2 + 0.1, 0.1)
r_vals = np.arange(-0.3, 0.3 + 0.15, 0.15)

# Open a file to write the task list
with open("tasklist_feature_offset.sh", "w") as f:
    for q in q_vals:
        for r in r_vals:
            for tr in [0.7, 0.8, 0.9, 1]:
                cmd = f"python3 extract_features_xysampling.py -tr {tr} -x {q:.2f} -y {r:.2f};\n" # change -tr from 0.7 to 1; formatting for floats
                f.write(cmd)

print("Task list saved to tasklist_feature_offset.sh")

