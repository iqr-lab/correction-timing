import numpy as np

# Define ranges
q_vals = np.arange(-0.2, 0.2 + 0.01, 0.01)
r_vals = np.arange(-0.3, 0.3 + 0.01, 0.01)

# Open a file to write the task list
with open("tasklist_feature_offset.sh", "w") as f:
    for q in q_vals:
        for r in r_vals:
            cmd = f"python3 extract_features_parallel.py -tr 0.7 -x {q:.2f} -y {r:.2f};\n"
            f.write(cmd)

print("Task list saved to task_list.sh")

# import math

# # Path to your original task list
# tasklist_file = "tasklist_feature_offset.sh"

# # Load all lines
# with open(tasklist_file, "r") as f:
#     lines = [line.strip() for line in f if line.strip()]

# num_chunks = 20
# chunk_size = math.ceil(len(lines) / num_chunks)

# for i in range(num_chunks):
#     chunk_lines = lines[i*chunk_size : (i+1)*chunk_size]
#     chunk_filename = f"tasklist_chunk_{i+1}.sh"
#     with open(chunk_filename, "w") as f_chunk:
#         f_chunk.write("\n".join(chunk_lines))
#     print(f"Created {chunk_filename} with {len(chunk_lines)} tasks")