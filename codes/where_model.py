import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse

# train MLP to infer where people leave the gripper from where they grab the gripper. 
# (relative - always start at 0 position, use vel to predict where it will end up and add starting point back to it later to avoid leaarning colors)

# -------- Dataset utilities --------

def filter_data(data):

    selected_data = []
    for i in range(len(data)):
        if data[i]["features"] != False: # important for training
            selected_data.append(data[i])
    print(len(selected_data))

    return selected_data


def load_data(timing_ratio, split, rep):

    # List of target numbers you have in the file names
    target_nums = [0, 1, 2, 3]  # adjust based on your files
    shapes = ["circle", "rectangle", "triangle", "square"]
    # target_nums = [3]

    train_idx_all = []
    val_idx_all = []
    test_idx_all = []

    for target in target_nums:
        for shape in shapes:
            filename = f'../splits/indices_{int(100*timing_ratio)}_{int(100*split)}_shape_{shape}_target_{int(target)}_{int(rep)}.pkl'
            
            with open(filename, 'rb') as f:
                idx_dict = pickle.load(f)
            
            train_idx_all.extend(idx_dict["train"])
            val_idx_all.extend(idx_dict["val"])
            test_idx_all.extend(idx_dict["test"])

    with open('../features/corrected_features_'+str(int(100*timing_ratio))+'.pkl', 'rb') as file:
        all_data_pre = pickle.load(file)
    all_data = filter_data(all_data_pre)  # corrected data has features = False, needs to be filtered before indexing

    training_data = [all_data[i] for i in train_idx_all]
    val_data = [all_data[i] for i in val_idx_all]
    test_data = [all_data[i] for i in test_idx_all]

    train_x, train_y = [], []
    for d in training_data:
        # if not d["corrected"]:
        #     continue
        pose_start = np.array(d["correction_pose_list"][0][:3], dtype=np.float32)
        vel = np.array(np.array(d["correction_pose_list"][1][:3]) - np.array(d["correction_pose_list"][0][:3]), dtype=np.float32)
        pose_target = np.array(d["correction_pose_list"][-1][:3], dtype=np.float32)
        # train_x.append(np.concatenate([pose_start, vel]))
        # train_y.append(pose_target)

        # --- Change here ---
        # Input: zero pose start, keep vel
        train_x.append(np.concatenate([np.zeros(3, dtype=np.float32), vel]))

        # Output: relative target
        train_y.append(pose_target - pose_start)

    val_x, val_y = [], []
    for d in val_data:
        # if not d["corrected"]:
        #     continue
        pose_start = np.array(d["correction_pose_list"][0][:3], dtype=np.float32)
        vel = np.array(np.array(d["correction_pose_list"][1][:3]) - np.array(d["correction_pose_list"][0][:3]), dtype=np.float32)
        pose_target = np.array(d["correction_pose_list"][-1][:3], dtype=np.float32)
        # val_x.append(np.concatenate([pose_start, vel]))
        # val_y.append(pose_target)

        # --- Change here ---
        # Input: zero pose start, keep vel
        val_x.append(np.concatenate([np.zeros(3, dtype=np.float32), vel]))

        # Output: relative target
        val_y.append(pose_target - pose_start)

    test_x, test_y = [], []
    for d in test_data:
        # if not d["corrected"]:
        #     continue
        pose_start = np.array(d["correction_pose_list"][0][:3], dtype=np.float32)
        vel = np.array(np.array(d["correction_pose_list"][1][:3]) - np.array(d["correction_pose_list"][0][:3]), dtype=np.float32)
        pose_target = np.array(d["correction_pose_list"][-1][:3], dtype=np.float32)
        # test_x.append(np.concatenate([pose_start, vel]))
        # test_y.append(pose_target)

        # --- Change here ---
        # Input: zero pose start, keep vel
        test_x.append(np.concatenate([np.zeros(3, dtype=np.float32), vel]))

        # Output: relative target
        test_y.append(pose_target - pose_start)

    return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32), np.array(val_x, dtype=np.float32), np.array(val_y, dtype=np.float32), np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)


# -------- Helpers (only used inside train/inference) --------
def normalize(X, y):
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    return X_norm, y_norm, (X_mean, X_std, y_mean, y_std)

def set_seed(seed=42):
    random.seed(seed)                 # Python RNG
    np.random.seed(seed)              # NumPy RNG
    torch.manual_seed(seed)           # PyTorch RNG (CPU)
    torch.cuda.manual_seed(seed)      # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(input_dim, output_dim, hidden_dim = 64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


# -------- Training --------
def train_model(train_x, train_y, val_x, val_y, epochs, lr, save_path, seed):
    """
    Train a model with validation and save the best model based on lowest val loss.

    Args:
        train_x (np.array): Training features
        train_y (np.array): Training targets
        val_x (np.array): Validation features
        val_y (np.array): Validation targets
        epochs (int): Number of training epochs
        lr (float): Learning rate
        save_path (str): Path to save model and normalization stats
    """

    set_seed(seed)

    # --- Normalize based on training set ---
    X_norm, y_norm, stats = normalize(train_x, train_y)
    train_x_tensor = torch.tensor(X_norm, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    val_x_tensor = torch.tensor((val_x - stats[0]) / stats[1], dtype=torch.float32)
    val_y_tensor = torch.tensor((val_y - stats[2]) / stats[3], dtype=torch.float32)

    print(f"Training size: {train_x_tensor.shape[0]} | Validation size: {val_x_tensor.shape[0]}")
    print(f"Input dim: {train_x_tensor.shape[1]} | Output dim: {train_y_tensor.shape[1]}")

    # --- Build model ---
    model = build_model(train_x_tensor.shape[1], train_y_tensor.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')  # Initialize best val loss

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        optimizer.zero_grad()
        preds = model(train_x_tensor)
        loss = criterion(preds, train_y_tensor)
        loss.backward()
        optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x_tensor)
            val_loss = criterion(val_preds, val_y_tensor)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # --- Save model if validation improves ---
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                "model_state": model.state_dict(),
                "X_mean": stats[0],
                "X_std": stats[1],
                "y_mean": stats[2],
                "y_std": stats[3]
            }, save_path)
            print(f"Validation improved. Model saved to {save_path}")

    print("Training complete.")


# -------- Inference --------
def run_inference(X, y, save_path):
    checkpoint = torch.load(save_path, weights_only=False)

    # Build model
    model = build_model(X.shape[1], y.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Normalize using saved stats
    X_mean, X_std = checkpoint["X_mean"], checkpoint["X_std"]
    y_mean, y_std = checkpoint["y_mean"], checkpoint["y_std"]

    X_norm = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        preds_norm = model(X_tensor)
        preds = preds_norm.numpy() * y_std + y_mean
        y_true = y

    print("Inference completed. Predictions vs Ground Truth (20 random samples):")
    idxs = random.sample(range(len(X)), min(20, len(X)))

    print(f"{'Sample':<8}{'Prediction':<40}{'Ground Truth':<40}")
    print("-" * 90)
    for i in idxs:
        pred_str = "[" + ", ".join(f"{v:.4f}" for v in preds[i]) + "]"
        gt_str   = "[" + ", ".join(f"{v:.4f}" for v in y_true[i]) + "]"
        print(f"{i:<8}{pred_str:<40}{gt_str:<40}")


# -------- Main --------
def main():
    parser = argparse.ArgumentParser(description="Train or run inference with MLP model")
    parser.add_argument("-m", "--mode", type=str, choices=["train", "inference"], default="train", help="Run mode")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-r','--repeat', type = int)

    args = parser.parse_args()

    train_x, train_y, val_x, val_y, test_x, test_y = load_data(args.timing_ratio, args.split, args.repeat)
    save_path = f"../goal_infer_files/where_infer/model_{str(int(100*args.timing_ratio))}_{str(int(100*args.split))}_{str(int(args.repeat))}.pt"

    if args.mode == "train":
        train_model(train_x, train_y, val_x, val_y, args.epochs, args.lr, save_path, args.repeat)
    else:
        run_inference(test_x, test_y, save_path)


if __name__ == "__main__":
    main()
