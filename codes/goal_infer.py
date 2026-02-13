from tensorflow.keras.models import load_model
import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
# import shap
from tensorflow.keras.layers import LayerNormalization, Attention
from scipy.stats import zscore
import argparse
from tensorflow.keras.callbacks import Callback
# from tf_explain.core.integrated_gradients import IntegratedGradients
import joblib
import yaml
# from tensorflow.keras.models import load_model
# from keras.layers import TFSMLayer
from sklearn.mixture import GaussianMixture
import copy
import torch
import torch.nn as nn
from scipy import stats
from itertools import combinations
import bz2

# infer goal with pwhen, pwhere and combining pwhen and pwhere

class pgoal():
    def __init__(self, timing_ratio, split, target, shape, rep, t):

        # goal positions
        # with open('../../experiment/config/target_position.yaml','r') as file:
        with open('../config/target_position.yaml','r') as file:

            self.target_position = yaml.safe_load(file)
            
        c_gs = np.array([self.target_position['circle']['size_1'][:3],self.target_position['circle']['size_2'][:3],
                        self.target_position['circle']['size_3'][:3],self.target_position['circle']['size_4'][:3]])
        s_gs = np.array([self.target_position['square']['size_1'][:3],self.target_position['square']['size_2'][:3],
                        self.target_position['square']['size_3'][:3],self.target_position['square']['size_4'][:3]])
        t_gs = np.array([self.target_position['triangle']['size_1'][:3],self.target_position['triangle']['size_2'][:3],
                        self.target_position['triangle']['size_3'][:3],self.target_position['triangle']['size_4'][:3]])
        r_gs = np.array([self.target_position['rectangle']['size_1'][:3],self.target_position['rectangle']['size_2'][:3],
                        self.target_position['rectangle']['size_3'][:3],self.target_position['rectangle']['size_4'][:3]])
        
        self.gs = {'circle':c_gs,'square':s_gs,'triangle':t_gs,'rectangle':r_gs}



        all_pdfs = []       # list of lists, outer: goals, inner: trajectories
        # all_pdfs_ground = []
        all_puwhens = []   # same structure
        sampled_goals = []  # list of sampled goal offsets

        self.timing_ratio = timing_ratio
        self.split = split
        self.target = target
        self.shape = shape
        self.rep = rep
        self.t = t

        with open('../splits/indices_'+str(int(100*self.timing_ratio))+
                  '_'+str(int(100*self.split))+'_shape_'+str(self.shape)+'_target_'+str(self.target)+'_'+str(int(self.rep))+'.pkl', 'rb') as f:  # shape and target
            idx_dict = pickle.load(f)
        train_idx = idx_dict["train"]
        test_idx = idx_dict["test"]
        print("train length =", len(train_idx), " test length =", len(test_idx))


        # i_vals = np.arange(-0.2, 0.2 + 0.01, 0.01)
        # j_vals = np.arange(-0.3, 0.3 + 0.01, 0.01)
        # k_vals = [0]
    
        # make sample space smaller for example data, can change based on needs
        i_vals = np.arange(-0.2, 0.2 + 0.1, 0.1)
        j_vals = np.arange(-0.3, 0.3 + 0.15, 0.15)
        k_vals = [0]


        for i in i_vals:
            for j in j_vals:
                for k in k_vals:

                    offset = np.array([i, j, k])

                    sampled_goals.append(offset)
                    # Create the clean key string as used when saving
                    key_str = '_'.join([f"{off:.2f}".rstrip('0').rstrip('.') for off in [i, j, k]]) # no 0 after and all 0.1/0/0.02, rounding

                    with open(f"../features/sampled/sampled_{str(int(100*self.timing_ratio))}_finer/{key_str}.pkl",'rb') as file:
                        traj_data = pickle.load(file)
                    test_data = [traj_data[i] for i in test_idx]

                    self.test_data = test_data

                    # CHANGE HERE! phwere for using where people leave the gripper to infer goal
                    # and phwere_inference for using where people grab the gripper to infer where they leave it and where the goal is
                    if self.t == "grasp":
                        pdfs = self.pwhere_inference(test_data, offset) # self.offset? close enough numbers?
                    elif self.t == "release":
                        pdfs = self.pwhere(test_data, offset) # where people leave the gripper
                    self.get_features(test_data)
                    self.input_data_all, self.target_data_all, self.corrected_time = self.stack_data()
                    p_uwhen = self.pwhen()  # List: one value per traj

                    assert len(pdfs) == len(p_uwhen)
                    all_pdfs.append(pdfs)
                    all_puwhens.append(p_uwhen)
        print("Done")          

        # Convert to NumPy arrays for easier manipulation
        all_pdfs = np.array(all_pdfs)          # Shape: (num_goals, num_trajectories)
        all_puwhens = np.array(all_puwhens)  # Same shape

        normalized_all_puwhens = all_puwhens / np.sum(all_puwhens, axis=0, keepdims=True)  # Shape: (num_goals, num_trajs) normalize first before multiplying
        powered_pwhens = normalized_all_puwhens ** 1 # change the power
        normalized_powered_pwhens = powered_pwhens / np.sum(powered_pwhens, axis=0, keepdims=True) 
        
        # where only
        normalized_all_pdfs = all_pdfs / np.sum(all_pdfs, axis=0, keepdims=True) 
        # self.plot_xy_goal_distribution(normalized_probs=normalized_all_pdfs, x_vals=i_vals, y_vals=j_vals)
        
        beta = 0.8
        combined = (normalized_all_pdfs)**(1-beta) * (normalized_powered_pwhens)**beta

        # Combined 
        p_goals = combined / np.sum(combined, axis=0, keepdims=True)  # Shape: (num_goals, num_trajs)
        # self.plot_xy_goal_distribution(normalized_probs=p_goals, x_vals=i_vals, y_vals=j_vals)



        # compare models
        # Normalize each column (trajectory) across goals
        p_goals_where = normalized_all_pdfs # Shape: (num_goals, num_trajs)

        p_goals_ground = self.gmm_xy_pdf_at_z0_centered()
        print(p_goals_ground.shape)
        # self.plot_xy_goal_distribution(normalized_probs=p_goals_ground, x_vals=i_vals, y_vals=j_vals)


        res = self.kl_compare_multiple_preds(p_goals_ground, [normalized_all_puwhens, p_goals_where, p_goals])

        print("Mean KLs:", res['mean_kls'])
        for i, ci in enumerate(res['ci']):
            print(f"Pred {i+1} CI: {ci}")
        print("Overall Friedman p-value:", res['anova_p'], "| Significant:", res['overall_significant'])

        print("\nPairwise comparisons:")
        for pair, info in res['pairwise'].items():
            print(f"Pred {pair[0]+1} vs Pred {pair[1]+1}: p={info['p_value']:.4f}, significant={info['significant']}")

        if self.t == "grasp":
            with open(f"../results/KLDs/"+str(int(100*self.timing_ratio))+
                    '_'+str(int(100*self.split))+'_shape_'+str(self.shape)+'_target_'+str(self.target)+'_'+str(int(self.rep))+'.pkl', "wb") as f:
                pickle.dump(res, f)
        elif self.t == "release":
            with open(f"../results/KLDs_leaving/"+str(int(100*self.timing_ratio))+
                    '_'+str(int(100*self.split))+'_shape_'+str(self.shape)+'_target_'+str(self.target)+'_'+str(int(self.rep))+'.pkl', "wb") as f:
                pickle.dump(res, f)
        



    def get_features(self, data):

        self.total_a1 = []
        self.total_a2 = []
        self.total_dis = []
        self.total_v = []
        self.total_legi = []
        self.total_acc = []
        self.total_jerk = []
        self.total_curv = []
        self.total_a3 = []
        self.total_ps = []
        self.total_opt = [] # boltzmann

        self.total_time = []
        self.total_cor = []

        for i in range(len(data)):
            a1 = []
            a2 = []
            dis = []
            v = []
            time = []
            cor = []
            legi = []
            acc = []
            jerk = []
            curv = []
            a3 = []
            ps = []
            opt = []
            for t in range(len(data[i]["features"])):
                a1.append(data[i]["features"][t]['a1'])
                a2.append(data[i]["features"][t]['a2'])
                dis.append(data[i]["features"][t]['dis'])
                v.append(data[i]["features"][t]['v'])
                legi.append(data[i]["features"][t]['legi'])
                acc.append(data[i]["features"][t]['acc'])
                jerk.append(data[i]["features"][t]['jerk'])
                curv.append(data[i]["features"][t]['curv'])
                a3.append(data[i]["features"][t]['a3'])
                ps.append(data[i]["features"][t]['ps'])
                opt.append(data[i]["features"][t]['opt'])

                time.append(data[i]["features"][t]['time'])
                cor.append(data[i]["features"][t]['cor'])

            self.total_a1.append(a1)
            self.total_a2.append(a2)
            self.total_dis.append(dis)
            self.total_v.append(v)
            self.total_legi.append(legi)
            self.total_acc.append(acc)
            self.total_jerk.append(jerk)
            self.total_curv.append(curv)
            self.total_a3.append(a3)
            self.total_ps.append(ps)
            self.total_opt.append(opt)

            self.total_time.append(time)
            self.total_cor.append(cor)

            # print(len(a1))

            # input()

        self.total_a1_padded = pad_sequences(self.total_a1, padding='post', dtype='float32', value=-999.0) # to deferentiate from 0
        self.total_a2_padded = pad_sequences(self.total_a2, padding='post', dtype='float32', value=-999.0)
        self.total_dis_padded = pad_sequences(self.total_dis, padding='post', dtype='float32', value=-999.0)
        self.total_v_padded = pad_sequences(self.total_v, padding='post', dtype='float32', value=-999.0)
        self.total_legi_padded = pad_sequences(self.total_legi, padding='post', dtype='float32', value=-999.0)
        self.total_acc_padded = pad_sequences(self.total_acc, padding='post', dtype='float32', value=-999.0)
        self.total_jerk_padded = pad_sequences(self.total_jerk, padding='post', dtype='float32', value=-999.0)
        self.total_curv_padded = pad_sequences(self.total_curv, padding='post', dtype='float32', value=-999.0)
        self.total_a3_padded = pad_sequences(self.total_a3, padding='post', dtype='float32', value=-999.0)
        self.total_ps_padded = pad_sequences(self.total_ps, padding='post', dtype='float32', value=-999.0)
        self.total_opt_padded = pad_sequences(self.total_opt, padding='post', dtype='float32', value=-999.0)

        self.total_time_padded = pad_sequences(self.total_time, padding='post', dtype='float32', value=-999.0) # only for plotting
        self.total_cor_padded = pad_sequences(self.total_cor, padding='post', dtype='float32', value=-999.0)


    def stack_data(self):

        input_data = np.stack([self.total_ps_padded, self.total_a1_padded, self.total_a3_padded, 
                                    self.total_dis_padded, self.total_a2_padded, self.total_legi_padded,
                                    self.total_opt_padded], axis=-1) # num_datapoints, num_time, num_features
        input_data_all = input_data[:, :, :]

        target_data_all = self.total_cor_padded
        corrected_time = self.total_time_padded

        return input_data_all, target_data_all, corrected_time
    
    def rescale_features_masked(self, input_data):
        """
        Scales input features to [0, 1] using MinMaxScaler,
        only using valid (unmasked) time steps for fitting.

        Args:
            input_data (np.ndarray): shape (N, T, F)
            mask (np.ndarray): shape (N, T), 1 for valid, 0 for padded

        Returns:
            scaled_data (np.ndarray): shape (N, T, F), same shape as input
        """
        mask = self.generate_mask_from_trailing_zeros(input_data)
        N, T, F = input_data.shape
        reshaped_data = input_data.reshape(N * T, F)
        flat_mask = mask.flatten()  # shape: (N*T,)

        # Only use valid rows for fitting the scaler
        valid_data = reshaped_data[flat_mask == 1]  # shape: (num_valid, F)

        scaler = MinMaxScaler()
        scaler.fit(valid_data)

        # Transform everything (this includes padded steps too)
        # features_scaled = scaler.transform(reshaped_data)
        features_scaled = reshaped_data.copy()
        features_scaled[flat_mask == 1] = scaler.transform(reshaped_data[flat_mask == 1])

        # Reshape back to (N, T, F)
        features_scaled = features_scaled.reshape(N, T, F)

        return features_scaled
    
    def generate_mask_from_trailing_zeros(self, X):
        """
        Generate a binary mask from input data,
        assuming 0-padding occurs only at the end of each sequence.

        Args:
            X (np.ndarray): Input data of shape (N, T, F)

        Returns:
            mask (np.ndarray): Binary mask of shape (N, T)
        """
        N, T, F = X.shape
        mask = np.zeros((N, T), dtype=float)

        for i in range(N):
            # Find last index where any feature is non-zero
            non_zero_timesteps = np.any(X[i] != -999.0, axis=-1)
            last_valid_idx = np.argmax(non_zero_timesteps[::-1] == 1)
            last_valid_idx = T - last_valid_idx - 1 if last_valid_idx != 0 else T - 1

            # Set mask to 1 for all valid time steps
            mask[i, :last_valid_idx + 1] = 1.0

        return mask
    

    def build_and_load_transformer_for_finetune(
        self,
        model_weights_path,
        num_heads=8,
        ff_dim=64,
        num_layers=2,
        embed_dim=32,
        freeze_until_layer=None,  # optional: freeze all layers up to this index
    ):
        """
        Build the transformer model, load pretrained weights, optionally freeze layers,
        and prepare for fine-tuning.
        
        Args:
            input_dim (int): Number of features per timestep
            model_weights_path (str): Path to pretrained weights (.h5 or .keras)
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward layer dimension
            num_layers (int): Number of transformer blocks
            embed_dim (int): Embedding dimension
            freeze_until_layer (int or None): Freeze layers up to this index (optional)
            learning_rate (float): Learning rate for fine-tuning

        Returns:
            tf.keras.Model: Compiled model ready for fine-tuning
        """

        input_dim = self.input_data_all.shape[-1]

        # --- Input ---
        input_seq = Input(shape=(None, input_dim), name='input_seq')
        
        # Masking layer
        masked_input = layers.Masking(mask_value=-999.0, name='masking')(input_seq)

        # --- Positional Encoding ---
        class PositionalEncoding(layers.Layer):
            def __init__(self, max_len=5000, d_model=embed_dim):
                super().__init__()
                self.pos_encoding = self.get_pos_encoding(max_len, d_model)

            def get_pos_encoding(self, max_len, d_model):
                pos = np.arange(max_len)[:, np.newaxis]
                i = np.arange(d_model)[np.newaxis, :]
                angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
                angle_rads = pos * angle_rates
                angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                pos_encoding = angle_rads[np.newaxis, ...]
                return tf.cast(pos_encoding, dtype=tf.float32)

            def call(self, inputs, mask=None):
                seq_len = tf.shape(inputs)[1]
                return inputs + self.pos_encoding[:, :seq_len, :]

            def compute_mask(self, inputs, mask=None):
                return mask

        # Feature projection + positional encoding
        x = layers.Dense(embed_dim, name='feature_projection')(masked_input)
        x = PositionalEncoding()(x)

        # Boolean mask for attention
        boolean_mask = layers.Lambda(lambda x: tf.reduce_any(tf.not_equal(x, -999.0), axis=-1), name='boolean_mask')(input_seq)
        attn_mask = layers.Lambda(lambda m: tf.cast(m[:, tf.newaxis, :], tf.bool), name='attn_mask')(boolean_mask)

        # Transformer encoder blocks
        for layer_idx in range(num_layers):
            attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name=f'mha_{layer_idx}')(
                query=x, value=x, key=x, attention_mask=attn_mask
            )
            attn_output = layers.Dropout(0.1, name=f'dropout_attn_{layer_idx}')(attn_output)
            out1 = layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{layer_idx}')(x + attn_output)

            ffn = layers.Dense(ff_dim, activation='relu', name=f'ffn_relu_{layer_idx}')(out1)
            ffn = layers.Dense(embed_dim, name=f'ffn_out_{layer_idx}')(ffn)
            ffn = layers.Dropout(0.1, name=f'dropout_ffn_{layer_idx}')(ffn)
            x = layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{layer_idx}')(out1 + ffn)

        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Build model
        model = models.Model(inputs=input_seq, outputs=output, name='transformer_correction')

        # --- Load pretrained weights ---
        model.load_weights(model_weights_path)
        print("✅ Pretrained weights loaded successfully")
        # model = load_model(model_weights_path) 

        # --- Freeze layers if requested ---
        if freeze_until_layer is not None:
            for i, layer in enumerate(model.layers):
                if i <= freeze_until_layer:
                    layer.trainable = False
            print(f"✅ Layers up to index {freeze_until_layer} frozen")

        return model

    def pwhen(self):


        model = self.build_and_load_transformer_for_finetune(model_weights_path=f'../goal_infer_files/when_weights/model_weights/transformer_all_{str(int(100*self.timing_ratio))}_{str(int(self.rep))}.weights.h5')

        features_scaled_valid = self.rescale_features_masked(self.input_data_all)
        
        y_pred = model.predict(features_scaled_valid)
        
        all_y_pred = []
        all_y_ground = []
        all_time = []
        p_uwhen = []

        # Iterate through each trajectory
        for i in range(y_pred.shape[0]):
        # for i in range(20):
            for j in range(y_pred.shape[1]):
                if j == 0: 
                    continue
                if self.input_data_all[i, :, -1][j] == -999.0:  # Skip the padded input
                    ind = j
                    break
            
            y_pred_actual = y_pred[i].flatten()[:j]
            all_y_pred.append(y_pred_actual)

            y_ground_actual = self.target_data_all[i].flatten()[:j]
            all_y_ground.append(y_ground_actual)

            time_actual = self.corrected_time[i, :][:j]
            all_time.append(time_actual)



            # Compute raw prob_when - the derivative of the y_pred
            raw_prob_when = np.maximum(y_pred_actual[1:] - y_pred_actual[:-1], 0)

            # Smooth with average of neighbors in a [-2, +2] window
            smoothed_prob_when = np.zeros_like(raw_prob_when)

            # 5 step average
            for k in range(len(raw_prob_when)):
                # Define window bounds
                start = max(0, k - 6) # 3 times since og traj timestep is 3x smaller
                end = min(len(raw_prob_when), k + 2)  # +3 because Python slices are exclusive at the end
                smoothed_prob_when[k] = np.mean(raw_prob_when[start:end])

            # smoothed_prob_when = smoothed_prob_when / np.sum(smoothed_prob_when, axis=0, keepdims=True)

            correction_indices = np.where((y_ground_actual[:-1] == 0) & (y_ground_actual[1:] == 1))[0] # works if correction is not at the very last time step
            correction_time_step = correction_indices[0] + 1 if len(correction_indices) > 0 else None

            # fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # # First plot: derivative
            # axs[0].plot(smoothed_prob_when, label="Δ Prediction")
            # if correction_time_step is not None:
            #     axs[0].axvline(correction_time_step, color='red', linestyle='--', label='Correction Step')
            # axs[0].set_title("Derivative of Prediction")
            # axs[0].legend()

            # # Second plot: prediction vs ground truth
            # axs[1].plot(y_pred_actual, label="Prediction")
            # axs[1].plot(y_ground_actual, label="Ground Truth")
            # if correction_time_step is not None:
            #     axs[1].axvline(correction_time_step, color='red', linestyle='--', label='Correction Step')
            # axs[1].set_title("Prediction vs Ground Truth")
            # axs[1].legend()

            # plt.tight_layout()
            # part_id = self.test_data[i]["participant_id"]
            # plt.title(f"participant {part_id}")
            # plt.savefig(f"./{i}_{int(100*self.timing_ratio)}_{int(100*self.offset[0])}_{int(100*self.offset[1])}_{int(100*self.offset[2])}.png") 
            # plt.close()
            # # plt.show()

            if correction_time_step == None: # corrction is at very last time step
                p_uwhen.append(smoothed_prob_when[-1])
            elif correction_time_step < len(smoothed_prob_when):
                p_uwhen.append(smoothed_prob_when[correction_time_step]) # the prob at the time when the correction happens
            else:
                # print(correction_time_step, len(smoothed_prob_when)) # same lengths, same trajs
                p_uwhen.append(smoothed_prob_when[-1])
            # print(smoothed_prob_when[correction_time_step])

        return p_uwhen
    
    def compute_metrics(self, test_y, y_predict, all_time, threshold=0.5):
        assert len(test_y) == len(y_predict), "Mismatch in number of sequences"

        # Fast flatten
        y_true_flat = np.concatenate(test_y)
        y_prob_flat = np.concatenate(y_predict)
        y_pred_flat = (y_prob_flat >= threshold).astype(int)

        gt_times = []
        pred_times = []

        for y_true_seq, y_pred_seq, time_seq in zip(test_y, y_predict, all_time):
            y_pred_binary_seq = (np.asarray(y_pred_seq) >= threshold).astype(int)

            gt_idx = self.get_first_sustained_one_index(y_true_seq)
            pred_idx = self.get_first_sustained_one_index(y_pred_binary_seq)

            if gt_idx is not None and pred_idx is not None:
                gt_times.append(time_seq[gt_idx])
                pred_times.append(time_seq[pred_idx])

        # Framewise metrics
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        roc_auc = roc_auc_score(y_true_flat, y_prob_flat)
        mc = matthews_corrcoef(y_true_flat, y_pred_flat)

        # Correction timing error
        if gt_times:
            mae = np.mean(np.abs(np.array(gt_times) - np.array(pred_times)))
        else:
            mae = None

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'Matthews Coefficient': mc,
            'Correction MAE': mae,
            'Correction Count': len(gt_times),
            'Actual Correction Count': int(len(test_y))
        }
    
    def get_first_sustained_one_index(self, seq):
        seq = np.asarray(seq)
        ones = np.where(seq == 1)[0]
        if len(ones) == 0:
            return None

        # Check where from that index onward all are 1
        for idx in ones:
            if np.all(seq[idx:] == 1):
                return idx
        return None

    
    def pwhere(self, corrected_data, offset):

        u_wheres = []

        for j in range(len(corrected_data)):
            center = np.array([self.gs[self.shape][0][0], 0, 0])
            target_pose = center - offset # dont know where the goal is but te abosolute position wrt offset off center

            correction_traj = np.array(corrected_data[j]["correction_pose_list"]).copy()
            u_where = correction_traj[-1][:3] - target_pose # where people leave the gripper
            u_wheres.append(u_where)
 

        u_wheres = np.array(u_wheres)  # Shape: (N, 3)

        gmm_3d = joblib.load('../goal_infer_files/gmms/leaving/gmm_uwhere_leaving_'+str(int(100*self.timing_ratio))+'_'+str(int(100*self.split))+'_'+str(int(self.rep))+'.pkl')
        pdf_vals = np.exp(gmm_3d.score_samples(u_wheres))


        return pdf_vals
    
    # -------- Inference --------
    def pwhere_inference(self, corrected_data, offset):

        save_path = f"../goal_infer_files/where_infer/model_{str(int(100*self.timing_ratio))}_{str(int(100*self.split))}_{str(int(self.rep))}.pt"

        test_x, test_y = [], []
        pose_starts = []
        for d in corrected_data:
            pose_start = np.array(d["correction_pose_list"][0][:3], dtype=np.float32)
            pose_starts.append(pose_start)
            vel = np.array(np.array(d["correction_pose_list"][1][:3]) - np.array(d["correction_pose_list"][0][:3]), dtype=np.float32)
            pose_target = np.array(d["correction_pose_list"][-1][:3], dtype=np.float32)

            # --- Change here ---
            # Input: zero pose start, keep vel
            test_x.append(np.concatenate([np.zeros(3, dtype=np.float32), vel]))

            # Output: relative target
            test_y.append(pose_target - pose_start)

        pose_starts = np.array(pose_starts)
        test_x = np.array(test_x)
        test_y = np.array(test_y)


        checkpoint = torch.load(save_path, weights_only=False)

        def build_model(input_dim, output_dim, hidden_dim = 64):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
                )

        # use model to predict where people leave the gripper
        # Build model
        model = build_model(test_x.shape[1], test_y.shape[1])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Normalize using saved stats
        X_mean, X_std = checkpoint["X_mean"], checkpoint["X_std"]
        y_mean, y_std = checkpoint["y_mean"], checkpoint["y_std"]

        X_norm = (test_x - X_mean) / X_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        with torch.no_grad():
            preds_norm = model(X_tensor)
            preds = preds_norm.numpy() * y_std + y_mean # predicted points
            y_true = test_y

        center = np.array([self.gs[self.shape][0][0], 0, 0])
        target_pose = center - offset # dont know where the goal is but te abosolute position wrt offset off center
        u_future = pose_starts + preds - target_pose # predicted points + the starts
        
        gmm_3d = joblib.load('../goal_infer_files/gmms/leaving/gmm_uwhere_leaving_'+str(int(100*self.timing_ratio))+'_'+str(int(100*self.split))+'_'+str(int(self.rep))+'.pkl')
        pdf_vals = np.exp(gmm_3d.score_samples(u_future))

        return pdf_vals


    
    
    def gmm_xy_pdf_at_z0_centered(self,  n: int = 5, m: int = 5, # change dim here to match w sample space dim
                                x_range=(-0.20, 0.20), y_range=(-0.30, 0.30),
                                ) -> np.ndarray:
        """
        Evaluate a 3D GMM on a 2D XY grid at Z=0, shifted to a new center.
        Rows correspond to X, columns to Y. Returns PDF as (nxn, 1).

        Args:
            gmm_3d: Trained sklearn GaussianMixture model (3D)
            n: Number of points along X and Y
            x_range: Tuple (min, max) for X axis (before shift)
            y_range: Tuple (min, max) for Y axis (before shift)
            center: Tuple (x_center, y_center) where the grid will be shifted

        Returns:
            pdf_values: Numpy array of shape (n*n, 1) with PDF values
        """

        gmm_3d = joblib.load(f'../goal_infer_files/gmms/ground_truth/gmm_uwhere_model_ground_truth_{self.shape}.pkl')


        gmm_shifted = copy.deepcopy(gmm_3d)
        gmm_shifted.means_ -= np.array([0.0, self.gs[self.shape][self.target][1], 0.0]) # y pose

        # --- Create grid ---
        x = np.linspace(x_range[0], x_range[1], n)   # rows
        y = np.linspace(y_range[0], y_range[1], m)   # columns
        xx, yy = np.meshgrid(x, y)  # meshgrid: rows correspond to x

        # --- Flatten grid for GMM evaluation ---
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n*m)])

        # --- Evaluate PDF ---
        pdf_values = np.exp(gmm_shifted.score_samples(grid_points))
        pdf_values /= pdf_values.sum()  # ensure sum=1
        
        # --- Reshape back to original grid shape ---
        pdf_grid = pdf_values.reshape(xx.shape)
        pdf_grid_inverted = pdf_grid.T # it is y first for the ijk loop so invert it here

        # --- Reshape to (nxn, 1) ---
        return pdf_grid_inverted.reshape(-1, 1)


    
    def kl_compare_multiple_preds(self, ground, preds, alpha=0.05):
        """
        Compare KL divergence of multiple predictions against the same ground.

        Args:
            ground: np.array (num_goals, num_trajs)
            preds: list of np.array, each (num_goals, num_trajs)
            alpha: significance level

        Returns:
            dict containing:
                'mean_kls': list of mean KL per prediction
                'ci': list of confidence intervals per prediction
                'kl_per_traj': list of per-traj KL arrays
                'pairwise': dict with pairwise comparison results
                    keys: (i, j), values: {'p_value', 'significant'}
                'overall_significant': bool, whether Friedman test shows significant difference
                'anova_p': p-value from Friedman test
        """
        EPS = 1e-12
        ground = np.clip(ground, EPS, 1.0)
        ground_broadcast = np.repeat(ground, preds[0].shape[1], axis=1)  # shape (n_goals, n_trajs)

        kl_per_traj = []
        mean_kls = []
        ci_list = []

        n_trajs = preds[0].shape[1]

        # --- Compute KL per prediction ---
        for pred in preds:
            pred = np.clip(pred, EPS, 1.0)
            kl_traj = np.sum(ground_broadcast * (np.log(ground_broadcast) - np.log(pred)), axis=0)
            kl_per_traj.append(kl_traj)
            mean_kl = np.mean(kl_traj)
            mean_kls.append(mean_kl)
            sem = stats.sem(kl_traj)
            ci = stats.t.interval(1-alpha, df=n_trajs-1, loc=mean_kl, scale=sem)
            ci_list.append(ci)

        # --- Overall significance (Friedman test) ---
        kl_matrix = np.vstack(kl_per_traj)  # shape: (num_preds, num_trajs)
        F_stat, anova_p = stats.friedmanchisquare(*kl_matrix)
        overall_significant = anova_p < alpha

        # --- Pairwise significance (Wilcoxon signed-rank test) ---
        pairwise_results = {}
        for i, j in combinations(range(len(preds)), 2):
            stat, p_val = stats.wilcoxon(kl_per_traj[i], kl_per_traj[j])
            pairwise_results[(i, j)] = {
                'p_value': p_val,
                'significant': p_val < alpha
            }

        return {
            'all_kls': kl_per_traj, # save all the kls for each traj
            'mean_kls': mean_kls,
            'ci': ci_list,
            'kl_per_traj': kl_per_traj,
            'pairwise': pairwise_results,
            'overall_significant': overall_significant,
            'anova_p': anova_p
        }


    def plot_xy_goal_distribution(self, normalized_probs, x_vals, y_vals, z_vals=None, traj_idx=None, show=True):
        """
        Plots the probability distribution of goals in the XY plane.

        Args:
            normalized_probs (np.ndarray): Array of shape (num_goals, num_trajs), normalized probabilities.
            x_vals (list or array): List of x offsets.
            y_vals (list or array): List of y offsets.
            z_vals (list or array, optional): List of z offsets (default: [0]).
            traj_idx (int, optional): Index of trajectory to visualize. If None, averages over all trajectories.
            show (bool): Whether to show the plot immediately.
        
        Returns:
            prob_grid (np.ndarray): 2D grid of probabilities in XY plane.
        """
        if z_vals is None:
            z_vals = [0]

        # Build all offsets
        offsets = np.array([[-x, -y, -z] for x in x_vals for y in y_vals for z in z_vals]) # the opposite of the offset for actual location
        num_goals = offsets.shape[0]

        # Select trajectory or average
        if traj_idx is not None:
            if traj_idx < 0 or traj_idx >= normalized_probs.shape[1]:
                raise ValueError(f"traj_idx={traj_idx} is out of bounds for {normalized_probs.shape[1]} trajectories")
            probs = normalized_probs[:, traj_idx]
        else:
            probs = np.mean(normalized_probs, axis=1)  # Average over trajectories

        # Prepare grid
        prob_grid = np.zeros((len(x_vals), len(y_vals)))
        for i, (x, y, z) in enumerate(offsets):
            xi = x_vals.index(x)
            yi = y_vals.index(y)
            prob_grid[xi, yi] = probs[i]

        # Plot heatmap
        plt.figure(figsize=(6,5))
        im = plt.imshow(prob_grid, origin='lower',
                        extent=[min(y_vals), max(y_vals), min(x_vals), max(x_vals)],
                        cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar(im, label='Probability')
        plt.xlabel('y offset')
        plt.ylabel('x offset')
        plt.title('Goal Probability Distribution in XY plane')

        if show:
            plt.show()
        
        return prob_grid



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-tg','--target', type = int)
    parser.add_argument('-sh','--shape', type = str)
    parser.add_argument('-r','--repeat', type = int)
    parser.add_argument('-t','--type', type = str) # use grasp or release poses to infer goal poses

    args = parser.parse_args()

    pg = pgoal(timing_ratio=args.timing_ratio, split = args.split, target = args.target, shape = args.shape, rep = args.repeat, t = args.type)