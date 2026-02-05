import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
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

# train the transformer for boltzmann model

class TerminateOnNaN(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"\nEpoch {epoch}: Invalid loss ({loss}), stopping training.")
            self.model.stop_training = True

class CustomMultiHeadAttention(layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)

    def call(self, query, value, attention_mask=None, training=False):
        # Call the parent MultiHeadAttention layer
        output, attention_weights = super().call(query, value, attention_mask=attention_mask, training=training)
        return output, attention_weights  # Return both the output and the attention weights


class transformer_train:
    def __init__(self, timing_ratio, split, rep):

        self.timing_ratio = timing_ratio
        self.split = split
        self.rep = rep


        # List of target numbers you have in the file names
        target_nums = [0, 1, 2, 3]  # adjust based on your files
        shapes = ["circle", "rectangle", "triangle", "square"]
        # target_nums = [3]

        train_idx_all = []
        val_idx_all = []
        test_idx_all = []

        for target in target_nums:
            for shape in shapes:
                # filename = f'/Users/anjiabei/Documents/research/features/splits/indices_{int(100*self.timing_ratio)}_{int(100*self.split)}_shape_{shape}_target_{int(target)}.pkl'
                filename = f'./splits/repeats/indices_{int(100*self.timing_ratio)}_{int(100*self.split)}_shape_{shape}_target_{int(target)}_{int(self.rep)}.pkl'
                
                with open(filename, 'rb') as f:
                    idx_dict = pickle.load(f)
                
                train_idx_all.extend(idx_dict["train"])
                val_idx_all.extend(idx_dict["val"])
                test_idx_all.extend(idx_dict["test"])


        with open('./corrected_features_'+str(int(100*self.timing_ratio))+'.pkl', 'rb') as file:
            all_data_pre = pickle.load(file)
        all_data = self.filter_data(all_data_pre)  # corrected data has features = False, needs to be filtered before indexing

        corrected_data = [all_data[i] for i in train_idx_all]
        val_data = [all_data[i] for i in val_idx_all]
        test_data = [all_data[i] for i in test_idx_all]
        print("corrected length =", len(corrected_data), " val_length = ", len(val_data), " test length =",len(test_data))
        self.val_data = val_data
        self.test_data = test_data


        with open('./uncorrected_features.pkl', 'rb') as file:
            uncorrected_data_pre = pickle.load(file)
        uncorrected_data = self.filter_data(uncorrected_data_pre)

        random.seed(self.rep)
        np.random.seed(self.rep)
        uncorrected_sampled = random.sample(uncorrected_data, min(len(corrected_data), len(uncorrected_data))) # sometimes corrected data is longer than uncorrected data (low comp)
        # uncorrected_sampled = random.sample(uncorrected_data, 2)
        training_data = corrected_data + uncorrected_sampled

        ind = len(corrected_data)
        self.get_features(training_data)

        self.devide_data(ind)
        # self.devide_data_corrected()


        uncorrected_sampled = random.sample(uncorrected_data, min(len(val_data), len(uncorrected_data))) # sometimes corrected data is longer than uncorrected data (low comp)
        # uncorrected_sampled = random.sample(uncorrected_data, 2)
        combined_val_data = val_data + uncorrected_sampled

        self.get_val_features(combined_val_data)
        self.devide_data_val()

        uncorrected_sampled = random.sample(uncorrected_data, min(len(test_data), len(uncorrected_data))) # sometimes corrected data is longer than uncorrected data (low comp)
        # uncorrected_sampled = random.sample(uncorrected_data, 2)
        combined_test_data = test_data + uncorrected_sampled

        self.get_test_features(combined_test_data)
        self.devide_data_test()


    def filter_data(self, data):

        selected_data = []
        for i in range(len(data)):
            # if data[i]["comp"].strip() == str(self.comp).strip() and data[i]["features"] != False:
            if data[i]["features"] != False: # important for training
                selected_data.append(data[i])
        print(len(selected_data))
        # input()

        return selected_data



    def get_features(self, data): # for training data

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

        self.total_time_padded = pad_sequences(self.total_time, padding='post', dtype='float32', value=-999.0)
        self.total_cor_padded = pad_sequences(self.total_cor, padding='post', dtype='float32', value=-999.0)

    def get_val_features(self, data):

        self.total_a1_val = []
        self.total_a2_val = []
        self.total_dis_val = []
        self.total_v_val = []
        self.total_legi_val = []
        self.total_acc_val = []
        self.total_jerk_val = []
        self.total_curv_val = []
        self.total_a3_val = []
        self.total_ps_val = []
        self.total_opt_val = [] # boltzmann

        self.total_time_val = []
        self.total_cor_val = []

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

            self.total_a1_val.append(a1)
            self.total_a2_val.append(a2)
            self.total_dis_val.append(dis)
            self.total_v_val.append(v)
            self.total_legi_val.append(legi)
            self.total_acc_val.append(acc)
            self.total_jerk_val.append(jerk)
            self.total_curv_val.append(curv)
            self.total_a3_val.append(a3)
            self.total_ps_val.append(ps)
            self.total_opt_val.append(opt)

            self.total_time_val.append(time)
            self.total_cor_val.append(cor)

            # print(len(a1))

            # input()

        self.total_a1_padded_val = pad_sequences(self.total_a1_val, padding='post', dtype='float32', value=-999.0) # to deferentiate from 0
        self.total_a2_padded_val = pad_sequences(self.total_a2_val, padding='post', dtype='float32', value=-999.0)
        self.total_dis_padded_val = pad_sequences(self.total_dis_val, padding='post', dtype='float32', value=-999.0)
        self.total_v_padded_val = pad_sequences(self.total_v_val, padding='post', dtype='float32', value=-999.0)
        self.total_legi_padded_val = pad_sequences(self.total_legi_val, padding='post', dtype='float32', value=-999.0)
        self.total_acc_padded_val = pad_sequences(self.total_acc_val, padding='post', dtype='float32', value=-999.0)
        self.total_jerk_padded_val = pad_sequences(self.total_jerk_val, padding='post', dtype='float32', value=-999.0)
        self.total_curv_padded_val = pad_sequences(self.total_curv_val, padding='post', dtype='float32', value=-999.0)
        self.total_a3_padded_val = pad_sequences(self.total_a3_val, padding='post', dtype='float32', value=-999.0)
        self.total_ps_padded_val = pad_sequences(self.total_ps_val, padding='post', dtype='float32', value=-999.0)
        self.total_opt_padded_val = pad_sequences(self.total_opt_val, padding='post', dtype='float32', value=-999.0)

        self.total_time_padded_val = pad_sequences(self.total_time_val, padding='post', dtype='float32', value=-999.0)
        self.total_cor_padded_val = pad_sequences(self.total_cor_val, padding='post', dtype='float32', value=-999.0)

    
    def get_test_features(self, data):

        self.total_a1_test = []
        self.total_a2_test = []
        self.total_dis_test = []
        self.total_v_test = []
        self.total_legi_test = []
        self.total_acc_test = []
        self.total_jerk_test = []
        self.total_curv_test = []
        self.total_a3_test = []
        self.total_ps_test = []
        self.total_opt_test = [] # boltzmann

        self.total_time_test = []
        self.total_cor_test = []

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

            self.total_a1_test.append(a1)
            self.total_a2_test.append(a2)
            self.total_dis_test.append(dis)
            self.total_v_test.append(v)
            self.total_legi_test.append(legi)
            self.total_acc_test.append(acc)
            self.total_jerk_test.append(jerk)
            self.total_curv_test.append(curv)
            self.total_a3_test.append(a3)
            self.total_ps_test.append(ps)
            self.total_opt_test.append(opt)

            self.total_time_test.append(time)
            self.total_cor_test.append(cor)

            # print(len(a1))

            # input()

        self.total_a1_padded_test = pad_sequences(self.total_a1_test, padding='post', dtype='float32', value=-999.0) # to deferentiate from 0
        self.total_a2_padded_test = pad_sequences(self.total_a2_test, padding='post', dtype='float32', value=-999.0)
        self.total_dis_padded_test = pad_sequences(self.total_dis_test, padding='post', dtype='float32', value=-999.0)
        self.total_v_padded_test = pad_sequences(self.total_v_test, padding='post', dtype='float32', value=-999.0)
        self.total_legi_padded_test = pad_sequences(self.total_legi_test, padding='post', dtype='float32', value=-999.0)
        self.total_acc_padded_test = pad_sequences(self.total_acc_test, padding='post', dtype='float32', value=-999.0)
        self.total_jerk_padded_test = pad_sequences(self.total_jerk_test, padding='post', dtype='float32', value=-999.0)
        self.total_curv_padded_test = pad_sequences(self.total_curv_test, padding='post', dtype='float32', value=-999.0)
        self.total_a3_padded_test = pad_sequences(self.total_a3_test, padding='post', dtype='float32', value=-999.0)
        self.total_ps_padded_test = pad_sequences(self.total_ps_test, padding='post', dtype='float32', value=-999.0)
        self.total_opt_padded_test = pad_sequences(self.total_opt_test, padding='post', dtype='float32', value=-999.0)

        self.total_time_padded_test = pad_sequences(self.total_time_test, padding='post', dtype='float32', value=-999.0)
        self.total_cor_padded_test = pad_sequences(self.total_cor_test, padding='post', dtype='float32', value=-999.0)



    def devide_data(self, ind): #50/50 ind is the length of teh corrected data; for training data


        input_data_all = np.stack([self.total_opt_padded], axis=-1) # num_datapoints, num_time, num_features


        print(input_data_all.shape)
        target_data_all = self.total_cor_padded

        corrected_x = input_data_all[:ind]
        uncorrected_x = input_data_all[ind:]
        corrected_y = target_data_all[:ind]
        uncorrected_y = target_data_all[ind:]
        corrected_time = self.total_time_padded[:ind]
        uncorrected_time = self.total_time_padded[ind:]

        # randomize
        rng = np.random.default_rng()  # Automatically uses system entropy
        indices = rng.permutation(corrected_x.shape[0])
        # indices = np.random.permutation(input_data.shape[0])
        # print(indices[:10])
        shuffled_corrected_x = corrected_x[indices]
        shuffled_corrected_y = corrected_y[indices]
        shuffled_corrected_time = corrected_time[indices]

        # randomize
        rng = np.random.default_rng()  # Automatically uses system entropy
        indices = rng.permutation(uncorrected_x.shape[0])
        # indices = np.random.permutation(input_data.shape[0])
        # print(indices[:10])
        shuffled_uncorrected_x = uncorrected_x[indices]
        shuffled_uncorrected_y = uncorrected_y[indices]
        shuffled_uncorrected_time = uncorrected_time[indices]


        training_x = np.concatenate((shuffled_corrected_x, shuffled_uncorrected_x), axis=0)
        training_y = np.concatenate((shuffled_corrected_y, shuffled_uncorrected_y), axis=0)

        # randomize
        rng = np.random.default_rng()  # Automatically uses system entropy
        indices = rng.permutation(training_x.shape[0])
        # indices = np.random.permutation(input_data.shape[0])
        # print(indices[:10])
        self.training_input = training_x[indices]
        self.training_target = training_y[indices]

        print(self.training_input.shape)



    def devide_data_val(self): 



        input_data_all = np.stack([self.total_opt_padded_val], axis=-1) # num_datapoints, num_time, num_features

        self.val_input = input_data_all
        self.val_target = self.total_cor_padded_val



    def devide_data_test(self): 


        input_data_all = np.stack([self.total_opt_padded_test], axis=-1) # num_datapoints, num_time, num_features

        self.valid_input = input_data_all # test input actually
        self.valid_target = self.total_cor_padded_test
        self.test_time = self.total_time_padded_test


    
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
        
    

    def set_seed(self, seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Optional: Force deterministic GPU ops (slower, only if you want full reproducibility)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    def training_transformer_masked_with_val(
        self,
        num_heads=8,
        ff_dim=64,
        num_layers=2,
        embed_dim=32,
    ):
        """
        Transformer with masking support, trained with a separate validation set.
        """

        self.set_seed(self.rep) # set the same seed here
        # self.set_seed(0)

        # --- Input ---
        input_seq = layers.Input(shape=(None, self.training_input.shape[-1]), name='input_seq')
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

        x = layers.Dense(embed_dim, name='feature_projection')(masked_input)
        x = PositionalEncoding()(x)

        # --- Attention mask ---
        boolean_mask = layers.Lambda(lambda x: tf.reduce_any(tf.not_equal(x, -999.0), axis=-1), name='boolean_mask')(input_seq)
        attn_mask = layers.Lambda(lambda m: tf.cast(m[:, tf.newaxis, :], tf.bool), name='attn_mask')(boolean_mask)

        # --- Transformer Encoder blocks ---
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

        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = tf.keras.models.Model(inputs=input_seq, outputs=output, name='transformer_correction')

        # --- Compile ---
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])

        # --- Preprocess training features ---
        features_scaled = self.rescale_features_masked(self.training_input)
        mask = self.generate_mask_from_trailing_zeros(self.training_input)
        y = np.expand_dims(self.training_target, axis=-1)

        # --- Preprocess validation features ---
        val_features_scaled = self.rescale_features_masked(self.val_input)
        val_mask = self.generate_mask_from_trailing_zeros(self.val_input)
        val_y = np.expand_dims(self.val_target, axis=-1)

        # --- Callbacks ---
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'best_transformer_b_model_{str(int(100*self.timing_ratio))}_{str(int(self.rep))}.keras',  # can end in .keras
                save_weights_only=False,                  # save the full model
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
                )

        # --- Train ---
        model.fit(
            features_scaled,
            y,
            sample_weight=mask,
            validation_data=(val_features_scaled, val_y, val_mask),
            epochs=40,
            batch_size=20,
            verbose=2,
            callbacks=[TerminateOnNaN(), checkpoint]
        )

        # --- Load best weights ---
        model.load_weights(f'best_transformer_b_model_{str(int(100*self.timing_ratio))}_{str(int(self.rep))}.keras')
        model.save_weights('./model_weights_bol_com/transformer_all_'+str(int(100*self.timing_ratio))+'_'+str(int(self.rep))+'.h5')

        return model
    
    
    def testing(self, model):

        features_scaled_valid = self.rescale_features_masked(self.valid_input)
        
        y_pred = model.predict(features_scaled_valid)
        
        all_y_pred = []
        all_y_ground = []
        all_time = []
        all_saliency = []

        # Iterate through the test samples
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if j == 0: 
                    continue
                if self.valid_input[i, :, -1][j] == -999.0: # skip the padded input
                    ind = j
                    break
            # print(self.valid_input[i, :, -1], y_pred[i])
            y_pred_actual = y_pred[i].flatten()[:j]
            # y_pred_binary_actual = y_pred_binary[i].flatten()[:j]
            all_y_pred.append(y_pred_actual)

            y_ground_actual = self.valid_target[i].flatten()[:j]
            all_y_ground.append(y_ground_actual)

            # time_actual = self.valid_input[i, :, -1][:j]
            time_actual = self.test_time[i, :][:j]
            all_time.append(time_actual)


        
        results = self.compute_metrics(all_y_ground, all_y_pred, all_time)
        for k, v in results.items():
            print(f"{k}: {v}")

        return results
    


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
            'Actual Correction Count': int(0.5 * len(test_y))
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-r','--repeat', type = int)

    args = parser.parse_args()

    tt = transformer_train(timing_ratio=args.timing_ratio, split = args.split, rep = args.repeat)
    model = tt.training_transformer_masked_with_val()
    results = tt.testing(model)
    with open(f"./results/evaluation_bol_com/"+str(int(100*args.timing_ratio))+'_'+str(int(100*args.split))+'_'+str(int(args.repeat))+".pkl", "wb") as f:
        pickle.dump(results, f)