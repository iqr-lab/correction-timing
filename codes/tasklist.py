# training gmms
# with open('tasklist.sh','w') as file:
#     for shape in ['rectangle','square','triangle','circle']:
#         for lamda in [1e-1, 1e-2, 1e-3]:
#             for sample_size in [120]:
#                 for particle_num in [15, 30, 45]:
#                     file.write('python3 train_gmms.py -s '+shape+' -c high -lam '+str(lamda)+
#                                ' -ss '+str(sample_size)+' -pn '+str(particle_num)+' \n')

#to calculate KLD
# with open('gmmlist.sh','w') as file:
#     for shape in ['rectangle', 'square', 'triangle', 'circle']:
#         for lamda in [1e-1, 1e-2, 1e-3]:
#             for sample_size in [15]:
#                 for particle_num in [15, 30, 45]:
#                     file.write('python3 read_gmm.py -s '+shape+' -c high -lam '+str(lamda)+
#                                ' -ss '+str(sample_size)+' -pn '+str(particle_num)+' \n')
                    

# #for making plots
# with open('boxlist.sh','w') as file:
#     for shape in ['rectangle', 'square', 'triangle', 'circle']:
#         for lamda in [1e-1, 1e-2, 1e-3]:
#             for particle_num in [15, 30, 45]:
#                 file.write('python3 boxnwhisker.py -s '+shape+' -c high -lam '+str(lamda)+
#                             ' -pn '+str(particle_num)+' \n')


# # KLD from Gaussian for when 
# with open('tasklist.sh','w') as file:
#     for shape in ['triangle']:
#         for legi in ['legible']:
#             for data_point in range(54):
#                 file.write('python3 metro_hastings_distance.py -s '+shape+' -l '+legi+
#                             ' -di '+str(data_point)+' \n')
                




# # train val test split 10 times
# with open('tasklist.sh','w') as file:
#     for shape in ['triangle', 'circle', 'square', 'rectangle']:
#         for target in range(4):
#             for tr in [0.7, 0.8, 0.9, 1]:
#                 file.write(f'python3 train_test_split.py -s 0.6 -tr {tr} -sh {shape} -tg {target} \n')

# # train val test split 10 times
# with open('tasklist.sh','w') as file:
#     for shape in ['triangle', 'circle', 'square', 'rectangle']:
#         for target in range(4):
#             for tr in [0.7, 0.8, 0.9, 1]:
#                 for rep in range(10):
#                     file.write(f'python3 split_data.py -s 0.6 -tr {tr} -sh {shape} -tg {target} -r {str(int(rep))} \n')

# # train transformer 50 times
# with open('tasklist_transformer.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for i in range(10):
#             file.write(f'python3 transformer_training_all_feature.py -s 0.6 -tr {tr} -r {i} \n')

# train transformer (boltzmann) 50 times
with open('tasklist_transformer_bol.sh','w') as file:
    for tr in [0.7, 0.8, 0.9, 1]:
        for i in range(10):
            file.write(f'python3 transformer_bol.py -s 0.6 -tr {tr} -r {i} \n')

# # ablation
# with open('tasklist_transformer_ablation.sh','w') as file:
#     for i in range(200):
#         for tr in [0.7, 0.8, 0.9, 1]:
#             for j in range(7):
#                 file.write(f'python3 transformer_ablation.py -s 0.6 -tr {tr} -r {i} -d {j} \n')


# # get the evaluation from the loaded weights
# with open('tasklist_eval.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for i in range(200):
#             file.write(f'python3 load_weights.py -s 0.6 -tr {tr} -r {i} \n')

# # get the evaluation from the loaded weights
# with open('tasklist_eval_ablation.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for i in range(50):
#             for j in range(7):
#                 file.write(f'python3 load_weights_ablation.py -s 0.6 -tr {tr} -r {i} -d {j} \n')


# # train gmm on 50 different splits
# with open('tasklist_gmm.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for i in range(10):
#             file.write(f'python3 gmm_training.py -s 0.6 -tr {tr} -r {i} \n')


# # train gmm on 50 different splits
# with open('tasklist_where_infer.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for i in range(10):
#             file.write(f'python3 where_model.py -s 0.6 -tr {tr} -r {i} \n')

# # goal infer on 50 different splits
# with open('tasklist_goal_infer.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for rep in range(10):
#             for shape in ['triangle', 'circle', 'square', 'rectangle']:
#                 for target in range(4):
                    
#                         file.write(f'python3 goal_infer_para.py -s 0.6 -tr {tr} -sh {shape} -tg {int(target)} -r {int(rep)} \n')

# # goal infer on 50 different splits
# with open('tasklist_goal_infer_alpha.sh','w') as file:
#     for tr in [0.7, 0.8, 0.9, 1]:
#         for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             for rep in range(5):
#                 for shape in ['triangle', 'circle', 'square', 'rectangle']:
#                     for target in range(4):
                        
#                             file.write(f'python3 goal_infer_alpha.py -s 0.6 -tr {tr} -sh {shape} -tg {int(target)} -r {int(rep)} -a {alpha}\n')