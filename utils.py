import os.path
import re
import numpy as np
import pandas as pd
from configs import *
from selfies import encoder
from tqdm import tqdm

# raw_datas
dataset_path = current_dataset.data_paths['dataset_path']
interactions_path = current_dataset.data_paths['interactions_path']
drug_datas_path = current_dataset.data_paths['drug_datas_path']
target_datas_path = current_dataset.data_paths['target_datas_path']
# new datas
drug_indexes_path = current_dataset.data_paths['drug_indexes_path']
target_indexes_path = current_dataset.data_paths['target_indexes_path']
positive_pairs_path = current_dataset.data_paths['positive_pairs_path']
negative_pairs_path = current_dataset.data_paths['negative_pairs_path']
drug_features_path = current_dataset.data_paths['drug_features_path']
target_features_path = current_dataset.data_paths['target_features_path']
drug_topological_similarity_path = current_dataset.data_paths['drug_topological_similarity_path']
target_topological_similarity_path = current_dataset.data_paths['target_topological_similarity_path']
drug_sequence_similarity_path = current_dataset.data_paths['drug_sequence_similarity_path']
target_sequence_similarity_path = current_dataset.data_paths['target_sequence_similarity_path']


def get_dataset():
    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path)
        return dataset
    else:
        print(dataset_path, ' not exist')
        return None


def get_interactions():
    if os.path.exists(interactions_path):
        return pd.read_csv(interactions_path, index_col=0, header=0)
    else:
        print('Get interactions')
        dataset = get_dataset()
        drug_ids = list(set(dataset['Drug_ID']))
        target_ids = list(set(dataset['Target_ID']))

        interactions = pd.DataFrame(index=drug_ids, columns=target_ids)
        for index in tqdm(dataset.index):
            drug_id, target_id, score = dataset.loc[index, ['Drug_ID', 'Target_ID', 'Label']]
            label = 1 if score else 0
            interactions.loc[drug_id][target_id] = label
        interactions.fillna(0, inplace=True)
        interactions.to_csv(interactions_path)
        return interactions


def get_drug_datas():  # Drug_ID Smiles Selfies Others
    if os.path.exists(drug_datas_path):
        return pd.read_csv(drug_datas_path)
    else:
        print('Get drug datas')
        dataset = get_dataset()
        drug_datas = [[] for _ in range(3)]
        for index in tqdm(dataset.index):
            id, smiles = dataset.loc[index, ['Drug_ID', 'Drug']]
            if id not in drug_datas[0]:
                drug_datas[0].append(id)
                drug_datas[1].append(smiles)
                drug_datas[2].append(encoder(smiles, strict=False))
        drug_datas = pd.DataFrame({'Drug_ID': drug_datas[0], 'Smiles': drug_datas[1], 'Selfies': drug_datas[2]})
        drug_datas.to_csv(drug_datas_path)
        return drug_datas


def get_target_datas():  # Target_ID Amino_acids
    if os.path.exists(target_datas_path):
        return pd.read_csv(target_datas_path)
    else:
        print('Get target datas')
        dataset = get_dataset()
        target_datas = [[] for _ in range(2)]
        for index in tqdm(dataset.index):
            id, amino_acids = dataset.loc[index, ['Target_ID', 'Target']]
            if id not in target_datas[0]:
                target_datas[0].append(id)
                target_datas[1].append(amino_acids)
        target_datas = pd.DataFrame({'Target_ID': target_datas[0], 'Amino_acids': target_datas[1]})
        target_datas.to_csv(target_datas_path)
        return target_datas


# new_datas : index -> data , feature

def get_drug_indexes():  # drug_id -> drug_index
    if os.path.exists(drug_indexes_path):
        with open(drug_indexes_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get drug indexes')
        drug_datas = get_drug_datas()
        drug_ids = drug_datas['Drug_ID'].to_list()
        drug_indexes = {drug_ids[index]: index for index in range(len(drug_ids))}
        with open(drug_indexes_path, 'w') as json_file:
            json.dump(drug_indexes, json_file)
        return drug_indexes


def get_target_indexes():  # target_id -> target_index
    if os.path.exists(target_indexes_path):
        with open(target_indexes_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get target indexes')
        target_datas = get_target_datas()
        target_ids = target_datas['Target_ID'].to_list()
        target_indexes = {target_ids[index]: index for index in range(len(target_ids))}
        with open(target_indexes_path, 'w') as json_file:
            json.dump(target_indexes, json_file)
        return target_indexes


def get_positive_pairs_by_index():  # [[drug_index] , [target_index]] , node_1 ->node_2
    if os.path.exists(positive_pairs_path):
        with open(positive_pairs_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get positive pairs by index')
        dataset = get_dataset()
        drug_indexes = get_drug_indexes()
        target_indexes = get_target_indexes()
        positive_pairs = [[], []]
        for index in tqdm(dataset.index):
            drug_id, target_id = dataset.loc[index, ['Drug_ID', 'Target_ID']]
            if dataset.loc[index, 'Label']:
                positive_pairs[0].append(drug_indexes[drug_id])
                positive_pairs[1].append(target_indexes[target_id])
        with open(positive_pairs_path, 'w') as json_file:
            json.dump(positive_pairs, json_file)
        return positive_pairs


def get_negative_pairs_by_index():
    if os.path.exists(negative_pairs_path):
        with open(negative_pairs_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get negative pairs by index')
        dataset = get_dataset()
        drug_indexes = get_drug_indexes()
        target_indexes = get_target_indexes()
        negative_pairs = [[], []]
        for index in tqdm(dataset.index):
            drug_id, target_id = dataset.loc[index, ['Drug_ID', 'Target_ID']]
            if not dataset.loc[index, 'Label']:
                negative_pairs[0].append(drug_indexes[drug_id])
                negative_pairs[1].append(target_indexes[target_id])
        with open(negative_pairs_path, 'w') as json_file:
            json.dump(negative_pairs, json_file)
        return negative_pairs


def get_drug_features_by_index():
    if os.path.exists(drug_features_path):
        with open(drug_features_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get drug features by index')
        drug_datas = get_drug_datas()
        smiles_dict = []
        selfies_dict = []
        for index in tqdm(drug_datas.index):
            smiles = list(drug_datas.loc[index, 'Smiles'])
            smiles_dict = list(set(smiles_dict + smiles))
            selfies = re.findall(r'\[.*?\]', drug_datas.loc[index, 'Selfies'])
            selfies_dict = list(set(selfies_dict + selfies))  # 36
        smiles_dict = sorted(smiles_dict, key=lambda x: len(x))
        smiles_dict = {smiles_dict[index]: index + 1 for index in range(len(smiles_dict))}
        selfies_dict = sorted(selfies_dict, key=lambda x: len(x))
        selfies_dict = {selfies_dict[index]: index + 1 for index in range(len(selfies_dict))}

        all_smiles_features = []
        all_selfies_features = []
        for index in tqdm(drug_datas.index):
            smiles = list(drug_datas.loc[index, 'Smiles'])
            smiles_feature = [smiles_dict[data] for data in smiles]
            all_smiles_features.append(smiles_feature)
            selfies = re.findall(r'\[.*?\]', drug_datas.loc[index, 'Selfies'])
            selfies_feature = [selfies_dict[data] for data in selfies]
            all_selfies_features.append(selfies_feature)
        with open(drug_features_path, 'w') as json_file:
            json.dump([all_smiles_features, all_selfies_features], json_file)
        return [all_smiles_features, all_selfies_features]


def get_target_features_by_index():
    if os.path.exists(target_features_path):
        with open(target_features_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get target features by index')
        target_datas = get_target_datas()
        amino_acids_dict = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'U', 'X']
        amino_acids_dict = {amino_acids_dict[index]: index + 1 for index in range(len(amino_acids_dict))}

        target_features_1 = []
        for index in tqdm(target_datas.index):
            amino_acids = target_datas.loc[index, 'Amino_acids']
            feature_1 = [amino_acids_dict[data] for data in amino_acids]
            target_features_1.append(feature_1)
        with open(target_features_path, 'w') as json_file:
            json.dump(target_features_1, json_file)
        return target_features_1


def get_topological_similarity(array_1, array_2):
    pearson_corr = np.corrcoef(array_1, array_2)[0, 1]
    # dot_product = np.dot(array_1, array_2)
    # norm_a = np.linalg.norm(array_1)
    # norm_b = np.linalg.norm(array_2)
    # cosine_similarity = dot_product / (norm_a * norm_b)
    # similarity = (pearson_corr / 2 + 0.5) * (cosine_similarity / 2 + 0.5)
    # if np.isnan(similarity):
    #     similarity = 0.001
    return pearson_corr


def get_drug_topological_similarity_by_index():
    if os.path.exists(drug_topological_similarity_path):
        with open(drug_topological_similarity_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get drug topological similarity by index')
        interactions = get_interactions()
        drug_indexes = get_drug_indexes()
        drug_topological_similarity = [[0 for _ in range(len(drug_indexes.keys()))] for _ in range(len(drug_indexes.keys()))]
        for drug_id_1 in tqdm(interactions.index):
            drug_array_1 = interactions.loc[drug_id_1].values
            for drug_id_2 in interactions.index[interactions.index.get_loc(drug_id_1) + 1:]:
                drug_array_2 = interactions.loc[drug_id_2].values
                topological_similarity = round(get_topological_similarity(drug_array_1, drug_array_2), 3)
                drug_topological_similarity[drug_indexes[drug_id_1]][drug_indexes[drug_id_2]] = topological_similarity
                drug_topological_similarity[drug_indexes[drug_id_2]][drug_indexes[drug_id_1]] = topological_similarity
        with open(drug_topological_similarity_path, 'w') as json_file:
            json.dump(drug_topological_similarity, json_file)
        return drug_topological_similarity


def get_target_topological_similarity_by_index():
    if os.path.exists(target_topological_similarity_path):
        with open(target_topological_similarity_path, 'r') as json_file:
            return json.load(json_file)
    else:
        print('Get target topological similarity by index')
        interactions = get_interactions()
        target_indexes = get_target_indexes()
        target_topological_similarity = [[0 for _ in range(len(target_indexes.keys()))] for _ in range(len(target_indexes.keys()))]
        for target_id_1 in tqdm(interactions.columns):
            target_array_1 = interactions[target_id_1].values
            for target_id_2 in interactions.columns[interactions.columns.get_loc(target_id_1) + 1:]:
                target_array_2 = interactions[target_id_2].values
                topological_similarity = round(get_topological_similarity(target_array_1, target_array_2), 3)
                target_topological_similarity[target_indexes[target_id_1]][target_indexes[target_id_2]] = topological_similarity
                target_topological_similarity[target_indexes[target_id_2]][target_indexes[target_id_1]] = topological_similarity
        with open(target_topological_similarity_path, 'w') as json_file:
            json.dump(target_topological_similarity, json_file)
        return target_topological_similarity


# def smith_waterman(seq_1, seq_2, match=2, mismatch=-1, gap=-1):
#     m, n = len(seq_1), len(seq_2)
#     score_matrix = np.zeros((m + 1, n + 1))
#     max_score = 0
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             match_score = match if seq_1[i - 1] == seq_2[j - 1] else mismatch
#             score = max(score_matrix[i - 1][j - 1] + match_score, score_matrix[i - 1][j] + gap, score_matrix[i][j - 1] + gap, 0)
#             score_matrix[i][j] = score
#             if score > max_score:
#                 max_score = score
#     return max_score
#
#
# def get_sequence_similarity(seq_1, seq_2):
#     return smith_waterman(seq_1, seq_2, match=2, mismatch=-2, gap=1)
#
#
# def get_drug_sequence_similarity_by_index():
#     if os.path.exists(drug_sequence_similarity_path):
#         with open(drug_sequence_similarity_path, 'r') as json_file:
#             return json.load(json_file)
#     else:
#         print('Get drug structure similarity by index')
#         drug_datas = get_drug_datas()
#         drug_indexes = get_drug_indexes()
#         drug_sequence_similarity = [[0 for _ in range(len(drug_indexes.keys()))] for _ in range(len(drug_indexes.keys()))]
#         for index_1 in tqdm(drug_datas.index):
#             drug_id_1 = drug_datas.loc[index_1, 'Drug_ID']
#             drug_sequence_1 = drug_datas.loc[index_1, 'Smiles']
#             for index_2 in drug_datas.index[index_1:]:
#                 drug_id_2 = drug_datas.loc[index_2, 'Drug_ID']
#                 drug_sequence_2 = drug_datas.loc[index_2, 'Smiles']
#                 sequence_similarity = round(get_sequence_similarity(drug_sequence_1, drug_sequence_2), 3)
#                 drug_sequence_similarity[drug_indexes[drug_id_1]][drug_indexes[drug_id_2]] = sequence_similarity
#                 drug_sequence_similarity[drug_indexes[drug_id_2]][drug_indexes[drug_id_1]] = sequence_similarity
#         with open(drug_sequence_similarity_path, 'w') as json_file:
#             json.dump(drug_sequence_similarity, json_file)
#         return drug_sequence_similarity
#
#
# def get_target_sequence_similarity_by_index():
#     if os.path.exists(target_sequence_similarity_path):
#         with open(target_sequence_similarity_path, 'r') as json_file:
#             return json.load(json_file)
#     else:
#         print('Get target structure similarity by index')
#         target_datas = get_target_datas()
#         target_indexes = get_target_indexes()
#         target_sequence_similarity = [[0 for _ in range(len(target_indexes.keys()))] for _ in range(len(target_indexes.keys()))]
#         for index_1 in tqdm(target_datas.index):
#             target_id_1 = target_datas.loc[index_1, 'Target_ID']
#             target_sequence_1 = target_datas.loc[index_1, 'Amino_acids']
#             for index_2 in target_datas.index[index_1:]:
#                 target_id_2 = target_datas.loc[index_2, 'Target_ID']
#                 target_sequence_2 = target_datas.loc[index_2, 'Amino_acids']
#                 sequence_similarity = round(get_sequence_similarity(target_sequence_1, target_sequence_2), 3)
#                 target_sequence_similarity[target_indexes[target_id_1]][target_indexes[target_id_2]] = sequence_similarity
#                 target_sequence_similarity[target_indexes[target_id_2]][target_indexes[target_id_1]] = sequence_similarity
#         with open(target_sequence_similarity_path, 'w') as json_file:
#             json.dump(target_sequence_similarity, json_file)
#         return target_sequence_similarity


if __name__ == '__main__':
    get_dataset()
    get_drug_indexes()
    get_target_indexes()
    get_positive_pairs_by_index()
    get_negative_pairs_by_index()
    get_drug_features_by_index()
    get_target_features_by_index()
    get_drug_topological_similarity_by_index()
    get_target_topological_similarity_by_index()
