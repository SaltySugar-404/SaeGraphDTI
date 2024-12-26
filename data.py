import heapq

from utils import *

drug_feature_length = current_dataset.configs['drug_feature_length']
target_feature_length = current_dataset.configs['target_feature_length']
drug_top_percentage = current_dataset.configs['drug_top_percentage']
target_top_percentage = current_dataset.configs['target_top_percentage']
drug_max_new_num_neighbors = current_dataset.configs['drug_max_new_num_neighbors']
target_max_new_num_neighbors = current_dataset.configs['target_max_new_num_neighbors']


class GraphData:  # node_0 -> node_1
    def __init__(self):
        self.num_drugs = None
        self.num_targets = None
        self.num_positive = None
        self.num_negative = None
        self.drug_feature_length = drug_feature_length
        self.target_feature_length = target_feature_length
        self.drug_top_percentage = drug_top_percentage
        self.target_top_percentage = target_top_percentage

        self.drug_features = []
        self.target_features = []
        self.positive_edges = []
        self.negative_edges = []
        self.__get_raw_edges()

        self.drug_edges = []
        self.target_edges = []
        self.__get_new_edges()

    def __get_raw_edges(self):
        drug_features = get_drug_features_by_index()
        self.drug_features = drug_features[0]  # smiles or selfies
        self.num_drugs = len(self.drug_features)
        for index in range(len(self.drug_features)):
            if len(self.drug_features[index]) < self.drug_feature_length:
                self.drug_features[index] += [0] * (self.drug_feature_length - len(self.drug_features[index]))
            else:
                self.drug_features[index] = self.drug_features[index][:self.drug_feature_length]

        self.target_features = get_target_features_by_index()
        self.num_targets = len(self.target_features)
        for index in range(len(self.target_features)):
            if len(self.target_features[index]) < self.target_feature_length:
                self.target_features[index] += [0] * (self.target_feature_length - len(self.target_features[index]))
            else:
                self.target_features[index] = self.target_features[index][:self.target_feature_length]

        self.positive_edges = get_positive_pairs_by_index()
        for index in range(len(self.positive_edges[1])):
            self.positive_edges[1][index] += self.num_drugs
        self.num_positive = len(self.positive_edges[0])

        self.negative_edges = get_negative_pairs_by_index()
        for index in range(len(self.negative_edges[1])):
            self.negative_edges[1][index] += self.num_drugs
        self.num_negative = len(self.negative_edges[0])

    def __get_new_edges(self):
        # get drug threshold
        drug_similarity = get_drug_topological_similarity_by_index()
        all_drug_similarity = []
        for data in drug_similarity:
            all_drug_similarity += data
        all_drug_similarity = sorted(all_drug_similarity, reverse=True)[self.num_drugs:]
        drug_threshold = all_drug_similarity[int(len(all_drug_similarity) * self.drug_top_percentage)]

        # add drug edges > drug threshold
        self.drug_edges = [[], []]
        drug_indexes = list(get_drug_indexes().values())
        drug_frequency_by_index = {index: 0 for index in drug_indexes}
        for index_0 in range(self.num_drugs):
            drug_index_0 = drug_indexes[index_0]
            for index_1 in range(index_0 + 1, self.num_drugs):
                drug_index_1 = drug_indexes[index_1]
                if drug_similarity[drug_index_0][drug_index_1] > drug_threshold:
                    self.drug_edges[0].append(drug_index_0)
                    self.drug_edges[1].append(drug_index_1)
                    drug_frequency_by_index[drug_index_0] += 1
                    drug_frequency_by_index[drug_index_1] += 1

        # add remain drug edges
        for drug_index, frequency in drug_frequency_by_index.items():
            if frequency < drug_max_new_num_neighbors:
                most_similarity_drug_values = heapq.nlargest(drug_max_new_num_neighbors - frequency, drug_similarity[drug_index])
                most_similarity_drug_indexes = [drug_similarity[drug_index].index(element) for element in most_similarity_drug_values]
                for most_similarity_drug_index in most_similarity_drug_indexes:
                    self.drug_edges[0].append(drug_index)
                    self.drug_edges[1].append(most_similarity_drug_index)

        # get target threshold
        target_similarity = get_target_topological_similarity_by_index()
        all_target_similarity = []
        for data in target_similarity:
            all_target_similarity += data
        all_target_similarity = sorted(all_target_similarity, reverse=True)[self.num_targets:]
        target_threshold = all_target_similarity[int(len(all_target_similarity) * self.target_top_percentage)]

        # add target edges > target threshold
        self.target_edges = [[], []]
        target_indexes = list(get_target_indexes().values())
        target_frequency_by_index = {index: 0 for index in target_indexes}
        for index_0 in range(self.num_targets):
            target_index_0 = target_indexes[index_0]
            for index_1 in range(index_0 + 1, self.num_targets):
                target_index_1 = target_indexes[index_1]
                if target_similarity[target_index_0][target_index_1] > target_threshold:
                    self.target_edges[0].append(target_index_0 + self.num_drugs)
                    self.target_edges[1].append(target_index_1 + self.num_drugs)
                    target_frequency_by_index[target_index_0] += 1
                    target_frequency_by_index[target_index_1] += 1

        # add remain target nodes
        for target_index, frequency in target_frequency_by_index.items():
            if frequency < target_max_new_num_neighbors:
                most_similarity_target_values = heapq.nlargest(target_max_new_num_neighbors - frequency, target_similarity[target_index])
                most_similarity_target_indexes = [target_similarity[target_index].index(element) for element in most_similarity_target_values]
                for most_similarity_target_index in most_similarity_target_indexes:
                    self.target_edges[0].append(target_index + self.num_drugs)
                    self.target_edges[1].append(most_similarity_target_index + self.num_drugs)

    def show_graph_data(self):
        print('num drugs = {} , num targets = {}'.format(self.num_drugs, self.num_targets))
        print('num positive = {} , num negative = {}'.format(self.num_positive, self.num_negative))
        max_edges = (self.num_drugs + self.num_targets) * (self.num_drugs + self.num_targets - 1) * 0.5
        print('sparsity = {}'.format(1 - (self.num_positive / max_edges)))
        print('average degree of drug = {} , average degree of target = {}'.format(self.num_positive / self.num_drugs, self.num_positive / self.num_targets))


def convert_to_bidirectional(raw_edge_indexes, insert_or_append='append'):  # node_1 <-> node_2
    new_edge_indexes = [[], []]
    if insert_or_append == 'insert':
        for index in range(len(raw_edge_indexes[0])):
            new_edge_indexes[0] += [raw_edge_indexes[0][index], raw_edge_indexes[1][index]]
            new_edge_indexes[1] += [raw_edge_indexes[1][index], raw_edge_indexes[0][index]]
    elif insert_or_append == 'append':
        new_edge_indexes[0] = raw_edge_indexes[0] + raw_edge_indexes[1]
        new_edge_indexes[1] = raw_edge_indexes[1] + raw_edge_indexes[0]
    else:
        print('Error')
    return new_edge_indexes


if __name__ == '__main__':
    GraphData().show_graph_data()
