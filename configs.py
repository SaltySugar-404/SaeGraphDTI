import json
import os
import shutil


class DatasetWithConfigs:
    def __init__(self, dataset_id: str, load_best_configs=True):
        self.dataset_id = dataset_id
        # data
        raw_data_folder = 'data/' + dataset_id + '/raw_datas'
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
            for filename in os.listdir('data/' + dataset_id):
                file_path = os.path.join('data/' + dataset_id, filename)
                if os.path.isfile(file_path):
                    target_path = os.path.join(raw_data_folder, filename)
                    shutil.move(file_path, target_path)
        new_data_folder = 'data/' + dataset_id + '/new_datas'
        if not os.path.exists(new_data_folder):
            os.makedirs(new_data_folder)
        all_results_folder = 'data/' + dataset_id + '/all_results'
        if not os.path.exists(all_results_folder):
            os.makedirs(all_results_folder)
        self.data_paths = {
            # raw_data
            'dataset_path': raw_data_folder + '/dataset.csv',
            'interactions_path': raw_data_folder + '/interactions.csv',
            'drug_datas_path': raw_data_folder + '/drug_datas.csv',
            'target_datas_path': raw_data_folder + '/target_datas.csv',
            # new_data
            'drug_indexes_path': new_data_folder + '/drug_indexes.json',
            'target_indexes_path': new_data_folder + '/target_indexes.json',
            'positive_pairs_path': new_data_folder + '/positive_pairs.json',
            'negative_pairs_path': new_data_folder + '/negative_pairs.json',
            'drug_features_path': new_data_folder + '/drug_features.json',
            'target_features_path': new_data_folder + '/target_features.json',
            'drug_topological_similarity_path': new_data_folder + '/drug_topological_similarity.json',
            'target_topological_similarity_path': new_data_folder + '/target_topological_similarity.json',
            'drug_sequence_similarity_path': new_data_folder + '/drug_sequence_similarity.json',
            'target_sequence_similarity_path': new_data_folder + '/target_sequence_similarity.json',
            # all_results
            'all_results': all_results_folder
        }
        # configs
        self.configs = {
            # graph_data_configs
            'drug_feature_length': 128,
            'target_feature_length': 2000,
            'drug_top_percentage': 0.1,
            'target_top_percentage': 0.1,
            'drug_max_new_num_neighbors': 1,
            'target_max_new_num_neighbors': 1,
            # model_configs
            'drug_embedding_dim': 32,
            'target_embedding_dim': 32,
            'drug_filter_sizes': [3, 6, 9, 12],
            'target_filter_sizes': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
            'graph_encoder_in_length': 32,
            'graph_encoder_num_layers': 6,
            'graph_encoder_node_dropout': 0.2,
            'graph_encoder_edge_dropout': 0.5,
            # train_configs
            'num_folds': 5,
            'num_epoch': 1000,
            'lr': 0.001,
            'weight_decay': 0.0,
            'step_size': 100,
            'gamma': 0.8,
        }
        if load_best_configs:
            self.load_best_configs()

    def load_best_configs(self):
        base_path = self.data_paths['all_results']
        all_files = os.listdir(base_path)
        if not len(all_files):
            print('Fail to load best configs , best configs not exist')
        else:
            best_file = max(all_files, key=lambda file: float(file[3:9]))
            best_file_path = base_path + '/' + best_file
            with open(best_file_path + '/configs.json', 'r') as json_file:
                self.configs = json.load(json_file)
            print('Best configs loaded')

    def print_best_configs(self):
        base_path = self.data_paths['all_results']
        all_files = os.listdir(base_path)
        if not len(all_files):
            print('Fail to print best configs , best configs not exist')
        else:
            best_file = max(all_files, key=lambda file: float(file[3:9]))
            best_file_path = base_path + '/' + best_file
            with open(best_file_path + '/configs.json', 'r') as json_file:
                configs = json.load(json_file)
            max_length = max(len(param) for param in configs.keys())
            print('\nBest configs : ')
            for param, value in configs.items():
                print(f'{param:{max_length}}   {value}')

    def print_current_configs(self):
        max_length = max(len(param) for param in self.configs.keys())
        print('Current configs : ')
        for param, value in self.configs.items():
            print(f'{param:{max_length}}   {value}')

    def save_new_configs_with_f1_score(self, f1_score: float, ):
        base_path = self.data_paths['all_results']
        new_folder = f'{base_path}/f1_{f1_score}'
        file_index = 1
        while os.path.exists(new_folder):
            new_folder = f'{base_path}/f1_{f1_score}({file_index})'
            file_index += 1
        os.makedirs(new_folder)
        with open(new_folder + '/configs.json', 'w') as json_file:
            json.dump(self.configs, json_file)
        print('New configs saved at ' + new_folder)


current_dataset = DatasetWithConfigs('E', load_best_configs=True)
# current_dataset = DatasetWithConfigs('GPCR', load_best_configs=True)
# current_dataset = DatasetWithConfigs('IC', load_best_configs=True)
# current_dataset = DatasetWithConfigs('DAVIS', load_best_configs=True)


if __name__ == '__main__':
    current_dataset.print_best_configs()
