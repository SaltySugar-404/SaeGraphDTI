import random

import torch.backends.cudnn as cudnn
import torch.optim as optim

from data import *
from evaluation import *
from model import *

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

cudnn.deterministic = True
cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'


def get_train_test_in_folds(all_positive_edges, all_negative_edges, num_folds, shuffle=True):
    all_edges_0 = all_positive_edges[0] + all_negative_edges[0]
    all_edges_1 = all_positive_edges[1] + all_negative_edges[1]
    all_labels = [1] * len(all_positive_edges[0]) + [0] * len(all_negative_edges[0])
    if shuffle:
        all_edges_and_labels = list(zip(all_edges_0, all_edges_1, all_labels))
        random.shuffle(all_edges_and_labels)
        all_edges_0, all_edges_1, all_labels = zip(*all_edges_and_labels)

    num_elements = len(all_edges_0)
    fold_size = num_elements // num_folds
    all_edges_0_folds, all_edges_1_folds, all_labels_folds = [], [], []
    for fold_index in range(num_folds):
        start_index = fold_index * fold_size
        end_index = (fold_index + 1) * fold_size if fold_index < num_folds - 1 else num_elements
        all_edges_0_folds.append(list(all_edges_0[start_index:end_index]))
        all_edges_1_folds.append(list(all_edges_1[start_index:end_index]))
        all_labels_folds.append(list(all_labels[start_index:end_index]))
    all_train_folds, all_test_folds = [], []
    for fold_index in range(num_folds):
        current_train_edges_0 = sum(all_edges_0_folds[:fold_index] + all_edges_0_folds[fold_index + 1:], [])
        current_test_edges_0 = all_edges_0_folds[fold_index]
        current_train_edges_1 = sum(all_edges_1_folds[:fold_index] + all_edges_1_folds[fold_index + 1:], [])
        current_test_edges_1 = all_edges_1_folds[fold_index]
        current_train_labels = sum(all_labels_folds[:fold_index] + all_labels_folds[fold_index + 1:], [])
        current_test_labels = all_labels_folds[fold_index]
        all_train_folds.append(([current_train_edges_0, current_train_edges_1], current_train_labels))
        all_test_folds.append(([current_test_edges_0, current_test_edges_1], current_test_labels))
    return all_train_folds, all_test_folds


# train_configs
num_folds = current_dataset.configs['num_folds']
num_epoch = current_dataset.configs['num_epoch']
lr = current_dataset.configs['lr']
weight_decay = current_dataset.configs['weight_decay']
step_size = current_dataset.configs['step_size']
gamma = current_dataset.configs['gamma']

show_predict_score = False

if __name__ == '__main__':
    graph = GraphData()
    drug_features = torch.tensor(graph.drug_features, dtype=torch.int32).to(device)
    target_features = torch.tensor(graph.target_features, dtype=torch.int32).to(device)
    drug_edges = convert_to_bidirectional(graph.drug_edges)
    target_edges = convert_to_bidirectional(graph.target_edges)
    positive_edges = convert_to_bidirectional(graph.positive_edges)
    negative_edges = convert_to_bidirectional(graph.negative_edges)
    all_train_folds, all_test_folds = get_train_test_in_folds(positive_edges, negative_edges, num_folds, shuffle=True)
    all_accuracy, all_precision, all_recall, all_f1_score, all_auc_score, all_aupr_score = [], [], [], [], [], []

    for fold_index in range(num_folds):
        print(f'Fold {fold_index + 1}')
        train_fold, test_fold = all_train_folds[fold_index], all_test_folds[fold_index]
        train_edges, train_labels = train_fold
        test_edges, test_labels = test_fold
        train_positive_edges_0 = [train_edges[0][index] for index in range(len(train_labels)) if train_labels[index] == 1]
        train_positive_edges_1 = [train_edges[1][index] for index in range(len(train_labels)) if train_labels[index] == 1]
        train_positive_edges = [train_positive_edges_0, train_positive_edges_1]

        train_edges = torch.tensor(train_edges, dtype=torch.int64).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
        test_edges = torch.tensor(test_edges, dtype=torch.int64).to(device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)

        model = AllModel(drug_num_embeddings=torch.max(drug_features) + 1,
                         target_num_embeddings=torch.max(target_features) + 1,
                         drug_edges=torch.tensor(drug_edges, dtype=torch.int64).to(device),
                         target_edges=torch.tensor(target_edges, dtype=torch.int64).to(device),
                         train_positive_edges=torch.tensor(train_positive_edges, dtype=torch.int64).to(device)).to(device)
        loss_fc = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        last_labels = np.array([])
        last_predicts = np.array([])
        for epoch in tqdm(range(num_epoch)):
            model.train()
            train_predicts = model(drug_features, target_features, train_edges).to(device)
            train_accuracy = float(np.sum(train_labels.cpu().detach().numpy() == np.round(train_predicts.cpu().detach().numpy())) / train_labels.shape[0])
            train_loss = loss_fc(train_predicts, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            model.eval()
            test_predicts = model.predict(drug_features, target_features, test_edges).to(device)
            test_loss = loss_fc(test_predicts, test_labels)

            if epoch == num_epoch - 1:
                last_labels = test_labels.cpu().detach().numpy()
                last_predicts = test_predicts.cpu().detach().numpy()
                if show_predict_score:
                    for index in range(len(test_edges[0])):
                        print(f'({test_edges[0][index]:<4}, {test_edges[1][index]:<4}) -> {last_predicts[index]:<5.3f}', end='   ')
                        if not (index + 1) % 10:
                            print()
                    print()

        accuracy, precision, recall, f1_score = get_confusion_matrix_results(labels=last_labels, predicts=np.round(last_predicts))
        auc_score, aupr_score = get_curve_results(labels=last_labels, predicts=last_predicts)
        print(f'accuracy = {accuracy:.4f} , precision = {precision:.4f} , recall = {recall:.4f} , F1_score = {f1_score:.4f} , AUC = {auc_score:.4f} , AUPR = {aupr_score:.4f}')
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1_score.append(f1_score)
        all_auc_score.append(auc_score)
        all_aupr_score.append(aupr_score)

    accuracy_mean, accuracy_std = get_mean_and_std(all_accuracy)
    precision_mean, precision_std = get_mean_and_std(all_precision)
    recall_mean, recall_std = get_mean_and_std(all_recall)
    f1_score_mean, f1_score_std = get_mean_and_std(all_f1_score)
    auc_score_mean, auc_score_std = get_mean_and_std(all_auc_score)
    aupr_score_mean, aupr_score_std = get_mean_and_std(all_aupr_score)
    print(f'accuracy = {accuracy_mean}({accuracy_std})')
    print(f'precision = {precision_mean}({precision_std})')
    print(f'recall = {recall_mean}({recall_std})')
    print(f'f1_score = {f1_score_mean}({f1_score_std})')
    print(f'auc_score = {auc_score_mean}({auc_score_std})')
    print(f'aupr_score = {aupr_score_mean}({aupr_score_std})')
    current_dataset.save_new_configs_with_f1_score(f1_score=f1_score_mean)
