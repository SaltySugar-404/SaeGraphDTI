import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

from configs import *

drug_embedding_dim = current_dataset.configs['drug_embedding_dim']
target_embedding_dim = current_dataset.configs['target_embedding_dim']
drug_filter_sizes = current_dataset.configs['drug_filter_sizes']
target_filter_sizes = current_dataset.configs['target_filter_sizes']
graph_encoder_in_length = current_dataset.configs['graph_encoder_in_length']
graph_encoder_num_layers = current_dataset.configs['graph_encoder_num_layers']
graph_encoder_node_dropout = current_dataset.configs['graph_encoder_node_dropout']


class SeqFilter(nn.Module):
    def __init__(self, num_embedding, embedding_dim, out_length, all_filter_size: list):
        super(SeqFilter, self).__init__()
        self.out_length = out_length
        self.all_filter_size = all_filter_size

        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim // 2, kernel_size=5),
            nn.AvgPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=embedding_dim // 2, out_channels=embedding_dim // 4, kernel_size=3),
            nn.AvgPool1d(kernel_size=3),
            nn.ReLU()
        )
        self.all_filter = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim // 4, out_channels=out_length, kernel_size=kernel_size, stride=1, padding=0)
            for kernel_size in all_filter_size
        ])
        self.linear = nn.Sequential(
            nn.Linear(in_features=len(all_filter_size) * out_length, out_features=len(all_filter_size) // 2 * out_length),
            nn.ReLU(),
            nn.Linear(in_features=len(all_filter_size) // 2 * out_length, out_features=out_length)
        )
        self._weight_initialize()

    def _weight_initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        # x, _ = self.self_attention(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        all_x = []
        for layer_index in range(len(self.all_filter_size)):
            filter = self.all_filter[layer_index]
            current_x = filter(x)
            current_x, _ = torch.max(current_x, dim=2)
            all_x.append(current_x)
        # x = torch.stack(all_x, dim=1)
        x = torch.cat(all_x, dim=-1)
        x = self.linear(x)
        # x, _ = torch.max(x, dim=1)
        return x


class GraphModel(nn.Module):
    def __init__(self, drug_edges, target_edges, train_positive_edges, in_length, num_layers, node_dropout):
        super(GraphModel, self).__init__()
        self.all_edges = torch.cat([drug_edges, target_edges, train_positive_edges], dim=-1)
        self.all_edge_weights = nn.Parameter(torch.ones(self.all_edges.shape[1]))
        self.selected_edges = self.select_with_weight()
        self.num_layers = num_layers
        self.node_dropout = node_dropout

        self.all_gat = nn.ModuleList([
            GATConv(in_channels=in_length, out_channels=in_length)
            for _ in range(num_layers)
        ])
        self.all_sage = nn.ModuleList([
            SAGEConv(in_channels=in_length, out_channels=in_length)
            for _ in range(num_layers)
        ])
        self.all_linear = nn.ModuleList([
            nn.Linear(in_features=in_length * 3, out_features=in_length)
            for _ in range(num_layers)
        ])

    def select_with_weight(self):
        all_edge_weights = torch.sigmoid(self.all_edge_weights)
        mask = all_edge_weights > 0.5
        return self.all_edges[:, mask]

    def encoder(self, drug_features, target_features):
        node_features = torch.cat([drug_features, target_features])
        for layer_index in range(self.num_layers):
            gat = self.all_gat[layer_index]
            sage = self.all_sage[layer_index]
            linear = self.all_linear[layer_index]
            node_features_with_gat = gat(node_features, self.selected_edges)
            node_features_with_sage = sage(node_features, self.selected_edges)
            node_features_with_gat = F.dropout(node_features_with_gat, p=self.node_dropout)
            node_features_with_sage = F.dropout(node_features_with_sage, p=self.node_dropout)
            node_features = linear(torch.cat([node_features, node_features_with_gat, node_features_with_sage], dim=-1))
            node_features = F.tanh(node_features)
        return node_features

    def decoder(self, encoder_features, edges):
        return (encoder_features[edges[0]] * encoder_features[edges[1]]).sum(dim=-1)

    def forward(self, drug_features, target_features, edges):
        train_encoder_features = self.encoder(drug_features, target_features)
        predicts = self.decoder(train_encoder_features, edges)
        return predicts


class AllModel(nn.Module):
    def __init__(self, drug_num_embeddings, target_num_embeddings, drug_edges, target_edges, train_positive_edges):
        super(AllModel, self).__init__()
        self.drug_model = SeqFilter(num_embedding=drug_num_embeddings, embedding_dim=drug_embedding_dim, out_length=graph_encoder_in_length,
                                    all_filter_size=drug_filter_sizes)
        self.target_model = SeqFilter(num_embedding=target_num_embeddings, embedding_dim=target_embedding_dim, out_length=graph_encoder_in_length,
                                      all_filter_size=target_filter_sizes)
        self.graph_model = GraphModel(drug_edges, target_edges, train_positive_edges, in_length=graph_encoder_in_length,
                                      num_layers=graph_encoder_num_layers, node_dropout=graph_encoder_node_dropout)

    def forward(self, drug_features, target_features, edges):
        drug_features = self.drug_model(drug_features)
        target_features = self.target_model(target_features)
        edge_predicts = self.graph_model(drug_features, target_features, edges)
        return F.sigmoid(edge_predicts)

    def predict(self, drug_features, target_features, edges):
        drug_features = self.drug_model(drug_features)
        target_features = self.target_model(target_features)
        encoder_features = self.graph_model.encoder(drug_features, target_features)
        edge_predicts = self.graph_model.decoder(encoder_features, edges)
        return F.sigmoid(edge_predicts)
