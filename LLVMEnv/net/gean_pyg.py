import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

from LLVMEnv.net.edge_attn import EdgeEncoding, EdgeAttn

class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size,
        node_hidden_size=32,
        n_message_passes=8,
        edge_emb_dim=64,
        num_actions=100,
        num_heads: int = 4,
    ):
        super(GNNEncoder, self).__init__()

        self.node_vocab_size = node_vocab_size
        self.node_hidden_size = node_hidden_size
        self.n_message_passes = n_message_passes

        self.num_actions = num_actions

        num_embedding = node_vocab_size
        self.node_embedding = nn.Embedding(num_embedding, node_hidden_size)

        embed_dim = self.node_hidden_size

        self.gnn = nn.ModuleList(
            [
                EdgeAttn(
                    out_channels=self.node_hidden_size,
                    edge_dim=edge_emb_dim,
                    num_heads=num_heads,
                    **({})
                )
                for i in range(self.n_message_passes)
            ]
        )

        self.edge_encoder = EdgeEncoding(edge_emb_dim)

        q_dim = embed_dim
        q_dim += edge_emb_dim

        self.Q = nn.Sequential(
            nn.Linear(q_dim, self.node_hidden_size),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size, self.num_actions),
        )

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.Q.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                module.reset_parameters()

    def forward(
        self,
        g,
        gnn=None,
    ):
        # Set default
        gnn = self.gnn
        q_net = self.Q

        self.featurize_nodes(g)

        edge_feat = self.get_edge_encoding(g)

        edge_index = g.edge_index

        res = g["feat"]

        layers = gnn

        res, edge_feat = self.layers_encode(
            res, edge_feat, layers, edge_index,
        )

        g["feat"] = res

        instruct_node = g["type"].flatten() == 0
        assert g.batch.ndim == 1
        batch = g.batch[instruct_node]
        feat = g["feat"][instruct_node]
        graph_agg = scatter_mean(feat, batch, dim=0, dim_size=g.num_graphs)

        edges_batch_idx = g.batch[g.edge_index[0]]
        edge_agg = scatter_mean(edge_feat, edges_batch_idx, dim=0, dim_size=g.num_graphs)

        graph_agg = torch.cat([graph_agg, edge_agg], dim=1)
        res = q_net(graph_agg)

        res = F.log_softmax(res, dim=-1)

        return res

    def layers_encode(
        self, 
        res,
        edge_feat,
        layers, 
        edge_index, 
    ):
        for i, layer in enumerate(layers):

            res, edge_feat = layer(
                res,
                edge_index,
                edge_attr=edge_feat,
            )

        return res, edge_feat

    def featurize_nodes(self, g):
        # This is very CompilerGym specific, can be rewritten for other tasks
        features = []

        features.append(self.node_embedding(g["x"].flatten()))

        g["feat"] = torch.cat(features)

    def get_edge_embedding(self, g, edge_index, num_origin_edges=None):
        if self.use_edge_embedding:
            function = g["function"].flatten()
            block = g["block"].flatten()
            i = edge_index[0]
            j = edge_index[1]
            func_diff = function[i] - function[j]
            func_diff = func_diff.bool().float().unsqueeze(-1)
            blk_diff = block[i] - block[j]
            blk_diff = blk_diff.bool().float().unsqueeze(-1)
            edge_attr = torch.cat([func_diff, blk_diff], dim=1)

            return edge_attr

    def get_edge_encoding(self, g):
        edge_types = g["flow"].flatten()
        edge_pos = g["position"].flatten()
        edge_index = g.edge_index
        block = g["block"].flatten()
        src_block = block[edge_index[0]]
        tgt_block = block[edge_index[1]]
        block_idx = torch.stack([src_block, tgt_block])
        edge_enc = self.edge_encoder(edge_types, edge_pos, block_idx)
        return edge_enc
