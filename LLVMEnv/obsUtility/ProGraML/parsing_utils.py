import csv
import os
import dgl

def update_networkx_feature(graph_fn, idx, feature_name, feature):
    graph_fn[idx][feature_name] = feature

def convert_networkx_to_dgl(graph, node_attrs=["text_idx", "type"], edge_attrs=["flow", "position"]):
    return dgl.from_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)

# utility for feature extraction
class FeatureExtractor:
    def __init__(self, vocab_csv_path=None, online_update=False):
        self.node_feature_list = ["text", "type", "function", "block"]
        self.node_feature_list_dgl = ["text_idx", "type", "function", "block"]
        self.edge_feature_list = ["flow", "position"]
        self.online_update = online_update

        if vocab_csv_path is None:
            self.vocab_mapping = {}
        else:
            self.vocab_csv_path = vocab_csv_path
            self.vocab_mapping = self.load_vocab_from_csv(vocab_csv_path)

    def load_vocab_from_csv(self, csv_path):
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                vocabs = [row[0] for row in reader]
            return {"text": {v: i for i, v in enumerate(vocabs)}}
        else:
            return {"text": {}}

    def save_vocab_to_csv(self):
        with open(self.vocab_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for token in self.vocab_mapping["text"].keys():
                writer.writerow([token])

    def process_nx_graph(self, graph):
        self.update_graph_with_vocab(
            graph.nodes, self.node_feature_list, self.vocab_mapping
        )

        dgl_graph = convert_networkx_to_dgl(
            graph,
            node_attrs=self.node_feature_list_dgl,
            edge_attrs=self.edge_feature_list,
        )
        return dgl_graph

    def update_vocabs(self, token):
        self.vocab_mapping["text"][token] = len(self.vocab_mapping["text"])
        self.save_vocab_to_csv()

    def update_graph_with_vocab(self, graph_fn, features, vocab):
        for feature_name in features:
            curr_vocab = None
            if feature_name in vocab:
                curr_vocab = vocab[feature_name]
            len_curr_vocab = len(curr_vocab) if curr_vocab is not None else 0
            for graph_item in graph_fn(data=feature_name):
                feature = graph_item[-1]
                idx = graph_item[0]

                if feature_name in vocab:
                    # this is for nodes feature "text", convert this feature to idx for embedding later
                    # aggregate all functions to a single type
                    if (
                        feature.endswith(")")
                        and feature.find(" (") >= 0
                    ):
                        feature = "__function__"
                    token_idx = curr_vocab.get(feature, len_curr_vocab)
                    if (
                        feature_name == "text"
                        and self.online_update
                        and token_idx == len_curr_vocab
                    ):
                        self.update_vocabs(feature)
                        curr_vocab = self.vocab_mapping["text"]
                        token_idx = curr_vocab.get(feature, len_curr_vocab)
                    update_networkx_feature(graph_fn, idx, f"{feature_name}_idx", token_idx)
                elif isinstance(feature, str):
                    assert len(vocab) == 0 and feature_name == "text"
                    update_networkx_feature(graph_fn, idx, f"{feature_name}_idx", -1)
                else:
                    assert isinstance(
                        feature, int
                    ), f"{(feature_name, feature)} is not an int"
