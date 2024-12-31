import os
import torch
import json
import requests
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from utils import set_seed, set_config_args, print_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from explainer import PaGELink

class LinkPredictionPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_dir = 'datasets'
        self.dataset_name = 'sales_process.dgl'
        self.valid_ratio = 0.1
        self.test_ratio = 0.2
        self.emb_dim = 64
        self.hidden_dim = 64
        self.out_dim = 64
        self.src_ntype, self.tgt_ntype = 'account', 'product'
        self.pred_etype = 'buys'
        self.link_pred_op = 'dot'
        self.lr = 0.01
        self.alpha = 1.0
        self.beta = 1.0
        self.num_hops = 2
        self.num_epochs = 20
        self.num_paths = 1
        self.max_path_length = 5
        self.k_core = 2
        self.prune_max_degree = 200
        self.save_explanation = True
        self.saved_explanation_dir = 'saved_explanations'
        self.pred_kwargs = {} if self.link_pred_op != 'cat' else {"in_feats": self.out_dim, "out_feats": 1}
        self.paths = {}
        set_seed(0)

    def load_model(self):
        processed_g = load_dataset(self.dataset_dir, self.dataset_name, self.valid_ratio, self.test_ratio)[1]
        graphs = [g.to(self.device) for g in processed_g]
        self.mp_g, self.train_pos_g, self.train_neg_g, self.val_pos_g, self.val_neg_g, self.test_pos_g, self.test_neg_g = graphs


        print("graphhhhhhhhhhhhh", self.mp_g)

        # encoder = HeteroRGCN(self.mp_g, self.emb_dim, self.hidden_dim, self.out_dim)
        # self.model = HeteroLinkPredictionModel(encoder, self.src_ntype, self.tgt_ntype, self.link_pred_op, **self.pred_kwargs)
        # state = torch.load('saved_models/synthetic_model.pth', map_location='cpu')
        # self.model.load_state_dict(state)
        # self.model.to(self.device)

    def explain_link(self, src_nid, tgt_nid):
        """
        Explain the prediction for a given source and target node.
        """
        pagelink = PaGELink(self.model, lr=self.lr, alpha=self.alpha, beta=self.beta, num_epochs=self.num_epochs, log=True).to(self.device)

        src_nid, tgt_nid = torch.tensor([src_nid], device=self.device), torch.tensor([tgt_nid], device=self.device)

        with torch.no_grad():
            # pred = self.model(src_nid, tgt_nid, self.mp_g).sigmoid().item() > 0.5
            pred = True

        if not pred:
            return {}, f"No link predicted between {src_nid.item()} and {tgt_nid.item()}."

        src_tgt = ((self.src_ntype, int(src_nid)), (self.tgt_ntype, int(tgt_nid)))
        paths, comp_g_edge_mask_dict = pagelink.explain(
            src_nid, tgt_nid, self.mp_g, self.num_hops, self.prune_max_degree, self.k_core, self.num_paths, self.max_path_length, return_mask=True
        )


        pred_edge_to_paths = {src_tgt: paths}
        return paths, pred_edge_to_paths, None

    def visualize_subgraph(self, paths, src_nid, tgt_nid, explanations):
        G = nx.DiGraph()

        for path in paths:
            for edge in path:
                edge_type, src, tgt = edge
                G.add_node(src)
                G.add_node(tgt)
                if edge_type == ('user', 'buys', 'item'):
                    G.add_edge(src, tgt, label='buys')
                elif edge_type == ('item', 'has', 'attr'):
                    G.add_edge(src, tgt, label='has attribute')
                elif edge_type == ('attr', 'of', 'item'):
                    G.add_edge(src, tgt, label='of item')
                elif edge_type == ('item', 'bought_by', 'user'):
                    G.add_edge(src, tgt, label='item bought by')
                else:
                    G.add_edge(src, tgt, label='other')
        G.add_edge(paths[0][0][1], paths[len(paths) - 1][2][2])

        explanations = {}
        for path in paths:
            for edge in path:
                edge_type, src, tgt = edge
                if edge_type == ('user', 'buys', 'item'):
                    explanations[src] = 'user'
                    explanations[tgt] = 'item'
                elif edge_type == ('item', 'has', 'attr'):
                    explanations[src] = 'item'
                    explanations[tgt] = 'attribute'
                elif edge_type == ('attr', 'of', 'item'):
                    explanations[src] = 'attribute'
                    explanations[tgt] = 'item'
                elif edge_type == ('item', 'bought_by', 'user'):
                    explanations[src] = 'item'
                    explanations[tgt] = 'user'
                else:
                    explanations[src] = 'other'
                    explanations[tgt] = 'other'

        node_labels = {}
        node_colors = []
        for node in G.nodes():
            tag = explanations.get(node, 'other')
            if tag == 'user':
                node_labels[node] = f'U ({node})'
                node_colors.append('green')
            elif tag == 'item':
                node_labels[node] = f'I ({node})'
                node_colors.append('red')
            elif tag == 'attribute':
                node_labels[node] = f'A ({node})'
                node_colors.append('blue')
            else:
                node_labels[node] = node
                node_colors.append('skyblue')

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, node_color=node_colors, font_size=7, font_color="white", font_weight='bold', arrowsize=20)

        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=10)

        final_edge = (src_nid, tgt_nid)
        if final_edge in G.edges():
            nx.draw_networkx_edges(G, pos, edgelist=[final_edge], edge_color='orange', width=2.0)

        plt.title('Subgraph Visualization of Explanation Paths')
        plt.show()

    def __llm_request(self, data):
        url = "http://crmdi-gpu4:5000/gpu/llm/text/api/llm_generation/generate"
        prompt = f"<INST><s>{data}</INST>"
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": 1500,
            "temperature": 0.1
        })
        headers = {
            'Content-Type': 'application/json',
            'signature': 'OMGXFyWnG/9ftNG+UKkpaCYmqHM6HMuPTyHeFyFgtlATdBji4SvzExcFd/3U1iWJZXcIH7dQ1RmvO50myp7u69Be7rNY27BZNmonBxQxxYQrbzRxM5G2g4queh4GoYTUNDOARp+TFbUFSY4uVuKqWvvSBCKh5sZJQI5kmGkyAAYo00nxwcoXZYOk9sNVpLwkUnT8OdjWtWIWGGNYLcIjHkWfL8hn9KR3yu7g8CVhoi4C2pKM3WeVbc44+OKdePJunHbmuPZ/W0cf6ksR1i/VVDQzBgLaiUGohp4yA4iFDneg2I4+4BMaEcWUlcToqpBjeGKgIOGLk0R9rRjOYuuctg==-1710142016958'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()

        data = result['text'][0]

        return data

    def save_explanations(self, pred_edge_to_paths):
        if not os.path.exists(self.saved_explanation_dir):
            os.makedirs(self.saved_explanation_dir)

        saved_path_explanation_file = f'pagelink_pred_edge_to_paths.txt'
        saved_path_explanation_path = Path.cwd().joinpath(self.saved_explanation_dir, saved_path_explanation_file)

        with open(saved_path_explanation_path, "w") as f:
            for path, explanation in pred_edge_to_paths.items():
                f.write(f"Path: {path}\n")
                for edge in explanation:
                    f.write(f"  {edge}\n")
                f.write("\n")

    def run_pipeline(self, src_nid=None, tgt_nid=None, index=None):
        self.load_model()

        if index is not None:
        # if src_nid is not None and tgt_nid is not None:
            test_src_nids, test_tgt_nids = self.test_pos_g.edges()
            src_nid, tgt_nid = test_src_nids[index].item(), test_tgt_nids[index].item()
            # src_nid, tgt_nid = src_nid, tgt_nid

        if src_nid is None or tgt_nid is None:
            raise ValueError("Either `src_nid` and `tgt_nid` or `index` must be provided.")

        print(f"\n\nExplaining link prediction for src_nid: {src_nid}, tgt_nid: {tgt_nid}")
        paths, pred_edge_to_paths, error = self.explain_link(src_nid, tgt_nid)


        if error:
            print(error)
            return

        if self.save_explanation:
            print("Saving explanations...")
            self.save_explanations(pred_edge_to_paths)

        for paths in pred_edge_to_paths.values():
            explanations = {}
            for path in paths:
                for edge in path:
                    for node in [edge[1], edge[2]]:
                        if 'user' in str(node):
                            explanations[node] = 'user'
                        elif 'item' in str(node):
                            explanations[node] = 'item'
                        elif 'attr' in str(node):
                            explanations[node] = 'attribute'
                        else:
                            explanations[node] = 'other'
            # self.visualize_subgraph(paths, src_nid, tgt_nid, explanations)

        prompt = f""" 
            Given an explanation in the form of paths connecting users to items via attributes, explain the reasoning behind the recommendation in natural language.

            Sample Input: {{(('user', 27), ('item', 33)): [[(('user', 'buys', 'item'), 27, 10), (('item', 'has', 'attr'), 10, 18), (('attr', 'of', 'item'), 18, 33)]]}}
            Sample Reasoning: We recommended this product because you previously bought Item 10, which shares the same attribute (Attribute 18) with Item 33. This suggests that Item 33 may also appeal to you            
            
            Input Format:
            Explanation: Dictionary where the key is a tuple representing a user and an item, and the value is a list of paths. Each path is a list of tuples representing the relationships (edges) and the nodes connected by these edges.

            Input: {pred_edge_to_paths}
            Reason: Based on the input explanation path,
        """

        # reason = self.__llm_request(prompt)
        # print("Explanation: ", reason)

if __name__ == '__main__':
    pipeline = LinkPredictionPipeline()

    pipeline.run_pipeline(index=18)

    # pipeline.run_pipeline(src_nid=51, tgt_nid=10)
