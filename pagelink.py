import os
import torch
import argparse
from tqdm.auto import tqdm
from pathlib import Path

import requests
import json

import networkx as nx
import matplotlib.pyplot as plt

from utils import set_seed, print_args, set_config_args, plot_hetero_graph
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from explainer import PaGELink


parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=-1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='aug_citation')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')

'''
Explanation args
'''
parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate') 
parser.add_argument('--alpha', type=float, default=1.0, help='explainer on-path edge regularizer weight') 
parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight') 
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--num_epochs', type=int, default=20, help='How many epochs to learn the mask')
parser.add_argument('--num_paths', type=int, default=2, help='How many paths to generate')
parser.add_argument('--max_path_length', type=int, default=5, help='max length of generated paths')
parser.add_argument('--k_core', type=int, default=2, help='k for the k-core graph') 
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1') 
parser.add_argument('--save_explanation', default=False, action='store_true', 
                    help='Whether to save the explanation')
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pagelink')

if 'citation' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'

elif 'synthetic' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'item'    

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}
    
if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'
    
print_args(args)
set_seed(0)

processed_g = load_dataset(args.dataset_dir, args.dataset_name, args.valid_ratio, args.test_ratio)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)  

pagelink = PaGELink(model, 
                    lr=args.lr,
                    alpha=args.alpha, 
                    beta=args.beta, 
                    num_epochs=args.num_epochs,
                    log=True).to(device)


test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = range(test_src_nids.shape[0])
ind = 18

pred_edge_to_comp_g_edge_mask = {}
pred_edge_to_paths = {}

# for ind in tqdm(test_ids):
src_nid, tgt_nid = test_src_nids[ind].unsqueeze(0), test_tgt_nids[ind].unsqueeze(0)

print("src_nid: ", src_nid)
print("tgt_nid: ", tgt_nid)

with torch.no_grad():
    pred = model(src_nid, tgt_nid, mp_g).sigmoid().item() > 0.5

if pred:
    src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
    paths, comp_g_edge_mask_dict = pagelink.explain(src_nid, 
                                                    tgt_nid, 
                                                    mp_g,
                                                    args.num_hops,
                                                    args.prune_max_degree,
                                                    args.k_core, 
                                                    args.num_paths, 
                                                    args.max_path_length,
                                                    return_mask=True)
    
    pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict 
    pred_edge_to_paths[src_tgt] = paths

def visualize_subgraph_with_labels(paths, src_nid, tgt_nid, explanations):
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

# visualize_subgraph_with_labels(paths, src_nid.item(), tgt_nid.item(), explanations)



def __llm_request(data):
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

prompt = f""" 
            Given an explanation in the form of paths connecting users to items via attributes, explain the reasoning behind the recommendation in natural language.

            Sample Input: {{(('user', 27), ('item', 33)): [[(('user', 'buys', 'item'), 27, 10), (('item', 'has', 'attr'), 10, 18), (('attr', 'of', 'item'), 18, 33)]]}}
            Sample Reasoning: We recommended this product because you previously bought Item 10, which shares the same attribute (Attribute 18) with Item 33. This suggests that Item 33 may also appeal to you            
            
            Input Format:
            Explanation: Dictionary where the key is a tuple representing a user and an item, and the value is a list of paths. Each path is a list of tuples representing the relationships (edges) and the nodes connected by these edges.

            Input: {pred_edge_to_paths}
            Reason: Based on the input explanation path,
        """

# reason = __llm_request(prompt)
# print(reason)




if args.save_explanation:
    if not os.path.exists(args.saved_explanation_dir):
        os.makedirs(args.saved_explanation_dir)

    saved_path_explanation_file = f'pagelink_{args.saved_model_name}_pred_edge_to_paths1.txt'
    saved_path_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_path_explanation_file)
    
    with open(saved_path_explanation_path, "w") as f:
        num_paths = len(pred_edge_to_paths)
        print(f"Total number of paths: {num_paths}")
        
        for path, explanation in pred_edge_to_paths.items():
            f.write(f"Path: {path}\n")
            
            for edge in explanation:
                f.write(f"  {edge}\n")
            
            f.write("\n")










