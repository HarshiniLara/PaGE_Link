import os
import torch
import requests
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from tqdm.auto import tqdm
from pathlib import Path

from utils import set_seed, print_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from explainer import PaGELink

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
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, node_color=node_colors, font_size=7, font_color="white", font_weight='bold', arrowsize=20, ax=ax)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=10, ax=ax)

    final_edge = (src_nid, tgt_nid)
    if final_edge in G.edges():
        nx.draw_networkx_edges(G, pos, edgelist=[final_edge], edge_color='orange', width=2.0, ax=ax)

    ax.set_title('Subgraph Visualization of Explanation Paths')
    st.pyplot(fig)

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

def main():
    st.title('PaGE Link for recommended product')
    args = argparse.Namespace(
        device_id=-1, dataset_dir='datasets', dataset_name='synthetic', valid_ratio=0.1,
        test_ratio=0.2, max_num_samples=-1, emb_dim=64, hidden_dim=64, out_dim=64,
        saved_model_dir='saved_models', saved_model_name='synthetic_model', src_ntype='user',
        tgt_ntype='item', pred_etype='likes', link_pred_op='dot', lr=0.01, alpha=1.0,
        beta=1.0, num_hops=2, num_epochs=20, num_paths=1, max_path_length=5,
        k_core=2, prune_max_degree=200, save_explanation=True,
        saved_explanation_dir='saved_explanations', config_path='config.yaml'
    )
    
    index = st.number_input('Enter index for source and target node:', min_value=0, step=1)

    # if 'citation' in args.dataset_name:
    #     args.src_ntype = 'author'
    #     args.tgt_ntype = 'paper'
    # elif 'synthetic' in args.dataset_name:
    #     args.src_ntype = 'user'
    #     args.tgt_ntype = 'item'

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
    src_nid, tgt_nid = test_src_nids[index].unsqueeze(0), test_tgt_nids[index].unsqueeze(0)

    st.write("Source Node ID: ", src_nid.item())
    st.write("Target Node ID: ", tgt_nid.item())

    # if st.button('Explain'):
    pred_edge_to_comp_g_edge_mask = {}
    pred_edge_to_paths = {}
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

    if pred_edge_to_paths == {}:
        st.write("No explanation paths found.")
        return

    st.write("Explanation Paths:", str(pred_edge_to_paths))

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

    visualize_subgraph_with_labels(paths, src_nid.item(), tgt_nid.item(), explanations)

    prompt = f""" 
    Given an explanation in the form of paths connecting users to items via attributes, explain the reasoning behind the recommendation.
    Need reasoning for the recommendations in natural language for the explanation in the form of paths.

    Sample Input: {{(('user', 27), ('item', 33)): [[(('user', 'buys', 'item'), 27, 10), (('item', 'has', 'attr'), 10, 18), (('attr', 'of', 'item'), 18, 33)]]}}
    Sample Reasoning: We recommended this product because you previously bought Item 10, which shares the same attribute (Attribute 18) with Item 33. This suggests that Item 33 may also appeal to you.

    Input Format:
    Explanation: Dictionary where the key is a tuple representing a user and an item, and the value is a list of paths. Each path is a list of tuples representing the relationships (edges) and the nodes connected by these edges.

    Input: {pred_edge_to_paths}
    Reason: Based on the input path,
    """
    reason = __llm_request(prompt)
    st.write("LLM Reasoning:", reason)

    if args.save_explanation:
        if not os.path.exists(args.saved_explanation_dir):
            os.makedirs(args.saved_explanation_dir)

        saved_path_explanation_file = f'pagelink_{args.saved_model_name}_pred_edge_to_paths1.txt'
        saved_path_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_path_explanation_file)

        with open(saved_path_explanation_path, "w") as f:
            for path, explanation in pred_edge_to_paths.items():
                f.write(f"Path: {path}\n")
                f.write(f"Explanation: {explanation}\n")
                f.write("\n")

if __name__ == '__main__':
    main()


