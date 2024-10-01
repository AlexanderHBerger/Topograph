from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float
    
import enum

import itertools
import matplotlib.pyplot as plt
from skimage import color, graph, measure
from matplotlib.collections import LineCollection
import networkx as nx
from tqdm import tqdm

class FiltrationType(enum.Enum):
    SUPERLEVEL = "superlevel"
    SUBLEVEL = "sublevel"
    BOTHLEVELS = "bothlevels"

class ActivationType(enum.Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class DiceType(enum.Enum):
    DICE = "dice"
    CLDICE = "cldice"

class AggregationType(enum.Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"

class ThresholdDistribution(enum.Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"

def convert_to_one_vs_rest( # this is more of a one-vs-max strategy
        prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"],
) -> Float[torch.Tensor, "batch channel *spatial_dimensions"]:
    """
    Converts a multi-class prediction tensor into a one-vs-rest format by building 
    the softmax over each class (one) and the max of all other classes (rest).

    Args:
        prediction (torch.Tensor): The input prediction tensor of shape (batch, channel, *spatial_dimensions).

    Returns:
        torch.Tensor: The converted prediction tensor of shape (batch, channel, *spatial_dimensions).
    """
    converted_prediction = torch.zeros_like(prediction)

    for channel in range(prediction.shape[1]):
        # Get logits for the channel class
        channel_logits = prediction[:,channel].unsqueeze(1)

        # For each pixel, get the class with the highest probability but exclude the channel class
        rest_logits = torch.max(prediction[:, torch.arange(prediction.shape[1]) != channel], dim=1).values.unsqueeze(1)

        # Apply softmax to get probabilities and select the probability of the channel class
        converted_prediction[:, channel] = torch.softmax(torch.cat([rest_logits, channel_logits], dim=1), dim=1)[:,1]

    return converted_prediction


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    else:
        raise ValueError('input tensor must have 4D with shape: (batch, channel, height, width)')


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    else:
        raise ValueError('input tensor must have 4D with shape: (batch, channel, height, width)')


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

def rev_cantor_pairing(z):
    w = np.floor((np.sqrt(8*z + 1) - 1) / 2)
    t = (w**2 + w) / 2
    y = z - t
    x = w - y
    return int(x), int(y)

def compute_diffs(labelled_components):
    h_diff = labelled_components[:-1, :] - labelled_components[1:, :]
    v_diff = labelled_components[:, :-1] - labelled_components[:, 1:]
    return h_diff, v_diff

def new_compute_diffs(paired_img_batch: torch.Tensor):
    h_diff = paired_img_batch[:,:-1, :] - paired_img_batch[:,1:, :]
    v_diff = paired_img_batch[:,:, :-1] - paired_img_batch[:,:, 1:]
    h_diff = h_diff != 0
    v_diff = v_diff != 0
    return h_diff, v_diff

def new_compute_diag_diffs(paired_img_batch: torch.Tensor, th: int = 11 ):
    weight = torch.tensor([[1, -1], [-1, 1]], device=paired_img_batch.device).unsqueeze(0).unsqueeze(0)
    diag_connections = F.conv2d(paired_img_batch.unsqueeze(1).float(), weight.float()).squeeze(1)
    diagr = diag_connections > th
    diagl = diag_connections < -th
    special_case_r = torch.logical_or(diag_connections == 7, diag_connections == 4)
    special_case_l = torch.logical_or(diag_connections == -7, diag_connections == -4)
    # special_case_r = torch.zeros_like(diagr)
    # special_case_l = torch.zeros_like(diagl)

    return diagr, diagl, special_case_r, special_case_l

def compute_diag_diffs(labelled_components):
    diagr_diff = labelled_components[:-1, :-1] - labelled_components[1:, 1:]
    diagl_diff = labelled_components[1:, :-1] - labelled_components[:-1, 1:]
    return diagr_diff, diagl_diff

def fill_adj_matr(adj_matrix, h_edges, v_edges):
    adj_matrix[*h_edges] = True
    adj_matrix[*h_edges[::-1]] = True  # Add the transposed edges
    adj_matrix[*v_edges] = True
    adj_matrix[*v_edges[::-1]] = True  # Add the transposed edges

    # remove self loops
    np.fill_diagonal(adj_matrix, False)
    
    return adj_matrix


def aufdickung(image, inc = 1):
    """
    Converts a binarized iamge (prediction or GT) into a 5x5 aufdickung image.

    Args:
        image (np.ndarray): The input image of shape (height, width).
        inc (int): The overlap of the forgound pixels to the neighboring pixels (e.g. 2 for GT and 1 for Prediction) .

    Returns:
        image (np.ndarray): The aufdickung image of shape (5*height, 5*width).
    """
    if inc != 1 and inc != 2:
        raise ValueError('inc must be 1 or 2')
    
    pad_value = 0 if inc == 1 else 1
    filter = torch.ones((7,7), device=image.device)

    filter = F.pad(filter, (1,1,1,1), value=pad_value)

    smooth_img = F.conv_transpose2d(image.unsqueeze(1).float(), filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_img[smooth_img > 0] = 1

    return smooth_img


def smoothing_5x5(preds, gts):
    """
    Performs the aufdickung with distinct aufdickungs levels for foreground pred/GT and combined foreground.of pred and GT.
    """

    intersection_filter = torch.ones((9,9), device=preds.device)
    rest_filter = torch.ones((7,7), device=preds.device)

    # pad the rest filter with zeros
    rest_filter = F.pad(rest_filter, (1,1,1,1), value=0)

    intersection = torch.logical_and(preds, gts)
    pred_fg = torch.logical_and(preds, ~intersection)
    gt_fg = torch.logical_and(gts, ~intersection)

    smooth_intersect = F.conv_transpose2d(intersection.unsqueeze(1).float(), intersection_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_intersect[smooth_intersect > 0] = 3

    smooth_pred = F.conv_transpose2d(pred_fg.unsqueeze(1).float(), rest_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_pred[smooth_pred > 0] = 1

    smooth_gt = F.conv_transpose2d(gt_fg.unsqueeze(1).float(), rest_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_gt[smooth_gt > 0] = 2


    smooth_img = smooth_pred + smooth_gt + smooth_intersect
    smooth_img[smooth_img > 3] = 3

    return smooth_img



def smoothing_7x7(preds, gts):
    base_filter = torch.ones((9,9), device=preds.device)
    filter_1 = F.pad(base_filter, (2,2,2,2), value=0)
    filter_2 = F.pad(base_filter, (1,1,1,1), value=1)
    filter_2 = F.pad(filter_2, (1,1,1,1), value=0)
    filter_3 = F.pad(base_filter, (2,2,2,2), value=1)


    intersection = torch.logical_and(preds, gts)
    pred_fg = torch.logical_and(preds, ~intersection)
    gt_fg = torch.logical_and(gts, ~intersection)

    smooth_pred = F.conv_transpose2d(pred_fg.unsqueeze(1).float(), filter_1.unsqueeze(0).unsqueeze(0), stride=7, padding=3)[:,0]
    smooth_pred[smooth_pred > 0] = 1

    smooth_gt = F.conv_transpose2d(gt_fg.unsqueeze(1).float(), filter_2.unsqueeze(0).unsqueeze(0), stride=7, padding=3)[:,0]
    smooth_gt[smooth_gt > 0] = 2

    smooth_inters = F.conv_transpose2d(intersection.unsqueeze(1).float(), filter_3.unsqueeze(0).unsqueeze(0), stride=7, padding=3)[:,0]
    smooth_inters[smooth_inters > 0] = 3

    smooth_img = smooth_pred + smooth_gt + smooth_inters
    smooth_img[smooth_img > 3] = 3

    return smooth_img

"""CHECKS FOR THE DIFFERENT AUFDICKUNG FUNCTIONS"""

def create_aufdickung_util(image, mode = "3diff_4color"):

    pred  = np.zeros_like(image)
    pred[image == 1] = 1
    pred[image == 3] = 1

    gt = np.zeros_like(image)
    gt[image == 2] = 1
    gt[image == 3] = 1

    # turn into tensor and add batch dimension
    pred_t = torch.tensor(pred).unsqueeze(0)
    gt_t = torch.tensor(gt).unsqueeze(0)


    if mode == "3diff_4color":
        auf = smoothing_7x7(pred_t, gt_t)
        # remove batch dimension
        auf = auf.squeeze(0)
        # turn to numpy
        auf = auf.numpy()


    elif mode == "2diff_2color":
        auf_p = smoothing_5x5(pred_t)
        auf_g = smoothing_5x5(gt_t)
        auf = auf_p + auf_g*2

        # remove batch dimension
        auf = auf.squeeze(0)
        # turn to numpy
        auf = auf.numpy()

    elif mode == "2diff_4color":
        auf = smoothing_5x5(pred_t, gt_t)

        # remove batch dimension
        auf = auf.squeeze(0)
        # turn to numpy
        auf = auf.numpy()

    else:
        raise ValueError("Mode not implemented")

    # turn auf to integer
    auf = auf.astype(int)
    return auf


def create_permuation_plot(perm_item_list, mode = "3diff_4color", product = False):
    # create permutation list
    if product:
        perms = list(itertools.product(perm_item_list, repeat=4))
        # remove combinations with only the same element
        perms = [perm for perm in perms if len(set(perm)) > 1]
    else:
        perms = list(itertools.permutations(perm_item_list))

    # create listed colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'black', 'purple'])
    
    # count the permutations and assign the number of rows and columns
    num_perms = len(perms)
    rows = num_perms // 6
    cols = 6

    # add a row if the number of permutations is not divisible by 6
    if num_perms % 6 != 0:
        rows += 1

    size = (20, 20 * rows / cols)
    fig, ax = plt.subplots(rows, cols, figsize=size )

    # add indices the rows and columns of the plots as row headers and column headers
    for i in range(rows):
        ax[i, 0].set_ylabel(i)
    for i in range(cols):
        ax[0, i].set_title(i)
        
    for i, perm in enumerate(perms):
        image = np.zeros((2, 2))
        image[0, 0] = perm[0]
        image[0, 1] = perm[1]
        image[1, 0] = perm[2]
        image[1, 1] = perm[3]

        auf = create_aufdickung_util(image, mode = mode)
        j = i // cols
        k = i % cols

        ax[j, k].imshow(auf, cmap=cmap, vmin = 0, vmax = 3)
        lab = measure.label(auf, background=None)
        rbg = color.label2rgb(lab)    
        rag = graph.rag_mean_color(rbg, lab, connectivity=1)
        lc = show_rag_mod(lab, rag)
        # plot the line collection of the RAG
        ax[j, k].add_collection(lc)


    plt.show()

def show_rag_mod(
    labels,
    rag,
):
 
    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))


    edge_cmap = plt.get_cmap("summer")
    edge_width=1.5
    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    lines = [
        [rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
        for (n1, n2) in rag.edges()
    ]

    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    edge_weights = [d['weight'] for x, y, d in rag.edges(data=True)]
    lc.set_array(np.array(edge_weights))

    return lc

def visualise_diags(diag_r_label, diag_l_label):
    diag_images = get_diag_perms()
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'black', 'purple'])
    fig, ax = plt.subplots(14, 6, figsize=(20, 60))

    for i, image in enumerate(diag_images):
        j = i // 6
        k = i % 6
        ax[j, k].imshow(image[0], cmap=cmap, vmin = 0, vmax = 3)
        # add information about the diag_r and diag_l labels
        ax[j, k].set_title(f"Diag_r: {diag_r_label[i]}, Diag_l: {diag_l_label[i]}")


def find_value(diag_r_label, diag_l_label):
    diag_images = get_diag_perms()
    for threshold in tqdm(range(1, 300)):
        for zero_value in range (0, -100, -1):
            for three_value in range(3, 100):
                if check_filter(diag_images, diag_r_label, diag_l_label, zero_value, three_value, threshold):
                    print("Zero value: ", zero_value)
                    print("Three value: ", three_value)
                    print("Threshold: ", threshold)
                    return
                

def get_diag_perms():
    perm_item_list = [0,1,2,3]
    perms = list(itertools.product(perm_item_list, repeat=4))
    # remove combinations with only the same element
    perms = [perm for perm in perms if len(set(perm)) > 1]
    diag_images = []

    for i, perm in enumerate(perms):
        image = np.zeros((2, 2))
        image[0, 0] = perm[0]
        image[0, 1] = perm[1]
        image[1, 0] = perm[2]
        image[1, 1] = perm[3]

        torch_image = torch.tensor(image, dtype=torch.int).unsqueeze(0)
        h_diff, v_diff = new_compute_diffs(torch_image)

        if np.any(h_diff.numpy() == 0) or np.any(v_diff.numpy() == 0):
            continue

        diag_images.append(torch_image)

    return diag_images

def check_filter(diag_images, diag_r_label, diag_l_label, zero_value, three_value, threshold, special_r_label=None, special_l_label=None):

    for i, image in enumerate(diag_images):
        image[image==0] = zero_value
        image[image==3] = three_value
        filter = torch.tensor([[1, -1], [-1, 1]], dtype=torch.int)

        diag_r = (image * filter).sum() > threshold
        diag_l = (image * filter).sum() < -threshold

        special_r = (image * filter).sum() == 7 or (image * filter).sum() == 4
        special_l = (image * filter).sum() == -7 or (image * filter).sum() == -4

        if diag_r.item() != diag_r_label[i] or diag_l.item() != diag_l_label[i]:
            return False
        
        if special_r_label is not None and special_l_label is not None:
            if special_r.item() != special_r_label[i] or special_l.item() != special_l_label[i]:
                print("Special case")
                return False
        
    return True


def value_check(perm_item_list, product = False):
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'black', 'purple'])
    counter = 0

    # create permutation list
    if product:
        perms = list(itertools.product(perm_item_list, repeat=4))
        # remove combinations with only the same element
        perms = [perm for perm in perms if len(set(perm)) > 1]
    else:
        perms = list(itertools.permutations(perm_item_list))

    for i, perm in enumerate(perms):
        image = np.zeros((2, 2))
        image[0, 0] = perm[0]
        image[0, 1] = perm[1]
        image[1, 0] = perm[2]
        image[1, 1] = perm[3]

        # image[image==0] = -8
        # image[image==3] = 8


        image_old_filter = np.zeros((2, 2))
        image_old_filter[0, 0] = perm[0]
        image_old_filter[0, 1] = perm[1]
        image_old_filter[1, 0] = perm[2]
        image_old_filter[1, 1] = perm[3]

        image_old_filter[image_old_filter==0] = -5
        image_old_filter[image_old_filter==3] = 5

        torch_image = torch.tensor(image).unsqueeze(0)
        torch_image_old_filter = torch.tensor(image_old_filter).unsqueeze(0)

        # a,b, special_case_r, special_case_l = new_compute_diag_diffs(torch_image_old_filter, 5)
        # a2,b2, special_case_r2, special_case_l2 = new_compute_diag_diffs(torch_image, 11)

        # check if either two horizontal nbrs have the same value or two vertical nbrs have the same value
        h_diff, v_diff = new_compute_diffs(torch_image)

        if np.any(h_diff.numpy() == 0) or np.any(v_diff.numpy() == 0):
            continue

        pred = torch.logical_or(torch_image == 1, torch_image == 3)
        gt = torch.logical_or(torch_image == 2, torch_image==3)
        auf = smoothing_5x5(pred, gt)
        
        print("Counter ", counter)
        plt.imshow(auf[0], cmap=cmap, vmin = 0, vmax = 3)
        plt.show()
        counter += 1
        # if np.all(a.numpy() == a2.numpy()) and np.all(b.numpy() == b2.numpy()):
        #     print("Permutation ", i, " is equivalent")
        # elif np.any(h_diff.numpy() == 0) or np.any(v_diff.numpy() == 0):
        #     print("Permutation ", i, " is not equivalent but it doesn't matter")
        #     #print(image)
        # else:
        #     print("WARNING: Permutation ", i, " is not equivalent")
        #     print(image)


        


def equivalence_check(perm_item_list, modeA = "3diff_4color", modeB = "2diff_2color", product = False):

    # warn that the isomorphism check does not consider the labels of the nodes
    print("Warning: The isomorphism check does not consider the labels of the nodes")

    # create permutation list
    if product:
        perms = list(itertools.product(perm_item_list, repeat=4))
        # remove combinations with only the same element
        perms = [perm for perm in perms if len(set(perm)) > 1]
    else:
        perms = list(itertools.permutations(perm_item_list))

    for i, perm in enumerate(perms):
        image = np.zeros((2, 2))
        image[0, 0] = perm[0]
        image[0, 1] = perm[1]
        image[1, 0] = perm[2]
        image[1, 1] = perm[3]

        aufA = create_aufdickung_util(image, mode = modeA)
        aufB = create_aufdickung_util(image, mode = modeB)

        # creat the RAGs
        labA = measure.label(aufA, background=None)
        rbgA = color.label2rgb(labA)
        ragA = graph.rag_mean_color(rbgA, labA, connectivity=1)


        labB = measure.label(aufB, background=None)
        rbgB = color.label2rgb(labB)
        ragB = graph.rag_mean_color(rbgB, labB, connectivity=1)

        # check if the RAGs are equivalent
        if nx.is_isomorphic(ragA, ragB):
            print("Permutation ", i, " is equivalent")
        else:
            print("Permutation ", i, " is not equivalent")

        print(perm)

"""OLD CODE"""
def relabel_func(graph, cluster):
    for node in cluster:
        return {
            "contracted_nodes": cluster,
            "predicted_classes": graph.nodes[node]['predicted_classes'],
            "gt_classes": graph.nodes[node]['gt_classes']
        }

def fast_contract_nodes1(graph, node_to_cluster):
    # This time using networkx's quotient_graph function
    partition = []
    for node, cluster in node_to_cluster.items():
        if cluster is None:
            continue
        partition.append(cluster)
        for n in cluster:
            node_to_cluster[n] = None

    contracted_graph = nx.quotient_graph(
        graph, 
        partition, 
        node_data=lambda cluster: relabel_func(graph, cluster),
        relabel=False
    )

    contracted_graph.graph['diag_adj'] = graph.graph['diag_adj']

    return contracted_graph


def fast_contract_nodes2(graph, node_to_cluster):
    # create 1-to-1 mapping between nodes and clusters
    node_to_contracted_nodes = {}
    contracted_node_to_node = {}
    seen = set()
    node_count = 0
    for node, cluster in node_to_cluster.items():
        if node in seen:
            continue
        contracted_node_to_node[node] = node  # here we can pick an arbitrary node from the cluster because they are all have the same node features
        for n in cluster:
            node_to_contracted_nodes[n] = node
        node_count += 1
        seen.update(cluster)

    edge_list = nx.to_pandas_edgelist(graph, edge_key='edgeData')
    
    edge_list['source'] = edge_list['source'].map(node_to_contracted_nodes)
    edge_list['target'] = edge_list['target'].map(node_to_contracted_nodes)

    edge_list = edge_list[edge_list['source'] != edge_list['target']]

    contracted_graph = nx.from_pandas_edgelist(edge_list, 'source', 'target', create_using=nx.Graph())

    # recover node features
    for node in contracted_graph.nodes:
        origin_node = contracted_node_to_node[node]
        contracted_graph.nodes[node]['predicted_classes'] = graph.nodes[origin_node]['predicted_classes']
        contracted_graph.nodes[node]['gt_classes'] = graph.nodes[origin_node]['gt_classes']
        cluster = node_to_cluster[origin_node]
        # delete the origin node from the cluster
        cluster.remove(origin_node)
        contracted_graph.nodes[node]['contracted_nodes'] = cluster

    contracted_graph.graph['diag_adj'] = graph.graph['diag_adj']

    return contracted_graph