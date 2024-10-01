from __future__ import annotations

from functools import partial
import random
import time
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import os
import torch.multiprocessing as mp

# set launch blocking
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import Topograph

from losses.utils import DiceType, AggregationType, ThresholdDistribution, fill_adj_matr, new_compute_diag_diffs, new_compute_diffs
from .dice_losses import Multiclass_CLDice
from scipy.ndimage import label
from scipy.cluster.hierarchy import DisjointSet
import timeit

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

def reverse_pairing(pairing: int) -> tuple[int, int]:
    match pairing:
        case 0: return 0, 0
        case 1: return 1, 0
        case 2: return 0, 1
        case 3: return 1, 1
        case _: return -1, -1

def label_regions(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Labels the regions in the predicted image based on the intersection of the predicted and ground truth images by assign a unique class to each connected component.

    Args:
        pred (ndarray, [H, W]): The predicted and binarized image in argmax encoding of shape [H, W].
        gt (ndarray, [H, W]): The ground truth image in argmax encoding of shape [H, W].

    Returns:
        tuple: A tuple containing the labeled regions, masks, prediction labels, and ground truth labels.
            - all_labels (ndarray, [H, W]): The labeled regions starting at 1 of shape [H, W].
            - pred_labels (ndarray, [N]): The predicted classes for each region.
            - gt_labels (ndarray, [N]): The ground truth classes for each region.
    """
    # create one hot encoding for each intersection class
    paired_img = (pred + 2 * gt)
    masked_imgs = np.eye(4)[paired_img].transpose(2, 0, 1).astype(np.int32)

    # use map to iterate through all possible combinations of classes and create connected component labeling with masks
    cc_result = map(label, masked_imgs)

    all_labels = np.zeros(pred.shape, dtype=np.int32)
    label_counter = 0 # counter for the number of classes that have already been set

    gt_labels = []
    pred_labels = []

    # iterate through all possible combinations of classes and aggregate all labels
    for inters_class, (labeled_regions, num_nodes) in enumerate(cc_result):
        # add the labeled mask to the final image
        all_labels += labeled_regions + (masked_imgs[inters_class] * label_counter)
        label_counter += num_nodes

        # get pred and gt class via reverse cantor pairing
        pred_class, gt_class = reverse_pairing(inters_class)

        # append pred and gt classes
        pred_labels.append(np.zeros((num_nodes)) + pred_class)
        gt_labels.append(np.zeros((num_nodes)) + gt_class)

    # convert label lists to numpy arrays
    pred_labels = np.concatenate(pred_labels)
    gt_labels = np.concatenate(gt_labels)

    if all_labels.max() > 0:
        all_labels -= 1

    return all_labels, pred_labels, gt_labels

def rag(labelled_regions, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl):
    max_label = labelled_regions.max()

    # if all voxel have the same class, there are no edges
    if max_label == 0:
        edges = np.empty((2, 0))
    else:
        # get the classes of each edge
        h_edges = np.stack([labelled_regions[1:, :][h_diff], labelled_regions[:-1, :][h_diff]])
        v_edges = np.stack([labelled_regions[:, 1:][v_diff], labelled_regions[:, :-1][v_diff]])

        # create adjacency matrix
        adj = np.zeros((max_label+1, max_label+1), dtype=bool)
        special_adj = np.zeros((max_label+1, max_label+1), dtype=bool)
        adj = fill_adj_matr(adj, h_edges, v_edges)

        dr_edges = np.stack([labelled_regions[:-1, :-1][diagr], labelled_regions[1:, 1:][diagr]])
        dl_edges = np.stack([labelled_regions[:-1, 1:][diagl], labelled_regions[1:, :-1][diagl]])
        special_dr_edges = np.stack([labelled_regions[:-1, :-1][special_diagr], labelled_regions[1:, 1:][special_diagr]])
        special_dl_edges = np.stack([labelled_regions[:-1, 1:][special_diagl], labelled_regions[1:, :-1][special_diagl]])
        adj = fill_adj_matr(adj, dr_edges, dl_edges)
        special_adj = fill_adj_matr(special_adj, special_dr_edges, special_dl_edges)

        # convert to edge index list
        edges = np.stack(np.nonzero(adj))
        special_edges = np.stack(np.nonzero(special_adj))

    return edges, special_edges

def contract_graph(graph):
    # identify clusters of nodes that all have the same predicted and gt class
    same_nodes = DisjointSet(graph.nodes)

    for node in graph.nodes:
        # skip correct background nodes because they never have only a diagonal edge
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue

        # get the node's cluster
        cur_node_cluster = same_nodes[node]
        
        # iterate through all neighbors of the current node
        for neighbor in graph[node]:
            # visit each edge only once or if it is a special edge, skip it
            if neighbor < node or graph[node][neighbor].get('special', False):
                continue
            # check if the neighbor has the same predicted and gt class as the current node
            if graph.nodes[neighbor]['predicted_classes'] == graph.nodes[node]['predicted_classes'] and graph.nodes[neighbor]['gt_classes'] == graph.nodes[node]['gt_classes']:
                nbr_cluster = same_nodes[neighbor]

                if nbr_cluster != cur_node_cluster:
                    same_nodes.merge(cur_node_cluster, nbr_cluster)

    # contract nodes in the graph based on the clusters
    for cluster in same_nodes.subsets():
        if len(cluster) == 1:
            continue

        # get the first node in the cluster
        first_node = cluster.pop()

        # Save the contracted nodes in the first node of each cluster
        graph.nodes[first_node]['contracted_nodes'] = cluster

        # contract all other nodes in the cluster to the first node
        for node in cluster:
            nx.contracted_nodes(graph, first_node, node, self_loops=False, copy=False)

    return graph

def identify_clusters(graph):
    pred_cluster = DisjointSet(graph.nodes)
    gt_cluster = DisjointSet(graph.nodes)

    for node in graph.nodes:
        # skip correct background nodes because they're never part of a cluster
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue

        # get the node's clusters  
        cur_pred_cluster = pred_cluster[node]
        cur_gt_cluster = gt_cluster[node]

        # iterate through all neighbors of the current node
        for neighbor in graph[node]:
            # visit each edge only once
            if neighbor < node:
                continue
            # # if it is a special edge, skip it
            # if graph[node][neighbor].get('special', False):
            #     continue
            # if they are both predicted foreground, merge pred cluster
            if graph.nodes[neighbor]['predicted_classes'] == 1 and graph.nodes[node]['predicted_classes'] == 1:
                pred_nbr_cluster = pred_cluster[neighbor]

                if pred_nbr_cluster != cur_pred_cluster:
                    pred_cluster.merge(cur_pred_cluster, pred_nbr_cluster)
            
            # if they have the same gt class, merge gt cluster
            if graph.nodes[neighbor]['gt_classes'] == 1 and graph.nodes[node]['gt_classes'] == 1:
                gt_nbr_cluster = gt_cluster[neighbor]

                if gt_nbr_cluster != cur_gt_cluster:
                    gt_cluster.merge(cur_gt_cluster, gt_nbr_cluster)

    # add pred cluster to each node
    for cluster in pred_cluster.subsets():
        node = cluster.pop()
        root = pred_cluster[node]
        graph.nodes[node]['pred_cluster'] = root

        for node in cluster:
            graph.nodes[node]['pred_cluster'] = root

    # add gt cluster to each node
    for cluster in gt_cluster.subsets():
        node = cluster.pop()
        root = gt_cluster[node]
        graph.nodes[node]['gt_cluster'] = root

        for node in cluster:
            graph.nodes[node]['gt_cluster'] = root

    return graph


def create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl):
    labelled_regions, predicted_classes, gt_classes = label_regions(argmax_pred, argmax_gt)

    # create a graph from the labelled regions
    if labelled_regions.max() == 0:  # if there is only one class, create a graph with a single node
        graph = nx.Graph()
        graph.add_node(0)
        edge_index = torch.tensor([[],[]])
        special_edge_index = torch.tensor([[],[]])
    else:
        edge_index, special_edge_index = rag(labelled_regions, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl)

    graph = nx.Graph()
    graph.add_edges_from(edge_index.T)
    graph.add_edges_from(special_edge_index.T, special=True)

    # add node attributes
    for node in graph.nodes:
        graph.nodes[node]['predicted_classes'] = predicted_classes[node]
        graph.nodes[node]['gt_classes'] = gt_classes[node]

    graph.graph['predicted_classes'] = predicted_classes

    graph = contract_graph(graph)

    graph = identify_clusters(graph)

    return graph, labelled_regions

def get_critical_nodes(graph):
    critical_nodes = []
    cluster_lengths = []

    for node in graph.nodes:
        # skip correctly predicted nodes
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])

        fg_nbr_clusters = set()
        correct_bg_nbrs_count = 0
        counter_class_str = "gt_cluster" if graph.nodes[node]['predicted_classes'] == 1 else "pred_cluster"

        for nbr in all_nbrs:
            # if it is a special edge, skip it
            if graph[node][nbr].get('special', False):
                continue

            nbr_gt_class = graph.nodes[nbr]['gt_classes']

            # if neighbor is correctly predicted, add the 
            if nbr_gt_class == 0 and graph.nodes[nbr]['predicted_classes'] == 0:    # correct background case
                correct_bg_nbrs_count += 1
                # If we have more than one correct background neighbor, we can stop here
                if correct_bg_nbrs_count > 1:
                    break
            else:  # all other nbrs are either incorrect foreground in the counter class or correct foreground
                fg_nbr_clusters.add(graph.nodes[nbr][counter_class_str])

        # if cur_node does not have exactly one correct background neighbor or not exactly one foreground nbr cluster, add it to critical nodes
        if correct_bg_nbrs_count != 1 or len(fg_nbr_clusters) != 1:
            critical_nodes.append(node)
            if "contracted_nodes" in graph.nodes[node]:
                critical_nodes += graph.nodes[node]["contracted_nodes"]
                cluster_lengths.append(len(graph.nodes[node]["contracted_nodes"]) + 1)
            else:
                cluster_lengths.append(1)
            continue

    return critical_nodes,cluster_lengths

def get_critical_nbrs(graph):
    error_count = 0

    for node in graph.nodes:
        # skip correctly predicted nodes
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])

        fg_nbr_clusters = set()
        bg_nbrs = set()
        counter_class_str = "gt_cluster" if graph.nodes[node]['predicted_classes'] == 1 else "pred_cluster"
        class_str = "gt_visisted" if graph.nodes[node]['predicted_classes'] == 1 else "pred_visited"

        for nbr in all_nbrs:
            # if it is a special edge, skip it
            if graph[node][nbr].get('special', False):
                continue

            nbr_gt_class = graph.nodes[nbr]['gt_classes']

            # if neighbor is correctly predicted, add the 
            if nbr_gt_class == 0 and graph.nodes[nbr]['predicted_classes'] == 0:    # correct background case
                bg_nbrs.add(nbr)
            else:  # all other nbrs are either incorrect foreground in the counter class or correct foreground
                fg_nbr_clusters.add(graph.nodes[nbr][counter_class_str])

        if len(bg_nbrs) == 1 and len(fg_nbr_clusters) == 1:
            continue
            
        # if a correct nbr is missing, add one to the error count
        if len(bg_nbrs) == 0:
            error_count += 1
        elif len(bg_nbrs) > 1:
            # if there are too many nbrs, count each as error that has not been counted yet
            seen_nodes = 0
            for error_node in bg_nbrs:
                if not class_str in graph.nodes[error_node]:
                    graph.nodes[error_node][class_str] = True
                else:
                    seen_nodes += 1
            
            error_count += len(bg_nbrs) - max(seen_nodes, 1)

        if len(fg_nbr_clusters) == 0:
            error_count += 1
        elif len(fg_nbr_clusters) > 1:
            seen_nodes = 0
            for error_node in fg_nbr_clusters:
                if not class_str in graph.nodes[error_node]:
                    graph.nodes[error_node][class_str] = True
                else:
                    seen_nodes += 1

            error_count += len(fg_nbr_clusters) - max(seen_nodes, 1)

    return error_count

def create_relabel_masks(critical_node_list, cluster_lengths, all_labels):
    region_error_infos = []
    remaining_nodes_in_cluster = 0
    i = 0
    cluster_counter = -1

    while i < len(critical_node_list):
        cluster_counter += 1
        node_set = [critical_node_list[i]]
        i += 1
        remaining_nodes_in_cluster = cluster_lengths[cluster_counter] - 1

        while remaining_nodes_in_cluster > 0:
            node_set.append(critical_node_list[i])
            i += 1
            remaining_nodes_in_cluster -= 1

        # get indices from all positions where all_labels is equal to any node in the node_set
        relabel_mask = np.isin(all_labels, node_set)

        index_relabel_mask = np.nonzero(relabel_mask)

        region_error_infos.append(index_relabel_mask)

    return region_error_infos

def create_relabel_masks_c(critical_node_list, cluster_lengths, all_labels):
    
    # convert to fortran storage order
    all_labels = np.asfortranarray(all_labels).astype(np.int32)
    # convert list to 1-dim numpy array in fortran storage
    critical_nodes = np.asfortranarray(critical_node_list).astype(np.int32)
    cluster_lengths = np.asfortranarray(cluster_lengths).astype(np.int32)

    relabel_indices = Topograph.get_relabel_indices(all_labels, critical_nodes, cluster_lengths)

    return relabel_indices

def _single_sample_class_loss(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl, sample_no, use_c=True):
    # create graph
    graph, labelled_regions = create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl)

    #time graph creation
    # graph_time = timeit.timeit("new_create_graph(paired_img, argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl)", globals=locals() | globals(), number=1)
    # print(f"Small: Graph creation time: {graph_time}")

    # identify critical nodes
    critical_nodes, cluster_lengths = get_critical_nodes(graph)

    # time critical node identification
    # critical_node_time = timeit.timeit("new_get_critical_nodes(graph)", globals=locals() | globals(), number=1)
    # print(f"Small: Critical node identification time: {critical_node_time}")

    # create relabel masks for all classes
    #error_region_infos = new_create_relabel_masks(one_hot_pred, graph, critical_nodes, labelled_regions)
    if use_c:
        error_region_infos = create_relabel_masks_c(critical_nodes, cluster_lengths, labelled_regions)
    else:
        error_region_infos = create_relabel_masks(critical_nodes, cluster_lengths, labelled_regions)

    # time relabel mask creation
    # relabel_mask_time = timeit.timeit("new_create_relabel_masks(one_hot_pred, graph, critical_nodes, labelled_regions)", globals=locals() | globals(), number=1)
    # print(f"Small: Reabel mask creation time: {relabel_mask_time}")

    return error_region_infos, sample_no

def _single_sample_class_metric(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl, sample_no):
    # create graph
    graph, labelled_regions = create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl,special_diagr, special_diagl)

    # get error causing neighbors
    error_count = get_critical_nbrs(graph)

    return error_count, sample_no

def single_sample_class_loss(args: dict):
    #time = timeit.timeit("_new_single_sample_class_loss(**args)", globals=locals() | globals(), number=1)
    #print(f"Time: {time}")
    return _single_sample_class_loss(**args)

def single_sample_class_metric(args: dict):
    #time = timeit.timeit("_new_single_sample_class_metric(**args)", globals=locals() | globals(), number=1)
    #print(f"Time: {time}")
    return _single_sample_class_metric(**args)


class TopographLoss(_Loss):
    def __init__(self, 
                 softmax=True, 
                 num_processes=1, 
                 include_background=True, 
                 use_c=True, 
                 sphere=False, 
                 eight_connectivity=True, 
                 aggregation=AggregationType.MEAN,
                 thres_distr=ThresholdDistribution.NONE,
                 thres_var=0.0,
        ):
        super(TopographLoss, self).__init__()
        self.softmax = softmax
        self.num_processes = num_processes
        self.include_background = include_background
        self.use_c = use_c
        self.sphere = sphere
        self.eight_connectivity = eight_connectivity
        self.thres_distr = thres_distr
        self.thres_var = thres_var
        self.aggregation = aggregation
        if self.num_processes > 1:
            self.pool = mp.Pool(num_processes)


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the topological loss.

        Args:
            input (Tensor): Input tensor of shape (batch_size, num_classes, H, W).
            target (Tensor): Target tensor of shape (batch_size, num_classes, H, W).

        Returns:
            Tensor: The calculated topological loss.

        """
        target = target.float()
        num_classes = input.shape[1]

        # Apply softmax to the input
        if self.softmax:
            input = F.softmax(input, dim=1)

        num_classes = input.shape[1]

        single_calc_inputs = []
        relabel_masks = []
        skip_index = 0 if self.include_background else 1

        if self.thres_distr != ThresholdDistribution.NONE:
            # Get the random probability to add to a class
            match self.thres_distr:
                case ThresholdDistribution.UNIFORM:
                    thres_noise = torch.rand(size=[input.shape[0], 1, 1], requires_grad=False, device=input.device) * (self.thres_var / (num_classes - 1))
                case ThresholdDistribution.GAUSSIAN:
                    thres_noise = torch.randn(size=[input.shape[0], 1, 1], requires_grad=False, device=input.device) * (self.thres_var / (num_classes - 1))
            
            # Detach the original input from the computation graph
            input_detached = input.detach().clone()

            # get class that is being reinforced
            noise_class = torch.randint(0, num_classes, (input.shape[0],), device=input.device)

            neg_noise = (thres_noise / (num_classes - 1))
            
            # Randomly add noise to the input (we also add neg_noise bc we substract it later)
            input_detached[:, noise_class] += thres_noise + neg_noise

            input_detached -= neg_noise.unsqueeze(1)

            # Re-attach the modified input to the computation graph without affecting gradients
            modified_input = input_detached
        else:
            modified_input = input
        
        # create argmax encoding using torch
        argmax_preds = torch.argmax(modified_input, dim=1)
        argmax_gts = torch.argmax(target, dim=1)

        if self.sphere:
            argmax_preds = F.pad(argmax_preds, (1, 1, 1, 1), value=0)
            argmax_gts = F.pad(argmax_gts, (1, 1, 1, 1), value=0)

        # get critical nodes for each class
        for class_index in range(skip_index, num_classes):
            # binarize image
            bin_preds = torch.zeros_like(argmax_preds)
            bin_gts = torch.zeros_like(argmax_gts)
            bin_preds[argmax_preds == class_index] = 1
            bin_gts[argmax_gts == class_index] = 1

            paired_imgs = bin_preds + 2 * bin_gts

            diag_val_1, diag_val_2 = (-4, 16) if self.eight_connectivity else (16, -4)

            paired_imgs[paired_imgs==0] = diag_val_1
            paired_imgs[paired_imgs==3] = diag_val_2

            h_diff, v_diff = new_compute_diffs(paired_imgs)
            diagr, diagl, special_diag_r, special_diag_l = new_compute_diag_diffs(paired_imgs, th=7)

            # move all to cpu
            # TODO: Fix device handling
            bin_preds = bin_preds.cpu().numpy()
            bin_gts = bin_gts.cpu().numpy()
            h_diff = h_diff.cpu().numpy()
            v_diff = v_diff.cpu().numpy()
            diagr = diagr.cpu().numpy()
            diagl = diagl.cpu().numpy()
            special_diag_r = special_diag_r.cpu().numpy()
            special_diag_l = special_diag_l.cpu().numpy()

            for i in range(input.shape[0]):
                # create dict with function arguments
                single_calc_input = {
                    "argmax_pred": bin_preds[i],
                    "argmax_gt": bin_gts[i],
                    "h_diff": h_diff[i],
                    "v_diff": v_diff[i],
                    "diagr": diagr[i],
                    "diagl": diagl[i],
                    "special_diagr": special_diag_r[i],
                    "special_diagl": special_diag_l[i],
                    "sample_no": i,
                    "use_c": self.use_c,
                }
                single_calc_inputs.append(single_calc_input)
                
        relabel_masks = []

        if self.num_processes > 1:
            chunksize = len(single_calc_inputs) // self.num_processes if len(single_calc_inputs) > self.num_processes else 1
            relabel_masks = self.pool.imap_unordered(single_sample_class_loss, single_calc_inputs, chunksize=chunksize)
        else:
            relabel_masks = map(single_sample_class_loss, single_calc_inputs)

        # calculate the topological loss for each class
        g_loss = torch.tensor(0.0, device=input.device)

        for region_error_infos, sample_no in relabel_masks:
            for region_indices in region_error_infos:
                if self.sphere:
                    region_indices = torch.tensor(region_indices)
                    region_indices -= 1

                if self.aggregation != AggregationType.CE:
                    class_indices = argmax_preds[sample_no, region_indices[0], region_indices[1]]
                    nominator = input[sample_no,class_indices,region_indices[0], region_indices[1]]

                match self.aggregation:
                    case AggregationType.MEAN:
                        g_loss += nominator.mean()
                    case AggregationType.RMS:
                        g_loss += torch.sqrt((nominator**2).mean())
                    case AggregationType.SUM:
                        g_loss += nominator.sum()
                    case AggregationType.MAX:
                        g_loss += nominator.max()
                    case AggregationType.MIN:
                        g_loss += nominator.min()
                    case AggregationType.CE:
                        masked_input = input[sample_no, :, region_indices[0], region_indices[1]].unsqueeze(0)
                        masked_target = target[sample_no, :, region_indices[0], region_indices[1]].unsqueeze(0)
                        g_loss += F.cross_entropy(masked_input, masked_target, reduction='mean')
                    case _:
                        raise ValueError(f"Invalid aggregation type: {self.aggregation}")
                
        # normalize by number of classes and batch size
        g_loss /= (input.shape[0] * (num_classes - skip_index))
        
        return g_loss

class DiceTopographLoss(_Loss):
    def __init__(self,
                 softmax: bool=True,
                 dice_type: DiceType=DiceType.CLDICE,
                 num_processes: int=1,
                 cldice_alpha: float = 0.5,
                 include_background: bool = True,
                 use_c=True,
                 sphere=False,
                 eight_connectivity=True,
                 aggregation=AggregationType.MEAN,
                 thres_distr=ThresholdDistribution.NONE,
                 thres_var=0.0) -> None:
        super().__init__()
        if dice_type == DiceType.DICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=softmax, 
                include_background=True, # irrelevant because pure Dice always uses background 
                smooth=1e-5, 
                alpha=0.0,
                convert_to_one_vs_rest=False,
                batch=True,
            )
        elif dice_type == DiceType.CLDICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=softmax, 
                include_background=include_background, 
                smooth=1e-5, 
                alpha=cldice_alpha, 
                iter_=5, 
                convert_to_one_vs_rest=False,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")
        
        self.TopographLoss = TopographLoss(
            softmax=softmax,
            num_processes=num_processes,
            include_background=include_background,
            use_c=use_c,
            sphere=sphere,
            eight_connectivity=eight_connectivity,
            aggregation=aggregation,
            thres_var=thres_var,
            thres_distr=thres_distr
        )

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"],
                alpha: float = 0.5
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute multiclass BM losses
        losses = {}
        if alpha > 0:
            topograph_loss = self.TopographLoss(prediction, target)
        else:
            topograph_loss = torch.zeros(1, device=prediction.device)

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)
        
        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["topograph_loss"] = alpha * topograph_loss

        return dice_loss + alpha * topograph_loss, losses
