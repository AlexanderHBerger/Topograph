import numpy as np
import networkx as nx
from losses.utils import smoothing_7x7, smoothing_5x5, compute_diffs, fill_adj_matr
from losses.topograph import create_relabel_masks, create_relabel_masks_c
from scipy.ndimage import label
import torch
import torch.nn.functional as F
import timeit
import torch.multiprocessing as mp
from torch.nn.modules.loss import _Loss
import cv2
from skimage import measure

import Topograph

# def single_sample_metric(data: tuple[np.ndarray, np.ndarray, int]) -> tuple[tuple[list, list] | int, int]:
#     input, target, sample_no = data

#     # create the union of the images
#     union = np.logical_or(input, target)

#     # create tje intersection of the images
#     intersection = np.logical_and(input, target)

#     union = union.astype(int)
#     intersection = intersection.astype(int)

#     # count the number of foreground components in the union
#     _, F_union_num_labels = label(union)
#     # count the number of foreground components in the intersection
#     _, F_intersection_num_labels = label(intersection)

#     # count the number of background components in the union
#     _, B_union_num_labels = label(np.logical_not(union))
#     # count the number of background components in the intersection
#     _, B_intersection_num_labels = label(np.logical_not(intersection))

#     # combine the difference in foreground and background components
#     diff = abs(F_union_num_labels - F_intersection_num_labels) + abs(B_union_num_labels - B_intersection_num_labels)


#     # return total number of critical nodes
#     # for i, critical_node_list in enumerate(critical_node_lists):
#     #     print(f"Critical nodes for class {i}: {len(critical_node_list)}")
#     return diff, sample_no

def simple_rag(labelled_components: np.ndarray):
    """
    Computes the region adjacency graph (RAG) for the given labelled components.

    Parameters:
    - labelled_components (ndarray, [H,W]): A 2D array representing the labelled components.

    Returns: 
    - edges (ndarray, [2, num_edges]): An array containing the edge index list.
    - diag_adj (ndarray, [num_nodes, num_nodes]): A boolean adjacency matrix for diagonal connections.
    """
    max_label = labelled_components.max()
    max_label = int(max_label)

    # if all voxel have the same class, there are no edges
    if max_label == 0:
        edges = np.empty((2, 0))
    else:
        # horizontal and vertical boarder
        h_diff, v_diff = compute_diffs(labelled_components)

        # get the classes of each edge
        h_edges = np.stack([labelled_components[1:, :][h_diff != 0], labelled_components[:-1, :][h_diff != 0]])
        v_edges = np.stack([labelled_components[:, 1:][v_diff != 0], labelled_components[:, :-1][v_diff != 0]])

        # create adjacency matrix
        adj = np.zeros((max_label+1, max_label+1), dtype=bool)

        # turn the edge arrays into integer arrays
        h_edges = h_edges.astype(int)
        v_edges = v_edges.astype(int)

        adj = fill_adj_matr(adj, h_edges, v_edges)

        # convert to edge index list
        edges = np.stack(np.nonzero(adj))

    return edges



def get_critical_nodes(graph: nx.Graph):
    """
    Returns a list of critical nodes in the given graph.

    A node is considered critical if it does not meet the following conditions:
    - It does not have exactly two correctly predicted neighbors of different classes.

    Args:
        graph (networkx.Graph): The input graph.
    Returns:
        list: A list of critical nodes in the graph.
    """
    critical_nodes = []
    cluster_lengths = []
    for node in graph.nodes:
        # skip correctly predicted nodes
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])
        correct_predicted_neighbors = []
        for nbr in all_nbrs:
            nbr_gt_class = graph.nodes[nbr]['gt_classes']
            if graph.nodes[nbr]['predicted_classes'] == nbr_gt_class: # correct predicted
                correct_predicted_neighbors.append((nbr, nbr_gt_class))

        if len(correct_predicted_neighbors) != 2 or correct_predicted_neighbors[0][1] == correct_predicted_neighbors[1][1]: 
            critical_nodes.append(node)
            cluster_lengths.append(1)
            continue


    return critical_nodes, cluster_lengths

def label_regions_paired_img(paired_img):
    labelled_regions = np.zeros_like(paired_img, dtype="uint16")

    # regions where both are 1
    intersection = paired_img == 3
    FF_num_labels, FF_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    # add the correct regions to the final image
    labelled_regions[FF_regions > 0] = FF_regions[FF_regions > 0]

    # regions where both are 0
    intersection = paired_img == 0
    BB_num_labels, BB_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    # add the correct regions to the final image
    labelled_regions[BB_regions > 0] = BB_regions[BB_regions > 0] + FF_num_labels

    # regions where pred is 0 and gt is 1
    intersection = paired_img == 2
    FB_num_labels, FB_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    # add the correct regions to the final image
    labelled_regions[FB_regions > 0] = FB_regions[FB_regions > 0] + FF_num_labels + BB_num_labels

    # regions where pred is 1 and gt is 0
    intersection = paired_img == 1
    BF_num_labels, BF_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    # add the correct regions to the final image
    labelled_regions[BF_regions > 0] = BF_regions[BF_regions > 0] + FF_num_labels + BB_num_labels + FB_num_labels

    # get the predicted and ground truth classes for each region
    predicted_classes = np.zeros(FF_num_labels + BB_num_labels + FB_num_labels + BF_num_labels)
    gt_classes = np.zeros(FF_num_labels + BB_num_labels + FB_num_labels + BF_num_labels)

    predicted_classes[:FF_num_labels] = 1
    gt_classes[:FF_num_labels] = 1

    predicted_classes[FF_num_labels:FF_num_labels + BB_num_labels] = 0
    gt_classes[FF_num_labels:FF_num_labels + BB_num_labels] = 0

    predicted_classes[FF_num_labels + BB_num_labels:FF_num_labels + BB_num_labels + FB_num_labels] = 0
    gt_classes[FF_num_labels + BB_num_labels:FF_num_labels + BB_num_labels + FB_num_labels] = 1

    predicted_classes[FF_num_labels + BB_num_labels + FB_num_labels:] = 1
    gt_classes[FF_num_labels + BB_num_labels + FB_num_labels:] = 0

    return labelled_regions, predicted_classes, gt_classes

def label_regions_paired_img_skimage(paired_img):
    # make the arrays boolean
    pred = np.zeros_like(paired_img, dtype=bool)
    gt = np.zeros_like(paired_img, dtype=bool)

    pred[paired_img == 1] = 1
    gt[paired_img == 2] = 1
    pred[paired_img == 3] = 1
    gt[paired_img == 3] = 1


    # perform the connected components analysis
    labelled_regions, num_reg = measure.label(paired_img.astype("uint8"), connectivity=1, background=None, return_num=True)
    #print(np.unique(labelled_regions))
    predicted_classes = None
    gt_classes = None

    # get the predicted and ground truth classes for each region
    predicted_classes = np.zeros(num_reg, dtype = "uint8")
    gt_classes = np.zeros(num_reg, dtype = "uint8")

    # do multiplication instead
    pred_lab = labelled_regions* pred
    gt_lab = labelled_regions* gt

    # multiply pred and gt with labelled regions
    #pred_lab = labelled_regions[pred >0]
    #gt_lab = labelled_regions[gt >0]

    # get the unique regions
    pred_regions = np.unique(pred_lab)
    gt_regions = np.unique(gt_lab)


    # make the arrays int
    pred_regions = pred_regions.astype(int)
    gt_regions = gt_regions.astype(int)

    # assign them to 
    predicted_classes[pred_regions-1] = 1
    gt_classes[gt_regions-1] = 1
    

    return labelled_regions, predicted_classes, gt_classes


def create_graph(paired_img) -> tuple[nx.Graph, np.ndarray]:
    labelled_region, predicted_classes, gt_classes = label_regions_paired_img(paired_img)

    edge_index = simple_rag(labelled_region)

    # create networkx graph directly
    graph = nx.Graph()
    graph.add_edges_from(edge_index.T)

    # add node attributes
    graph.graph["predicted_classes"] = predicted_classes[1:]
    for node in graph.nodes:
        graph.nodes[node]['predicted_classes'] = predicted_classes[node-1]
        graph.nodes[node]['gt_classes'] = gt_classes[node-1]

    return graph, labelled_region

def _single_sample_loss(paired_img, sample_no, use_c=True):
    binary_graph, labelled_regions = create_graph(paired_img)
    critical_node_list, cluster_lengths = get_critical_nodes(binary_graph)
    #filter = torch.zeros((7,7))
    filter = torch.zeros((5,5))
    filter[filter.shape[0]//2,filter.shape[0]//2] = 1
    original_region_labels = F.conv2d(torch.tensor(labelled_regions).unsqueeze(0).unsqueeze(0).float(), filter.unsqueeze(0).unsqueeze(0).float(), stride=filter.shape[0]).squeeze(0).squeeze(0).int().numpy()
    if use_c:
        relabel_indices = create_relabel_masks_c(critical_node_list, cluster_lengths, original_region_labels)
    else:
        relabel_indices = create_relabel_masks(critical_node_list, cluster_lengths, original_region_labels)
    return relabel_indices, sample_no

def single_sample_loss(args: dict):
    return _single_sample_loss(**args)

class ExactTopographLoss(_Loss):
    def __init__(self, softmax=True, num_processes=1, include_background=True, use_c=True, sphere=False):
        super(ExactTopographLoss, self).__init__()
        self.softmax = softmax
        self.num_processes = num_processes
        self.include_background = include_background
        self.use_c = use_c
        self.sphere = sphere
        if self.num_processes > 1:
            #raise ValueError("num_processes > 1 is not supported yet.")
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

        # create argmax encoding using torch
        argmax_preds = torch.argmax(input.clone().detach(), dim=1)
        argmax_gts = torch.argmax(target.clone().detach(), dim=1)

        if self.sphere:
            argmax_preds = F.pad(argmax_preds, (1, 1, 1, 1), value=0)
            argmax_gts = F.pad(argmax_gts, (1, 1, 1, 1), value=0)

        single_calc_inputs = []
        relabel_masks = []
        skip_index = 0 if self.include_background else 1

        # get critical nodes for each class
        for class_index in range(skip_index, num_classes):
            # binarize image
            bin_preds = torch.zeros_like(argmax_preds)
            bin_gts = torch.zeros_like(argmax_gts)
            bin_preds[argmax_preds == class_index] = 1
            bin_gts[argmax_gts == class_index] = 1
            smooth_paired_imgs = smoothing_5x5(bin_preds, bin_gts).cpu().numpy()

            # remove unnecessary tensors from cuda and free up memory
            del bin_preds
            del bin_gts

            for i in range(input.shape[0]):
                single_calc_input = {
                    "paired_img": smooth_paired_imgs[i],
                    "sample_no": i,
                    "use_c": self.use_c,
                }
                single_calc_inputs.append(single_calc_input)
        
        relabel_masks = []

        if self.num_processes > 1:
            chunksize = len(single_calc_inputs) // self.num_processes if len(single_calc_inputs) > self.num_processes else 1
            relabel_masks = self.pool.imap_unordered(single_sample_loss, single_calc_inputs, chunksize=chunksize)
        else:
            relabel_masks = map(single_sample_loss, single_calc_inputs)

        # calculate the topological loss for each class
        g_loss = torch.tensor(0.0, device=input.device)
        #region_mask = torch.zeros_like(argmax_preds[0])
        for region_error_infos, sample_no in relabel_masks:
            for region_indices in region_error_infos:
                class_indices = argmax_preds[sample_no, region_indices[0], region_indices[1]]
                if self.sphere:
                    region_indices = torch.tensor(region_indices)
                    region_indices -= 1
                # single_region_mask = torch.zeros_like(argmax_preds[0])
                # single_region_mask[region_indices[0], region_indices[1]] = 1
                # # turn indices into boolean mask
                # region_mask += single_region_mask

                nominator = input[sample_no,class_indices,region_indices[0], region_indices[1]]
                num_pixels = len(class_indices)
                #print(nominator.sum(), ", num pixels: ", num_pixels)
                g_loss += (nominator / num_pixels).sum()
        

        # normalize by number of classes and batch size
        g_loss /= (input.shape[0] * (num_classes - skip_index))
        
        return g_loss