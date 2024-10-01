#include "topograph.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <stdexcept>
#include <queue>
#include <memory>
#include <ranges>

#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Vector<bool, Eigen::Dynamic> VectorXb;

/**
 * reverse the cantor pairing of a single integer
 *
 * @param z The cantor pairing of two integers.
 * @return The two integers that were paired.
 */
tuple<int, int> rev_cantor_pairing(int z)
{
    int w = (int)floor((sqrt(8 * z + 1) - 1) / 2);
    int t = (w * w + w) / 2;
    int y = z - t;
    int x = w - y;
    return make_tuple(x, y);
}

/**
 * Connected Component Labeling of all class combinations between predicted and ground truth arrays.
 *
 * This function takes in two Eigen::ArrayXXi objects, `pred` and `gt`, representing the predicted and ground truth arrays respectively.
 * It labels the regions in these arrays based on all class combinations and returns the labeled regions along with other information.
 *
 * @param pred The predicted array.
 * @param gt The ground truth array.
 * @return A tuple containing the labeled regions, the gt and pred labels of all regions, and the total number of labels.
 */
tuple<unique_ptr<Eigen::ArrayXXi>, unique_ptr<vector<int>>, unique_ptr<vector<int>>, int> label_regions(const Eigen::ArrayXXi &pred, const Eigen::ArrayXXi &gt)
{
    Eigen::ArrayXXi cantor_pairing = (((pred + gt) * (pred + gt + 1) / 2) + gt);
    Eigen::MatrixXi con_comps;
    unique_ptr<Eigen::ArrayXXi> all_labels = make_unique<Eigen::ArrayXXi>(Eigen::ArrayXXi::Zero(pred.rows(), pred.cols()));
    Eigen::ArrayXXi masked_img_int;
    MatrixXb masked_img;
    cv::Mat masked_img_cv;
    cv::Mat con_comps_cv;
    int num_labels;
    int label_counter = 0;
    int pred_class;
    int gt_class;
    int num_classes = cantor_pairing.maxCoeff() + 1;
    vector<int> num_regions(num_classes, 0);
    vector<int> pred_labels(num_classes, 0);
    vector<int> gt_labels(num_classes, 0);
    unique_ptr<vector<int>> pred_labels_unfold = make_unique<vector<int>>();
    unique_ptr<vector<int>> gt_labels_unfold = make_unique<vector<int>>();

    if (num_classes == 1)
    {
        return make_tuple(move(all_labels), move(pred_labels_unfold), move(gt_labels_unfold), 0);
    }

    // iterate over all class combinations
    for (int i = 0; i < num_classes; i++)
    {
        // create mask for class i
        masked_img = (cantor_pairing == i);

        // get connected components in that mask using opencv
        cv::eigen2cv(masked_img, masked_img_cv);
        cv::setNumThreads(0);
        num_labels = cv::connectedComponents(masked_img_cv, con_comps_cv, 4); // TODO: check which algorithm is most suitable
        cv::setNumThreads(-1);
        cv::cv2eigen(con_comps_cv, con_comps);

        // accumulate labels in all_labels
        masked_img_int = masked_img.cast<int>();
        *all_labels += (con_comps.array() + (masked_img_int * label_counter));

        // increment label counter
        label_counter += (num_labels - 1);

        // retain pred and gt class for this combined class by reversing cantor pairing
        tie(pred_class, gt_class) = rev_cantor_pairing(i);

        // append pred and gt class to pred_labels and gt_labels (equivalent to python code below)
        num_regions[i] = num_labels - 1;
        pred_labels[i] = pred_class;
        gt_labels[i] = gt_class;
    }

    pred_labels_unfold->reserve(label_counter);
    gt_labels_unfold->reserve(label_counter);

    // unfold pred_labels and gt_labels
    for (int i = 0; i < num_classes; i++)
    {
        pred_labels_unfold->insert(pred_labels_unfold->end(), num_regions[i], pred_labels[i]);
        gt_labels_unfold->insert(gt_labels_unfold->end(), num_regions[i], gt_labels[i]);
    }

    *all_labels -= 1;

    return make_tuple(move(all_labels), move(pred_labels_unfold), move(gt_labels_unfold), label_counter);
}

// tuple<unique_ptr<MatrixXb>, unique_ptr<MatrixXb>> bin_rag(const Eigen::ArrayXXi &labelled_comps, bool diagonal_connectivity, int max_label, int class_index)
// {
//     unique_ptr<MatrixXb> adj = make_unique<MatrixXb>(MatrixXb::Zero(max_label, max_label));
//     unique_ptr<MatrixXb> diag_adj = make_unique<MatrixXb>(MatrixXb::Zero(max_label, max_label));

//     if (max_label == 0)
//     {
//         return make_tuple(move(adj), move(diag_adj));
//     }

//     Eigen::ArrayXXi col;
//     Eigen::ArrayXXi right_col;
//     const Eigen::Block<int, 2, 2> block;
//     int cols = labelled_comps.cols();
//     int rows = labelled_comps.rows();

//     for (int j = 0; j < cols; j++)
//     {
//         for (int i = 0; i < rows; i++)
//         {
            
//         }    
//     }
// }

/**
 * Computes the Region Adjacency Graph (RAG) for a given matrix of labelled components.
 *
 * @param labelled_comps The matrix of labelled components.
 * @param diagonal_connectivity Flag indicating whether to consider diagonal connectivity.
 * @param max_label The maximum label value in the matrix.
 * @return A tuple containing two unique pointers to adjacency matrices: `adj` and `diag_adj`.
 *         - `adj` represents the adjacency matrix without diagonal connections.
 *         - `diag_adj` represents the diagonal connections in the adj matrix.
 */
tuple<unique_ptr<MatrixXb>, unique_ptr<MatrixXb>> rag(const Eigen::ArrayXXi &labelled_comps, bool diagonal_connectivity, int max_label)
{
    unique_ptr<MatrixXb> adj = make_unique<MatrixXb>(MatrixXb::Zero(max_label, max_label));
    unique_ptr<MatrixXb> diag_adj = make_unique<MatrixXb>(MatrixXb::Zero(max_label, max_label));

    if (max_label == 0)
    {
        return make_tuple(move(adj), move(diag_adj));
    }

    Eigen::ArrayXXi col;
    Eigen::ArrayXXi right_col;
    int el;
    int right_el;
    int bottom_el;
    int bottom_right_el;
    int bottom_left_el;
    int cols = labelled_comps.cols();
    int rows = labelled_comps.rows();

    // Iterate over the rows and columns of the matrix
    for (int j = 0; j < cols; j++)
    {
        col = labelled_comps.col(j);
        for (int i = 0; i < rows; i++)
        {
            el = col(i);

            if (j < cols - 1)
            {
                right_col = labelled_comps.col(j + 1);
                right_el = right_col(i);

                // Check the right neighbor
                if (el != right_el)
                {
                    (*adj)(right_el, el) = true;
                    (*adj)(el, right_el) = true;
                    (*diag_adj)(right_el, el) = false;
                    (*diag_adj)(el, right_el) = false;
                }
            }

            if (i < rows - 1)
            {
                bottom_el = col(i + 1);

                // Check the bottom neighbor
                if (el != bottom_el)
                {
                    (*adj)(bottom_el, el) = true;
                    (*adj)(el, bottom_el) = true;
                    (*diag_adj)(bottom_el, el) = false;
                    (*diag_adj)(el, bottom_el) = false;
                }
            }

            if (diagonal_connectivity && i < rows - 1 && j < cols - 1)
            {
                right_col = labelled_comps.col(j + 1);
                bottom_right_el = right_col(i + 1);

                // Check the bottom right neighbor
                if (el != bottom_right_el)
                {
                    if ((*adj)(bottom_right_el, el) == false)
                    {
                        (*diag_adj)(bottom_right_el, el) = true;
                        (*diag_adj)(el, bottom_right_el) = true;
                    }

                    (*adj)(bottom_right_el, el) = true;
                    (*adj)(el, bottom_right_el) = true;
                }
            }

            if (diagonal_connectivity && i < rows - 1 && j > 0)
            {
                bottom_left_el = labelled_comps(i + 1, j - 1);

                // Check the bottom left neighbor
                if (el != bottom_left_el)
                {
                    if ((*adj)(bottom_left_el, el) == false)
                    {
                        (*diag_adj)(bottom_left_el, el) = true;
                        (*diag_adj)(el, bottom_left_el) = true;
                    }

                    (*adj)(bottom_left_el, el) = true;
                    (*adj)(el, bottom_left_el) = true;
                }
            }
        }
    }

    return make_tuple(move(adj), move(diag_adj));
}

/**
 * Binarizes a graph based on a given class index using the efficient union find algorithm.
 * 
 * @param class_index The class index to binarize the graph.
 * @param graph The input graph to be binarized.
 * @return The binarized graph.
 */
BinaryGraph binarize_graph(int class_index, const Graph &graph)
{
    BinaryGraph binary_graph = {
        make_unique<MatrixXb>(MatrixXb::Zero(graph.num_nodes, graph.num_nodes)),
        make_unique<unordered_map<int, vector<int>>>(graph.num_nodes),
        make_unique<vector<int>>(),
        0};
    vector<int> rank(graph.num_nodes);
    vector<int> parent(graph.num_nodes);
    VectorXb neighbors;
    VectorXb diag_neighbors;
    boost::disjoint_sets<int *, int *> clusters(&rank[0], &parent[0]);
    unordered_map<int, vector<int>>::iterator contracted_nodes;
    int cur_cluster, neighbor_cluster;
    int pred_cur_node, pred_neighbor, gt_cur_node, gt_neighbor;
    int num_clusters = graph.num_nodes;

    // make set for each vertex
    for (int cur_node = 0; cur_node < graph.num_nodes; cur_node++)
    {
        clusters.make_set(cur_node);
    }

    for (int cur_node = 0; cur_node < graph.num_nodes; cur_node++)
    {
        // check if current node is already in a cluster and get the cluster
        cur_cluster = clusters.find_set(cur_node);

        // iterate through all neighbors of the current node (optimized to half the neighbors because graph is undirected)
        neighbors = graph.adj->col(cur_node);
        diag_neighbors = graph.diag_adj->col(cur_node);
        for (int neighbor = 0; neighbor < cur_node; neighbor++)
        {
            // if candidate neighbor is not actually a neighbor (i.e., if there's no connection in the adj matrix or if it's a diag connection where at least one node is background), skip
            if (
                neighbors[neighbor] == false || (diag_neighbors[neighbor] && !((*graph.gt_labels)[neighbor] == class_index && (*graph.gt_labels)[cur_node] == class_index)))
            {
                continue;
            }

            pred_cur_node = (*graph.pred_labels)[cur_node];
            pred_neighbor = (*graph.pred_labels)[neighbor];
            gt_cur_node = (*graph.gt_labels)[cur_node];
            gt_neighbor = (*graph.gt_labels)[neighbor];

            // if the neighbor has the same gt and pred class as the current node (in a binarized view), they should be in the same cluster
            if (
                // cur_node and neighbor either both predicted to be class_index or both not predicted to be class_index
                ((pred_cur_node != class_index && pred_neighbor != class_index) || (pred_cur_node == class_index && pred_neighbor == class_index)) && // AND cur_node and neighbor either both have gt class_index or both not have gt class_index
                ((gt_cur_node != class_index && gt_neighbor != class_index) || (gt_cur_node == class_index && gt_neighbor == class_index)))
            {
                neighbor_cluster = clusters.find_set(neighbor);

                if (cur_cluster != neighbor_cluster)
                {
                    clusters.link(cur_cluster, neighbor_cluster);
                    num_clusters--;
                }
            }
        }
    }

    binary_graph.num_nodes = num_clusters;

    for (int cur_node = 0; cur_node < graph.num_nodes; cur_node++)
    {
        cur_cluster = clusters.find_set(cur_node);
        // add node to cluster
        (*binary_graph.clusters)[cur_cluster].push_back(cur_node);

        if (cur_cluster == cur_node)
        {
            binary_graph.nodes->push_back(cur_cluster);
        }

        neighbors = graph.adj->col(cur_node);
        diag_neighbors = graph.diag_adj->col(cur_node);
        for (int neighbor = 0; neighbor < cur_node; neighbor++)
        {
            // if candidate neighbor is not actually a neighbor (i.e., if there's no connection in the adj matrix or if it's a diag connection where at least one node is background), skip
            if (
                neighbors[neighbor] == false || (diag_neighbors[neighbor] && !((*graph.gt_labels)[neighbor] == class_index && (*graph.gt_labels)[cur_node] == class_index)))
            {
                continue;
            }

            neighbor_cluster = clusters.find_set(neighbor);

            if (cur_cluster != neighbor_cluster)
            {
                (*binary_graph.adj)(cur_cluster, neighbor_cluster) = true;
                (*binary_graph.adj)(neighbor_cluster, cur_cluster) = true;
            }
        }
    }

    return binary_graph;
}

/**
 * Retrieves the critical nodes from the given graph and binary graph based on the specified class index and strong homotopy equivalence flag.
 *
 * @param graph The graph containing the predicted and ground truth labels.
 * @param binary_graph The binary graph representing the adjacency matrix.
 * @param class_index The index of the class to consider.
 * @param strong_homotopy_equivalence Flag indicating whether to use strong homotopy equivalence.
 * @return A vector of integers representing the critical nodes.
 */
vector<int> get_critical_nodes(const Graph &graph, const BinaryGraph &binary_graph, int class_index, bool strong_homotopy_equivalence)
{
    vector<int> critical_nodes;
    VectorXb neighbors;
    queue<int> wrong_neighbors;
    int wrong_neighbor, correct_fg_neigh, correct_bg_neigh;
    bool nbr_gt_class;

    // iterate over all nodes
    for (int cur_node : *binary_graph.nodes)
    {
        // skip correctly predicted nodes
        if (
            ((*graph.pred_labels)[cur_node] == (*graph.gt_labels)[cur_node]) || ((*graph.pred_labels)[cur_node] != class_index && (*graph.gt_labels)[cur_node] != class_index))
        {
            continue;
        }

        correct_fg_neigh = -1;
        correct_bg_neigh = -1;
        wrong_neighbors = queue<int>();

        // iterate over all neighbors
        neighbors = binary_graph.adj->col(cur_node);
        for (int neighbor : *binary_graph.nodes)
        {
            // skip if node is not a neighbor
            if (neighbors[neighbor] == false)
            {
                continue;
            }

            // if neighbor is correctly predicted and it is the first correct neighbor of this class, save it
            // if cur_node had already a correct neighbor of this class, add it to the critical nodes
            nbr_gt_class = (*graph.gt_labels)[neighbor] == class_index;
            if (nbr_gt_class == 0 && (*graph.pred_labels)[neighbor] != class_index)
            { // correct background case
                if (correct_bg_neigh == -1)
                {
                    correct_bg_neigh = neighbor;
                }
                else
                {
                    correct_bg_neigh = -1;
                    break;
                }
            }
            else if (nbr_gt_class == 1 && (*graph.pred_labels)[neighbor] == class_index)
            { // correct foreground case
                if (correct_fg_neigh == -1)
                {
                    correct_fg_neigh = neighbor;
                }
                else
                {
                    correct_fg_neigh = -1;
                    break;
                }
            }
            else
            { // wrong neighbor case
                wrong_neighbors.push(neighbor);
            }
        }

        // if cur_node does not have exactly two correct neighbors, add it to the critical nodes and go to the next node
        if (correct_fg_neigh == -1 || correct_bg_neigh == -1)
        {
            critical_nodes.push_back(cur_node);
            continue;
        }

        // if cur_node has exactly two correct neighbors, check if each wrong neighbor is connected to the correct neighbor of the same class
        while (!wrong_neighbors.empty())
        {
            wrong_neighbor = wrong_neighbors.front();
            wrong_neighbors.pop();
            // if it has gt foreground, it must be connected to the correct foreground neighbor
            if ((*graph.gt_labels)[wrong_neighbor] == class_index && (*binary_graph.adj)(correct_fg_neigh, wrong_neighbor) == false)
            {
                critical_nodes.push_back(cur_node);
                break;
            }
            else if ((*graph.gt_labels)[wrong_neighbor] != class_index && (*binary_graph.adj)(correct_bg_neigh, wrong_neighbor) == false)
            {
                critical_nodes.push_back(cur_node);
                break;
            }
        }
    }

    // for (const auto& [node, correct_nbrs] : flagged_nodes) {
    //     neighbors = binary_graph.adj->col(node);

    //     // node is not critical if all false predicted neighbors are also flagged
    //     for (int neighbor = 0; neighbor < binary_graph.num_nodes; neighbor++) {
    //         // skip non-nbrs
    //         if (neighbors[neighbor] == false) {
    //             continue;
    //         }

    //         // skip correctly predicted neighbors
    //         if (
    //             ((*graph.pred_labels)[neighbor] != class_index && (*graph.gt_labels)[neighbor] != class_index)
    //             || ((*graph.pred_labels)[neighbor] == class_index && (*graph.gt_labels)[neighbor] == class_index)
    //         ) {
    //             continue;
    //         }

    //         // if false nbr is not flagged, node is critical
    //         if (flagged_nodes.find(neighbor) == flagged_nodes.end()) {
    //             critical_nodes.push_back(node);
    //             break;
    //         }

    //         // if false nbrs does not have the same two true nbrs, node is critical
    //         if (flagged_nodes[neighbor].first != correct_nbrs.first || flagged_nodes[neighbor].second != correct_nbrs.second) {
    //             critical_nodes.push_back(node);
    //             break;
    //         }
    //     }
    // }

    // if (!strong_homotopy_equivalence) {
    //     // throw not implemented error
    //     throw std::runtime_error("Only strong homotopy equivalence is implemented");
    // }

    return critical_nodes;
}

vector<tuple<vector<int>, vector<int>>> get_relabel_indices(
    const Eigen::ArrayXXi &labelled_comps,
    const Eigen::VectorXi &critical_nodes,
    const Eigen::VectorXi &cluster_lengths)
{
    vector<tuple<vector<int>, vector<int>>> relabel_indices(cluster_lengths.size());
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> dense_node_mask;
    Eigen::SparseMatrix<bool> sparse_node_mask;
    int critical_node, remaining_nodes_in_cluster = 0, cluster_counter = -1;

    for (int i = 0; i < critical_nodes.size(); i++)
    {
        critical_node = critical_nodes[i];

        if (remaining_nodes_in_cluster == 0)
        {
            remaining_nodes_in_cluster = cluster_lengths[++cluster_counter] - 1;
        } else {
            remaining_nodes_in_cluster--;
        }

        dense_node_mask = (labelled_comps == critical_node);

        // transform to sparse matrix
        sparse_node_mask = dense_node_mask.matrix().sparseView();

        // for each entry in sparse matrix (i.e. one pixel belonging to the node), add a triplet of form (pred_class, row, col) to relabel list
        for (int k = 0; k < sparse_node_mask.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<bool>::InnerIterator it(sparse_node_mask, k); it; ++it)
            {
                // add triplet to relabel list
                (get<0>(relabel_indices[cluster_counter])).push_back(it.row());
                (get<1>(relabel_indices[cluster_counter])).push_back(it.col());
            }
        }
    }

    return relabel_indices;
}


/**
 * @brief Adds relabel masks to the relabel_indices vector.
 *
 * This function takes argmax_pred, binary_graph, critical_nodes, graph, and num_classes.
 * It iterates over each critical node and its corresponding cluster, and for each node in the cluster, it creates a dense node mask
 * based on the labeled components in the graph. It then transforms the dense node mask into a sparse matrix and adds triplets
 * of the form (pred_class, row, col) to the relabel list for each entry in the sparse matrix. The relabel list is stored in the
 * relabel_indices vector.
 *
 * @param relabel_indices A reference to a vector of tuples representing the relabel indices.
 * @param argmax_pred The argmax prediction array.
 * @param binary_graph The binary graph object.
 * @param critical_nodes A vector of critical nodes.
 * @param graph The graph object.
 * @param num_classes The number of classes.
 */
void add_relabel_masks(
    vector<tuple<vector<int>, vector<int>, vector<int>>> &relabel_indices,
    const Eigen::ArrayXXi &argmax_pred,
    const BinaryGraph &binary_graph,
    const vector<int> &critical_nodes,
    const Graph &graph,
    const int num_classes)
{
    vector<int> node_cluster;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> dense_node_mask;
    Eigen::SparseMatrix<bool> sparse_node_mask;
    int pred_class;

    // allocate memory for relabel_indices by the curretn size plus number of critical nodes
    // relabel_indices.reserve(relabel_indices.size() + critical_nodes.size());

    // iterate over each critical node
    for (int critical_node : critical_nodes)
    {
        node_cluster = (*binary_graph.clusters)[critical_node];
        relabel_indices.push_back(make_tuple(vector<int>(), vector<int>(), vector<int>()));

        // iterate over each node in the cluster
        for (int node : node_cluster)
        {
            dense_node_mask = ((*graph.labelled_comps) == node);
            pred_class = (*graph.pred_labels)[node];

            // transform to sparse matrix
            sparse_node_mask = dense_node_mask.matrix().sparseView();

            // for each entry in sparse matrix (i.e. one pixel belonging to the node), add a triplet of form (pred_class, row, col) to relabel list
            for (int k = 0; k < sparse_node_mask.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<bool>::InnerIterator it(sparse_node_mask, k); it; ++it)
                {
                    // add triplet to relabel list
                    (get<0>(relabel_indices.back())).push_back(pred_class);
                    (get<1>(relabel_indices.back())).push_back(it.row());
                    (get<2>(relabel_indices.back())).push_back(it.col());
                }
            }
        }
    }

    return;
}

/**
 * Computes the single loss for a given prediction and ground truth using topograph algorithm.
 *
 * @param argmax_pred A reference to the prediction matrix in Eigen::ArrayXXi format.
 * @param argmax_gt A reference to the ground truth matrix in Eigen::ArrayXXi format.
 * @param num_classes The number of classes in the prediction and ground truth matrices.
 * @return A vector of tuples, where each tuple contains three vectors: the indices of the regions to be relabeled,
 * the indices of the critical nodes, and the indices of the regions in the graph.
 */
vector<tuple<vector<int>, vector<int>, vector<int>>> compute_single_loss(Eigen::Ref<Eigen::ArrayXXi> argmax_pred, Eigen::Ref<Eigen::ArrayXXi> argmax_gt, int num_classes)
{
    Graph graph;
    BinaryGraph binary_graph;
    vector<int> critical_nodes;
    vector<tuple<vector<int>, vector<int>, vector<int>>> relabel_indices;

    // Create region labels
    tie(graph.labelled_comps, graph.pred_labels, graph.gt_labels, graph.num_nodes) = label_regions(argmax_pred, argmax_gt);

    // Create graph from region labels
    tie(graph.adj, graph.diag_adj) = rag(*(graph.labelled_comps.get()), true, graph.num_nodes);

    for (int class_index = 0; class_index < num_classes; class_index++)
    {
        binary_graph = binarize_graph(class_index, graph);
        critical_nodes = get_critical_nodes(graph, binary_graph, class_index, true);
        add_relabel_masks(relabel_indices, argmax_pred, binary_graph, critical_nodes, graph, num_classes);
    }

    return relabel_indices;
}

/**
 * Computes the batch loss for a given set of predicted and ground truth arrays.
 * 
 * @param argmax_pred A vector of Eigen::Ref<Eigen::ArrayXXi> representing the predicted arrays.
 * @param argmax_gt A vector of Eigen::Ref<Eigen::ArrayXXi> representing the ground truth arrays.
 * @param num_classes The number of classes in the arrays.
 * @param num_threads The number of threads to use for parallel computation.
 * @return A vector of vector of tuples, where each tuple contains three vectors: 
 *         the indices of the predicted array, the indices of the ground truth array, 
 *         and the loss values for each index pair.
 */
vector<vector<tuple<vector<int>, vector<int>, vector<int>>>> compute_batch_loss(const vector<Eigen::Ref<Eigen::ArrayXXi>> &argmax_pred, const vector<Eigen::Ref<Eigen::ArrayXXi>> &argmax_gt, int num_classes, int num_threads)
{
    vector<vector<tuple<vector<int>, vector<int>, vector<int>>>> relabel_indices(argmax_pred.size());

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (long unsigned int i = 0; i < argmax_pred.size(); i++)
    {
        relabel_indices[i] = compute_single_loss(argmax_pred[i], argmax_gt[i], num_classes);
    }

    return relabel_indices;
}