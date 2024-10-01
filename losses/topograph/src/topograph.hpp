#include <Eigen/Dense>

#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

using namespace std;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

// return struct of label_regions
struct LabelledRegions {
    Eigen::ArrayXXi labelled_comps;
    vector<int> pred_labels;
    vector<int> gt_labels;
    int num_classes;
};

// return  struct of rag
struct Rag {
    MatrixXb adj;
    MatrixXb diag_adj;
};

struct Graph {
    unique_ptr<MatrixXb> adj;
    unique_ptr<MatrixXb> diag_adj;
    unique_ptr<vector<int>> pred_labels;
    unique_ptr<vector<int>> gt_labels;
    unique_ptr<Eigen::ArrayXXi> labelled_comps;
    int num_nodes;
};

struct BinaryGraph {
    unique_ptr<MatrixXb> adj;
    unique_ptr<unordered_map<int, vector<int>>> clusters;
    unique_ptr<vector<int>> nodes;
    int num_nodes;
};

tuple<int, int> rev_cantor_pairing(int z);
tuple<unique_ptr<Eigen::ArrayXXi>, unique_ptr<vector<int>>, unique_ptr<vector<int>>, int> label_regions(const Eigen::ArrayXXi &pred, const Eigen::ArrayXXi &gt);
tuple<unique_ptr<MatrixXb>, unique_ptr<MatrixXb>> rag(const Eigen::ArrayXXi &labelled_comps, bool diagonal_connectivity, int max_label);
BinaryGraph binarize_graph(int class_index, const Graph &graph);
vector<int> get_critical_nodes(const Graph &graph, const BinaryGraph &binary_graph, int class_index, bool strong_homotopy_equivalence = true);
void add_relabel_masks(
    vector<tuple<vector<int>>> &relabel_indices,
    const Eigen::ArrayXXi &argmax_pred, 
    const BinaryGraph &binary_graph, 
    const vector<int> &critical_nodes,
    const Graph &graph, 
    const int num_classes
);
vector<vector<tuple<vector<int>, vector<int>, vector<int>>>> compute_batch_loss(
    const vector<Eigen::Ref<Eigen::ArrayXXi>> &argmax_pred, 
    const vector<Eigen::Ref<Eigen::ArrayXXi>> &argmax_gt, 
    int num_classes, 
    int num_threads
);
vector<tuple<vector<int>, vector<int>, vector<int>>> compute_single_loss(
    Eigen::Ref<Eigen::ArrayXXi> argmax_pred, 
    Eigen::Ref<Eigen::ArrayXXi> argmax_gt, 
    int num_classes
);
vector<tuple<vector<int>, vector<int>>> get_relabel_indices(
    const Eigen::ArrayXXi &labelled_comps, 
    const Eigen::VectorXi &critical_nodes,
    const Eigen::VectorXi &cluster_lengths
);