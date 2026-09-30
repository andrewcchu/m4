#include "npy.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include "Type.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include <iomanip>
#include <algorithm>
#include <ATen/Context.h>
#include <unordered_map>


// flowsim parameters
std::vector<int64_t> fat;
std::vector<int64_t> fsize;
std::vector<int64_t> fct_i;
std::vector<double> params;
std::unordered_map<int, int64_t> fct_map;
//std::vector<int64_t> fid;
//torch::Tensor fid_tensor;
uint64_t limit;

uint32_t num_tors = 70;
uint32_t num_per_tor = 16;

std::vector<int> host_ids;
uint32_t flow_limit;
std::unordered_map<uint32_t, uint32_t> flow_counts;
std::queue<uint32_t> flow_queue;
std::unordered_map<uint32_t, std::queue<uint32_t>> tor_queue;

int32_t n_flows;

std::vector<int32_t> flowid_to_linkid_flat;
std::vector<int32_t> flowid_to_linkid_offsets;
std::vector<int32_t> edges_flow_ids;
std::vector<int32_t> edges_link_ids;

std::vector<float> res_fct;
std::vector<float> res_sldn;

// m4 options
auto options_int64 = torch::TensorOptions().dtype(torch::kInt64);


// m4 model
torch::Device device(torch::kCPU);

static torch::jit::script::Module lstmcell_time;
static torch::jit::script::Module lstmcell_rate;
static torch::jit::script::Module lstmcell_time_link;
static torch::jit::script::Module lstmcell_rate_link;
static torch::jit::script::Module output_layer;
static torch::jit::script::Module gnn_layer_0;
static torch::jit::script::Module gnn_layer_1;
static torch::jit::script::Module gnn_layer_2;

// m4 tensors
torch::Tensor size_tensor;
torch::Tensor fat_tensor;
torch::Tensor i_fct_tensor;
torch::Tensor params_tensor;

torch::Tensor release_time_tensor;
bool queued;

torch::Tensor flowid_to_linkid_flat_tensor;
torch::Tensor flowid_to_linkid_offsets_tensor;
torch::Tensor flowid_to_nlinks_tensor;

torch::Tensor edges_flow_ids_tensor;
torch::Tensor edges_link_ids_tensor;

torch::Tensor edge_index;

torch::Tensor h_vec;
torch::Tensor z_t_link;

torch::Tensor link_to_graph_id;
torch::Tensor link_to_nflows;
torch::Tensor flow_to_graph_id;

torch::Tensor time_last;
torch::Tensor flowid_active_mask;

torch::Tensor res_fct_tensor;
torch::Tensor res_sldn_tensor;

torch::Tensor sldn_est;

static torch::Tensor ones_cache;

int graph_id_counter;
int graph_id_cur;

int flow_id_in_prop;
int current_flow;
int n_flows_active;
int n_flows_arrived;
int n_flows_completed;
float time_clock;
int completed_flow_id;
int min_idx;

float flow_arrival_time;
float flow_completion_time;
float raw_flow_completion_time;
uint64_t completion_time_clamps;
float max_completion_backtrack_ns;
static constexpr const char* event_time_policy = "completion_not_before_current_time_v1";
static constexpr const char* execution_policy = "deterministic_algorithms_cublas_4096_8_v1";
static constexpr const char* cublas_workspace_config = ":4096:8";

static void configure_execution() {
    // Set this before CUDA initialization or the first cuBLAS handle. GNN
    // scatter sums otherwise vary across prefix replays on CUDA, and small
    // state differences can alter the subsequent event order.
    if (::setenv("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config, 1) != 0)
        throw std::runtime_error("cannot configure deterministic cuBLAS workspace");
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
}

int get_tor(int flow_id) {
    return host_ids.at(flow_id) / num_per_tor;
}

void setup_m4(const std::string& model_dir) {
    if (device.is_cuda() && !torch::cuda::is_available())
        throw std::runtime_error("CUDA device requested but CUDA is unavailable");

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load models
    static bool models_loaded = false;
    if (!models_loaded) {
        try {
            lstmcell_time = torch::jit::load(model_dir + "/lstmcell_time.pt", device);
            lstmcell_rate = torch::jit::load(model_dir + "/lstmcell_rate.pt", device);
            lstmcell_rate_link = torch::jit::load(model_dir + "/lstmcell_rate_link.pt", device);
            lstmcell_time_link = torch::jit::load(model_dir + "/lstmcell_time_link.pt", device);
            output_layer = torch::jit::load(model_dir + "/output_layer.pt", device);
            gnn_layer_0 = torch::jit::load(model_dir + "/gnn_layer_0.pt", device);
            gnn_layer_1 = torch::jit::load(model_dir + "/gnn_layer_1.pt", device);
            gnn_layer_2 = torch::jit::load(model_dir + "/gnn_layer_2.pt", device);
        }
        catch (const c10::Error& e) {
            throw std::runtime_error(std::string("failed to load TorchScript modules: ") + e.what());
        }

        // Set models to evaluation mode
        lstmcell_time.eval();
        lstmcell_rate.eval();
        lstmcell_rate_link.eval();
        lstmcell_time_link.eval();
        output_layer.eval();
        gnn_layer_0.eval();
        gnn_layer_1.eval();
        gnn_layer_2.eval();

        // Optimize models for inference
        lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
        lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
        lstmcell_time_link = torch::jit::optimize_for_inference(lstmcell_time_link);
        lstmcell_rate_link = torch::jit::optimize_for_inference(lstmcell_rate_link);
        output_layer = torch::jit::optimize_for_inference(output_layer);
        gnn_layer_0 = torch::jit::optimize_for_inference(gnn_layer_0);
        gnn_layer_1 = torch::jit::optimize_for_inference(gnn_layer_1);
        gnn_layer_2 = torch::jit::optimize_for_inference(gnn_layer_2);
    }
}

void setup_m4_tensors(torch::Device device, int32_t n_edges, int32_t n_links, int32_t h_vec_dim) {
    // Define tensor options
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool);
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64);

    // Clone tensors to ensure ownership
    size_tensor = torch::from_blob(fsize.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    size_tensor = torch::log2(size_tensor / 1000.0f + 1.0f);

    fat_tensor = torch::from_blob(fat.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    i_fct_tensor = torch::from_blob(fct_i.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    params_tensor = torch::from_blob(params.data(), {n_flows, 13}, options_double).to(torch::kFloat32).to(device);

    //fid_tensor = torch::from_blob(fid.data(), {n_flows}, options_int64).to(device);

    // Convert flowid_to_linkid to tensors
    flowid_to_linkid_flat_tensor = torch::from_blob(flowid_to_linkid_flat.data(), {n_edges}, options_int32).to(torch::kInt64).to(device);
    flowid_to_linkid_offsets_tensor = torch::from_blob(flowid_to_linkid_offsets.data(), {n_flows + 1}, options_int32).to(device);
    flowid_to_nlinks_tensor = flowid_to_linkid_offsets_tensor.slice(0, 1, n_flows+1) - flowid_to_linkid_offsets_tensor.slice(0, 0, n_flows);
    
    // Convert edges_flow_ids and edges_link_ids to tensors
    edges_flow_ids_tensor = torch::from_blob(edges_flow_ids.data(), {n_edges}, options_int32).to(torch::kInt64).to(device);
    edges_link_ids_tensor = torch::from_blob(edges_link_ids.data(), {n_edges}, options_int32).to(torch::kInt64).to(device);

    // Construct edge_index tensor [2, 2 * n_edges] for bidirectional edges
    edge_index = torch::stack({edges_flow_ids_tensor, edges_link_ids_tensor}, 0); // [2, n_edges]

    // Initialize tensors for active flows
    h_vec = torch::zeros({n_flows, h_vec_dim}, options_float).to(device);
    h_vec.index_put_({torch::arange(n_flows, device=device), 0}, 1.0f);
    h_vec.index_put_({torch::arange(n_flows, device=device), 2}, size_tensor);
    h_vec.index_put_({torch::arange(n_flows, device=device), 3}, flowid_to_nlinks_tensor.to(options_float));

    // Initialize z_t_link as in Python
    z_t_link = torch::zeros({n_links, h_vec_dim}, options_float).to(device); // [n_links, h_vec_dim]
    z_t_link.index_put_({torch::arange(n_links, device=device), 1}, 1.0f);
    z_t_link.index_put_({torch::arange(n_links, device=device), 2}, 1.0f);

    // Initialize graph management tensors
    link_to_graph_id = -torch::ones({n_links}, options_int32).to(device);
    link_to_nflows = torch::zeros({n_links}, options_int32).to(device);
    flow_to_graph_id = -torch::ones({n_flows}, options_int32).to(device);

    graph_id_counter = 0;
    graph_id_cur = 0;

    // Initialize time_last and flowid_active_mask
    time_last = torch::zeros({n_flows}, options_float).to(device);
    flowid_active_mask = torch::zeros({n_flows}, options_bool).to(device);

    release_time_tensor = torch::zeros({n_flows}, options_float).to(device);

    // Initialize result tensors
    res_fct_tensor = torch::zeros({n_flows, 2}, options_float).to(device);
    res_sldn_tensor = torch::zeros({n_flows, 2}, options_float).to(device);

    // Initialize counters
    flow_id_in_prop = 0;
    current_flow = 0;
    n_flows_active = 0;
    n_flows_arrived = 0;
    n_flows_completed = 0;
    time_clock = 0.0f;
    completed_flow_id = -1; // Initialize with invalid ID
    min_idx = -1;
    completion_time_clamps = 0;
    max_completion_backtrack_ns = 0.0f;

    ones_cache = torch::ones({n_links}, options_int32).to(device);
}

void update_times_m4() {
    torch::NoGradGuard no_grad;
    // Determine next flow arrival and completion times
    if (flow_limit == 0) {
        if (current_flow < n_flows) {
            flow_arrival_time = fat_tensor[current_flow].item<float>();
            flow_id_in_prop = current_flow;
        } else {
            flow_arrival_time = std::numeric_limits<float>::infinity();
            flow_id_in_prop = -1;
        }
    } else {
        flow_id_in_prop = -1;
        flow_arrival_time = std::numeric_limits<float>::infinity();
        int queue_size = 0;
        for (int i = 0; i < num_tors; i++) {
            queue_size += tor_queue[i].size();
        }
        if (!flow_queue.empty()) {
            flow_id_in_prop = flow_queue.front();
            flow_arrival_time = fat_tensor[flow_id_in_prop].item<float>() < time_clock ? time_clock : fat_tensor[flow_id_in_prop].item<float>();
            queued = true;
        }
        else {
            while (current_flow < n_flows) {
                int tor = get_tor(current_flow);
                if (flow_counts[tor] < flow_limit) {
                    flow_id_in_prop = current_flow;
                    flow_arrival_time = fat_tensor[current_flow].item<float>();
                    break;
                } else {
                    tor_queue[get_tor(current_flow)].push(current_flow);
                    current_flow++;
                }
            }
            queued = false;
        }
    }
    flow_completion_time = std::numeric_limits<float>::infinity();
    raw_flow_completion_time = flow_completion_time;

    if (n_flows_active > 0) {
        // Get indices of active flows
        auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
        auto h_vec_active = h_vec.index_select(0, flowid_active_indices);
        auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1).to(torch::kFloat32); // [n_active,1]
        auto params_data_cur = params_tensor.index_select(0, flowid_active_indices);
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_active}, 1);

        // Perform inference
        sldn_est = output_layer.forward({ input_tensor }).toTensor().view(-1);; // [n_active]
        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());

        auto fct_stamp_est = release_time_tensor.index_select(0, flowid_active_indices) + sldn_est * i_fct_tensor.index_select(0, flowid_active_indices);

        // Find the flow with the minimum estimated completion time
        min_idx = torch::argmin(fct_stamp_est).item<int>();
        raw_flow_completion_time = fct_stamp_est[min_idx].item<float>();
        if (!std::isfinite(raw_flow_completion_time))
            throw std::runtime_error("nonfinite completion prediction");
        // FCT is predicted from release, and a new state can shorten it below
        // elapsed time. Process that completion now; never rewind the clock.
        // Keep the raw argmin ordering when several predictions are overdue.
        flow_completion_time = std::max(time_clock, raw_flow_completion_time);
        completed_flow_id = flowid_active_indices[min_idx].item<int>();
    }
}


void step_m4() {
    torch::NoGradGuard no_grad;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Decide whether the next event is a flow arrival or completion
    if (flow_arrival_time <= flow_completion_time) {
        // New flow arrives before the next completion


        if (queued) {
            flow_queue.pop();
        } else {
            current_flow++;
        }

        time_clock = flow_arrival_time;

        flowid_active_mask[flow_id_in_prop] = true;
        
        time_last[flow_id_in_prop] = time_clock;
        release_time_tensor.index_put_({flow_id_in_prop}, flow_arrival_time);
        n_flows_arrived++;

        // Assign graph IDs
        int start_idx = flowid_to_linkid_offsets[flow_id_in_prop];
        int end_idx = flowid_to_linkid_offsets[flow_id_in_prop + 1];
        auto links_tensor = flowid_to_linkid_flat_tensor.slice(0, start_idx, end_idx);

        link_to_nflows.index_add_(0, links_tensor, ones_cache.slice(0, 0, links_tensor.size(0)));

        // Extract graph IDs for valid links
        auto graph_ids_tensor = link_to_graph_id.index({links_tensor});
        auto graph_mask = graph_ids_tensor != -1;
        auto valid_graph_ids_tensor = graph_ids_tensor.masked_select(graph_mask);

        // Convert unique graph IDs to a CPU vector for iteration
        auto unique_graph_ids_tensor = std::get<0>(torch::_unique(valid_graph_ids_tensor, false, false));
        int64_t num_unique_ids = unique_graph_ids_tensor.size(0);

        // Define `graph_id_cur` to use for assigning IDs
        if (num_unique_ids == 0) {
            // Case: No existing graph ID, assign a new one
            graph_id_cur = graph_id_counter;
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
            graph_id_counter += 1;
        } else if (num_unique_ids == 1) {
            // Case: One unique graph ID exists, reuse it
            graph_id_cur = unique_graph_ids_tensor.item<int64_t>();
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
        } else {
            // Case: Multiple graph IDs need to be merged into a new one
            graph_id_cur = graph_id_counter;

            // Update all flows and links with old graph IDs to the new ID
            auto old_graph_ids = unique_graph_ids_tensor;

            // Create masks for flows and links with old graph IDs
            auto flows_with_old_ids_mask = torch::isin(flow_to_graph_id, old_graph_ids);
            auto links_with_old_ids_mask = torch::isin(link_to_graph_id, old_graph_ids);

            // Update graph IDs in a single operation
            flow_to_graph_id.masked_fill_(flows_with_old_ids_mask, graph_id_cur);
            link_to_graph_id.masked_fill_(links_with_old_ids_mask, graph_id_cur);

            // Assign the new graph ID to the current flow and its links
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);

            // Increment the graph ID counter for the next assignment
            graph_id_counter += 1;
        }
        flow_counts[get_tor(flow_id_in_prop)] += 1;
        n_flows_active += 1;
    }
    else {
        // Flow completes before the next arrival
        if (raw_flow_completion_time < time_clock) {
            ++completion_time_clamps;
            max_completion_backtrack_ns = std::max(
                max_completion_backtrack_ns, time_clock - raw_flow_completion_time);
        }
        time_clock = flow_completion_time;
        // Actual FCT and SLDN
        res_fct_tensor[completed_flow_id][0] = flow_completion_time - release_time_tensor[completed_flow_id].item<float>();
        res_sldn_tensor[completed_flow_id][0] = sldn_est[min_idx];
        // Update active flow mask to mark the flow as completed
        flowid_active_mask[completed_flow_id] = false;

        // Decrement the count of active flows and increment completed flows
        n_flows_active--;
        n_flows_completed++;
        flow_counts[get_tor(completed_flow_id)] -= 1;
        if (!tor_queue[get_tor(completed_flow_id)].empty()) {
            flow_queue.push(tor_queue[get_tor(completed_flow_id)].front());
            tor_queue[get_tor(completed_flow_id)].pop();
        }

        // Get graph ID of the completed flow
        graph_id_cur = flow_to_graph_id[completed_flow_id].item<int64_t>();
        // Get links for this flow
        int start_idx = flowid_to_linkid_offsets[completed_flow_id];
        int end_idx = flowid_to_linkid_offsets[completed_flow_id + 1];
        auto links_tensor = flowid_to_linkid_flat_tensor.slice(0, start_idx, end_idx);

        link_to_nflows.index_add_(0, links_tensor, -ones_cache.slice(0, 0, links_tensor.size(0)));
        flow_to_graph_id.index_put_({completed_flow_id}, -1);

        // Find links with no active flows using tensor operations
        auto no_flow_mask = (link_to_nflows.index({links_tensor}) == 0);
        auto no_flow_links_tensor = links_tensor.masked_select(no_flow_mask);

        // Update link_to_graph_id and reset z_t_link for links with no active flows in bulk

        // Assign -1 to 'link_to_graph_id' for all 'no_flow_links' at once
        link_to_graph_id.index_put_({no_flow_links_tensor}, -1);

        // // Create tensors with desired reset values for 'z_t_link'
        auto reset_values = torch::zeros({no_flow_links_tensor.size(0), z_t_link.size(1)}, options_float).to(device);
        auto ones = torch::ones({no_flow_links_tensor.size(0)}, options_float).to(device);

        auto slice = z_t_link.index({no_flow_links_tensor, torch::indexing::Slice()});
        z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice()}, reset_values);
        z_t_link.index_put_({no_flow_links_tensor, 1}, ones);
        z_t_link.index_put_({no_flow_links_tensor, 2}, ones);
    }
    // Update h_vec for active flows
    auto flowid_active_mask_cur = torch::logical_and(flowid_active_mask, flow_to_graph_id == graph_id_cur);
    auto flowid_active_list_cur = torch::nonzero(flowid_active_mask_cur).flatten();
    if (flowid_active_list_cur.numel() > 0 && flow_arrival_time < flow_completion_time) {
        
        // Calculate time deltas for active flows
        auto time_deltas = (time_clock - time_last.index_select(0, flowid_active_list_cur).squeeze()).view({-1, 1});

        // Create a mask for the edges corresponding to the active flows
        auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
        auto selected_indices = edge_mask.nonzero().flatten();
        auto edge_index_cur = edge_index.index_select(1, selected_indices);

        // Determine the number of active flows
        auto n_flows_active_cur = flowid_active_list_cur.size(0);
        auto new_flow_indices=torch::searchsorted(flowid_active_list_cur,edge_index_cur[0]);

        // Extract return_inverse from the tuple (index 1 of the tuple)
        auto unique_result_tuple = torch::_unique(edge_index_cur[1], true, true);
        auto active_link_idx = std::get<0>(unique_result_tuple); // Unique link IDs
        auto new_link_indices = std::get<1>(unique_result_tuple); // Inverse indices for reindexing

        new_link_indices += n_flows_active_cur;
        auto edges_list_active=torch::cat({ torch::stack({new_flow_indices, new_link_indices}, 0), torch::stack({new_link_indices, new_flow_indices}, 0)}, 1);

        // Check if any time delta is greater than zero
        auto h_vec_time_updated = h_vec.index_select(0, flowid_active_list_cur);
        auto h_vec_time_link_updated = z_t_link.index_select(0, active_link_idx);
        auto max_time_delta = torch::max(time_deltas).item<float>();
        if (max_time_delta>0.0f) {
            // Update time using lstmcell_time
            time_deltas.fill_(max_time_delta/1000.0f);
            h_vec_time_updated = lstmcell_time.forward({ time_deltas, h_vec_time_updated}).toTensor();

            auto time_deltas_link = torch::zeros({active_link_idx.size(0), 1}, options_float).to(device);
            time_deltas_link.fill_(max_time_delta / 1000.0f);
            h_vec_time_link_updated = lstmcell_time_link.forward({ time_deltas_link, h_vec_time_link_updated }).toTensor();
        }

        // Forward pass through the GNN layers
        auto z_t_link_cur=z_t_link.index_select(0,active_link_idx);
        auto x_combined=torch::cat({h_vec_time_updated, h_vec_time_link_updated}, 0);

        auto gnn_output_0 = gnn_layer_0.forward({x_combined, edges_list_active}).toTensor();
        auto gnn_output_1 = gnn_layer_1.forward({gnn_output_0, edges_list_active}).toTensor();
        auto gnn_output_2 = gnn_layer_2.forward({gnn_output_1, edges_list_active}).toTensor();

        // Update rate using lstmcell_rate
        auto h_vec_rate_updated = gnn_output_2.slice(0,0,n_flows_active_cur);
        auto h_vec_rate_link = gnn_output_2.slice(0, n_flows_active_cur, gnn_output_2.size(0));

        auto params_data = params_tensor.index_select(0, flowid_active_list_cur);
        h_vec_rate_updated = torch::cat({h_vec_rate_updated, params_data}, 1);

        h_vec_rate_updated = lstmcell_rate.forward({ h_vec_rate_updated, h_vec_time_updated }).toTensor();
        h_vec_rate_link = lstmcell_rate_link.forward({ h_vec_rate_link, h_vec_time_link_updated }).toTensor();

        // Update h_vec with the new hidden states
        h_vec.index_copy_(0, flowid_active_list_cur, h_vec_rate_updated);

        //auto z_t_link_updated = h_vec_rate_link.slice(0, n_flows_active_cur, n_flows_active_cur + active_link_idx.size(0));
        new_link_indices -= n_flows_active_cur;
        auto long_indices = active_link_idx.to(torch::kInt64);
        z_t_link.index_copy_(0, long_indices, h_vec_rate_link);

        // Update time_last to the current time for active flows
        time_last.index_put_({flowid_active_list_cur}, time_clock);
    }
}



static void sync_device() {
    if (device.is_cuda()) torch::cuda::synchronize(device.index());
}

static std::vector<int> route_line(const std::string& line) {
    std::istringstream input(line);
    std::vector<int> values;
    int value;
    while (input >> value) values.push_back(value);
    if (!input.eof() || values.empty() || values[0] < 1 ||
        static_cast<size_t>(values[0]) != values.size() - 1)
        throw std::runtime_error("malformed route row: " + line);
    return values;
}

struct ScenarioInput {
    std::vector<int64_t> fat, fsize, fct_i;
    std::vector<double> params;
    std::vector<int> host_ids;
    std::vector<int32_t> flat, offsets, edge_flows, edge_links;
    int n_links;
};

static ScenarioInput read_scenario(const std::filesystem::path& scenario) {
    ScenarioInput input;
    auto d_fat = npy::read_npy<int64_t>((scenario / "fat.npy").string());
    auto d_size = npy::read_npy<int64_t>((scenario / "fsize.npy").string());
    auto d_ideal = npy::read_npy<int64_t>((scenario / "fct_i_topology_flows.npy").string());
    auto d_fid = npy::read_npy<int32_t>((scenario / "fid_topology_flows.npy").string());
    input.fat = d_fat.data; input.fsize = d_size.data; input.fct_i = d_ideal.data;
    const int count = static_cast<int>(input.fat.size());
    if (count < 1 || d_fat.shape.size() != 1 || d_size.shape != d_fat.shape ||
        d_ideal.shape != d_fat.shape || d_fid.shape != d_fat.shape)
        throw std::runtime_error("flow arrays have different or invalid shapes");
    for (int i = 0; i < count; ++i) {
        if (d_fid.data[i] != i || input.fsize[i] <= 0 || input.fct_i[i] <= 0 ||
            (i && input.fat[i] < input.fat[i - 1]))
            throw std::runtime_error("invalid flow ID, size, ideal FCT, or arrival order");
    }
    auto d_param = npy::read_npy<double>((scenario / "param_topology_flows.npy").string());
    input.params = d_param.data;
    if (d_param.shape != npy::shape_t{static_cast<size_t>(count), 13})
        throw std::runtime_error("parameters must have shape [n_flows, 13]");
    for (double p : input.params) if (!std::isfinite(p)) throw std::runtime_error("nonfinite parameter");
    std::ifstream topology(scenario / "topology.txt");
    int nodes, switches, undirected_links;
    if (!(topology >> nodes >> switches >> undirected_links) || nodes < 2 || undirected_links < 1)
        throw std::runtime_error("invalid topology header");
    input.n_links = 2 * undirected_links;
    std::string line;
    std::getline(topology, line);
    if (!std::getline(topology, line)) throw std::runtime_error("missing topology switch list");
    std::vector<std::pair<int, int>> endpoints(input.n_links);
    for (int lid = 0; lid < undirected_links; ++lid) {
        if (!std::getline(topology, line)) throw std::runtime_error("topology has too few link rows");
        std::istringstream row(line);
        int a, b; std::string rate, delay, error;
        if (!(row >> a >> b >> rate >> delay >> error) || a < 0 || a >= nodes ||
            b < 0 || b >= nodes || a == b)
            throw std::runtime_error("malformed topology link row");
        endpoints[2 * lid] = {a, b}; endpoints[2 * lid + 1] = {b, a};
    }
    std::ifstream node_routes(scenario / "flow_to_path.txt");
    std::ifstream link_routes(scenario / "flow_to_links.txt");
    if (!node_routes || !link_routes) throw std::runtime_error("missing native route file");
    std::string node_line, link_line;
    int32_t offset = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::getline(node_routes, node_line) || !std::getline(link_routes, link_line))
            throw std::runtime_error("native route file has too few rows");
        auto route = route_line(node_line), links = route_line(link_line);
        if (route[0] != links[0] + 1 || route[1] < 0 || route.back() < 0 ||
            route[1] >= nodes || route.back() >= nodes)
            throw std::runtime_error("native node/link route length or endpoint invalid");
        input.host_ids.push_back(route[1]); input.offsets.push_back(offset);
        for (size_t j = 1; j < links.size(); ++j) {
            if (links[j] < 0 || links[j] >= input.n_links ||
                endpoints[links[j]] != std::make_pair(route[j], route[j + 1]))
                throw std::runtime_error("native link route invalid");
            input.flat.push_back(links[j]); input.edge_flows.push_back(i);
            input.edge_links.push_back(links[j]); ++offset;
        }
    }
    if (std::getline(node_routes, node_line) || std::getline(link_routes, link_line))
        throw std::runtime_error("native route file has extra rows");
    input.offsets.push_back(offset);
    return input;
}

static void activate(const ScenarioInput& input) {
    fat = input.fat; fsize = input.fsize; fct_i = input.fct_i; params = input.params;
    host_ids = input.host_ids; flowid_to_linkid_flat = input.flat;
    flowid_to_linkid_offsets = input.offsets; edges_flow_ids = input.edge_flows;
    edges_link_ids = input.edge_links;
    n_flows = static_cast<int32_t>(fat.size()); limit = fat.size();
}

struct RecurrentState {
    torch::Tensor h, link, link_graph, link_count, flow_graph, last, active;
    torch::Tensor release, fct, sldn;
    int graph_counter, graph_current, current, arrived, completed, active_count;
    float clock;
    uint64_t completion_clamps;
    float max_backtrack;
    std::unordered_map<uint32_t, uint32_t> counts;
    std::queue<uint32_t> queue;
    std::unordered_map<uint32_t, std::queue<uint32_t>> tor_queues;
};

static RecurrentState save_state() {
    return {h_vec.clone(), z_t_link.clone(), link_to_graph_id.clone(),
            link_to_nflows.clone(), flow_to_graph_id.clone(), time_last.clone(),
            flowid_active_mask.clone(), release_time_tensor.clone(),
            res_fct_tensor.clone(), res_sldn_tensor.clone(), graph_id_counter,
            graph_id_cur, current_flow, n_flows_arrived, n_flows_completed,
            n_flows_active, time_clock, completion_time_clamps,
            max_completion_backtrack_ns, flow_counts, flow_queue, tor_queue};
}

static void restore_state(const RecurrentState& s, int prefix_count) {
    h_vec = s.h.clone(); z_t_link = s.link.clone();
    link_to_graph_id = s.link_graph.clone(); link_to_nflows = s.link_count.clone();
    flow_to_graph_id = s.flow_graph.clone(); time_last = s.last.clone();
    flowid_active_mask = s.active.clone(); release_time_tensor = s.release.clone();
    res_fct_tensor = s.fct.clone(); res_sldn_tensor = s.sldn.clone();
    // Future-flow sizes are action-specific, including the hidden-state input.
    if (prefix_count < n_flows)
        h_vec.slice(0, prefix_count, n_flows).select(1, 2).copy_(
            size_tensor.slice(0, prefix_count, n_flows));
    graph_id_counter = s.graph_counter; graph_id_cur = s.graph_current;
    current_flow = s.current; n_flows_arrived = s.arrived;
    n_flows_completed = s.completed; n_flows_active = s.active_count;
    time_clock = s.clock; flow_counts = s.counts; flow_queue = s.queue;
    completion_time_clamps = s.completion_clamps;
    max_completion_backtrack_ns = s.max_backtrack;
    tor_queue = s.tor_queues; queued = false;
}

static void reset_engine(const ScenarioInput& input, int hidden_size) {
    activate(input); flow_counts.clear();
    while (!flow_queue.empty()) flow_queue.pop();
    tor_queue.clear(); queued = false; flow_limit = 0;
    setup_m4_tensors(device, static_cast<int32_t>(input.flat.size()),
                     input.n_links, hidden_size);
}

static void step_checked(int& steps) {
    if (++steps > 2 * n_flows + 8) throw std::runtime_error("rollout stalled: event limit exceeded");
    update_times_m4();
    if (!std::isfinite(flow_arrival_time) && !std::isfinite(flow_completion_time))
        throw std::runtime_error("rollout stalled: no finite next event");
    if (flow_completion_time < time_clock || flow_arrival_time < time_clock)
        throw std::runtime_error("rollout stalled: event time moved backward");
    step_m4();
}

static void replay_cutoff(int64_t cutoff_ns) {
    int steps = 0;
    while (n_flows_completed < n_flows) {
        update_times_m4();
        if (std::min(flow_arrival_time, flow_completion_time) >= static_cast<float>(cutoff_ns))
            break;
        if (!std::isfinite(flow_arrival_time) && !std::isfinite(flow_completion_time))
            throw std::runtime_error("prefix stalled");
        if (++steps > 2 * n_flows + 8) throw std::runtime_error("prefix event limit exceeded");
        if (std::min(flow_arrival_time, flow_completion_time) < time_clock)
            throw std::runtime_error("prefix event time moved backward");
        step_m4();
    }
}

static void complete_rollout() {
    int steps = 0;
    while (n_flows_completed < n_flows) step_checked(steps);
}

static std::vector<float> materialize() {
    auto result = res_fct_tensor.select(1, 0).to(torch::kCPU).contiguous();
    std::vector<float> values(result.data_ptr<float>(), result.data_ptr<float>() + n_flows);
    for (float value : values)
        if (!std::isfinite(value) || value <= 0)
            throw std::runtime_error("rollout produced incomplete or nonfinite FCT");
    return values;
}

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct ActionInput { std::string id; ScenarioInput data; };
struct ArmSample {
    double full_set_s = 0, shared_prefix_s = 0;
    std::vector<double> per_action_prefix_s, continuation_s, single_s;
    std::vector<std::vector<float>> predictions;
    std::vector<uint64_t> completion_clamps;
    std::vector<float> max_backtrack;
};

static ArmSample measure_arm(const std::vector<ActionInput>& actions, bool cached,
                            int64_t cutoff_ns, int hidden_size) {
    ArmSample sample;
    sync_device();
    const auto full_start = std::chrono::steady_clock::now();
    RecurrentState prefix;
    int prefix_count = 0;
    if (cached) {
        reset_engine(actions.front().data, hidden_size);
        replay_cutoff(cutoff_ns);
        prefix_count = current_flow;
        prefix = save_state();
        sync_device();
        sample.shared_prefix_s = seconds_since(full_start);
    }
    for (const auto& action : actions) {
        sync_device();
        const auto action_start = std::chrono::steady_clock::now();
        reset_engine(action.data, hidden_size);
        if (cached) restore_state(prefix, prefix_count);
        else replay_cutoff(cutoff_ns);
        sync_device();
        const double prefix_s = seconds_since(action_start);
        const auto continuation_start = std::chrono::steady_clock::now();
        complete_rollout();
        auto prediction = materialize();
        sync_device();
        const double continuation_s = seconds_since(continuation_start);
        sample.per_action_prefix_s.push_back(prefix_s);
        sample.continuation_s.push_back(continuation_s);
        sample.single_s.push_back(prefix_s + continuation_s);
        sample.predictions.push_back(std::move(prediction));
        sample.completion_clamps.push_back(completion_time_clamps);
        sample.max_backtrack.push_back(max_completion_backtrack_ns);
    }
    sync_device();
    sample.full_set_s = seconds_since(full_start);
    return sample;
}

static void write_numbers(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << std::setprecision(12) << values[i];
    }
    out << ']';
}

static void verify_predictions(const std::vector<float>& reference,
                               const std::vector<float>& observed,
                               const std::string& phase, const std::string& action,
                               const std::filesystem::path& out_dir) {
    if (reference.size() != observed.size())
        throw std::runtime_error(phase + " coverage mismatch: " + action);
    size_t first = reference.size();
    double max_error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = std::abs(observed[i] - reference[i]);
        max_error = std::max(max_error, error);
        if (error > std::max(16.0f, 1e-5f * std::abs(reference[i])) && first == reference.size())
            first = i;
    }
    if (first == reference.size()) return;
    std::filesystem::create_directories(out_dir);
    const auto diagnostic_path = out_dir / "parity_failure.json";
    std::ofstream diagnostic(diagnostic_path);
    if (!diagnostic) throw std::runtime_error("cannot write native parity diagnostics");
    const float tolerance = std::max(16.0f, 1e-5f * std::abs(reference[first]));
    diagnostic << std::setprecision(12)
               << "{\"phase\":\"" << phase << "\",\"action\":\"" << action
               << "\",\"flow_index\":" << first << ",\"n_flows\":" << reference.size()
               << ",\"reference_fct_ns\":" << reference[first]
               << ",\"observed_fct_ns\":" << observed[first]
               << ",\"absolute_error_ns\":" << std::abs(observed[first] - reference[first])
               << ",\"tolerance_ns\":" << tolerance << ",\"max_absolute_error_ns\":" << max_error
               << ",\"execution_policy\":\"" << execution_policy
               << "\",\"event_time_policy\":\"" << event_time_policy
               << "\",\"deterministic_algorithms\":true,\"cublas_workspace_config\":\""
               << cublas_workspace_config << "\"}\n";
    for (const auto& side : {std::string("reference"), std::string("observed")}) {
        npy::npy_data<float> array;
        array.data = side == "reference" ? reference : observed;
        array.shape = {array.data.size()}; array.fortran_order = false;
        npy::write_npy((out_dir / ("parity_" + side + "_fct_ns.npy")).string(), array);
    }
    std::ostringstream error;
    error << std::setprecision(12) << phase << " prediction mismatch: " << action
          << "; flow_index=" << first << "; reference_fct_ns=" << reference[first]
          << "; observed_fct_ns=" << observed[first] << "; tolerance_ns=" << tolerance
          << "; diagnostics=" << diagnostic_path.string();
    throw std::runtime_error(error.str());
}

static int matched_request(const std::unordered_map<std::string, std::string>& args) {
    for (const auto& key : {"--models", "--request", "--output-dir", "--timings", "--device", "--repeats"})
        if (!args.count(key)) throw std::runtime_error(std::string("missing ") + key);
    const int repeats = std::stoi(args.at("--repeats"));
    const int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 1;
    const int hidden_size = args.count("--hidden-size") ? std::stoi(args.at("--hidden-size")) : 200;
    const int64_t cutoff_ns = args.count("--cutoff-ns") ? std::stoll(args.at("--cutoff-ns")) : 1500000;
    if (repeats < 1 || warmup < 0 || hidden_size < 4 || cutoff_ns <= 0)
        throw std::runtime_error("invalid matched request parameters");
    device = torch::Device(args.at("--device"));
    if (device.is_cuda() && torch::cuda::device_count() != 1)
        throw std::runtime_error("matched benchmark requires exactly one GPU");
    std::ifstream request(args.at("--request"));
    if (!request) throw std::runtime_error("cannot open request list");
    std::vector<ActionInput> actions;
    std::string line;
    while (std::getline(request, line)) {
        const auto sep = line.find('\t');
        if (sep == std::string::npos || line.find('\t', sep + 1) != std::string::npos)
            throw std::runtime_error("malformed request row");
        const auto id = line.substr(0, sep);
        const auto path = line.substr(sep + 1);
        if (path.empty()) throw std::runtime_error("empty scenario path");
        if (id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos)
            throw std::runtime_error("unsafe action ID");
        for (const auto& action : actions) if (action.id == id) throw std::runtime_error("duplicate action ID");
        actions.push_back({id, read_scenario(path)});
    }
    if (actions.empty() || actions.size() > 5) throw std::runtime_error("request needs one to five actions");
    const auto& base = actions.front().data;
    const int prefix_count = std::lower_bound(base.fat.begin(), base.fat.end(), cutoff_ns) - base.fat.begin();
    for (const auto& action : actions) {
        const auto& x = action.data;
        if (x.n_links != base.n_links || x.fat.size() != base.fat.size() ||
            x.host_ids != base.host_ids || x.flat != base.flat || x.offsets != base.offsets ||
            x.edge_flows != base.edge_flows || x.edge_links != base.edge_links ||
            !std::equal(x.fat.begin(), x.fat.begin() + prefix_count, base.fat.begin()) ||
            !std::equal(x.fsize.begin(), x.fsize.begin() + prefix_count, base.fsize.begin()) ||
            !std::equal(x.fct_i.begin(), x.fct_i.begin() + prefix_count, base.fct_i.begin()) ||
            !std::equal(x.params.begin(), x.params.begin() + 13 * prefix_count, base.params.begin()) ||
            (prefix_count < static_cast<int>(x.fat.size()) && x.fat[prefix_count] < cutoff_ns))
            throw std::runtime_error("actions differ before cutoff or in route/flow order");
    }
    setup_m4(args.at("--models"));
    std::vector<ArmSample> cached_samples, uncached_samples;
    for (int rep = -warmup; rep < repeats; ++rep) {
        auto cached = measure_arm(actions, true, cutoff_ns, hidden_size);
        auto uncached = measure_arm(actions, false, cutoff_ns, hidden_size);
        for (size_t i = 0; i < actions.size(); ++i) {
            auto single = measure_arm({actions[i]}, true, cutoff_ns, hidden_size);
            cached.single_s[i] = single.full_set_s;
            const auto& a = single.predictions.front();
            const auto& b = cached.predictions[i];
            verify_predictions(b, a, "single-request", actions[i].id, args.at("--output-dir"));
        }
        for (size_t i = 0; i < actions.size(); ++i) {
            const auto& a = cached.predictions[i]; const auto& b = uncached.predictions[i];
            verify_predictions(b, a, "cached/uncached", actions[i].id, args.at("--output-dir"));
        }
        if (rep >= 0) {
            cached_samples.push_back(std::move(cached));
            uncached_samples.push_back(std::move(uncached));
        }
    }
    const auto out_dir = std::filesystem::path(args.at("--output-dir"));
    std::filesystem::create_directories(out_dir);
    for (size_t i = 0; i < actions.size(); ++i) {
        for (const auto& arm : {std::string("cached"), std::string("uncached")}) {
            const auto& prediction = (arm == "cached" ? cached_samples : uncached_samples).back().predictions[i];
            npy::npy_data<float> output;
            output.data = prediction; output.shape = {prediction.size()}; output.fortran_order = false;
            npy::write_npy((out_dir / (arm + "__" + actions[i].id + ".npy")).string(), output);
        }
    }
    std::ofstream timings(args.at("--timings"));
    if (!timings) throw std::runtime_error("cannot open matched timings output");
    timings << "{\"unit\":\"nanoseconds\",\"precision\":\"fp32_tf32_off\",\"device\":\""
            << args.at("--device") << "\",\"cutoff_ns\":" << cutoff_ns
            << ",\"event_time_policy\":\"" << event_time_policy << "\""
            << ",\"execution_policy\":\"" << execution_policy
            << "\",\"deterministic_algorithms\":true,\"cublas_workspace_config\":\""
            << cublas_workspace_config << "\""
            << ",\"n_flows\":" << base.fat.size() << ",\"prefix_flows\":" << prefix_count
            << ",\"cached_uncached_parity\":true,\"warmup\":" << warmup
            << ",\"repeats\":" << repeats << ",\"arms\":{";
    for (const auto& arm : {std::string("cached"), std::string("uncached")}) {
        if (arm == "uncached") timings << ',';
        const auto& samples = arm == "cached" ? cached_samples : uncached_samples;
        timings << '\"' << arm << "\":{\"full_set_s\":";
        std::vector<double> values;
        for (const auto& s : samples) values.push_back(s.full_set_s);
        write_numbers(timings, values); values.clear();
        timings << ",\"shared_prefix_s\":";
        for (const auto& s : samples) values.push_back(s.shared_prefix_s);
        write_numbers(timings, values);
        timings << ",\"actions\":{";
        for (size_t i = 0; i < actions.size(); ++i) {
            if (i) timings << ',';
            timings << '\"' << actions[i].id << "\":{";
            for (const auto& field : {std::string("per_action_prefix_s"), std::string("continuation_s"), std::string("single_s")}) {
                if (field != "per_action_prefix_s") timings << ',';
                timings << '\"' << field << "\":"; values.clear();
                for (const auto& s : samples) values.push_back(
                    field == "per_action_prefix_s" ? s.per_action_prefix_s[i] :
                    field == "continuation_s" ? s.continuation_s[i] : s.single_s[i]);
                write_numbers(timings, values);
            }
            timings << ",\"completion_time_clamps\":"; values.clear();
            for (const auto& s : samples) values.push_back(s.completion_clamps[i]);
            write_numbers(timings, values);
            timings << ",\"max_completion_backtrack_ns\":"; values.clear();
            for (const auto& s : samples) values.push_back(s.max_backtrack[i]);
            write_numbers(timings, values);
            timings << '}';
        }
        timings << "}}";
    }
    timings << "}}\n";
    return 0;
}

int main(int argc, char *argv[]) {
  try {
    configure_execution();
    std::unordered_map<std::string, std::string> args;
    if (argc < 3 || argc % 2 == 0)
        throw std::runtime_error("usage: no_flowsim --models DIR --scenario DIR ... | --request FILE --output-dir DIR ...");
    for (int i = 1; i < argc; i += 2) args[argv[i]] = argv[i + 1];
    if (args.count("--request")) return matched_request(args);
    for (const auto& key : {"--models", "--scenario", "--output", "--device", "--timings", "--repeats"})
        if (!args.count(key)) throw std::runtime_error(std::string("missing ") + key);
    device = torch::Device(args.at("--device"));
    const int repeats = std::stoi(args.at("--repeats"));
    const int hidden_size = args.count("--hidden-size") ? std::stoi(args.at("--hidden-size")) : 200;
    if (repeats < 1 || hidden_size < 4) throw std::runtime_error("invalid repeats or hidden size");
    auto input = read_scenario(args.at("--scenario"));
    activate(input);
    const int n_links = input.n_links;
    const int32_t offset = static_cast<int32_t>(input.flat.size());
    flow_limit = 0;
    setup_m4(args.at("--models"));
    std::vector<double> rollout_seconds, warm_wall_seconds, completion_clamps, max_backtrack;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        flow_counts.clear();
        while (!flow_queue.empty()) flow_queue.pop();
        tor_queue.clear();
        queued = false;
        setup_m4_tensors(device, offset, n_links, hidden_size);
        sync_device();
        auto started = std::chrono::steady_clock::now();
        for (int steps = 0; n_flows_completed < n_flows; ++steps) {
            if (steps > 2 * n_flows + 8) throw std::runtime_error("rollout stalled: event limit exceeded");
            update_times_m4();
            if (!std::isfinite(flow_arrival_time) && !std::isfinite(flow_completion_time))
                throw std::runtime_error("rollout stalled: no finite next event");
            if (flow_completion_time < time_clock || flow_arrival_time < time_clock)
                throw std::runtime_error("rollout stalled: event time moved backward");
            step_m4();
        }
        auto finished = std::chrono::steady_clock::now();
        warm_wall_seconds.push_back(std::chrono::duration<double>(finished - started).count());
        sync_device();
        rollout_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        completion_clamps.push_back(completion_time_clamps);
        max_backtrack.push_back(max_completion_backtrack_ns);
    }
    auto result = res_fct_tensor.select(1, 0).to(torch::kCPU).contiguous();
    std::vector<float> values(result.data_ptr<float>(), result.data_ptr<float>() + n_flows);
    for (float value : values)
        if (!std::isfinite(value) || value <= 0)
            throw std::runtime_error("rollout produced incomplete or nonfinite FCT");
    npy::npy_data<float> output;
    output.data = values; output.shape = {static_cast<size_t>(n_flows)};
    output.fortran_order = false;
    npy::write_npy(args.at("--output"), output);
    std::ofstream timings(args.at("--timings"));
    if (!timings) throw std::runtime_error("cannot open timing output");
    timings << "{\"unit\":\"nanoseconds\",\"device\":\"" << args.at("--device")
            << "\",\"execution_policy\":\"" << execution_policy
            << "\",\"deterministic_algorithms\":true,\"cublas_workspace_config\":\""
            << cublas_workspace_config
            << "\",\"event_time_policy\":\"" << event_time_policy
            << "\",\"n_flows\":" << n_flows << ",\"warm_wall_s\":[";
    for (size_t i = 0; i < warm_wall_seconds.size(); ++i) {
        if (i) timings << ',';
        timings << std::setprecision(12) << warm_wall_seconds[i];
    }
    timings << "],\"gpu_sync_rollout_s\":[";
    for (size_t i = 0; i < rollout_seconds.size(); ++i) {
        if (i) timings << ',';
        timings << std::setprecision(12) << rollout_seconds[i];
    }
    timings << "],\"completion_time_clamps\":";
    write_numbers(timings, completion_clamps);
    timings << ",\"max_completion_backtrack_ns\":";
    write_numbers(timings, max_backtrack);
    timings << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "no_flowsim: " << error.what() << '\n';
    return 1;
  }
}
