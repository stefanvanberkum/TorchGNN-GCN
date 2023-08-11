#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include "Model/inc/Model.hxx"
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <iostream>
#include <set>
#include <thread>

int main() {
    openblas_set_num_threads(1);

    // Read data.
    std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::vector<std::vector<float>> Xs;
    std::vector<std::vector<float>> edge_indices;
    std::vector<std::vector<float>> batches;
    
    std::string line, word;

    std::filesystem::directory_iterator batch_it = std::filesystem::directory_iterator(current_path.string() + "/data");
    std::set<std::filesystem::path> sorted_paths;
    for (const std::filesystem::directory_entry& dir_entry: batch_it) {
      sorted_paths.insert(dir_entry.path());
    }
    for (std::filesystem::path batch_path: sorted_paths) {
      batch_path.append("X.csv");

      std::vector<float> X;
      std::ifstream f = std::ifstream(batch_path, std::ios::in);
      while(getline(f, line)) {
        std::stringstream str(line);
  
        while(getline(str, word, ',')) {
          X.push_back(std::stof(word));
        }
      }
      f.close();
      Xs.push_back(X);

      batch_path.replace_filename("edge_index.csv");

      std::vector<float> edge_index;
      f = std::ifstream(batch_path, std::ios::in);
      while(getline(f, line)) {
        std::stringstream str(line);
  
        while(getline(str, word, ',')) {
          edge_index.push_back(std::stof(word));
        }
      }
      f.close();
      edge_indices.push_back(edge_index);

      batch_path.replace_filename("batch.csv");

      std::vector<float> batch;
      f = std::ifstream(batch_path, std::ios::in);
      while(getline(f, line)) {
        std::stringstream str(line);
  
        while(getline(str, word, ',')) {
          batch.push_back(std::stof(word));
        }
      }
      f.close();
      batches.push_back(batch);
    }

    // Generate TorchGNN model.
    Model torchGNN_model = Model();

    // Load PyTorch script.
    torch::Device device(torch::kCPU);
    torch::jit::script::Module torch_model = torch::jit::load(current_path.string() + "/model_script.pt", device);

    int in_features = 32;
    int n_classes = 5;

    std::vector<float> out;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::chrono::duration<double> torch_time;
    std::chrono::duration<double> torchGNN_time;

    std::filesystem::remove(current_path.string() + "/torch_result.csv");
    std::filesystem::remove(current_path.string() + "/torchGNN_result.csv");
    std::ofstream out_f;
    
    // Evaluate all batches using TorchScript.
    std::size_t n_batches = Xs.size();
    for (std::size_t i = 0; i < n_batches; i++) {
      std::vector<float> X = Xs[i];
      std::vector<int64_t> edge_index = std::vector<int64_t>(edge_indices[i].begin(), edge_indices[i].end());
      std::vector<int64_t> batch = std::vector<int64_t>(batches[i].begin(), batches[i].end());

      int n_nodes = batch.size();
      int n_edges = edge_index.size() / 2;
      int n_graphs = std::set<int64_t>(batch.begin(), batch.end()).size(); 
      
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::from_blob(X.data(), {n_nodes, in_features}));
      inputs.push_back(torch::from_blob(edge_index.data(), {2, n_edges}, torch::kInt64));
      inputs.push_back(torch::from_blob(batch.data(), {n_nodes}, torch::kInt64));
      at::Tensor out_batch = torch_model.forward(inputs).toTensor();

      float* out_batch_arr = out_batch.data_ptr<float>();
      for (int graph = 0; graph < n_graphs; graph++) {
        for (int c = 0; c < n_classes; c++) {
          out.push_back(*out_batch_arr++);
        }
      }
    }

    // Write result to file.
    out_f.open(current_path.string() + "/torch_result.csv", std::ios::app);
    for (std::size_t i = 0; i < out.size(); i += n_classes) {
      bool first_feat = true;
      for (int j = 0; j < n_classes; j++) {
        if (!first_feat) {
          out_f << ",";
        } else {
          first_feat = false;
        }
        out_f << out[i + j];
      }
      out_f << "\n";
    }
    out_f.close();

    // Evaluate all batches using TorchGNN.
    out.clear();
    for (std::size_t i = 0; i < n_batches; i++) {
      std::vector<float> X = Xs[i];
      std::vector<float> edge_index = edge_indices[i];
      std::vector<float> batch = batches[i];

      std::vector<float> out_batch = torchGNN_model.forward(X, edge_index, batch);

      for (float e: out_batch) {
        out.push_back(e);
      }
    }

    // Write result to file.
    out_f.open(current_path.string() + "/torchGNN_result.csv", std::ios::app);
    for (std::size_t i = 0; i < out.size(); i += n_classes) {
      bool first_feat = true;
      for (int j = 0; j < n_classes; j++) {
        if (!first_feat) {
          out_f << ",";
        } else {
          first_feat = false;
        }
        out_f << out[i + j];
      }
      out_f << "\n";
    }
    out_f.close();

    // Collect timing data.
    for (int round = 0; round < 100; round++) {
      for (std::size_t i = 0; i < n_batches; i++) {
        std::vector<float> X = Xs[i];
        std::vector<int64_t> edge_index = std::vector<int64_t>(edge_indices[i].begin(), edge_indices[i].end());
        std::vector<int64_t> batch = std::vector<int64_t>(batches[i].begin(), batches[i].end());

        int n_nodes = batch.size();
        int n_edges = edge_index.size() / 2;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::from_blob(X.data(), {n_nodes, in_features}));
        inputs.push_back(torch::from_blob(edge_index.data(), {2, n_edges}, torch::kInt64));
        inputs.push_back(torch::from_blob(batch.data(), {n_nodes}, torch::kInt64));

        start = std::chrono::high_resolution_clock::now();
        at::Tensor out_batch = torch_model.forward(inputs).toTensor();
        end = std::chrono::high_resolution_clock::now();
        torch_time += end - start;
      }

      for (std::size_t i = 0; i < n_batches; i++) {
        std::vector<float> X = Xs[i];
        std::vector<float> edge_index = edge_indices[i];
        std::vector<float> batch = batches[i];

        start = std::chrono::high_resolution_clock::now();
        std::vector<float> out_batch = torchGNN_model.forward(X, edge_index, batch);
        end = std::chrono::high_resolution_clock::now();
        torchGNN_time += end - start;
      }
    }
    
    // Write timings.
    std::ofstream time_f;
    time_f.open(current_path.string() + "/timings.csv", std::ios::trunc);
    time_f << "PyTorch," << std::chrono::duration_cast<std::chrono::milliseconds>(torch_time).count() << std::endl;
    time_f << "TorchGNN," << std::chrono::duration_cast<std::chrono::milliseconds>(torchGNN_time).count() << std::endl;
    time_f.close();

    return 0;
}