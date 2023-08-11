#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include "Model/inc/Model.hxx"
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <iostream>

int main() {
    openblas_set_num_threads(1);

    // Read data.
    std::string const HOME = getenv("HOME");
    std::vector<float> X;
    
    std::string line, word;

    std::ifstream X_f = std::ifstream(HOME + "/TorchGNN/data/X.csv", std::ios::in);
    while(getline(X_f, line)) {
			std::stringstream str(line);
 
			while(getline(str, word, ',')) {
				X.push_back(std::stof(word));
		  }
    }
    X_f.close();

    // Generate TorchGNN model.
    Model torchGNN_model = Model();

    // Load PyTorch script.
    torch::Device device(torch::kCPU);
    torch::jit::script::Module torch_model = torch::jit::load(HOME + "/TorchGNN/model_script.pt", device);

    int batch_size = 64;
    std::vector<float> out;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::chrono::duration<double> torch_time;
    std::chrono::duration<double> torchGNN_time;

    std::filesystem::remove(HOME + "/TorchGNN/torch_result.csv");
    std::filesystem::remove(HOME + "/TorchGNN/torchGNN_result.csv");
    std::ofstream out_f;
    
    out.clear();
    for (int i = 0; i < 10000; i += batch_size) {
      std::vector<float> X_batch(X.begin() + 3072 * i, X.begin() + 3072 * (i + batch_size));
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::from_blob(X_batch.data(), {batch_size, 3072}));
      at::Tensor out_batch = torch_model.forward(inputs).toTensor();

      float* out_batch_arr = out_batch.data_ptr<float>();
      for (int j = 0; j < batch_size * 10; j++) {
        out.push_back(*out_batch_arr++);
      }
    }
    out_f.open(HOME + "/TorchGNN/torch_result.csv", std::ios::app);
    for (int batch = 0; batch < int(out.size() / (batch_size * 10)); batch++) {
      for (int obs = 0; obs < batch_size; obs++) {
        bool first_feat = true;
        for (int feat = 0; feat < 10; feat++) {
          if (!first_feat) {
            out_f << ",";
          } else {
            first_feat = false;
          }
          out_f << out[batch_size * batch + 10 * obs + feat];
        }
        out_f << "\n";
      }
    }
    out_f.close();

    out.clear();
    for (int i = 0; i < 10000; i += batch_size) {
      std::vector<float> X_batch(X.begin() + 3072 * i, X.begin() + 3072 * (i + batch_size));
      std::vector<float> out_batch = torchGNN_model.forward(X_batch);

      for (float e: out_batch) {
        out.push_back(e);
      }
    }
    out_f.open(HOME + "/TorchGNN/torchGNN_result.csv", std::ios::app);
    for (int batch = 0; batch < int(out.size() / (batch_size * 10)); batch++) {
      for (int obs = 0; obs < batch_size; obs++) {
        bool first_feat = true;
        for (int feat = 0; feat < 10; feat++) {
          if (!first_feat) {
            out_f << ",";
          } else {
            first_feat = false;
          }
          out_f << out[batch_size * batch + 10 * obs + feat];
        }
        out_f << "\n";
      }
    }
    out_f.close();

    for (int round = 0; round < 100; round++) {
      for (int i = 0; i < 10000; i += batch_size) {
        std::vector<float> X_batch(X.begin() + 3072 * i, X.begin() + 3072 * (i + batch_size));
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::from_blob(X_batch.data(), {batch_size, 3072}));

        start = std::chrono::high_resolution_clock::now();
        at::Tensor out_batch = torch_model.forward(inputs).toTensor();
        end = std::chrono::high_resolution_clock::now();
        torch_time += end - start;
      }

      for (int i = 0; i < 10000; i += batch_size) {
        std::vector<float> X_batch(X.begin() + 3072 * i, X.begin() + 3072 * (i + batch_size));

        start = std::chrono::high_resolution_clock::now();
        std::vector<float> out_batch = torchGNN_model.forward(X_batch);
        end = std::chrono::high_resolution_clock::now();
        torchGNN_time += end - start;
      }
    }
    
    // Write timings.
    std::ofstream time_f;
    time_f.open(HOME + "/TorchGNN/timings.csv", std::ios::trunc);
    time_f << "PyTorch," << std::chrono::duration_cast<std::chrono::milliseconds>(torch_time).count() << std::endl;
    time_f << "TorchGNN," << std::chrono::duration_cast<std::chrono::milliseconds>(torchGNN_time).count() << std::endl;
    time_f.close();

    return 0;
}