#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <omp.h> // OpenMP for parallel processing
#include "core/session/onnxruntime_cxx_api.h"

using namespace std;

struct User {
    int id;
    double channel_gain;
};

struct UserPair {
    User user1;
    User user2;
    double power1;
    double power2;
};

// Generate random users
vector<User> generateUsers(int num_users) {
    vector<User> users;
    srand(time(0));
    for (int i = 0; i < num_users; i++) {
        users.push_back({i + 1, (double)(rand() % 100 + 1) / 100.0});
    }
    return users;
}

// K-Means Clustering for User Pairing (O(N))
pair<vector<User>, vector<User>> kMeansClustering(vector<User>& users, int iterations = 10) {
    double centroid1 = 0.8, centroid2 = 0.2;
    vector<User> strong, weak;

    for (int it = 0; it < iterations; it++) {
        strong.clear();
        weak.clear();
        for (auto& user : users) {
            if (fabs(user.channel_gain - centroid1) < fabs(user.channel_gain - centroid2))
                strong.push_back(user);
            else
                weak.push_back(user);
        }
        double sum1 = 0, sum2 = 0;
        for (auto& u : strong) sum1 += u.channel_gain;
        for (auto& u : weak) sum2 += u.channel_gain;
        if (!strong.empty()) centroid1 = sum1 / strong.size();
        if (!weak.empty()) centroid2 = sum2 / weak.size();
    }

    return {strong, weak};
}

// Pairing Strong and Weak Users
vector<UserPair> pairUsers(vector<User>& strong, vector<User>& weak) {
    vector<UserPair> pairs;
    int num_pairs = min(strong.size(), weak.size());
    for (int i = 0; i < num_pairs; i++) {
        pairs.push_back({strong[i], weak[i], 0.0, 0.0});
    }
    return pairs;
}

// Load DRL Model for Power Allocation (ONNX)
vector<double> getPowerAllocationFromDRL(UserPair pair) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PowerAllocationModel");
    Ort::SessionOptions session_options;
    std::string model_path_str = "drl_power_allocation.onnx";
    std::wstring model_path(model_path_str.begin(), model_path_str.end());
    Ort::Session session(env, model_path.c_str(), session_options);


    Ort::AllocatorWithDefaultOptions allocator;

    // Prepare input tensor
    vector<int64_t> input_shape = {1, 2}; // 1 row, 2 features
    array<float, 2> input_data = {static_cast<float>(pair.user1.channel_gain), static_cast<float>(pair.user2.channel_gain)};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Run inference
    vector<const char*> input_names = {"input"};
    vector<const char*> output_names = {"output"};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    
    // Extract power allocation output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    return {output_data[0], output_data[1]}; // {Power for Strong, Power for Weak}
}

// Assign DRL-Based Power Allocation
void assignPower(vector<UserPair>& pairs) {
    cout << "🔄 Assigning Power..." << endl;

    for (size_t i = 0; i < pairs.size(); i++) {
        cout << "⚡ Getting power allocation for User Pair " << i + 1 << endl;

        vector<double> power_alloc = getPowerAllocationFromDRL(pairs[i]);

        cout << "✅ Received power allocation: " << power_alloc[0] << ", " << power_alloc[1] << endl;

        pairs[i].power1 = power_alloc[0];
        pairs[i].power2 = power_alloc[1];
    }

    cout << "✅ Power Assignment Completed" << endl;
}


// Simulate 6G Network - New Users Entering the System Every Second
void simulate6G(int total_users, int users_per_second) {
    vector<User> all_users;
    vector<UserPair> all_pairs;

    for (int t = 0; t < total_users / users_per_second; t++) {
        cout << "Time Step: " << t + 1 << "s - New Users Entering: " << users_per_second << endl;
        
        vector<User> new_users = generateUsers(users_per_second);
        cout << "✅ Users Generated" << endl;

        auto [strong_users, weak_users] = kMeansClustering(new_users);
        cout << "✅ Users Clustered: " << strong_users.size() << " strong, " << weak_users.size() << " weak" << endl;

        if (strong_users.empty() || weak_users.empty()) {
            cout << "❌ No valid user pairs found, skipping this time step." << endl;
            continue;
        }

        vector<UserPair> new_pairs = pairUsers(strong_users, weak_users);
        cout << "✅ User Pairs Formed: " << new_pairs.size() << endl;

        assignPower(new_pairs);
        cout << "✅ Power Assigned" << endl;

        all_pairs.insert(all_pairs.end(), new_pairs.begin(), new_pairs.end());
    }

    cout << "✅ Simulation Completed. Writing to file..." << endl;

    ofstream output_file("6G_simulation_data.csv");
    output_file << "User1_ID,User1_ChannelGain,User2_ID,User2_ChannelGain,Power1,Power2\n";
    for (const auto& pair : all_pairs) {
        output_file << pair.user1.id << "," << pair.user1.channel_gain << ","
                    << pair.user2.id << "," << pair.user2.channel_gain << ","
                    << pair.power1 << "," << pair.power2 << "\n";
    }
    output_file.close();
    cout << "✅ Data saved in 6G_simulation_data.csv\n";
}



int main() {
    int total_users = 50; // Total users in simulation
    int users_per_second = 5; // Dynamic arrival of users per second

    cout << "Starting 6G PD-NOMA Simulation..." << endl;
    simulate6G(total_users, users_per_second);
    return 0;
}
