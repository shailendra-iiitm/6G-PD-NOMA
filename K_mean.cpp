#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>

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

// K-Means Clustering (K=2 for strong and weak users)
pair<vector<User>, vector<User>> kMeansClustering(vector<User>& users, int iterations = 10) {
    if (users.size() < 2) { // Ensure at least 2 users
        return {{}, {}};
    }

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
        
        // Ensure we have at least one strong and one weak user
        if (strong.empty() || weak.empty()) {
            return {{}, {}};  // No valid pairing possible
        }

        double sum1 = 0, sum2 = 0;
        for (auto& u : strong) sum1 += u.channel_gain;
        for (auto& u : weak) sum2 += u.channel_gain;
        if (!strong.empty()) centroid1 = sum1 / strong.size();
        if (!weak.empty()) centroid2 = sum2 / weak.size();
    }

    return {strong, weak};
}



// Pair strong users with weak users
vector<UserPair> pairUsers(vector<User>& strong, vector<User>& weak) {
    vector<UserPair> pairs;
    int num_pairs = min(strong.size(), weak.size());

    for (int i = 0; i < num_pairs; i++) {
        pairs.push_back({strong[i], weak[i], 0.0, 0.0});
    }
    return pairs;
}

// Assign power allocation (initial rule-based model)
void assignPower(vector<UserPair>& pairs) {
    for (auto& pair : pairs) {
        pair.power1 = 0.8;  // Stronger user gets less power
        pair.power2 = 0.2;  // Weaker user gets more power
    }
}

// Generate users with random channel gains
vector<User> generateUsers(int num_users) {
    vector<User> users;
    srand(time(0));
    for (int i = 0; i < num_users; i++) {
        users.push_back({i + 1, (double)(rand() % 100 + 1) / 100.0});
    }
    return users;
}

// Save dataset to CSV
void saveDataset(vector<UserPair>& pairs, string filename) {
    ofstream file(filename);
    file << "User1_ID,User1_ChannelGain,User2_ID,User2_ChannelGain,Power1,Power2\n";
    for (const auto& pair : pairs) {
        file << pair.user1.id << "," << pair.user1.channel_gain << ","
             << pair.user2.id << "," << pair.user2.channel_gain << ","
             << pair.power1 << "," << pair.power2 << "\n";
    }
    file.close();
}

int main() {
    int num_users = 1000000;  // Large dataset for 6G simulation

    vector<User> users = generateUsers(num_users);
    
    // Use explicit std::pair handling instead of structured bindings
    pair<vector<User>, vector<User>> user_groups = kMeansClustering(users);
    vector<User> strong_users = user_groups.first;
    vector<User> weak_users = user_groups.second;

    vector<UserPair> pairs = pairUsers(strong_users, weak_users);
    assignPower(pairs);
    saveDataset(pairs, "kmeans_data.csv");

    cout << "K-Means optimized dataset generated and saved as kmeans_data.csv\n";
    return 0;
}
