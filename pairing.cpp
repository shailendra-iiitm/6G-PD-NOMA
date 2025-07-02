#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
using namespace std;

struct User {
    int id;
    double channel_gain;
    double power_alloc;
};

// Fixed Power Allocation (80%-20%)
void fixedPowerAllocation(vector<User>& users) {
    for (size_t i = 0; i < users.size(); i++) {
        users[i].power_alloc = (i % 2 == 0) ? 0.8 : 0.2; // Alternate power allocation
    }
}

// Channel Gain-Based Pairing
void channelGainPairing(vector<User>& users) {
    sort(users.begin(), users.end(), [](User a, User b) { return a.channel_gain > b.channel_gain; });
    for (size_t i = 0; i < users.size() / 2; i++) {
        users[i].power_alloc = 0.8;
        users[users.size() - 1 - i].power_alloc = 0.2;
    }
}

// K-Means Clustering-Based Pairing
void kMeansClustering(vector<User>& users) {
    double centroid1 = 0.8, centroid2 = 0.2;
    vector<User> strong, weak;
    for (auto& user : users) {
        if (fabs(user.channel_gain - centroid1) < fabs(user.channel_gain - centroid2))
            strong.push_back(user);
        else
            weak.push_back(user);
    }
    for (size_t i = 0; i < min(strong.size(), weak.size()); i++) {
        strong[i].power_alloc = 0.8;
        weak[i].power_alloc = 0.2;
    }
}

// Function to generate random users
vector<User> generateUsers(int num_users) {
    vector<User> users(num_users);
    for (int i = 0; i < num_users; i++) {
        users[i] = {i + 1, (double)(rand() % 100 + 1) / 100.0, 0};
    }
    return users;
}

// Function to measure execution time
void measureExecutionTime(void (*func)(vector<User>&), vector<User>& users, string method) {
    auto start = chrono::high_resolution_clock::now();
    func(users);
    auto end = chrono::high_resolution_clock::now();
    cout << method << " Execution Time: " 
         << chrono::duration<double, milli>(end - start).count() << " ms\n";
}

int main() {
    const int NUM_USERS = 50000;  // Increase users for better measurement
    srand(time(0));

    vector<User> users;

    // Measure execution times
    users = generateUsers(NUM_USERS);
    measureExecutionTime(fixedPowerAllocation, users, "Fixed Power Allocation");

    users = generateUsers(NUM_USERS);
    measureExecutionTime(channelGainPairing, users, "Channel Gain-Based Pairing");

    users = generateUsers(NUM_USERS);
    measureExecutionTime(kMeansClustering, users, "K-Means Clustering-Based Pairing");

    return 0;
}
