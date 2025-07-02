// ass2.cpp
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>   // for completeness, though not strictly needed below

using namespace std;

// 1) Huffman tree node
struct Node {
    string symbol;
    double prob;
    Node *left, *right;
    Node(const string& s, double p)
      : symbol(s), prob(p), left(nullptr), right(nullptr) {}
};

// 2) Comparator for min‑heap (lowest prob at top)
struct Compare {
    bool operator()(const Node* a, const Node* b) const {
        return a->prob > b->prob;
    }
};

// 3) Recursively walk the Huffman tree and build codewords
void assignCodes(Node* root, const string& prefix, map<string,string>& codes) {
    if (!root) return;
    // Leaf node
    if (!root->left && !root->right) {
        codes[root->symbol] = prefix;
    } else {
        assignCodes(root->left,  prefix + "0", codes);
        assignCodes(root->right, prefix + "1", codes);
    }
}

int main() {
    int M;
    cout << "Enter number of symbols M: ";
    if (!(cin >> M) || M <= 0) {
        cerr << "Invalid M\n"; 
        return 1;
    }

    vector<string> symbols(M);
    vector<double> probs(M);
    cout << "Enter each symbol and its probability (e.g. A 0.25):\n";
    for (int i = 0; i < M; i++) {
        cin >> symbols[i] >> probs[i];
        if (!cin || probs[i] < 0) {
            cerr << "Bad input\n";
            return 1;
        }
    }

    // Build priority queue of leaf nodes
    priority_queue<Node*, vector<Node*>, Compare> pq;
    for (int i = 0; i < M; i++) {
        pq.push(new Node(symbols[i], probs[i]));
    }

    // Merge until one tree remains
    while (pq.size() > 1) {
        Node* x = pq.top(); pq.pop();
        Node* y = pq.top(); pq.pop();
        Node* z = new Node("", x->prob + y->prob);
        z->left  = x;
        z->right = y;
        pq.push(z);
    }
    Node* root = pq.top();

    // Generate codes
    map<string,string> codes;
    assignCodes(root, "", codes);

    // Print symbol, probability, code
    cout << "\nSymbol\tProb\tCode\n";
    for (int i = 0; i < M; i++) {
        cout 
          << symbols[i] << '\t'
          << probs[i]   << "\t"
          << codes[symbols[i]] 
          << "\n";
    }

    // Compute average length L and entropy H
    double L = 0.0, H = 0.0;
    for (int i = 0; i < M; i++) {
        double p = probs[i];
        int    ℓ = codes[symbols[i]].length();
        L += p * ℓ;
        if (p > 0) H -= p * log2(p);
    }

    cout << "\nAverage code length L = " << L << "\n";
    cout << "Entropy H            = " << H << " bits/symbol\n";

    // (Optional) clean up allocated nodes omitted for brevity

    return 0;
}
