#!/usr/bin/env python3
import heapq, math
from itertools import count

class Node:
    def __init__(self, symbol=None, prob=0.0, left=None, right=None):
        self.symbol = symbol
        self.prob   = prob
        self.left   = left
        self.right  = right

def build_huffman_tree(symbols, probs):
    """Builds Huffman tree; returns its root."""
    heap = []
    counter = count()  # tie‑breaker
    for sym, p in zip(symbols, probs):
        heapq.heappush(heap, (p, next(counter), Node(symbol=sym, prob=p)))

    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        merged = Node(prob=p1 + p2, left=n1, right=n2)
        heapq.heappush(heap, (merged.prob, next(counter), merged))

    return heap[0][2]

def assign_codes(node, prefix="", codebook=None):
    """Recursively builds codebook {symbol: code}."""
    if codebook is None:
        codebook = {}
    # Leaf?
    if node.left is None and node.right is None:
        # If only one symbol, give it code "0"
        codebook[node.symbol] = prefix or "0"
    else:
        assign_codes(node.left,  prefix + "0", codebook)
        assign_codes(node.right, prefix + "1", codebook)
    return codebook

def main():
    # 1) Read inputs
    M = int(input("Enter number of symbols M: "))
    symbols, probs = [], []
    print("Enter each symbol and its probability (e.g. A 0.25):")
    for _ in range(M):
        s, p = input().split()
        symbols.append(s)
        probs.append(float(p))

    # 2) Build tree and codebook
    root  = build_huffman_tree(symbols, probs)
    codes = assign_codes(root)

    # 3) Print symbol, prob, code
    print("\nSymbol\tProb\tCode")
    for s, p in zip(symbols, probs):
        print(f"{s}\t{p:.4f}\t{codes[s]}")

    # 4) Compute average length and entropy
    L = sum(p * len(codes[s]) for s, p in zip(symbols, probs))
    H = -sum(p * math.log2(p) for p in probs if p > 0)

    print(f"\nAverage code length L = {L:.4f}")
    print(f"Entropy H            = {H:.5f} bits/symbol")

if __name__ == "__main__":
    main()
