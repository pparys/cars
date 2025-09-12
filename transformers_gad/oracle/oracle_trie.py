import torch
import torch.nn.functional as F
import json
import logging

logger = logging.getLogger(__name__)

class TrieNode:
    def __init__(self, parent = None):
        self.children = {}
        self.parent = parent
        self.raw_logprob = None
        self.log_theta = None

    def create_child(self, token_id : int):
        assert token_id not in self.children
        self.children[token_id] = TrieNode(self)

class Trie:
    def __init__(self):
        self.root = TrieNode()
        print("Creating new oracle trie")
