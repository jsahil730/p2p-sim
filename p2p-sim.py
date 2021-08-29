import sys
import numpy as np

#Parameters
n = 6               #number of nodes in P2P network
z = 0.3             #probability that a node is slow
p_edge = 0.3        #probability with which edge is drawn between two nodes
peers = {}          #will store adjacency lists, it is a dictionary of lists
seed = 102          #seed for random functions

# Structs
class Txn:
    def __init__(self, sender, receiver, amount):
        self.sender = sender                ## Node ID of the sender -- if == -1 
                                            # this is a mining transaction ##
        self.receiver = receiver            # Node ID of the receiver
        self.amount = amount                # amount of bitcoins being transferred

class Block:
    def __init__(self, blkID, parent_blkID, size, txns):
        self.blkID = blkID                  # Block ID
        self.parent_blkID = parent_blkID    # Parent Block ID
        self.size = size                    # Size of the Block
        self.txns = txns                    ## List of transactions --
                                            #  one transaction having "-1" 
                                            #  as sender is mining transaction ##

class Node:
  def __init__(self, alpha, label):
    self.alpha = alpha                # Fraction of Hashing Power
    self.label = label                # Slow or Fast
    self.blockchain = Blockchain()    # Blockchain -- tree of blocks
    self.unused = []                  # List of Unused transactions
    self.used = []                    # List of Used transactions


class BlockChain:
    def __init__(self):
        self.parent_info = {}                       # Dictionary of Block ID mapped to Parent Block ID
        self.block_info = {-1 : Block(-1,-1,0,[])}  ## Dictionary of Block ID mapped to Block structure 
                                                    #  Initializing it with the genesis block by default ##
        self.block_depth = {}                       # Dictionary of Block ID mapped to its depth in the tree
        self.toa = {}                               # Dictionary of Block ID mapped to time of arrival
        self.node_coins = {}                        ## Dictionary of (Block ID, Node ID) mapped to number of  
                                                    #  coins owned by Node till this Block ##
        self.mining_block = -1                      ## Block ID of the last block of the longest chain on 
                                                    #  which mining will take place ##
            

np.random.seed(seed)


def coin_flip(p):
    """Generates 1 with p probabilty"""
    rand_num = np.random.random()
    if(rand_num <= p):
        return 1
    else:
        return 0

def connect_dfs(i, visited):
    """Performs dfs on node i"""
    visited[i] = True
    for j in peers[i]:
        if(not visited[j]):
            connect_dfs(j, visited)

def connect_graph():
    """
    Calls dfs and connects dfs roots
    """
    roots = []
    visited = {}
    for i in range(n):
        visited[i] = False
    for i in range(n):
        if(not visited[i]):
            roots.append(i)
            connect_dfs(i, visited)
    for i in range(len(roots)-1):
        peers[roots[i]].append(roots[i+1])
        peers[roots[i+1]].append(roots[i])

def gen_graph():
    """
    Samples the random graph for P2P network
    Algo: 
    i) Connect each pair with p_edge probabilty
    ii) Connect the graph by running a dfs and making edge between dfs roots
    """
    for i in range(n):
        peers[i] = []
    for i in range(n):
        for j in range(i+1, n):
            if(coin_flip(p_edge)):
                peers[i].append(j)
                peers[j].append(i)
    connect_graph()

def print_graph():
    """Prints adjacency list"""
    for i in range(n):
        print(i, ': ', end='')
        for j in peers[i]:
            print(j, ' ', end='')
        print()

        
