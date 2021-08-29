import sys
import numpy as np

#Parameters
n = 6               #number of nodes in P2P network
z = 0.7             #probability that a node is fast
p_edge = 0.3        #probability with which edge is drawn between two nodes
seed = 102          #seed for random functions
Ttx = 120           #mean time for transaction generation
total_sim_time = 10000      # total time the simulation will run

np.random.seed(seed)

peers = {}          #will store adjacency lists, it is a dictionary of lists
rho = {}            #stores speed of light propagation delay, accessed as rho[i,j]
c = {}              #stores link speed, used as c[i,j]
d_mean = {}         #stores mean of exponential dist for queuing delay, used as d_mean[i,j]
nodes = []          #stores nodes of the network
block_no = 0        #id for blocks
txn_no = 0          #id for txns
curr_time = 0       #current time in the simulation



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


def print_graph():
    """Prints adjacency list"""
    for i in range(n):
        print(i, ': ', end='')
        for j in peers[i]:
            print(j, ' ', end='')
        print()


def init_global_values():
    for i in range(n):
        for j in peers[i]:
            rho[i, j] = (10 + 490*np.random.random())/1000
            if(nodes[i].is_fast and nodes[j].is_fast):
                c[i, j] = 100
            else:
                c[i, j] = 5
            d_mean[i, j] = 96/(c[i, j]*1000)


def get_latency(i, j, m_size):
    """
    Calculate latency in sending message of size m_size from note i to j
    m_size is in kbits
    """
    latency = rho[i, j] + m_size / \
        (c[i, j]*1000) + np.random.exponential(d_mean[i, j])
    return latency



# 0-> txn generation: event has sen=-1, rec = creator of txn
# 1-> txn received: event has sen = sender of the event, rec = node that will forward
# 2-> block generation
# 3-> block received

def gen_txn(curr_time):
    """This generates random txn and adds to event queue"""
    global txn_no
    sen = np.random.randint(low=0, high=n)
    rec = np.random.randint(low=0, high=n)
    coins = 100*np.random.random()
    next_time = np.random.exponential(Ttx) + curr_time
    txn = Txn(txn_no, sen, rec, coins)
    txn_no += 1
    event_queue.add_event(next_time, Event(0, sender=-1, rec=sen, txn = txn, blk = None))
    return 


# Structs
class Event_Queue:
    def __init__(self):
        self.queue = []
    
    def add_event(self, time, event):
        index = len(self.queue)
        for i in range(len(self.queue)):
            (t, _) = self.queue[i]
            if(t>time):
                index = i
                break
        new_queue = self.queue[:index] + [(time, event)] + self.queue[index:]
        self.queue = new_queue

    def execute_event_queue(self):
        while(len(self.queue)):
            (t, e) = self.queue[0]
            self.queue = self.queue[1:]
            if(t > total_sim_time):
                return
            e.execute()


class Txn:
    def __init__(self, txnID, sender, receiver, amount):
        self.txnID = txnID
        self.sender = sender                ## Node ID of the sender -- if == -1 
                                            # this is a mining transaction ##
        self.receiver = receiver            # Node ID of the receiver
        self.amount = amount                # amount of bitcoins being transferred


class Block:
    def __init__(self, blkID, parent_blkID, txns):
        self.blkID = blkID                  # Block ID
        self.parent_blkID = parent_blkID    # Parent Block ID
        self.size = len(txns)*8             # Size of the Block in kbits
        self.txns = txns                    ## List of transactions --
                                            #  one transaction having "-1" 
                                            #  as sender is mining transaction ##

class Node:
  def __init__(self):
    self.alpha = 1/(800*n) + np.random.random()/(800*n)     # Fraction of Hashing Power
                                                            #Selected uniformly from [1/800n, 1/400n]
    self.is_fast = coin_flip(z)       # Slow or Fast
    self.blockchain = BlockChain()    # Blockchain -- tree of blocks
    self.unused = []                  # List of Unused transactions
    self.used = []                    # List of Used transactions
    self.rec_txn = {}                 # Dict of txn to sender
    self.rec_blk = {}                 # Dict of blk to sender


class BlockChain:
    def __init__(self):
        self.parent_info = {}                       # Dictionary of Block ID mapped to Parent Block ID
        self.block_info = {-1 : Block(blkID=-1, parent_blkID=-1, txns=[])}      ## Dictionary of Block ID mapped to Block structure 
                                                    #  Initializing it with the genesis block by default ##
        self.block_depth = {-1: 0}                       # Dictionary of Block ID mapped to its depth in the tree
        self.toa = {}                               # Dictionary of Block ID mapped to time of arrival
        self.node_coins = {}                        ## Dictionary of (Block ID, Node ID) mapped to number of  
                                                    #  coins owned by Node till this Block ##
        self.mining_block = -1                      ## Block ID of the last block of the longest chain on 
                                                    #  which mining will take place ##


event_queue = Event_Queue()


class Event:
    def __init__(self, type, sender, rec, txn=None, blk=None):
        self.type = type
        self.txn = txn
        self.blk = blk
        self.sender = sender
        self.rec = rec

    def execute(self):
        sen = self.sender
        rec = self.rec
        if(self.type == 0):
            # add receive txn event for peers of sender
            
            nodes[rec].rec_txn[self.txn.txnID] = -1
            #TODO: add to unused list
            # nodes[rec].unused.append(self.txn.txnID)

            for peer in peers[rec]:
                event_queue.add_event(
                    curr_time + get_latency(rec, peer, 8), Event(1, rec, peer, self.txn))
            # generate new
            gen_txn(curr_time)
        elif(self.type == 1):
            if(nodes[rec].rec_txn.has_key(self.txn.txnID)):
                return
            nodes[rec].rec_txn[self.txn.txnID] = sen
            #TODO: add to unused list, consider if a block containing txn has already been received 
            # nodes[rec].unused

            for peer in peers[rec]:
                if(peer == sen):
                    continue
                else:
                    event_queue.add_event(
                        curr_time + get_latency(rec, peer, 8), Event(1, rec, peer, self.txn))
        elif(self.type == 2):
            pass
        else: #block received
            pass
            
    
if __name__ == '__main__':
    gen_graph()                 #graph is sampled
    for i in range(n):          #node objects are assigned to each node id
        nodes.append(Node())
    init_global_values()        #parameters for calculating latency are assigned
