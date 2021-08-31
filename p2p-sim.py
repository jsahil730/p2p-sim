import sys
import numpy as np
from enum import Enum

#Parameters
n = 6               #number of nodes in P2P network
z = 0.7             #probability that a node is fast
p_edge = 0.3        #probability with which edge is drawn between two nodes
seed = 102          #seed for random functions
Ttx = 60           #mean time for transaction generation
total_sim_time = 1000    # total time the simulation will run

class Event_type(Enum):
    gen_txn = 0
    rec_txn = 1
    gen_blk = 2
    rec_blk = 3
    broadcast_invalid_block = 4

#constants
MINING_FEES = 50
MINING_FEE_SENDER = -1
MAX_TXN_NUM = 1000

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


def gen_txn():
    """This generates random txn and adds to event queue"""
    global txn_no
    sen = np.random.randint(low=0, high=n)
    rec = np.random.randint(low=0, high=n)
    coins = 100*np.random.random()
    next_time = np.random.exponential(Ttx) + curr_time
    txn = Txn(txn_no, sen, rec, coins)
    txn_no += 1
    event_queue.add_event(next_time, Event(Event_type.gen_txn, sender=-1, rec=sen, txn = txn, blk = None))
    return 

def gen_valid_blk(nodeID):
    global block_no
    txns = []
    txns.append(Txn(txn_no, MINING_FEE_SENDER, nodeID, MINING_FEES))
    txn_pool = nodes[nodeID].unused
    curr_longest_block = nodes[nodeID].blockchain.mining_block
    balance = {}
    for i in range(n):
        balance[i] = nodes[nodeID].blockchain.node_coins[curr_longest_block, i]
    for k in txn_pool.keys():
        if(len(txns) > MAX_TXN_NUM):
            break
        if(txn_pool[k].amount <= balance[txn_pool[k].sender]):
            txns.append(txn_pool[k])
            balance[txn_pool[k].sender] -= txn_pool[k].amount
            balance[txn_pool[k].receiver] += txn_pool[k].amount
    next_time = curr_time + np.random.exponential(nodes[nodeID].alpha)
    new_blk = Block(block_no, curr_longest_block, txns)
    block_no+= 1
    event_queue.add_event(next_time, Event(Event_type.gen_blk, -1, nodeID, txn= None, blk= new_blk))

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
        global curr_time
        while(len(self.queue)):
            (t, e) = self.queue[0]
            self.queue = self.queue[1:]
            if(t > total_sim_time):
                return
            curr_time = t
            e.execute()


class Txn:
    def __init__(self, txnID, sender, receiver, amount):
        self.txnID = txnID
        self.sender = sender                ## Node ID of the sender -- if == -1 
                                            # this is a mining transaction ##
        self.receiver = receiver            # Node ID of the receiver
        self.amount = amount                # amount of bitcoins being transferred

    def get_txn_str(self):
        """Returns the txn in string format"""
        return 'TxnID: {}; Node {} -> Node {}, {} coins'.format(self.txnID, self.sender, self.receiver, self.amount)


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
    self.alpha = 400*n + np.random.random()*(400*n)     #Average Mining time
                                                        #Selected uniformly from [400n, 800n]
    self.is_fast = coin_flip(z)       # Slow or Fast
    self.blockchain = BlockChain()    # Blockchain -- tree of blocks
    self.unused = {}                  # List of Unused transactions
    self.used = {}                    # List of Used transactions
    self.rec_txn = {}                 # Dict of txn to sender
    self.rec_blk = {}                 # Dict of blk to sender


class BlockChain:
    def __init__(self):
        self.block_info = {-1 : Block(blkID=-1, parent_blkID=-1, txns=[])}      ## Dictionary of Block ID mapped to Block structure 
                                                    #  Initializing it with the genesis block by default ##
        self.block_depth = {-1: 0}                       # Dictionary of Block ID mapped to its depth in the tree
        self.toa = {}                               # Dictionary of Block ID mapped to time of arrival
        self.node_coins = {}                        ## Dictionary of (Block ID, Node ID) mapped to number of  
                                                    #  coins owned by Node till this Block ##
        for i in range(n):
            self.node_coins[-1, i] = 0
        self.mining_block = -1                      ## Block ID of the last block of the longest chain on 
                                                    #  which mining will take place ##
        self.orphans = []                           # blocks whose parents are not in the blockchain

    def add_block(self, blk):
        #check if incoming block's parent is in node's blockchain
        if(blk.parent_blkID in self.block_info):
            self.block_info[blk.blkID] = blk
            self.block_depth[blk.blkID] = 1 + self.block_depth[blk.parent_blkID]
            self.toa[blk.blkID] = curr_time
            for i in range(n):
                self.node_coins[blk.blkID, i] = self.node_coins[blk.parent_blkID, i]
            for txn in blk.txns:
                if(not (txn.sender == -1)):     # check that txn is not mining fees
                    self.node_coins[blk.blkID, txn.sender] -= txn.amount
                self.node_coins[blk.blkID, txn.receiver] += txn.amount
            if(self.block_depth[blk.blkID] > self.block_depth[self.mining_block]):
                self.mining_block = blk.blk_ID

            #TODO: check for orphan in the orphan list in the node
        else:
            pass


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
        if(self.type == Event_type.gen_txn):
            
            # for testing txn frowarding
            # print('{0:.2f}'.format(curr_time), ': Node ', rec, ', Txn ', self.txn.txnID, ' generated, ', self.txn.get_txn_str())

            #store the sender of the event as -1
            nodes[rec].rec_txn[self.txn.txnID] = -1

            # add txn to unused map
            nodes[rec].unused[self.txn.txnID] = self.txn

            # add receive txn event for peers of sender
            msg_size = 8  # in kbits
            for peer in peers[rec]:
                event_queue.add_event(
                    curr_time + get_latency(rec, peer, msg_size), Event(Event_type.rec_txn, rec, peer, self.txn))
            # generate new txn
            gen_txn()
        elif(self.type == Event_type.rec_txn):

            # for testing txn frowarding
            # print('{0:.2f}'.format(curr_time), ': Node ', rec, ', Txn ',
                #   self.txn.txnID, ' received ', self.txn.get_txn_str())

            if(self.txn.txnID in nodes[rec].rec_txn):
                return
            
            #store in the node the sender of txn
            nodes[rec].rec_txn[self.txn.txnID] = sen

            # add to unused list
            if((self.txn.txnID not in nodes[rec].unused) and (self.txn.txnID not in nodes[rec].used)):
                nodes[rec].unused[self.txn.txnID] = self.txn

            #forward to all peers except for the sender
            msg_size = 8  # in kbits
            for peer in peers[rec]:
                if(peer == sen):
                    continue
                else:
                    event_queue.add_event(
                        curr_time + get_latency(rec, peer, 8), Event(Event_type.rec_txn, rec, peer, self.txn))
        elif(self.type == Event_type.gen_blk):
            # mark txns in the mined block as used, TODO: recheck this part
            for txn in self.blk.txns:
                if(txn.sender == -1):
                    continue
                nodes[rec].unused.pop(txn.txnID)
                nodes[rec].used[txn.txnID] = txn

            

            # check if longest chain block is parent of block in the event
            if(nodes[rec].blockchain.mining_block == self.blk.parent_blkID):
                #add block to node's blockchain
                nodes[rec].blockchain.add_block(self.blk)

                #broadcast the new block
                for peer in peers[rec]:
                    event_queue.add_event(curr_time+ get_latency(rec, peer, self.blk.size), 
                        Event(Event_type.rec_blk, rec, peer, txn= None, blk=self.blk))

                #create a new mining event, make sure to mine valid block
                if(True): #TODO: check for creating an invalid block 
                    gen_valid_blk(rec)
                else:
                    pass
        elif(self.type == Event_type.broadcast_invalid_block):
            #TODO: should rec generate a valid block here?
            for peer in peers[rec]:
                event_queue.add_event(curr_time + get_latency(rec, peer, self.blk.size),
                                      Event(Event_type.rec_blk, rec, peer, txn=None, blk=self.blk))
        else: #block received event
            pass
            
    
if __name__ == '__main__':
    gen_graph()                 #graph is sampled
    # print_graph()
    # print('Events:')
    for i in range(n):          #node objects are assigned to each node id
        nodes.append(Node())
    init_global_values()        #parameters for calculating latency are assigned
    gen_txn(curr_time)
    event_queue.execute_event_queue()
