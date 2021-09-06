from igraph import *
import sys
import numpy as np
from enum import Enum
from termcolor import colored
import time
import matplotlib.pyplot as plt
from bisect import bisect

# Parameters
n = 10  # number of nodes in P2P network
z = 0.7  # probability that a node is fast
p_edge = 0.3  # probability with which edge is drawn between two nodes
seed = 102  # seed for random functions
Ttx = 30  # mean time for transaction generation
total_sim_time = 10    # total time the simulation will run
compile_time = time.time()


class Event_type(Enum):
    gen_txn = 0
    rec_txn = 1
    gen_blk = 2
    rec_blk = 3
    broadcast_invalid_block = 4


# constants
MINING_FEES = 50
MINING_FEE_SENDER = -1
MAX_TXN_NUM = 1000
INVALID_BLOCK_FREQ = 50

np.random.seed(seed)

peers = {}  # will store adjacency lists, it is a dictionary of lists
rho = {}  # stores speed of light propagation delay, accessed as rho[i,j]
c = {}  # stores link speed, used as c[i,j]
# stores mean of exponential dist for queuing delay, used as d_mean[i,j]
d_mean = {}
nodes = []  # stores nodes of the network
block_no = 0  # id for blocks
txn_no = 0  # id for txns
curr_time = 0  # current time in the simulation


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
        print(f"{i} : {peers[i]}")


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
    event_queue.add_event(next_time, Event(
        Event_type.gen_txn, sender=-1, rec=sen, txn=txn, blk=None))
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
    balance[nodeID] += MINING_FEES
    for k in txn_pool.keys():
        if(len(txns) > MAX_TXN_NUM):
            break
        if(txn_pool[k].amount <= balance[txn_pool[k].sender]):
            txns.append(txn_pool[k])
            balance[txn_pool[k].sender] -= txn_pool[k].amount
            balance[txn_pool[k].receiver] += txn_pool[k].amount
    next_time = curr_time + np.random.exponential(nodes[nodeID].alpha)
    new_blk = Block(block_no, curr_longest_block, txns)
    block_no += 1
    event_queue.add_event(next_time, Event(
        Event_type.gen_blk, -1, nodeID, txn=None, blk=new_blk))

# Structs


class Event_Queue:
    def __init__(self):
        self.queue = []

    def add_event(self, time, event):
        index = bisect(self.queue,(time,event))
        self.queue.insert((time,event))

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
        self.sender = sender  # Node ID of the sender -- if == -1
        # this is a mining transaction ##
        self.receiver = receiver            # Node ID of the receiver
        self.amount = amount                # amount of bitcoins being transferred

    def __repr__(self):
        """Returns the txn in string format"""
        if (self.sender == MINING_FEE_SENDER):
            return f"TxnID: {self.txnID}; Node {self.receiver} mined, {self.amount} coins"
        else:
            return f"TxnID: {self.txnID}; Node {self.sender} -> Node {self.receiver}, {self.amount} coins"


class Block:
    def __init__(self, blkID, parent_blkID, txns):
        self.blkID = blkID                  # Block ID
        self.parent_blkID = parent_blkID    # Parent Block ID
        self.size = len(txns)*8             # Size of the Block in kbits
        self.txns = txns  # List of transactions --
        #  one transaction having "-1"
        #  as sender is mining transaction ##

    def __repr__(self):

        return f"BlkID: {self.blkID}; Parent : {self.parent_blkID}; Txns : {self.txns}"

    def log_data(self, toa=None):
        if (self.blkID == -1):
            return f"<genesis block hash>,N/A,N/A"
        else:
            return f"{self.blkID},{'' if toa is None else time.asctime(time.gmtime(toa))},{self.parent_blkID if self.parent_blkID != -1 else '<genesis block hash>'}"


class Node:
    def __init__(self):
        self.alpha = 120 + np.random.random()*(180)  # Average Mining time
        # Selected uniformly from [120,300]
        self.is_fast = False       # Slow or Fast
        self.blockchain = BlockChain()    # Blockchain -- tree of blocks
        self.unused = {}                  # List of Unused transactions
        self.used = {}                    # List of Used transactions
        self.rec_txn = {}                 # Dict of txn to sender
        self.rec_blk = {}                 # Dict of blk to sender
        self.toa = {}                     # Dictionary of Block ID mapped to time of arrival


class BlockChain:
    def __init__(self):
        # Dictionary of Block ID mapped to Block structure
        self.block_info = {-1: Block(blkID=-1, parent_blkID=-1, txns=[])}
        #  Initializing it with the genesis block by default ##
        # Dictionary of Block ID mapped to its depth in the tree
        self.block_depth = {-1: 0}
        # Dictionary of (Block ID, Node ID) mapped to number of
        self.node_coins = {}
        #  coins owned by Node till this Block ##
        for i in range(n):
            self.node_coins[-1, i] = 0
        self.mining_block = -1  # Block ID of the last block of the longest chain on
        #  which mining will take place ##
        # blocks whose parents are not in the blockchain
        self.orphans = []
        # self.longest_chain = [-1]                   # List of blockIDs in the longest chain

    def add_block(self, blk):
        """
        Adds block to blockchain is its parent exists in the chain already
        Otherwise adds to orphans
        If successfully added, checks for orphans to add
        Checks validity here for blocks in chain
        Returns a tuple -  (mining_block_has_changed,block_is_valid)
        """
        mining_block_changed = False
        # check if incoming block's parent is in node's blockchain
        if(blk.parent_blkID in self.block_info):
            if (not self.check_valid_block(blk)):
                return (False, False)
            self.block_info[blk.blkID] = blk
            self.block_depth[blk.blkID] = 1 + \
                self.block_depth[blk.parent_blkID]
            for i in range(n):
                self.node_coins[blk.blkID,
                                i] = self.node_coins[blk.parent_blkID, i]
            for txn in blk.txns:
                if(not (txn.sender == MINING_FEE_SENDER)):     # check that txn is not mining fees
                    self.node_coins[blk.blkID, txn.sender] -= txn.amount
                self.node_coins[blk.blkID, txn.receiver] += txn.amount
            if(self.block_depth[blk.blkID] > self.block_depth[self.mining_block]):
                # if(blk.parent_blkID == self.mining_block):
                #     self.longest_chain.append(blk.blkID)

                self.mining_block = blk.blkID
                mining_block_changed = True
            orphans_copy = self.orphans.copy()
            for orphan in orphans_copy:
                if(orphan.parent_blkID == blk.blkID):
                    self.orphans.remove(orphan)
                    mining_block_changed = mining_block_changed or self.add_block(
                        orphan)[0]

        else:
            self.orphans.append(blk)
        return (mining_block_changed, True)

    def check_valid_block(self, blk):
        balance = {}
        for i in range(n):
            balance[i] = self.node_coins[blk.parent_blkID, i]
        for txn in blk.txns:
            if(not (txn.sender == MINING_FEE_SENDER)):
                balance[txn.sender] -= txn.amount
                if(balance[txn.sender] < 0):
                    return False
            balance[txn.receiver] += txn.amount
        return True


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
            # print(
            #     f"{curr_time:.2f} : Node {sen} -> Node {rec} , Txn {self.txn.txnID} generated - {self.txn}", file=sys.stderr)

            # store the sender of the event as -1
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
            # print(
            #     f"{curr_time:.2f} : Node {sen} -> Node {rec} , Txn {self.txn.txnID} received - {self.txn}", file=sys.stderr)

            if(self.txn.txnID in nodes[rec].rec_txn):
                return

            # store in the node the sender of txn
            nodes[rec].rec_txn[self.txn.txnID] = sen

            # if rec has already got this txn through a block, then it just stores it in its pool
            # without forwarding
            # add to unused list
            # TODO: What exactly to do if we receive a block with new txn
            # if((self.txn.txnID not in nodes[rec].unused) and (self.txn.txnID not in nodes[rec].used)):
            #     nodes[rec].unused[self.txn.txnID] = self.txn

            nodes[rec].unused[self.txn.txnID] = self.txn

            # forward to all peers except for the sender
            msg_size = 8  # in kbits
            for peer in peers[rec]:
                if(peer == sen):
                    continue
                else:
                    event_queue.add_event(
                        curr_time + get_latency(rec, peer, 8), Event(Event_type.rec_txn, rec, peer, self.txn))
        elif(self.type == Event_type.gen_blk):

            print(
                f"{curr_time:.2f} : Node {rec} , Blk {self.blk.blkID} mined - {self.blk}", file=sys.stderr)

            # check if longest chain block is parent of block in the event
            # otherwise discard event
            if(nodes[rec].blockchain.mining_block == self.blk.parent_blkID):

                # mark txns in the mined block as used
                for txn in self.blk.txns:
                    if(txn.sender == MINING_FEE_SENDER):
                        continue
                    nodes[rec].unused.pop(txn.txnID)
                    nodes[rec].used[txn.txnID] = txn

                # add toa, add block to node's blockchain and add to rec_blk map
                nodes[rec].toa[self.blk.blkID] = curr_time
                nodes[rec].blockchain.add_block(self.blk)
                nodes[rec].rec_blk[self.blk.blkID] = -1

                # broadcast the new block
                for peer in peers[rec]:
                    event_queue.add_event(curr_time + get_latency(rec, peer, self.blk.size),
                                          Event(Event_type.rec_blk, rec, peer, txn=None, blk=self.blk))

                # create a new mining event
                if(False):
                    pass  # TODO:need to generate invalid block
                else:
                    gen_valid_blk(rec)
            else:
                print(colored(
                    f"{curr_time:.2f} : Terminated bad mined blk {self.blk.blkID} at Node {rec}, new leaf block is {nodes[rec].blockchain.mining_block} - {self.blk}", "red"), file=sys.stderr)
                return

        # elif(self.type == Event_type.broadcast_invalid_block):
        #     #TODO: should rec generate a valid block here?
        #     for peer in peers[rec]:
        #         event_queue.add_event(curr_time + get_latency(rec, peer, self.blk.size),
        #                               Event(Event_type.rec_blk, rec, peer, txn=None, blk=self.blk))
        else:  # block received event
            # check if havent received it already, otherwise discard

            # print(
            # f"{curr_time:.2f} : Node {sen} -> Node {rec} , Blk {self.blk.blkID} received - {self.blk}", file=sys.stderr)

            if(self.blk.blkID in nodes[rec].rec_blk):
                return

            prev_mining_block = nodes[rec].blockchain.mining_block

            # add block and its children if present in orphan lists
            # TODO: handle case if blk contains a txn not in the pool of node
            (mining_block_changed,
             block_is_valid) = nodes[rec].blockchain.add_block(self.blk)
            # check if block is valid, otherise discard
            if(not block_is_valid):
                print(colored(
                    f"{curr_time:.2f} : Discarded invalid received blk {self.blk.blkID} at Node {rec} - {self.blk}", "red"), file=sys.stderr)
                return

            # add to received block map
            nodes[rec].rec_blk[self.blk.blkID] = sen

            # add toa
            nodes[rec].toa[self.blk.blkID] = curr_time

            # forward to peers
            for peer in peers[rec]:
                if(peer == sen):
                    continue
                event_queue.add_event(curr_time + get_latency(rec, peer, self.blk.size),
                                      Event(Event_type.rec_blk, rec, peer, txn=None, blk=self.blk))

            for txn in self.blk.txns:
                if(txn.sender == MINING_FEE_SENDER):
                    continue
                if(txn.txnID not in nodes[rec].rec_txn):
                    print('Block containts unreceived txn')

            if(mining_block_changed):
                new_mining_block = nodes[rec].blockchain.mining_block
                # chain_changed = True # why this
                iter = new_mining_block
                while((iter != prev_mining_block) and (iter != -1)):
                    iter = nodes[rec].blockchain.block_info[iter].parent_blkID

                # chain extended, do nothing
                if (iter == -1):
                    # chain changed, reset all used txns
                    nodes[rec].unused.update(nodes[rec].used)
                    nodes[rec].used = {}

                # from iter (exclusive) till end of chain, make txns used
                curr_block = new_mining_block
                while (curr_block != iter):
                    for txn_it in nodes[rec].blockchain.block_info[curr_block].txns:
                        nodes[rec].used[txn_it.txnID] = txn_it
                        if (txn_it.txnID in nodes[rec].unused):
                            nodes[rec].unused.pop(txn_it.txnID)
                    curr_block = nodes[rec].blockchain.block_info[curr_block].parent_blkID

                # restart mining on the new mining block
                gen_valid_blk(rec)
            else:
                return


def finish_simulation():
    for i in range(len(nodes)):
        l_log = []
        with open(f"{i}.log", 'w') as f:
            for id, blk in nodes[i].blockchain.block_info.items():
                if (id == -1):
                    continue  # genesis block
                tm = None
                if (blk.blkID in nodes[i].toa):
                    tm = compile_time + nodes[i].toa[blk.blkID]
                l_log.append((tm, blk.log_data(tm)))
            for i in sorted(l_log):
                f.write(i[1])
                f.write("\n")


def make_graph():
    g = Graph(directed=True)
    bchain = nodes[0].blockchain
    for k in bchain.block_info.keys():
        g.add_vertex(name=f"{k}", label=k if k != -1 else "G")

    for (k, v) in bchain.block_info.items():
        if (k == -1):
            continue
        g.add_edge(f"{k}", f"{v.parent_blkID}")

    fig, ax = plt.subplots()

    root = g.vs.find(name="-1")
    style = {}
    style["vertex_size"] = 10
    style["vertex_label_dist"] = 1.5
    style["vertex_label_size"] = 10
    style["layout"] = g.layout_reingold_tilford(root=[root.index])
    style["target"] = ax

    plot(g, **style)
    plt.show()


if __name__ == '__main__':
    gen_graph()  # graph is sampled
    # print_graph()
    # print('Events:')
    for i in range(n):  # node objects are assigned to each node id
        nodes.append(Node())
    for i in range(int(n*z)):
        nodes[i].is_fast = True
    init_global_values()  # parameters for calculating latency are assigned
    gen_txn()
    for i in range(n):
        gen_valid_blk(i)
    event_queue.execute_event_queue()
    finish_simulation()
    make_graph()
