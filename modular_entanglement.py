# This module is meant to accompany the work in "Unitary Entanglement Generation
# in Hierarchical Quantum Networks" Bapat et al 2018
# IPython notebooks are provided which demonstrate the functions included by
# reproducing the results printed in the text
# Contact Zachary Eldredge at eldredge@umd.edu with questions

import matplotlib.pyplot as plt
import itertools as it
import random
import copy
import numpy as np
import networkx as nx
from functools import reduce
from operator import mul, add
from math import log, sqrt, ceil
import os
import pandas as pd

########################################
## PART 1: Functions for Probabilistic Unitary Simulation
########################################


#1A: Weighted Graph construction
def build_weighted_hierarchical(H : nx.Graph,levels : int, wfun): 
    """ Function to build a hierarchical weighted graph out of a small, simple
    graph

    Parameters
    ----------
    H, networkx.Graph:
        A graph which will be hierarchically expanded to a larger one
    levels, int:
        Number of times to repeat the nesting process
    wfun, function from int -> int or float:
        A function that defines what weight to use for connections at the lth
        level of the hierarchy.

    Returns
    -------
    hierarchy, Graph:
        The iterated hierarchical product of H with itself and the appropriate
        weight function
    """
    # Build a hierarchical product of one graph ('H') with 'levels' levels,
    # where the l-th level has weight wfun(l) Note that this identical to the
    # build_hierarchical function, and I refer you to that function for the
    # algorithm that constructs the graph. In this function, we use the same
    # logic except that we add the weight according to the passed function
    label_list = list(it.product(list(H.nodes()),repeat=levels))
    hierarchy = nx.Graph()
    for i in label_list: # Add all those nodes
        hierarchy.add_node(i)
    
    for node in label_list:
        node_num = hierarchyToNumber(node, len(H.nodes()))
        for l in range(levels):
            base = node[:-(1+l)] # get the first part of the node
            for h in H.edges(node[-(l+1)]): # for every edge in base graph
                connected_node = base + (h[-1],) + tuple(0 for i in range(l))
                connected_node_num = hierarchyToNumber(connected_node, 
                        len(H.nodes()))
                hierarchy.add_edge(node, connected_node, weight = wfun(l)) 
                # add the corresponding edge in this hierarchy graph
            if node[-(l+1)] != 0:
                break
    
    return hierarchy

def hierarchyToNumber(digits, b):
    """ Function to convert an address in the hierarchy to a number.

    Parameters
    ----------
    digits, tuple of ints:
        The address in the hierarchy.
    b, int:
       The order of the base graph for the hierarcical product.

    Returns
    -------
    count, int:
        A number which can be used to index the hierarchy.
    """

    count = 0
    for pos, d in enumerate(digits):
        count += d*(b**pos)
    return count

#1B: GHZ Creation ("Color Spread") Simulation
def nx_color_spread(init_pt, graph,maxtsteps):
    """ Function to perform the random simulation of spreading the GHZ state on
    a weighted graph

    Parameters
    ----------
    init_pt, any type:
        Label for the initial point we are spreading from. Should be present in
        the graph.
    graph, Graph:
        Weighted graph describing connections between nodes. Weight of each node
        will be the probability of success
    maxtsteps, int:
        Maximum number of timesteps to run simulation for

    Returns
    -------
    color_count, list of ints:
        List whose ith element is the number of converted nodes at the ith
        timestep
    boundary_count, list of ints:
        List whose ith element is the size of the boundary at the ith timestep
    graph, Graph:
        The graph (complete with converted nodes recolored blue) after the final
        timestep
    """
    # Try to probabilistically color the graph and see how long it takes
    # Function will return a list of how many points were colored at each time
    # step
    color_count = [1]
    color_dict = {}
    for x in graph.nodes():
        color_dict[x] = 'white'
    color_dict[init_pt] = 'blue' # I like blue.
    nodes = [init_pt]
    boundary = list(graph[init_pt].keys())
    boundary_count = [len(boundary)]
    for t in range(maxtsteps):
        new_boundary = copy.deepcopy(boundary) # Make a copy of the current
                                               # boundary to update as we go
        for b in boundary: # Look at each node b on the boundary
            for n in graph[b]: 
                # Look at each node connected to those nodes
                if color_dict[n] == 'blue': 
                # If they are blue, there is chance they spread the entanglement
                    if random.uniform(0,1) <= graph[n][b]['weight']:
                        # weight provides the probability of spread
                        color_dict[b] = 'blue' # color that node
                        nodes.append(b)
                        # Now that we have added the node b, we need to remove
                        # it from the list of the boundary
                        new_boundary.remove(b)
                        for new_b in graph[b]:
                            # Don't add already-added nodes to boundary!
                            if color_dict[new_b] != 'blue':
                                # Don't double up in the boundary
                                if new_b not in new_boundary: 
                                    new_boundary.append(new_b)
                if color_dict[b] == 'blue':
                    break # Break out of this loop
        boundary = new_boundary
        color_count.append(len(nodes))
        boundary_count.append(len(boundary))
    return color_count


def run_trials(graph, ntrials, init_pt, maxtsteps = 10000):
    """ Function to run many trials of the function nx_color_spread and report
    the average.

    Parameters
    ----------
    graph, nx.Graph:
        The graph which the GHZ state creation is being tried on.
    ntrials, int:
        Number of trials to run.
    init_pt:
        A node label in graph which is identified as the inital qubit in state
        |+> from which the GHZ state will be created.

    Returns
    -------
    mean_time, float:
        The average number of steps required to complete the state creation.
    """
    trials = []
    for trial in range(ntrials):
        trials.append(nx_color_spread(init_pt, graph, maxtsteps).index(len(graph)))

    mean_time = np.mean(trials)
    return mean_time

#1C Analytic functions for weighted diameters
def nn_fit_fn(x):
    """ Function to guess from analytics the time requried to complete the GHZ
    state creation on the nearest-neighbor 2D grid when starting from one
    corner.

    Parameters
    ----------
    x, int:
        Size of the grid, in total number of qubits.

    Returns
    -------
    The total distance that needs to be traversed by two-qubit gates to create
    the GHZ state.
    """
    return 2*(sqrt(x) - 1)

def hier_fit_fn(x, alpha,unit_size):
    """ Function to guess from analytics the time requried to complete the GHZ
    state creation on the hierarchy when starting from a bottom-level node.

    Parameters
    ----------
    x, int:
        Size of the graph, in total number of qubits.
    alpha, float:
        Scaling constant of the graph.
    unit_size, int:
        Order of the base graph in the hierarchy.

    Returns
    -------
    The total distance that needs to be traversed by two-qubit gates to create
    the GHZ state.
    """
    #Note that, as dicussed in the paper, we convert alpha (the weight that
    # scales the probabilities) to 1/alpha (to get estimated times)
    levels = log(x, unit_size)
    beta = 1/alpha
    hier_fit = (beta**levels + beta**(levels - 1)  - 2)/(beta - 1)
    return hier_fit

########################################
## PART 2: Functions for Circuit Placement 
########################################

# note the circuit placement notebook will also call some Part 1 functions


#2A: Functions that aren't (directly) related to parition and rotate algorithm
def get_random_comp_graph(nqubits, ngates):
    """ Function to build a random computational graph.

    Parameters
    ----------
    nqubits, int:
        Number of qubits which will be included in the graph
    ngates, int: 
        The total number of gates in the circuit the computational graph
        represents

    Returns
    -------
    comp_graph, nx.Graph:
        A computational graph for a random circuit with ngates nodes and total
        edge weigth ngates
    """
    comp_graph = nx.Graph() # Create an empty graph
    for q in range(nqubits): # Add all the desired nodes
        comp_graph.add_node(q)
    for g in range(ngates): # Now we will add all the edges
        # Pick two random points
        choose = np.random.choice(np.arange(nqubits),(2,),replace=False) 
         # If we've already got that edge, (there is already a gate)
        if comp_graph.has_edge(*choose):
             # increase its weight by one (add another gate)
            comp_graph[choose[0]][choose[1]]['weight'] += 1
        else:
            comp_graph.add_edge(*choose, weight = 1) # Otherwise make a new edge
    return comp_graph


def length_cost(c, metric, mapping):
    """ Function to evaluate the cost of a computational graph C being placed on
    physical architecture.

    Parameters
    ----------
    c, nx.Graph:
        The computational graph. c[i][j]['weight'] yields the total number of
        gates between qubit i and qubit j in the algorithm to be placed.

    metric, nx.Graph:
        The graph representing the physical architecture (the metric graph).
        metric[i][j] exists if and only if the nodes i and j can perform a
        two-qubit gate between them.

    mapping, dictionary:
        A dictionary where every key is a node label from the comptuational
        graph and every value is a node label from the metric graph,
        representing the proposed circuit placement. Should be a one-to-one map. 

    Returns
    -------
    cost, int or float:
        The total cost of the mapping, that is, the total distance traversed by
        all gates.
    """
    # First we reverse the mapping
    rev_mapping = {v:k for k,v in mapping.items()}  
    # A note from the authors: why use rev_mapping like this, why not make the
    # map in that order to start with?
    # The reason is that in other code we have developed, we have used
    # rev_mapping to relabel the nodes, which networkx allows you to do easily
    # when the dictionary is in this order. 
    cost = 0 # Initialize the cost
    for i, j in c.edges(): 
        # For every edge, accumulate the cost and multiply by the weight in c
        cost += nx.shortest_path_length(metric, source = rev_mapping[i], 
                target = rev_mapping[j])*c[i][j]['weight']
    return cost


#2B: Partition-and-Rotate Algorithm Functions
def pr_split_nodes(cgraph, node_list, k):
    """ Function which uses partition-and-rotate to produce a grouping of the
    nodes that can be used to place the circuit on a hierarchical graph. 

    Parameters
    ----------
    cgraph, nx.Graph:
        Computational graph to cluster

    node_list, list of labels from cgraph:
        Nodes in node_list which will be clustered (used because this function
        is recursive)

    k, int:
        Number of partitions to create in the "partition" part of the algorithm 

    Returns
    -------
    cluster_list, a list of lists of lists, etc:
        A list which can be fed to convert_split_list_to_dict() to produce the
        circuit mapping. In this list, nodes in the same sub-hierarchies are in
        the same sub-lists, and the first node or cluster in a list is at the
        root of that hierarchy.
    """
    
    # PARTITION

    if len(node_list) <=  k: # Everything in one cluster? no need to split!
        cluster_list = node_list # Return the list
        
    else:
        # Perform clustering on this set of nodes
        if len(nx.subgraph(cgraph,node_list).edges()) > 0: 
            # Assuming there are edges, hand it off to paritioning subroutine
            cluster_list = metis_partition(nx.subgraph(cgraph,node_list), k)
        else:
            # If there are no edges, just chop it into three, it doesn't matter
            cluster_list = list(np.array_split(node_list,3)) 
        # Now for each sublist, call pr_split_nodes (this function!) again
        for l in enumerate(cluster_list):
            # Now, we replace every list element with an element which is split.
            # this recursion continues until we only have k nodes in the
            # subgraph in question

            cluster_list[l[0]] = pr_split_nodes(cgraph,l[1], k)     

    # ROTATE

    # Next, we sort each cluster in terms of the number of connections it has
    # leading outside the cluster. So for instance, at this point cluster_list
    # should be:
    # [ [cluster 1 ], [cluster 2], [cluster 3]] 
    # We now look at each of them in turn and ask which one has the most
    # connections (in Cgraph) that lead to none of the others. That one gets the
    # 0-coordinate. See the function count_out_from_set for more info.

    # Each cluster starts out assuming it has no outward connection
    out_scores = [0]*k
    for l in range(k):
        # Then for each one we count how many outward connections it has
        out_scores[l] = count_out_from_set(cgraph, cluster_list[l], node_list) 

    # Now sort by those scores
    cluster_list = sorted(cluster_list, key = lambda x: 
            count_out_from_set(cgraph, x, node_list), reverse = True) 

    return cluster_list


def metis_partition(graph, nparts):
    """ Function which uses the Metis software package to partition a graph into
    several parts which are minimally connected and perfectly balanced. 

    Parameters
    ----------
    graph, nx.Graph:
        Graph to partition
    nparts, int:
        Number of parts to partition the graph into 

    Returns
    -------
    A list of nparts lists, where the elements which share a sublist belong to
    the same partition
    """

    # This function works through what I will confess is an ugly hack: it writes
    # the graph to file using my function write_metis and then calls the Metis
    # command-line tool

    # Write the file
    write_metis(graph, './temp')
    # Execute Metis
    os.system('gpmetis -ptype=rb ' + './temp ' + str(nparts) + ' > /dev/null')
    # Remove the file we wrote
    os.system('rm ./temp')
    
    # Now, we're going to read in the file and put it into a numpy array
    part_data = np.empty(len(graph), int)
    with open('./temp.part.' + str(nparts)) as f: # Open the file
        # The kth line has information on the kth node
        for k,line in enumerate(f): 
            part_data[k] = int(line[0]) 
            # Node k will now be in partition labeled by part_data[k]
    os.system('rm ./temp.part.' + str(nparts)) # Delete that file

    # On semi-rare occasions, METIS does not yield a balanced partition, so
    # here's some code to fix that. I think this tends to happen if the graph is
    # disconnected or something, I'm not sure.

    # First, count all the different values in part_data 
    vals, inds, counts =  np.unique(part_data, 
            return_index = True, return_counts = True)
    
    # IF all the partitions were the same size, every element of counts would be
    # the same, there'd be no standard deviation, so we use that as a diagnostic
    while np.std(counts) > 0: 
        # find indices with too many and too few
        k_to_reduce = np.where(counts == max(counts)) 
        k_to_increase = np.where(counts == min(counts)) 
        # flip the first in k_to_reduce to be in k_to_increase
        part_data[inds[k_to_reduce]] = k_to_increase 
        # redo standard dev calculation
        vals, inds, counts = np.unique(part_data, return_index = True, 
                return_counts = True)
    
    # Alright, now we want to turn our partition data into a list of lists (how
    # we handle the data elsewhere). partition will be that list of lists;
    # initialize it empty
    partition = np.empty((nparts, len(graph)//nparts), int)

    for i in range(nparts): 
        # For every partition, find every node whose corresponding part_data
        # indicates it belongs there and put it in that array
        partition[i,:] = np.array(graph.nodes())[np.where(
            np.array(part_data) == i)[0]]

    # Then return that array as a list-of-lists
    return [list(i) for i in partition] 

def write_metis(graph, filename):
    """ Function to write a NetworkX Graph object to file in a form that can
    then be acted on by the Metis command-line program.

    Parameters
    ----------
    graph, nx.Graph:
        Graph to write to file

    filename, string:
        filename to use

    Returns
    -------
    None
    """
    metis_file = [str(len(graph)) + ' ' + str(len(graph.edges())) + ' 001\n']
    
    # Metis doesn't want a label above number of nodes
    metis_node_label_dict = {j:i for i,j in enumerate(graph.nodes())} 
    
    for node in graph:
        node_string = ''
        for adj_node in graph[node]:
            node_string += ' ' + str(metis_node_label_dict[adj_node] + 
                    1) + ' ' + str(graph[node][adj_node]['weight'])
        node_string += '\n'
        metis_file.append(node_string)
        
    with open(filename,'w') as f:
        for n in metis_file:
            f.write(n)
            
    return

def count_out_from_set(cgraph, set_to_eval, set_to_leave):
    """ Function which counts how many connections lead out of a set. Used in
    rotation, since we want to ensure this set ends up being at the root of a
    hierarchy.  

    Parameters
    ----------
    cgraph, nx.Graph:
        Computational graph partition-and-rotate is being performed on.

    set_to_eval:
        Set we are interested in seeing the total number of outward connections
        from.

    set_to_leave:
        Set of nodes whose edges we want to discount
     

    Returns
    -------
    count, float:
        Total weight of all edges that originate in set_to_eval and end
        somewhere besides set_to_leave
    """
    count = 0 # initiate the count
   
    # we flatten so we have an easy to work with list of nodes of interest
    for node in flatten(set_to_eval): 
        for edge in cgraph[node]: # look at every one of those edges
            # make sure they don't go to where we're ignoring
            if edge not in set_to_leave:
                # add that weight to the count
                count+=cgraph[node][edge]['weight'] 
    return count

def flatten(l):
    """ Function for flattening lists-of-lists-of-lists in Python. 

    Parameters
    ----------
    l, lists of lists of lists

    Returns
    -------
    A flattened version, a list of the individual items
    """

    # We work with lists-of-lists-of-lists-etc rather than numpy arrays, because
    # we need to be able to replace the list elements with further lists for our
    # recursive scheme to work. BUT python lists can't be easily flattened
    # unlike numpy arrays. Since sometimes we just need the list without all the
    # smaller-scale detail, we need this function

    return list(np.array(l).flatten())


def convert_split_list_to_dict(in_list):
    """ Function to convert a "splitting list" of the type returned by our
    clustering algorithm into a dictionary with node mappings.

    Parameters
    ----------
    in_list, list of lists (of lists, etc):
        The output of a function like pr_split_nodes. This is a nested list of
        numbers in which qubits in the same sub-hierarchy are in the same
        sub-list, and the first entry in every element is the node/cluster which
        is at the root of that sub-hierarchy.

    Returns
    -------
    dictionary, a dictionary of tuple:int pairings:
        This tells us which node in the computational graph each node in the
        hierarchy corresponds to, with each node in the hierarchy represented by
        a tuple (equivalently, a base-k number where k is the order of the base
                graph).
    """
    splitting_array = np.array(in_list)
    nlevels = len(splitting_array.shape) # we can deduce the number of levels
    k = splitting_array.shape[0] # as well as the order of the base graph k
    # Now just use the fact that every node in hierarchy is a base-k number
    dictionary = {i:splitting_array[i] for i in 
            it.product(tuple(range(k)),  repeat = nlevels)}
    return dictionary
