from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import itertools


def centralty(A: np.matrix, rho: float) -> np.matrix:
    """

    Parameters
    ----------
    A : np matrix
    rho : network effect

    Returns
    -------
    Centrality vector as described in paper
    """
    n = A.shape[0]
    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    central = lin.inv(ident - (rho * ApA))
    central = central @ ones  
    return central

def price_vector(a, c, rho, G):
    n = len(G)
    frac1 = (a+c)/2
    frac2 = rho * ( (a-c)/2)
    return (frac1 * np.ones((n,1))) + (frac2 * (G - G.T) @ centralty(G, rho))
 

def consumption(n, rho, a, c, G):
    p = price_vector(a, c, rho, G)
    v = a * np.ones((n,1))
    v = v - p 
    mat = lin.inv(np.eye(n,n) - 2 * rho * G  )
    return 0.5 * mat @ v 

def optProfit(G, rho, a, c):
    n = len(G)
    price = price_vector(a, c, rho, G)
    consu = lin.inv(np.eye(n,n) - 2*rho*G)
    consu = 0.5 * consu @ (a * np.ones((n,1)) - price)
    profit = (price - c*np.ones((n,1))).T @ consu
    return profit[0,0]
 
def computeProfit(G, v, rho, a, c):
    """Profit in the graph G using prices v

    Args:
        G (Array): Graph
        v (vector): price vector to use
        rho (float): discount factor
        a (float): _description_
        c (flaot): _description_

    Returns:
        flaot: profit
    """
    n = len(G)
    consu = lin.inv(np.eye(n,n) - 2*rho*G)
    consu = 0.5 * consu @ (a * np.ones((n,1)) - v)
    profit = (v - c*np.ones((n,1))).T @ consu
    return profit[0,0]


## From itertools, but with modified initial args 
def product(args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def get_mesh(coords : np.arrary, steps : int) ->  itertools.product :
    ### 
    # coords: nx2 array where each row is the start and end index for that dimension 
    # steps : number of steps to take in each dimension. Total size of mesh is steps ^ dimension 
    # Returns a generator to all the coordinate space in this domain
    one_d_spaces = [np.linspace(i[0], i[1], steps) for i in coords]
    return product(one_d_spaces)


def make_n_layer(core, inner_weights, branching_factors):
    sizes = [ core ]
    cores = [ core ]
    for i in range(1, len(inner_weights)+1):
        sizes.append(sizes[i-1] * branching_factors[i-1])
        cores.append(cores[i-1] + sizes[i])
    n = sum(sizes)
    G = np.zeros((n,n))
    cur = sizes[0]
    for i in range(cores[0]):
        for j in range(inner_weights[0]):
            k = (i + j+1) % core
            G[i,k] = 1
        for j in range(branching_factors[0]):
            G[i, cur] = 1
            cur += 1
    for layer in range(1, len(inner_weights)  ):
        for i in range(cores[layer-1], cores[layer] ):
            for j in range(inner_weights[layer]):
                k = i + j  + 1 
                if(k >= cores[layer]):
                    overflow = k - cores[layer]
                    k = cores[layer - 1] + overflow
                    #print(n, k, j)
                G[i, k] = 1
            for j in range(branching_factors[layer]):
                G[i, cur] = 1
                cur += 1 
    ## Add in loop from last to first node
    #for node in range(cores[-2], cores[-1]):
    #    G[node, node % cores[0]] = 1
    return cores, G 

def slice(i, cores):
    n = cores[-1]
    slc = np.zeros((n,))
    if(i == 0):
        slc[0:cores[0]] = 1
        return slc 
    slc[cores[i-1]: cores[i]] = 1
    return slc 
    
def build_constraint(cores, B, a, c):
    n = len(cores)
    G = np.zeros((n,n))
    v = np.zeros((n,1))
    ones = np.ones((cores[-1],1))
    for i in range(n):
        for j in range(n):
            G[i,j] = slice(i, cores).T @ B @ slice(j, cores) + slice(j, cores).T @ B @ slice(i, cores)
        v[i] = a * slice(i, cores).T @ B @ ones + c * ones.T @ B @ slice(i, cores)
    return G, v

def price_from_small(cores, block_map, prices):
    n  = cores[-1]
    v = np.zeros((n,1))
    v[0:cores[0]] = prices[ block_map[0]]
    for i in range(1, len(cores)):
        v[cores[i-1]: cores[i]] = prices[ block_map[i]]
    return v 


def profit_from_spec(core, inner_weights, branching_factors):
    cores, G = make_n_layer(core, inner_weights, branching_factors)
    n = cores[-1]
    I = np.eye(n,n)
    rho = 0.95 / norm(G + G.T, ord=2)
    B = inv(I - 2 * rho * G)
    A, v = build_constraint(cores, B, a, c)
    small_prices = inv(A) @ v 
    price = price_from_small(cores, [i for i in range(len(cores))], small_prices)
    ones = np.ones((n,1))
    profit = 0.5 * (price - c*ones).T @ B @ (a*ones - price)
    print(a,c)
    return profit[0,0], price, B 


def profit_with_block(core, inner_weights, branching_factors, blocks):
    cores, G = make_n_layer(core, inner_weights, branching_factors)
    n = cores[-1]
    I = np.eye(n,n)
    rho = 0.95 / norm(G + G.T, ord=2)
    B = inv(I - 2 * rho * G)
    A, v = build_constraint(cores, B, a, c)
    P = np.zeros((len(blocks), len(A)))
    block_map = [0 for i in range(len(A))]
    for i in range(len(blocks)):
        for j in blocks[i]:
            P[i, j] = 1
            block_map[j] = i 

    small_prices = inv(P @ A @ P.T) @ P @ v
    price = price_from_small(cores, block_map, small_prices)
    ones = np.ones((n,1))
    profit = 0.5 * (price - c*ones).T @ B @ (a*ones - price)
    return profit[0,0], price


def myblock(k, blocks):
    if(k < blocks[0]):
        return 0 
    count = 1
    while(k >= sum(blocks[0:count])):
        count += 1
        
    return count -1

def stoch_block(block_sizes, prob_mat):
    n = sum(block_sizes)
    A = np.zeros((n,n))
    import time 
    np.random.seed(int(time.time()))

    for i in range(n):
        for j in range(n):
            commun1 = myblock(i, block_sizes)
            commun2 = myblock(j, block_sizes)
            coin_flip = np.random.rand()
            if(coin_flip < prob_mat[commun1, commun2]):
                A[i,j] = 1
    return A

def aver_stoch_block(block_sizes, prob_mat):
    n = sum(block_sizes)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            commun1 = myblock(i, block_sizes)
            commun2 = myblock(j, block_sizes)
            A[i,j] = prob_mat[commun1, commun2];
    return A  


def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def linearOrders(initial):
    order = [initial.copy()]
    for i in range(0,len(initial[0])):
        if(initial[0][:i] == []):
            temp = [initial[0][i:].copy()]
        else:
            temp = [ initial[0][:i].copy(), initial[0][i:].copy()]
        order.append(temp.copy())
      
    return order[1::]

def ancestors(collection):
    new_res = []
    for i  in range(len(collection)):
        base = collection.copy()
       
        for split in partition(collection[i]):
            if(len(split) == 2) and split[0] and split[1]:
                nw = base.copy()
                nw.pop(i)
                nw.append(split[0])
                nw.append(split[1])
                new_res.append(nw)
    return new_res