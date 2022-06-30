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
    price = price_vector(a, c, discount, G)
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

