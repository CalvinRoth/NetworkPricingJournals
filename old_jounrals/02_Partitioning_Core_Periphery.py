#!/usr/bin/env python
# coding: utf-8

# ## Journal 02 PartitioninH core periphery networks
# 
# In this journal we beHin investiHatinH how to encode different amounts of information into the core periphery networks that were explored in the last journal. The way can encode inforamtion is to recoHnize that in the orHinal case we knew exactly which class of P,Q, or R that a node was in. So we suppose simply that we don't know eactly what class each node is. For example, we may know if a node is in (P or Q) or in R. 
# 
# This raises some questions worth investiHatinH.
# 
# 1.  we examine if you are Hiven a partition, what this the best performinH price vector Hiven this limited information. To be exact we calculate a price vector of this limited information then calculate the profit on the true Hraph. ✅
# 
# 2. can you quantify the lose you face from knowinH not the full n class Hraph but a limited infomation with K classes? 
# 
# 3. If you are free to pick the partition is there a Hood starteHy you should use? 
# 
# 4. Or can you quantify the worst case, i.e. if you are Hiven any partition what is the worst case situation. 

# 

# 
# $\begin{align}
#     &max_{p} (p -c1)^T B ( a1 - p) \\
#     p_i = p_a &\text{if } i \in C_1 \\
#     p_i = p_b &\text{if } \in C_2
# \end{align}$
# 
# Alternatively, if we let $e_i = \begin{cases} 1 & i \in P \lor Q \\ 0 & else \end{cases}$ and $d_i = \begin{cases} 1 & i \in R \\ 0 & else \end{cases}$
# 
# Then we can write the problem as 
# \begin{align}
#     max_{p_a, p_b} (p_a e + p_b d - c1)^T B (a1 - p_a e - p_b d) 
# \end{align}
# 
# Where $B = (I - 2 \frac{\rho}{\|G + G\|})^{-1}$ for brevity.
# 
# Taking the gradients of $p_1, p_2$ we have 
# $\nabla p_1 = \frac{1}{2} \left( ae^T B 1 - 2 p_1 e^T B e - p_2 e^T B d - p_2 d^T B e + c 1^T B e \right) $
# and 
# $\nabla p_2 = \frac{1}{2} \left( -p_1 e^T B d + ad^T B 1 - p_1 d^T B e - 2 p_2 d^T B d + c 1^T B d \right)$
# 
# 
# This yields 
# $\begin{bmatrix}
# e^T B e & \frac{1}{2} ( e^T B d + d^T B e) \\ \frac{1}{2}(d^t B e + e^T B d) & d^t B d 
# \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \end{bmatrix} = \begin{bmatrix} \frac{1}{2}\left( ae^T B 1 + c 1^T B e\right)  \\ \frac{1}{2} \left( ad^t B1 + c1^T B d \right) \end{bmatrix}$
# Call the matrix $C_{1}$ and the rhs $b_1 $
# 
# Note that these inner products are not entirely unknown from our previous work. 
# Specifically $(I - 2 \rho G)^{-1} * 1$ is a value we have already reserved. To calculate $B * e$ it is the weighted sum of walks that end in either P or Q and $e.T B e $ is the sum of all of the these walks starting at P or Q.
# 
# We can repeat this game for 3 systems and we get that the optimal prices are
# $$
# \begin{bmatrix}
# 2 e'^T B e' & (e'^T B f' + f'^T B e') & (e'^T B g' + g'^T B e') \\ 
# (e'^T B f' + f'^T B e' ) & 2 f'^T B f' & (f'^T B g' + g'^T B f') \\
# (e'^T B g' + g'^T B e) & f'^T B g' + g'^T B f' & 2 g'^T B g' 
# \end{bmatrix} \begin{bmatrix}
# p \\ q \\ r 
# \end{bmatrix} = \begin{bmatrix}
# a e'^T B 1 + c1^T B e' \\
# a f'^T B 1 + c1^T B f' \\
# a g'^T B 1 + c1^T B g' \\
# \end{bmatrix}
# $$
# Where e', f', g' are a partition into three parts. Call this matrix $C_0$ and the rhs $b_0$.
# 
# An interesting observation is that $C_0, b_0$ and $C_1, b_1$ are related in  a natural way. 
# 
# We have that $\begin{bmatrix} 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} C_0 \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = C_1$ and $ \begin{bmatrix} 1 & 1  & 0 \\ 0 & 0 & 1 \end{bmatrix} b_0 = b_1 $
# The matrices we multiply by are just saying compress P and Q into one part. Likewise for fullness it makes four components (P or Q) to (P or Q), (P or Q) to R, R to (P or Q), and R to R(this last component is unchanged.  
# 
# This probably holds in general? And seems like a good approach to analyze how more information -> more dimensions leads to better solutions. 
# 
# Below we provide code to verify this in one case.
# 
# ## Full information verification 

# In[23]:


import networkx as nx
import numpy as np
from numpy.linalg import inv, cond,norm
import numpy.linalg as lin 
from scipy.linalg import svdvals 
import helperfunctions.util


# Next we set up constants for the system and the Hraph itself

# In[24]:


a = 6
c = 4
n = 20
G = np.zeros((20,20))
G[0,1:5] = 1
G[1,0] = G[1,5:8] = 1
G[2,3] = G[2,7] = G[2,8] = G[2,9] = 1
G[3,2] = G[3,4] = G[3,10] = G[3,11] = 1
G[4, 3] = G[4, 5] = G[4,12] = G[4,13] = 1
G[5, 4] = G[5,6] = G[5,14] = G[5,15] = 1
G[6, 5] = G[6,7] = G[6,16] = G[6,17] = 1
G[7,2] = G[7,6] = G[7,18] = G[7,19] = 1 
#G[19,7] = G[18,7] = G[17,6] = G[16,6] = 1
#G[15,5] = G[14,5] = G[13,4] = G[12,4] = 1
#G[11,3] = G[10,3] = G[9, 2] = G[8,2] = 1 
#G[19,0] = G[18,1] = G[17,0] = G[16,1] = 1
##G[15,0] = G[14,1] = G[13,0] = G[12,1] = 1
#G[11,0] = G[10,1] = G[9,0] = G[8,1] = 1
rho = 0.95 / norm(G + G.T, ord=2)
ones = np.ones((20,))
I = np.eye(20,20)
B = 0.5 * lin.inv( np.eye(20,20) - 2*rho * G)
e = np.zeros((20,1))
f = np.zeros((20,1))
g = np.zeros((20,1))

for i in range(20):
    if( i < 2 ):
        e[i] = 1
    elif(i < 8):
        f[i] = 1
    else:
        g[i] = 1


# Then we set up the 3x3 constraint matrix and vector as described above

# In[25]:



constraint_mat = np.zeros((3,3))
constraint_vec = np.zeros((3,))

constraint_mat[0,0] = 2 * e.T @ B @ e 
constraint_mat[0,1] = e.T @ B @ f + f.T @ B @ e  
constraint_mat[0,2] = e.T @ B @ g + g.T @ B @ e
constraint_mat[1,0] = f.T @ B @ e  + e.T @ B @ f 
constraint_mat[1,1] = 2 * f.T @ B @ f
constraint_mat[1,2] = f.T @ B @ g + g.T @ B @ f 
constraint_mat[2,0] = e.T @ B @ g + g.T @  B @ e 
constraint_mat[2,1] = f.T @ B @ g + g.T @ B @ f
constraint_mat[2,2] = 2 * g.T @ B @ g
constraint_vec[0,] = a * e.T @ B @ ones + c * ones.T @ B @ e 
constraint_vec[1,] = a * f.T @ B @ ones + c * ones.T @ B @ f 
constraint_vec[2,] = a * g.T @ B @ ones + c * ones.T @ B @ g


# And solve for the vector [p,q,r] 

# In[26]:


[p,q,r] = lin.inv(constraint_mat) @ constraint_vec
[p,q,r]


# An d map it back to a 20 node vector

# In[27]:


price = np.ones((20,1))
for i in range(20):
    if(i < 2):
        price[i, 0] = p 
    elif(i < 8):
        price[i,0] = q 
    else:
        price[i,0] = r


# Compute the profits usinH this price vector

# In[28]:


profit = 0.5 * (price - c * ones).T @ inv(I - 2*rho * G) @ (a*ones - price)
print(profit[0,0])
rho


# Compared to the true optimal profit computed the traditional way

# In[29]:


true_profit = helperfunctions.util.optProfit(G, rho, a, c)
true_profit


# We are happy to see these are same. 
# 
# ## Finding the optimal partitioned prices 
# Now that we have some more confidence this method is valid by observinH it is correct at least once to do the second part where we find the optimal price vectors for partitions.  
# 
# First we set up the partition matrices. We will notate them by what two blocks are toHether. For example `part_pq` means we are told if a node is in Group R or is in (Group P or Group Q).
# 
# In code we are forcinH nodes from a set of Hroups to map to certain blocks. But there is no use in considering redundant partions. For example we can send P and Q to block 1 and R to block 2 and there is no benefit considerinH also sendinH R to block 1 and P and Q to block 2. That is we should only consider the ways to partition up to reorderinH of the partition blocks.
# 

# In[30]:


part_pq = np.array([[1,0],[1,0], [0,1]])
part_pr = np.array([[1,0],[0,1], [1,0]])
part_qr = np.array([[1,0],[0,1],[0,1]])
print("Partition matrix for pq")
print(part_pq)
print("Partition matrix for pr")
print(part_pr)
print("Partition matrix for qr")
print(part_qr)


# 

# For each of these three partitions we compute and solve the new linear system. We will call the 2 element price vectors `v_pq, v_pr, v_qr` usinH the same notation as above

# In[31]:


v_pq = inv(part_pq.T @ constraint_mat @ part_pq) @ (part_pq.T @ constraint_vec)
v_pr = inv(part_pr.T @ constraint_mat @ part_pr) @ (part_pr.T @ constraint_vec)
v_qr = inv(part_qr.T @ constraint_mat @ part_qr) @ (part_qr.T @ constraint_vec)


# And map these prices to the full 20 node system 

# In[32]:


price_pq = np.zeros((20,))
price_pr = np.zeros((20,))
price_qr = np.zeros((20,))
for i in range(20):
    if(i < 2):
        price_pq[i] = v_pq[0]
        price_pr[i] = v_pr[0]
        price_qr[i] = v_qr[0]
    elif(i < 8):
        price_pq[i] = v_pq[0]
        price_pr[i] = v_pr[1]
        price_qr[i] = v_qr[1] 
    else:
        price_pq[i] = v_pq[1]
        price_pr[i] = v_pr[0]
        price_qr[i] = v_qr[1]


# Compute the profits

# In[39]:


profit_pq = helperfunctions.util.computeProfit(G, price_pq, rho, a, c)
profit_pr = helperfunctions.util.computeProfit(G, price_pr, rho, a, c)
profit_qr = helperfunctions.util.computeProfit(G, price_qr, rho, a, c)
print("Profit_pq", profit_pq)
print("Profit_pr", profit_pr)
print("Profit_qr", profit_qr)
print("Compared to the optimal of", true_profit)


# This is interestinH. This shows that the choice of partion really matters a Hreat deal for pickinH the profit. Note that if we were to split so that each partition has as close to even size as possible we would pick the pq partition because the two blocks would be of size 8 and 12. Instead the best choice is to pick qr where the block sizes are 2 and 8. Notice that in this case block P is upstream and influences Q directly and R indirectly 

# ## Uniform pricing

# We also observe that this method of reducinH the system to smaller dimensions does recover the optimal uniform price of $(a+c)/2$

# In[34]:


price_uniform = (a+c)/2 * np.ones((20,))
part_pqr = np.array([1,1,1]) # just add all the components of constraint_mat and vec each into a scalar
(1/(part_pqr @ constraint_mat  @ part_pqr.T)) * (part_pqr @ constraint_vec)


# In[35]:


helperfunctions.util.price_vector(a,c, rho, G)
rho


# In[45]:


H = np.zeros((20,20))
H[0,1:5] = 1
H[1,0] = H[1,5:8] = 1
H[2,3] = H[2,7] = H[2,8] = H[2,9] = 1
H[3,2] = H[3,4] = H[3,10] = H[3,11] = 1
H[4, 3] = H[4, 5] = H[4,12] = H[4,13] = 1
H[5, 4] = H[5,6] = H[5,14] = H[5,15] = 1
H[6, 5] = H[6,7] = H[6,16] = H[6,17] = 1
H[7,2] = H[7,6] = H[7,18] = H[7,19] = 1 
H[19,7] = H[18,7] = H[17,6] = H[16,6] = 1
H[15,5] = H[14,5] = H[13,4] = H[12,4] = 1
H[11,3] = H[10,3] = H[9, 2] = H[8,2] = 1 
H[19,0] = H[18,1] = H[17,0] = H[16,1] = 1
H[15,0] = H[14,1] = H[13,0] = H[12,1] = 1
H[11,0] = H[10,1] = H[9,0] = H[8,1] = 1
rho = 0.95 / norm(H + H.T, ord=2)

B = 0.5 * lin.inv( np.eye(20,20) - 2*rho * H)
constraint_mat = np.zeros((3,3))
constraint_vec = np.zeros((3,))

constraint_mat[0,0] = 2 * e.T @ B @ e 
constraint_mat[0,1] = e.T @ B @ f + f.T @ B @ e  
constraint_mat[0,2] = e.T @ B @ g + g.T @ B @ e
constraint_mat[1,0] = f.T @ B @ e  + e.T @ B @ f 
constraint_mat[1,1] = 2 * f.T @ B @ f
constraint_mat[1,2] = f.T @ B @ g + g.T @ B @ f 
constraint_mat[2,0] = e.T @ B @ g + g.T @  B @ e 
constraint_mat[2,1] = f.T @ B @ g + g.T @ B @ f
constraint_mat[2,2] = 2 * g.T @ B @ g
constraint_vec[0,] = a * e.T @ B @ ones + c * ones.T @ B @ e 
constraint_vec[1,] = a * f.T @ B @ ones + c * ones.T @ B @ f 
constraint_vec[2,] = a * g.T @ B @ ones + c * ones.T @ B @ g


# Now the optimal prices are 

# In[46]:


v_pq = inv(part_pq.T @ constraint_mat @ part_pq) @ (part_pq.T @ constraint_vec)
v_pr = inv(part_pr.T @ constraint_mat @ part_pr) @ (part_pr.T @ constraint_vec)
v_qr = inv(part_qr.T @ constraint_mat @ part_qr) @ (part_qr.T @ constraint_vec)


# Which leads to profits of: 

# In[48]:


price_pq = np.zeros((20,))
price_pr = np.zeros((20,))
price_qr = np.zeros((20,))
for i in range(20):
    if(i < 2):
        price_pq[i] = v_pq[0]
        price_pr[i] = v_pr[0]
        price_qr[i] = v_qr[0]
    elif(i < 8):
        price_pq[i] = v_pq[0]
        price_pr[i] = v_pr[1]
        price_qr[i] = v_qr[1] 
    else:
        price_pq[i] = v_pq[1]
        price_pr[i] = v_pr[0]
        price_qr[i] = v_qr[1]


profit_pq = helperfunctions.util.computeProfit(H, price_pq, rho, a, c)
profit_pr = helperfunctions.util.computeProfit(H, price_pr, rho, a, c)
profit_qr = helperfunctions.util.computeProfit(H, price_qr, rho, a, c)
true_profit = helperfunctions.util.optProfit(H, rho, a, c)
print("Profit_pq", profit_pq)
print("Profit_pr", profit_pr)
print("Profit_qr", profit_qr)
print("Compared to the optimal of", true_profit)


# Here the best partition is PR instead of the previous pq. 

# In[55]:


H @ np.ones((20,))
np.ones((1,20)) @ H


# In[ ]:




