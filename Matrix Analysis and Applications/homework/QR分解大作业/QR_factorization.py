import numpy as np

def vector_norm(x):
    return np.sqrt(np.sum(x**2))

def modified_Gram_Schmidt(input_matrix):
    # A is a matrix
    A = input_matrix.copy()
    m = A.shape[1] # number of columns
    n = A.shape[0] # number of rows
    k = min(n,m)
    Q = np.zeros((n,k)) # Q is an orthogonal matrix
    R = np.zeros((k,m)) # R is an upper triangular matrix
    # calculation of Q
    Q[:,0] = A[:,0] / vector_norm(A[:,0])
    for i in range(1,k):
        Q[:,i] = A[:,i]
    for i in range(1,k):
        for j in range(i,k):
            Q[:,j] = Q[:,j] - np.dot(Q[:,i-1],Q[:,j])*Q[:,i-1]
        Q[:,i] = Q[:,i] / vector_norm(Q[:,i])
    # calculation of R
    for i in range(k):
        for j in range(i,m):
            R[i,j] = np.dot(Q[:,i],A[:,j])
    return Q, R

def Householdr_reduction(input_matrix):
    # A is a matrix
    A = input_matrix.copy()
    n = A.shape[0] # number of rows
    m = A.shape[1] # number of columns
    k = min(n,m) # minimum of rows and columns
    Q = np.eye(n).astype(float) 
    # square matrix
    if n == m :
        iters = k - 1
    else:
        iters = k
    for i in range(iters):
        x = A[i:,i]
        e = np.zeros(n-i)
        e[0] = 1.
        u = (x - vector_norm(x)*e)
        H = np.eye(n)
        H[i:,i:] -= 2*np.outer(u,u)/np.sum(u**2)
        A[i:,i:] = np.matmul(H[i:,i:],A[i:,i:])
        A[i+1:,i] = 0
        Q = np.matmul(H,Q)
    Q = Q.T  # Q^-1 = Q.T
    return Q[:n,:k], A[:k,:m]

def Givens_reduction(input_matrix):
    # A is a matrix
    A = input_matrix.copy()
    n = A.shape[0] # number of rows
    m = A.shape[1] # number of columns
    k = min(n,m) # minimum of rows and columns
    Q = np.eye(n).astype(float)
    for i in range(m): #use i,i to eliminate the element below 
        for j in range(i+1,n):
            if A[j,i] != 0:
                c = A[i,i]/np.sqrt(A[i,i]**2 + A[j,i]**2)
                s = A[j,i]/np.sqrt(A[i,i]**2 + A[j,i]**2)
                G = np.eye(n)
                G[i,i] = c
                G[j,j] = c
                G[i,j] = s
                G[j,i] = -s
                A = np.matmul(G,A)
                A[j,i] = 0
                Q = np.matmul(G,Q)
    Q = Q.T  # Q^-1 = Q.T
    return Q[:n,:k], A[:k,:m]

def solver(R,b):
    # A is a matrix, b is a vector
    x = np.zeros(R.shape[1])
    for i in range(R.shape[1]-1,-1,-1):
        x[i] = (b[i] - np.dot(R[i,i+1:],x[i+1:]))/R[i,i]
    return x


def solve_equations(A,b,method = "modified_Gram_Schmidt"):
    # A is a matrix, b is a vector
    if method not in ["modified_Gram_Schmidt","Householder_reduction","Givens_reduction"]:
        print("Method not recognized")
        return
    print("Method:",method)
    if method == "modified_Gram_Schmidt":
        q,r = modified_Gram_Schmidt(A)
        print("QR factorization by modified Gram-Schmidt method:")
        print("q:\n",q)
        print("r:\n",r)
    elif method == "Householder_reduction":
        q,r = Householdr_reduction(A)
        print("QR factorization by Householder reduction method:")
        print("q:\n",q)
        print("r:\n",r)
    elif method == "Givens_reduction":
        q,r = Givens_reduction(A)
        print("QR factorization by Givens reduction method:")
        print("q:\n",q)
        print("r:\n",r)
    else:
        return
    
    if A.shape[0] < A.shape[1]: #row < column
        print("equations may have no solution!\n use least squares to solve")
        r = np.matmul(r.T,r)
        b = np.matmul(A.T,b)
        x = solver(r,b)
        print("x = ",x)
        return x
    
    elif A.shape[0] > A.shape[1]: #row > column
        print("equations may have multiple solutions!")
        freedom = A.shape[0] - r.shape[0]
        print("freedom:",freedom)
        x = solver(r,np.matmul(q.T,b))
        print("unique  x:\n",x)
        return x
        
    else : #row = column
        print("equations have unique solution!")
        x = solver(r,np.matmul(q.T,b))
        print("x = ",x)
        return x
    
def input_matrix(prompt):
    print(prompt)
    # 获取矩阵的行数和列数
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    # 初始化矩阵
    matrix = np.zeros((rows, cols)).astype(float)
    # 按行读取矩阵元素
    print("Enter the matrix elements (row by row):")
    for i in range(rows):
        #print(f"Enter row {i+1} (separated by space):")
        row = list(map(float, input().split()))
        matrix[i, :] = row
    
    return matrix

def input_vector(prompt):
    print(prompt)
    row = list(map(float, input().split()))
    return row


if __name__ == "__main__":
    methods = {
        1: "Givens_reduction",
        2: "Householder_reduction",
        3: "modified_Gram_Schmidt",
    }
    A = input_matrix("Enter the matrix A:")
    b = input_vector("Enter the vector b:")
    method = input("Enter the method (1:Givens_reduction, 2:Householdr_reduction, 3:modified_Gram_Schmidt):")
    method = methods[int(method)]

    x = solve_equations(A,b,method)
    print("Ax = ",np.matmul(A,x))


'''
测试用例1：
Enter the matrix A:
Enter the number of rows: 3
Enter the number of columns: 3
Enter the matrix elements (row by row):
0 -20 -14
3 27 -4
4 11 -2
Enter the vector b:
20 27 15
Enter the method (1:Givens_reduction, 2:Householdr_reduction, 3:modified_Gram_Schmidt):1
Method: Givens_reduction
QR factorization by Givens reduction method:
q:
 [[ 0.   -0.8  -0.6 ]
 [ 0.6   0.48 -0.64]
 [ 0.8  -0.36  0.48]]
r:
 [[ 5. 25. -4.]
 [ 0. 25. 10.]
 [ 0.  0. 10.]]
equations have unique solution!
x =  [ 1.1456  0.5456 -2.208 ]
Ax =  [20. 27. 15.]
'''

'''
测试用例2：
Enter the matrix A:
Enter the number of rows: 3
Enter the number of columns: 3
Enter the matrix elements (row by row):
0 -20 -14
3 27 -4
4 11 -2
Enter the vector b:
20 27 15
Enter the method (1:Givens_reduction, 2:Householdr_reduction, 3:modified_Gram_Schmidt):2
Method: Householder_reduction
QR factorization by Householder reduction method:
q:
 [[ 0.   -0.8  -0.6 ]
 [ 0.6   0.48 -0.64]
 [ 0.8  -0.36  0.48]]
r:
 [[ 5. 25. -4.]
 [ 0. 25. 10.]
 [ 0.  0. 10.]]
equations have unique solution!
x =  [ 1.1456  0.5456 -2.208 ]
Ax =  [20. 27. 15.]
'''


'''
测试用例3：
Enter the matrix A:
Enter the number of rows: 3
Enter the number of columns: 3
Enter the matrix elements (row by row):
0 -20 -14
3 27 -4
4 11 -2
Enter the vector b:
20 27 15
Enter the method (1:Givens_reduction, 2:Householdr_reduction, 3:modified_Gram_Schmidt):3
Method: modified_Gram_Schmidt
QR factorization by modified Gram-Schmidt method:
q:
 [[ 0.   -0.8  -0.6 ]
 [ 0.6   0.48 -0.64]
 [ 0.8  -0.36  0.48]]
r:
 [[ 5. 25. -4.]
 [ 0. 25. 10.]
 [ 0.  0. 10.]]
equations have unique solution!
x =  [ 1.1456  0.5456 -2.208 ]
Ax =  [20. 27. 15.]
'''