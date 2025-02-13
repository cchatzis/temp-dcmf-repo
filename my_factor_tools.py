# v 1.2

import matcouply

import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from copy import deepcopy
from tlviz.factor_tools import degeneracy_score, factor_match_score
from tensorly.cp_tensor import CPTensor
from matcouply.penalties import MatricesPenalty
from scipy.sparse import diags, csr_matrix, bmat
from scipy.sparse.linalg import spsolve


# from tensorly.random import random_parafac2
# from tensorly.parafac2_tensor import parafac2_to_tensor
# from tensorly.parafac2_tensor import apply_parafac2_projections

# from tlviz.factor_tools import factor_match_score, cosine_similarity,degeneracy_score

# # import math
# # from random import choice
# 
# # from scipy import spatial

# # from matplotlib.ticker import FormatStrFormatter

######################################################################################
# Synthetic data generation
######################################################################################


# Experiment 1
def create_random_smooth_parafac2_tensor(shape, rank, seed, H = None):

    I = shape[0]
    J = shape[1]
    K = shape[2]

    np.random.seed(seed)

    A = np.random.normal(0, 1, (I, rank))
    A = A / np.linalg.norm(A, axis=0)

    while True:
        C = np.random.uniform(1, 15, (K, rank))
        max_cosine_distance = 0
        for i in range(C.shape[1]):
            for j in range(i + 1, C.shape[1]):
                cosine_distance = np.dot(C[:, i], C[:, j]) / (
                    np.linalg.norm(C[:, i]) * np.linalg.norm(C[:, j])
                )
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance

        if max_cosine_distance < 0.8:
            break

    array = np.arange(J)

    # Split the array into 'rank' segments
    segments = np.array_split(array, rank)

    B = [np.zeros((J, rank))]
    for r in range(rank):
        for j in segments[r]:
            B[0][j, r] = np.random.normal(0, 1)

    for k in range(1,K):
        temp_B = deepcopy(B[-1])
        for r in range(rank):
            for j in segments[r]:
                temp_B[j, r] = temp_B[j, r] + np.random.normal(0, 0.25)
        
        B.append(temp_B)

    Bnorms = [tl.norm(b, axis=0) for b in B]
    B = [b / bnorm for b, bnorm in zip(B, Bnorms)]

    for k in range(K):
        for r in range(rank):
            C[k,r] = C[k,r] * Bnorms[k][r]

    tensor = tl.zeros(shape)

    for k in range(K):

        tensor[:, :, k] = A @ np.diag(C[k, :]) @ B[k].T

    tensor = tensor / tl.norm(tensor)

    return tensor, A, B, C

# Experiment 2
def generate_non_parafac2_tensor(shape,rank,seed):

    I = shape[0]
    J = shape[1]
    K = shape[2]

    np.random.seed(seed=seed)

    A = np.random.normal(0, 1, (I, rank))
    A = A / np.linalg.norm(A, axis=0)

    while True:
        C = np.random.uniform(1, 15, (K, rank))
        max_cosine_distance = 0
        for i in range(C.shape[1]):
            for j in range(i + 1, C.shape[1]):
                cosine_distance = np.dot(C[:, i], C[:, j]) / (
                    np.linalg.norm(C[:, i]) * np.linalg.norm(C[:, j])
                )
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance

        if max_cosine_distance < 0.8:
            break

    B = [np.random.normal(0, 1,(J,rank))]

    sign1 = np.random.choice([-1,1])
    sign2 = np.random.choice([-1,1])
    sign3 = np.random.choice([-1,1])

    for k in range(1,K):
        temp_B = deepcopy(B[-1])
        temp_B += np.random.normal(0, 0.1,(J,rank))

        scale1 = np.random.uniform(0.04,0.08)
        scale2 = np.random.uniform(0.04,0.08)
        scale3 = np.random.uniform(0.04,0.08)

        if 0 <= k < 20:
            temp_B[:,0] += sign1 * scale1 * (temp_B[:,1] - temp_B[:,0])
        else:
            temp_B[:,0] += -1 * sign1 * scale1 * (temp_B[:,1] - temp_B[:,0])

        if 5 <= k < 25:
            temp_B[:,1] += sign2 * scale2 * (temp_B[:,2] - temp_B[:,1])
        else:
            temp_B[:,1] += -1 * sign2 * scale2 * (temp_B[:,2] - temp_B[:,1])

        if 15 <= k < 35:
            temp_B[:,2] += sign3 * scale3 * (temp_B[:,2] - temp_B[:,0])
        else:
            temp_B[:,2] += -1 * sign3 * scale3 * (temp_B[:,2] - temp_B[:,0])
            
        B.append(deepcopy(temp_B))

    Bnorms = [tl.norm(b, axis=0) for b in B]
    B = [b / bnorm for b, bnorm in zip(B, Bnorms)]

    tensor = tl.zeros((I,J,K))

    for k in range(K):

        tensor[:, :, k] = A @ np.diag(C[k, :]) @ B[k].T

    tensor = tensor / tl.norm(tensor)

    return tensor,A,B,C

# Experiment 3A
def create_ortho_tensor_H(shape, rank, seed):

    I = shape[0]
    J = shape[1]
    K = shape[2]

    np.random.seed(seed)

    A = np.random.normal(0, 1, (I, rank))
    A = A / np.linalg.norm(A, axis=0)

    while True:
        C = np.random.uniform(1, 15, (K, rank))
        max_cosine_distance = 0
        for i in range(C.shape[1]):
            for j in range(i + 1, C.shape[1]):
                cosine_distance = np.dot(C[:, i], C[:, j]) / (
                    np.linalg.norm(C[:, i]) * np.linalg.norm(C[:, j])
                )
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance

        if max_cosine_distance < 0.8:
            break

    array = np.arange(J)

    # Split the array into 'rank' segments
    segments = np.array_split(array, rank)

    B = [np.zeros((J, rank))]
    for r in range(rank):
        for j in segments[r]:
            B[0][j, r] = np.random.normal(0, 1)

    Bnorms = [tl.norm(b, axis=0) for b in B]
    B = [b / bnorm for b, bnorm in zip(B, Bnorms)]

    # create a list of rank len(segment[i])xlen(segment[i]) orthogonal matrices
    list_of_H = []
    for r in range(rank):
        Q, _ = np.linalg.qr(np.random.normal(0, 1, (len(segments[r]), len(segments[r]))))
        list_of_H.append(Q)
    
    # create a block diagonal matrix from the orthogonal matrices
    H = np.zeros((J,J))
    for r in range(rank):
        H[np.ix_(segments[r],segments[r])] = list_of_H[r]

    new_Bs = [B[0]]

    for k in range(1,K):
        new_Bs.append(H @ new_Bs[-1])

    B = new_Bs

    tensor = tl.zeros(shape)

    for k in range(K):

        tensor[:, :, k] = A @ np.diag(C[k, :]) @ B[k].T

    tensor = tensor / tl.norm(tensor)

    return tensor, A, B, C, H

def generate_random_matrix_non_orthogonal(n, eigenvalues_range=(0.95, 0.99)):
    """
    Generate a random square matrix with eigenvalues within a specified range and non-orthogonal eigenvectors.

    Parameters:
        n (int): The size of the matrix (n x n).
        eigenvalues_range (tuple): Min and max values for eigenvalues.

    Returns:
        np.ndarray: The generated matrix.
    """
    # Step 1: Generate eigenvalues
    eigenvalues = np.random.uniform(eigenvalues_range[0], eigenvalues_range[1], size=n)
    
    # Step 2: Generate a random matrix (not orthogonalized) for eigenvectors
    V = np.random.randn(n, n)
    
    # Step 3: Ensure V is invertible by checking its determinant
    while np.linalg.det(V) == 0:
        V = np.random.randn(n, n)
    
    # Step 4: Construct the matrix using eigenvalue decomposition
    D = np.diag(eigenvalues)  # Create diagonal matrix of eigenvalues
    matrix = V @ D @ np.linalg.inv(V)  # A = V * D * V^-1
    
    return matrix

# Experiment 3B
def generate_non_ortho_dataset(shape,rank,seed):

    I = shape[0]
    J = shape[1]
    K = shape[2]

    np.random.seed(seed=seed)

    A = np.random.normal(0, 1, (I, rank))
    A = A / np.linalg.norm(A, axis=0)

    while True:
        C = np.random.uniform(1, 15, (K, rank))
        max_cosine_distance = 0
        for i in range(C.shape[1]):
            for j in range(i + 1, C.shape[1]):
                cosine_distance = np.dot(C[:, i], C[:, j]) / (
                    np.linalg.norm(C[:, i]) * np.linalg.norm(C[:, j])
                )
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance

        if max_cosine_distance < 0.8:
            break

    B = [np.random.normal(0, 1,(J,rank))]

    H = generate_random_matrix_non_orthogonal(J)

    for k in range(1,K):
        B_temp = H @ B[k-1]
        B.append(deepcopy(B_temp))

    # Bnorms = [tl.norm(b, axis=0) for b in B]
    # B = [b / bnorm for b, bnorm in zip(B, Bnorms)]

    tensor = tl.zeros(shape)

    for k in range(K):

        tensor[:, :, k] = A @ np.diag(C[k, :]) @ B[k].T

    tensor = tensor / tl.norm(tensor)

    return tensor, A, B, C, H


######################################################################################
# Metrics
######################################################################################

def check_degenerate(factors, threshold=-0.85):
    """
    Check solution for degenerecy (just a wrapper for tlviz degeneracy score).
    """

    A = factors[2]
    B = factors[1]
    D = factors[0]

    new_B = np.vstack(B)
    decomp = CPTensor((np.ones(A.shape[1]), (D, new_B, A)))

    if degeneracy_score(decomp) < threshold:
        return True
    else:
        return False

def get_fms(gnd_factors, est_factors, skip_mode=None):

    (A, B_is, C) = gnd_factors
    (A2, B_is2, C2) = est_factors

    cp_tensor1 = (
        (np.array([1.0] * A.shape[1])),
        (A, np.vstack(np.array(deepcopy(B_is))), C),
    )
    cp_tensor2 = (
        (np.array([1.0] * A2.shape[1])),
        (A2, np.vstack(np.array(deepcopy(B_is2))), C2),
    )

    return factor_match_score(
            cp_tensor1,
            cp_tensor2,
            absolute_value=True,
            consider_weights=False,
            skip_mode=skip_mode)


######################################################################################
# AO-ADMM penalties
######################################################################################


class myTemporalSmoothnessPenalty(MatricesPenalty):
    def __init__(
        self, smoothness_l, aux_init="random_uniform", dual_init="random_uniform"
    ):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.smoothness_l = smoothness_l

    @copy_ancestor_docstring
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):

        # factor_matrices: factor + mus
        # feasability_penalties: rhos
        # auxes: -||-

        # rhs = [rhos[i] * factor_matrices[i] for i in range(len(B_is))]

        B_is = factor_matrices
        rhos = feasibility_penalties

        rhs = [rhos[i] * factor_matrices[i] for i in range(len(B_is))]

        # Construct matrix A to peform gaussian elimination on

        A = np.zeros((len(B_is), len(B_is)))

        for i in range(len(B_is)):
            for j in range(len(B_is)):
                if i == j:
                    A[i, j] = 4 * self.smoothness_l + rhos[i]
                elif i == j - 1 or i == j + 1:
                    A[i, j] = -2 * self.smoothness_l
                else:
                    pass

        A[0, 0] -= 2 * self.smoothness_l
        A[len(B_is) - 1, len(B_is) - 1] -= 2 * self.smoothness_l

        # Peform GE

        for k in range(1, A.shape[-1]):
            m = A[k, k - 1] / A[k - 1, k - 1]

            A[k, :] = A[k, :] - m * A[k - 1, :]
            rhs[k] = rhs[k] - m * rhs[k - 1]  # Also update the respective rhs!

        # Back-substitution

        new_ZBks = [np.empty_like(B_is[i]) for i in range(len(B_is))]

        new_ZBks[-1] = rhs[-1] / A[-1, -1]
        q = new_ZBks[-1]

        for i in range(A.shape[-1] - 2, -1, -1):
            q = (rhs[i] - A[i, i + 1] * q) / A[i, i]
            new_ZBks[i] = q

        return new_ZBks

    def penalty(self, x):
        penalty = 0
        for x1, x2 in zip(x[:-1], x[1:]):
            penalty += np.sum((x1 - x2) ** 2)
        return self.smoothness_l * penalty
    

class myTemporallyOrthogonalPenalty(MatricesPenalty):
    def __init__(
        self, smoothness_l, H, aux_init="random_uniform", dual_init="random_uniform",verify=False,
    ):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.smoothness_l = smoothness_l
        self.H = H
        self.verify = verify

    @copy_ancestor_docstring
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):

        # factor_matrices: factor + mus
        # feasability_penalties: rhos
        # auxes: -||-

        B_is = factor_matrices
        rhos = feasibility_penalties
        R = B_is[0].shape[-1]
        J = B_is[0].shape[-2]
        K = len(B_is)

        rhs = [rhos[i] * factor_matrices[i].flatten() for i in range(len(B_is))]

        # Build the sparse block matrix A
        diagonal_blocks = []
        off_diagonal_terms = []  # Store off-diagonal connections

        # Step 1: Construct diagonal and off-diagonal blocks
        for k in range(K):
            # Diagonal block
            if k == 0 or k == K - 1:
                diag_coeff = 2 * self.smoothness_l + rhos[k]
            else:
                diag_coeff = 4 * self.smoothness_l + rhos[k]
            A_diag = diag_coeff * np.eye(J * R)
            diagonal_blocks.append(csr_matrix(A_diag))

            # Off-diagonal terms
            if k > 0:
                off_diag = -2 * self.smoothness_l * np.kron(self.H, np.eye(R))
                off_diagonal_terms.append(((k, k - 1), csr_matrix(off_diag)))
            if k < K - 1:
                off_diag = -2 * self.smoothness_l * np.kron(self.H.T, np.eye(R))
                off_diagonal_terms.append(((k, k + 1), csr_matrix(off_diag)))

        # Step 2: Build the sparse matrix A
        block_matrix = [[None for _ in range(K)] for _ in range(K)]

        # Fill diagonal blocks
        for i, diag in enumerate(diagonal_blocks):
            block_matrix[i][i] = diag

        # Fill off-diagonal blocks
        for (i, j), off_diag in off_diagonal_terms:
            block_matrix[i][j] = off_diag

        # Convert to sparse block matrix
        A_sparse = bmat(block_matrix, format="csr")

        B = np.hstack(rhs)

        # Step 3: Solve the sparse system
        Z_vec = spsolve(A_sparse, B)

        # Step 4: Reshape the solution back into matrices
        Z_B = [Z_vec[k * J * R : (k + 1) * J * R].reshape(J, R) for k in range(K)]

        if self.verify:

            # Verification
            residuals = []

            # Check first equation
            lhs_1 = (2 * self.smoothness_l + rhos[0]) * Z_B[0] - 2 * self.smoothness_l * self.H.T @ Z_B[1]
            rhs_1 = rhs[0].reshape(J, R)
            residuals.append(np.linalg.norm(lhs_1 - rhs_1))

            # Check middle equations (k = 2 to K-1)
            for k in range(1, K-1):
                lhs_k = (4 * self.smoothness_l + rhos[k]) * Z_B[k] \
                        - 2 * self.smoothness_l * (self.H @ Z_B[k-1] + self.H.T @ Z_B[k+1])
                rhs_k = rhs[k].reshape(J, R)
                residuals.append(np.linalg.norm(lhs_k - rhs_k))

            # Check last equation
            lhs_K = (2 * self.smoothness_l + rhos[-1]) * Z_B[-1] - 2 * self.smoothness_l * self.H @ Z_B[-2]
            rhs_K = rhs[-1].reshape(J, R)
            residuals.append(np.linalg.norm(lhs_K - rhs_K))

            # # Print residuals
            # for k, res in enumerate(residuals, start=1):
            #     print(f"Residual for equation {k}: {res}")

            # Check if all residuals are within tolerance
            tolerance = 1e-4
            if all(res < tolerance for res in residuals):
                print("The solution satisfies the system of equations.")
            else:
                print("The solution does not satisfy the system. Check for errors.")

        return Z_B

    def penalty(self, x):
        penalty = 0
        for x1, x2 in zip(x[:-1], x[1:]):
            penalty += np.sum((self.H @ x1 - x2) ** 2)
        return self.smoothness_l * penalty
    

class myTemporallyPenalty(MatricesPenalty):
    def __init__(
        self, smoothness_l, H, aux_init="random_uniform", dual_init="random_uniform",verify=False,
    ):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.smoothness_l = smoothness_l
        self.H = H
        self.verify = verify

    @copy_ancestor_docstring
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):

        # factor_matrices: factor + mus
        # feasability_penalties: rhos
        # auxes: -||-

        B_is = factor_matrices
        rhos = feasibility_penalties
        R = B_is[0].shape[-1]
        J = B_is[0].shape[-2]
        K = len(B_is)

        rhs = [rhos[i] * factor_matrices[i].flatten() for i in range(len(B_is))]

        # Build the sparse block matrix A
        diagonal_blocks = []
        off_diagonal_terms = []  # Store off-diagonal connections

        # Step 1: Construct diagonal and off-diagonal blocks
        for k in range(K):
            # Diagonal block
            if k == 0:
                A_diag = np.kron(2 * self.smoothness_l * self.H.T @ self.H + rhos[k] * np.eye(J), np.eye(R))
            elif k == K - 1:
                A_diag = (2 * self.smoothness_l + rhos[k] ) * np.eye(J * R)
            else:
                A_diag = np.kron(2 * self.smoothness_l * self.H.T @ self.H + rhos[k] * np.eye(J) + 2 * self.smoothness_l * np.eye(J),np.eye(R))

            diagonal_blocks.append(csr_matrix(deepcopy(A_diag)))

            # Off-diagonal terms
            if k > 0:
                off_diag = -2 * self.smoothness_l * np.kron(self.H, np.eye(R))
                off_diagonal_terms.append(((k, k - 1), csr_matrix(off_diag)))
            if k < K - 1:
                off_diag = -2 * self.smoothness_l * np.kron(self.H.T, np.eye(R))
                off_diagonal_terms.append(((k, k + 1), csr_matrix(off_diag)))

        # Step 2: Build the sparse matrix A
        block_matrix = [[None for _ in range(K)] for _ in range(K)]

        # Fill diagonal blocks
        for i, diag in enumerate(diagonal_blocks):
            block_matrix[i][i] = diag

        # Fill off-diagonal blocks
        for (i, j), off_diag in off_diagonal_terms:
            block_matrix[i][j] = off_diag

        # Convert to sparse block matrix
        A_sparse = bmat(block_matrix, format="csr")

        B = np.hstack(rhs)

        # Step 3: Solve the sparse system
        Z_vec = spsolve(A_sparse, B)

        # Step 4: Reshape the solution back into matrices
        Z_B = [Z_vec[k * J * R : (k + 1) * J * R].reshape(J, R) for k in range(K)]

        if self.verify:

            # Verification
            residuals = []

            # Check first equation
            lhs_1 = (2 * self.smoothness_l * self.H.T @ self.H + rhos[0] * np.eye(J)) @ Z_B[0] - 2 * self.smoothness_l * self.H.T @ Z_B[1]
            rhs_1 = rhs[0].reshape(J, R)
            residuals.append(np.linalg.norm(lhs_1 - rhs_1))

            # Check middle equations (k = 2 to K-1)
            for k in range(1, K-1):
                lhs_k = (2 * self.smoothness_l * np.eye(J) + 2 * self.smoothness_l * self.H.T @ self.H + rhos[k] * np.eye(J)) @ Z_B[k] \
                        - 2 * self.smoothness_l * (self.H @ Z_B[k-1] + self.H.T @ Z_B[k+1])
                rhs_k = rhs[k].reshape(J, R)
                residuals.append(np.linalg.norm(lhs_k - rhs_k))

            # Check last equation
            lhs_K = (2 * self.smoothness_l + rhos[-1]) * Z_B[-1] - 2 * self.smoothness_l * self.H @ Z_B[-2]
            rhs_K = rhs[-1].reshape(J, R)
            residuals.append(np.linalg.norm(lhs_K - rhs_K))

            # # Print residuals
            # for k, res in enumerate(residuals, start=1):
            #     print(f"Residual for equation {k}: {res}")

            # Check if all residuals are within tolerance
            tolerance = 1e-4
            if all(res < tolerance for res in residuals):
                print("The solution satisfies the system of equations.")
            else:
                print("The solution does not satisfy the system. Check for errors.")

        return Z_B

    def penalty(self, x):
        penalty = 0
        for x1, x2 in zip(x[:-1], x[1:]):
            penalty += np.sum((self.H @ x1 - x2) ** 2)
        return self.smoothness_l * penalty


# ######################################################################################
# # Plotting
# ######################################################################################

def form_plotting_B(B_list, pattern_no, J, K):
    """
    Takes as input a list of B factors and return a matrix containing
    the pattern_no-th column of each factor matrix.
    """

    matrix2return = np.zeros((K, J))

    for k in range(K):

        matrix2return[k, :] = B_list[k][:, pattern_no].T

    return matrix2return


def plot_factors(factors):

    import matplotlib.gridspec as gridspec

    # Normalize factors
    A = deepcopy(factors[0])
    B = deepcopy(factors[1])
    C = deepcopy(factors[2])

    A = A / np.linalg.norm(A, axis=0)
    C = C / np.linalg.norm(C, axis=0)
    for k in range(len(B)):
        B[k] = B[k] / np.linalg.norm(B[k], axis=0)

    # Create a GridSpec with 3 rows and 2 columns
    gs = gridspec.GridSpec(A.shape[1] + 1, 2)

    # Create the subplots with increased DPI
    fig = plt.figure(figsize=(20, 8), dpi=150)
    for i in range(A.shape[1]):
        ax_B1 = fig.add_subplot(gs[i, :])
        B1 = form_plotting_B(B, i, B[0].shape[0], len(B))
        cax = ax_B1.imshow(B1, cmap="viridis", aspect="auto")
        cbar = fig.colorbar(cax, ax=ax_B1, fraction=0.046, pad=0.04)

    ax_A = fig.add_subplot(gs[-1, 0])  # A is in the first column of the third row
    ax_C = fig.add_subplot(gs[-1, 1])  # C is in the second column of the third row

    # Adjust the spacing between the plots
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Adjust the spacing here

    # Prep A
    # barplot of each columns of A
    for i in range(A.shape[1]):
        ax_A.bar(np.arange(A.shape[0]), A[:, i], alpha=0.95)

    ax_A.set_title("A")
    ax_A.set_axisbelow(True)  # Set grid lines in the background
    ax_A.grid()

    # Prep C
    # lineplot of each columns of C
    for i in range(A.shape[1]):
        ax_C.plot(np.arange(C.shape[0]), C[:, i], alpha=0.95)

    ax_C.set_title("C")
    ax_C.set_axisbelow(True)  # Set grid lines in the background
    ax_C.grid()

    plt.show()