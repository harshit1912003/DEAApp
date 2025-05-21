import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def continuous_kernel(u, bandwidth):
    return np.exp(-(u ** 2) / (2 * bandwidth ** 2)) / (np.sqrt(2 * np.pi) * bandwidth)

def discrete_kernel(x_i, x_j, c):
    return 1 if np.array_equal(x_i, x_j) else 0

def product_kernel(sample1, sample2, bandwidth, continuous_idx, discrete_idx, c=0.5):
    kernel_vals = 1
    for idx in continuous_idx:
        kernel_vals *= continuous_kernel(sample1[idx] - sample2[idx], bandwidth[idx])
    for idx in discrete_idx:
        kernel_vals *= discrete_kernel(sample1[idx], sample2[idx], c)
    return kernel_vals

def compute_kernel_matrix(data, bandwidth, continuous_idx, discrete_idx):
    n = data.shape[0]
    kernel_matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            kernel_matrix[i, j] = product_kernel(data[i], data[j], bandwidth, continuous_idx, discrete_idx)
            kernel_matrix[j, i] = kernel_matrix[i, j]  
    np.fill_diagonal(kernel_matrix, 0)
    return kernel_matrix


def kernel_conv_cont(u, bandwidth):
    return np.exp(-(u**4) / (2 * bandwidth**2)) / (np.sqrt(4 * np.pi) * bandwidth)


def compute_cross_kernel_matrix(X, Y, bandwidth, continuous_idx, discrete_idx):
    n1, n2 = len(X), len(Y)
    cross_kernel_matrix = np.ones((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            kernel_val = 1
            for idx in continuous_idx:
                u = (X[i][idx] - Y[j][idx]) / bandwidth[idx]
                kernel_val *= continuous_kernel(u, bandwidth[idx])
            
            for idx in discrete_idx:
                kernel_val *= discrete_kernel(X[i][idx], Y[j][idx], c=0.5)
            
            cross_kernel_matrix[i, j] = kernel_val
    
    return cross_kernel_matrix


def lscv(bandwidth, data, continuous_idx, discrete_idx):
    n = len(data)
    total_error = 0
    for i in range(n):
        leave_out_data = np.delete(data, i, axis=0)
        density = product_kernel(data[i], leave_out_data, bandwidth, continuous_idx, discrete_idx)
        total_error += density ** 2
    return total_error / n - 2 * np.mean(density)

def select_bandwidth(data, continuous_idx, discrete_idx):
    initial_bandwidth = 1.06 * np.std(data, axis=0) * len(data) ** (-1/5)
    result = minimize(lscv, initial_bandwidth, args=(data, continuous_idx, discrete_idx), method='L-BFGS-B')
    return result.x

def compute_test_statistic(X, Y, bandwidth, continuous_idx, discrete_idx):
    n1, n2 = len(X), len(Y)
    KXX = compute_kernel_matrix(X, bandwidth, continuous_idx, discrete_idx)
    KYY = compute_kernel_matrix(Y, bandwidth, continuous_idx, discrete_idx)
    KXY = compute_cross_kernel_matrix(X, Y, bandwidth, continuous_idx, discrete_idx)
    
    I_X = np.sum(KXX) / (n1 * (n1 - 1))
    I_Y = np.sum(KYY) / (n2 * (n2 - 1))
    I_XY = np.sum(KXY) / (n1 * n2)
    I = I_X + I_Y - 2 * I_XY
    
    Omega = 2 * n1 * n2 * np.prod(bandwidth) * (
        (np.sum(KXX**2) / (n1**2 * (n1 - 1)**2)) +
        (np.sum(KYY**2) / (n2**2 * (n2 - 1)**2)) +
        (np.sum(KXY**2) / (n1**2 * n2**2))
    )
    
    Tn = np.sqrt(n1 * n2 * np.prod(bandwidth)) * I / np.sqrt(Omega)
    return Tn

def bootstrap_test(X, Y, bandwidth, continuous_idx, discrete_idx, n_boot):
    combined_data = np.vstack((X, Y))
    Tn_observed = compute_test_statistic(X, Y, bandwidth, continuous_idx, discrete_idx)
    
    Tn_bootstrap = []
    for _ in range(n_boot):
        X_boot = combined_data[np.random.choice(len(combined_data), size=len(X), replace=True)]
        Y_boot = combined_data[np.random.choice(len(combined_data), size=len(Y), replace=True)]
        Tn_bootstrap.append(compute_test_statistic(X_boot, Y_boot, bandwidth, continuous_idx, discrete_idx))
    
    # p_value = np.mean(np.array(Tn_bootstrap) >= Tn_observed)
    p_value=(np.sum(np.array(Tn_bootstrap) >= Tn_observed) + 1) / (n_boot + 1)
    return p_value, Tn_observed

def li_test(X, Y, continuous_idx, discrete_idx, alpha=0.05, n_boot=2000):
    bandwidth = select_bandwidth(np.vstack((X, Y)), continuous_idx, discrete_idx)
    p_value, Tn_observed = bootstrap_test(X, Y, bandwidth, continuous_idx, discrete_idx, n_boot=n_boot)
    h = int(p_value <= alpha)
    return p_value, h, Tn_observed