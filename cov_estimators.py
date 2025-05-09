import numpy as np
import pandas as pd

def SCM(ret_data, decay_function=None):
    """
    Compute (possibly weighted) sample covariance matrix.

    Parameters:
        ret_data (pd.DataFrame): Asset returns, shape (T, N)
        decay_function (callable): Function that returns a weight vector of length T
        
    Returns:
        pd.DataFrame: Covariance matrix, shape (N, N)
    """

    T, N = ret_data.shape

    if decay_function is not None:
        weights = decay_function(T)  # Should return an array of length T (normalized to 1)
    else:
        weights = np.full(T, 1/T)      # Equal weights if no decay function is provided

    # Center the returns
    weighted_mean = np.average(ret_data, axis=0, weights=weights)
    
    demeaned = ret_data - weighted_mean  # shape (T, N)

    # Compute weighted covariance matrix
    cov_matrix = (demeaned.T * weights) @ demeaned  # shape (N, N)
    
    return pd.DataFrame(cov_matrix, index=ret_data.columns, columns=ret_data.columns)


def LedoitWolf(ret_data: pd.DataFrame, alpha, decay_function=None) -> np.ndarray:
    """
    Vectorized implementation of the target covariance matrix F under the single-index (market) model
    for the Ledoit-Wolf estimator, incorporating a decay function to weight time observations.

    Parameters:
        ret_data: DataFrame of shape (T, N), where each column represents a stock's returns,
                  T is the number of time periods and N is the number of stocks.
        decay_function: A callable that takes an integer T and returns an array of weights of length T.
                        If None, equal weights are used.

    Returns:
        F: ndarray of shape (N, N), the covariance matrix under the single-index model, defined as:
           Diagonal: F_ii = β_i^2 * Var(m) + Var(ε_i)
           Off-diagonal: F_ij = β_i * β_j * Var(m)
    """
    T, N = ret_data.shape

    # 1. Compute the market return: using the equal-weight average of all stocks.
    #    (This is done on each time observation)
    market = ret_data.mean(axis=1)  # Series of length T

    # Use the decay function if provided, otherwise use equal weights.
    if decay_function is not None:
        weights = decay_function(T)  # Should return an array of length T (normalized to 1)
    else:
        weights = np.full(T, 1/T)      # Equal weights if no decay function is provided

    # 2. Compute the weighted mean of the market returns using decay weights.
    m_mean = np.average(market, weights=weights)
    # Compute the weighted variance of market returns.
    m_var = np.average((market - m_mean)**2, weights=weights)

    # 3. Compute each stock's weighted mean.
    asset_means = np.average(ret_data, axis=0, weights=weights)  # Array of length N

    # 4. Demean the stock returns.
    ret_data_demeaned = ret_data - asset_means

    # 5. Vectorized computation of each stock's beta:
    #    β_i = Cov(r_i, market) / Var(market)
    #    where Cov(r_i, market) = sum_{t=1}^T weights[t]*(r_{it} - E[r_i])*(market[t] - m_mean)
    cov_with_market = np.average(ret_data_demeaned.multiply(market - m_mean, axis=0), axis=0, weights=weights)
    beta = cov_with_market / m_var  # Series of length N

    # 6. Compute each stock's intercept (alpha):
    #    α_i = E[r_i] - β_i * m_mean
    alpha = asset_means - beta * m_mean

    # 7. Compute the predicted returns and residuals in a fully vectorized manner.
    #    For each stock i: predicted[t, i] = α_i + β_i * market[t]
    predicted = market.values.reshape(-1, 1) * beta.reshape(1, -1) + alpha.reshape(1, -1)
    residuals = ret_data.values - predicted
    # Compute the weighted variance of residuals (using the provided decay weights)
    resid_var = np.array([np.average(residuals[:, i]**2, weights=weights) for i in range(N)])

    # 8. Construct the target covariance matrix F.
    #    Use the outer product to compute inter-stock covariance terms.
    #    For i != j: F_ij = β_i * β_j * Var(m)
    #    For i == j: F_ii = β_i^2 * Var(m) + Var(ε_i)
    b = beta  # Convert beta Series to ndarray with shape (N,)
    F = m_var * np.outer(b, b) + np.diag(resid_var)

    F = pd.DataFrame(F, index=ret_data.columns, columns=ret_data.columns)

    resulting_cov=alpha*F+(1-alpha)*SCM(ret_data,decay_function=decay_function)

    return resulting_cov


def GerberCov(ret_data, c=0.5, decay_function=None):
    """
    Compute the Gerber covariance estimator as described in the referenced paper, 
    with the option to incorporate a decay function to weight the time observations.

    For a return matrix R with T time periods and N assets, let s be the weighted sample standard deviation 
    vector (length N) and set thresholds H = c * s (where c is a fraction such as 0.5).

    For each time t and asset pair (i, j), define the indicator:
        m_{ij}(t) = +1   if |r_t^i| > H_i, |r_t^j| > H_j and sign(r_t^i) == sign(r_t^j)
        m_{ij}(t) = -1   if |r_t^i| > H_i, |r_t^j| > H_j and sign(r_t^i) != sign(r_t^j)
        m_{ij}(t) =  0   otherwise.
    
    When a decay_function is provided, each observation at time t is weighted by w[t]. In that case,
    the weighted sample standard deviations and weighted summations are used:
    
        g_{ij} = (Σ_t w[t] · m_{ij}(t)) / (Σ_t w[t] · I{both i and j valid at time t}),
    
    where I{...} is the indicator function.
    
    Finally, the Gerber covariance matrix is defined as:
        Σ_Gerber = diag(s) * G * diag(s)
    
    Parameters:
        ret_data: DataFrame of asset returns with shape (T, N) (rows: time, columns: assets)
        c: float, threshold fraction (default 0.5)
        decay_function: callable or None. If provided, it should take an integer T and return a weight vector (length T)
                        normalized to sum to 1. If None, equal weights are used.
    
    Returns:
        DataFrame: Gerber covariance matrix (N x N), indexed and columned by asset names.
    """
    T, N = ret_data.shape

    # ---------------------
    # Obtain weights for time observations.
    # ---------------------
    if decay_function is not None:
        weights = decay_function(T)  # 1D array of length T
    else:
        weights = np.full(T, 1/T)
    weights = np.array(weights)  # Ensure numpy array

    # ---------------------
    # Compute weighted sample standard deviations for each asset.
    # ---------------------
    X = ret_data.values  # shape (T, N)
    # Compute weighted mean per asset.
    wm = np.average(X, axis=0, weights=weights)  # 1D array of length N
    # Compute weighted variance (note: weights sum to one).
    weighted_var = np.average((X - wm)**2, axis=0, weights=weights)
    s_values = np.sqrt(weighted_var)  # 1D array of length N
    # Wrap in a Series with asset names.
    s = pd.Series(s_values, index=ret_data.columns)
    
    # ---------------------
    # Define thresholds per asset: H_k = c * s_k.
    # ---------------------
    H = c * s

    # ---------------------
    # Compute the indicator m_{ij}(t)
    # ---------------------
    # R: returns array (T, N)
    R = X  # shape (T, N)
    # Broadcast thresholds: shape (T, N)
    H_mat = np.tile(H.values, (T, 1))
    # valid: True if |r_t^i| > H_i for each asset i at time t.
    valid = np.abs(R) > H_mat  # shape (T, N)
    # sign_R: sign of returns.
    sign_R = np.sign(R)  # shape (T, N)
    
    # For each time t compute an N×N matrix: valid_pair[t,i,j] True if both asset i and j are valid.
    valid_pair = valid[:, :, None] & valid[:, None, :]  # shape (T, N, N)
    # For each time t compute equality of signs for each pair (i, j).
    eq_sign = (sign_R[:, :, None] == sign_R[:, None, :])  # shape (T, N, N)
    eq_sign_int = eq_sign.astype(int)  # 1 if equal, 0 if not.
    
    # If both valid, set m_{ij}(t)=1 if signs equal, -1 if not; otherwise 0.
    m_t = valid_pair.astype(int) * (2 * eq_sign_int - 1)  # shape (T, N, N)
    
    # ---------------------
    # Weighted summation over time.
    # ---------------------
    # Numerator: weighted sum of m_{ij}(t)
    M_sum = (weights[:, None, None] * m_t).sum(axis=0)  # shape (N, N)
    # Denominator: weighted sum of valid_indicator
    count_matrix = (weights[:, None, None] * valid_pair.astype(float)).sum(axis=0)  # shape (N, N)
    
    # Gerber statistic: g_{ij} = M_sum / count (if count>0, else 0)
    G = np.zeros((N, N))
    mask = count_matrix > 0
    G[mask] = M_sum[mask] / count_matrix[mask]
    
    # ---------------------
    # Form the Gerber covariance matrix: diag(s) * G * diag(s)
    # ---------------------
    s_arr = s.values.reshape(-1, 1)  # shape (N, 1)
    Sigma_Gerber = s_arr * G * s_arr.T
    return pd.DataFrame(Sigma_Gerber, index=ret_data.columns, columns=ret_data.columns)