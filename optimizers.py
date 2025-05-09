from scipy.optimize import minimize
import numpy as np

def min_var_opt_inv(ret_data,cov_estimator):
    T,N=ret_data.shape
    cov_mat=cov_estimator(ret_data)
    e=np.ones(N)
    x0 = np.ones(N) / N

    try:
        cov_inv = np.linalg.inv(cov_mat)
        w = cov_inv @ e / (e.T @ cov_inv @ e)
    except:
        print("Covariance matrix is singular or ill-conditioned. Using scipy.optimize.minimize for QP.")
        
        # Define the quadratic objective function: (1/2)*w.T @ cov_mat @ w.
        def objective(w):
            return 0.5 * np.dot(w, np.dot(cov_mat, w))
        
        # Constraint: the portfolio weights must sum to 1.
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # Initial guess: equal weights.
        
        res = minimize(objective, x0, method='SLSQP', constraints=constraints)
        if not res.success:
            print("Optimization failed:", res.message)
        w = res.x

    w = x0 if np.isnan(w).any() else w
    
    return w           



def min_var_opt_stable(ret_data,cov_estimator):
    T,N=ret_data.shape
    cov_mat=cov_estimator(ret_data)
    e=np.ones(N)
    x0 = np.ones(N) / N


    try:
        u = np.linalg.solve(cov_mat, e) # u is an auxiliary vector, where cov_mat*u=e (i.e., u=cov_mat^{-1}*e)

        w=u/(e.T @ u)

    except:
        print("Covariance matrix is singular or ill-conditioned. Using scipy.optimize.minimize for QP.")
        
        # Define the quadratic objective function: (1/2)*w.T @ cov_mat @ w.
        def objective(w):
            return 0.5 * np.dot(w, np.dot(cov_mat, w))
        
        # Constraint: the portfolio weights must sum to 1.
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # Initial guess: equal weights.
        
        res = minimize(objective, x0, method='SLSQP', constraints=constraints)
        if not res.success:
            print("Optimization failed:", res.message)
        w = res.x

    w = x0 if np.isnan(w).any() else w
    
    return w           