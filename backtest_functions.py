
import os
import time
import pickle
import numpy as np
import pandas as pd
from decay_functions import *
from cov_estimators import *


def backtest_single(ret_data, optimizer, lbw):

    returns = []
    weights = []
    dates = []
    for row_i in range(lbw, len(ret_data)):
        train_window = ret_data.iloc[row_i - lbw:row_i, :]
        w = optimizer(train_window)
        current_return = ret_data.iloc[row_i, :].values
        port_return = np.dot(w, current_return)
        returns.append(port_return)
        weights.append(w)
        dates.append(ret_data.index[row_i])
    returns_df = pd.DataFrame({'Return': returns}, index=dates)
    weights_df = pd.DataFrame(weights, index=dates, columns=ret_data.columns)
    return returns_df, weights_df

def run_backtests(
    ret_data: pd.DataFrame,
    optimizer,
    lbw_list,
    decay_settings,
    cov_settings,
    var_cache_file,
    returns_cache_file,
    weights_cache_file,
    output_path='output/',
    autosave_interval=30*60  # seconds
):
    """
    Loop over all parameter combinations, backtest and cache results. 
    Every autosave_interval seconds, automatically overwrite the cache 
    files to guard against mid-run failures. Progress is printed after each iteration.

    Parameters:
    -----------
    ret_data : pd.DataFrame
        Historical returns dataframe (index = dates, columns = assets).
    optimizer : callable
        A function of signature optimizer(ret_data, cov_estimator=...) -> weight_vector.
    lbw_list : list[int]
        Look-back window sizes to test.
    decay_settings : dict
        Mapping decay_name -> {"func": decay_function, "params": [...]}
    cov_settings : dict
        Mapping cov_name -> {"func": cov_function, "params": [...]}
    var_cache_file : str
        Filename for pickled dict of variances.
    returns_cache_file : str
        Filename for pickled dict of return series.
    weights_cache_file : str
        Filename for pickled dict of weight series.
    output_path : str
        Directory in which to store cache files.
    autosave_interval : int
        How often (in seconds) to auto-save caches.

    Returns:
    --------
    var_df : pd.DataFrame
        DataFrame of all parameter combinations and their strategy variances.
    returns_cache : dict
        Mapping parameter-key -> pd.DataFrame of aligned return series.
    weights_cache : dict
        Mapping parameter-key -> pd.DataFrame of aligned weight series.
    """

    # 1) ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 2) full paths for the three cache files
    var_cache_path     = os.path.join(output_path, var_cache_file)
    returns_cache_path = os.path.join(output_path, returns_cache_file)
    weights_cache_path = os.path.join(output_path, weights_cache_file)

    # 3) load or initialize caches
    if os.path.exists(var_cache_path):
        with open(var_cache_path, "rb") as f:
            var_cache = pickle.load(f)
    else:
        var_cache = {}

    if os.path.exists(returns_cache_path):
        with open(returns_cache_path, "rb") as f:
            returns_cache = pickle.load(f)
    else:
        returns_cache = {}

    if os.path.exists(weights_cache_path):
        with open(weights_cache_path, "rb") as f:
            weights_cache = pickle.load(f)
    else:
        weights_cache = {}

    # 4) determine the common start date so all series are aligned
    common_lbw = max(lbw_list)
    common_start = ret_data.index[common_lbw]

    # 5) precompute total number of iterations for progress tracking
    total = 0
    decay_keys = list(decay_settings.keys())
    for _ in lbw_list:
        for cov_name, cov_info in cov_settings.items():
            n_cov = len(cov_info["params"])
            for d in decay_keys:
                n_dec = len(decay_settings[d]["params"])
                total += n_cov * n_dec

    counter = 0
    results = []
    last_save = time.time()

    def is_cached(key, cache):
        return key in cache

    # 6) main nested loops
    for lbw in lbw_list:
        for cov_name, cov_info in cov_settings.items():
            for cov_param in cov_info["params"]:
                for decay_type in decay_keys:
                    for decay_param in decay_settings[decay_type]["params"]:
                        key = (lbw, decay_type, str(decay_param), cov_name, str(cov_param))

                        # if already in all three caches, skip computation
                        if (is_cached(key, var_cache) 
                            and is_cached(key, returns_cache)
                            and is_cached(key, weights_cache)):
                            variance = var_cache[key]
                            print(f"[Cached] {key}")
                        else:
                            # select decay function
                            if decay_type == "EW":
                                chosen_decay = EW_decay
                            elif decay_type == "Exp":
                                chosen_decay = lambda T, base=decay_param: Exp_decay(T, base)
                            elif decay_type == "Lin":
                                chosen_decay = lambda T, ew=decay_param: Lin_decay(T, ew)
                            else:
                                chosen_decay = None

                            # build the covariance estimator
                            if cov_name == "SCM":
                                cov_est = lambda data: SCM(data, decay_function=chosen_decay)
                            elif cov_name == "LedoitWolf":
                                cov_est = lambda data, a=cov_param: LedoitWolf(data, a, decay_function=chosen_decay)
                            elif cov_name == "GerberCov":
                                cov_est = lambda data, c=cov_param: GerberCov(data, c=c, decay_function=chosen_decay)
                            else:
                                raise ValueError(f"Unknown cov_estimator {cov_name}")

                            # wrap into the optimizer interface
                            run_opt = lambda data: optimizer(data, cov_estimator=cov_est)

                            # run backtest: returns_df, weights_df
                            returns_df, weights_df = backtest_single(ret_data, run_opt, lbw)

                            # align start date
                            returns_df = returns_df[returns_df.index >= common_start]
                            weights_df = weights_df[weights_df.index >= common_start]

                            # compute variance
                            variance = np.var(returns_df["Return"].values)

                            # store into caches
                            var_cache[key]     = variance
                            returns_cache[key] = returns_df
                            weights_cache[key] = weights_df

                            print(f"[Computed] {key} -> var={variance:.6f}")

                        # record result
                        results.append({
                            "lbw": lbw,
                            "decay_function": decay_type,
                            "decay_parameter": decay_param,
                            "cov_estimator": cov_name,
                            "cov_param": cov_param,
                            "strategy_variance": variance
                        })

                        # update progress
                        counter += 1
                        pct = counter/total*100
                        print(f"Progress: {counter}/{total} ({pct:.1f}%)")

                        # autosave caches if enough time has passed
                        now = time.time()
                        if now - last_save > autosave_interval:
                            with open(var_cache_path,     "wb") as f: pickle.dump(var_cache, f)
                            with open(returns_cache_path, "wb") as f: pickle.dump(returns_cache, f)
                            with open(weights_cache_path, "wb") as f: pickle.dump(weights_cache, f)
                            print(f"Autosaved at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            last_save = now

    # final save
    with open(var_cache_path,     "wb") as f: pickle.dump(var_cache, f)
    with open(returns_cache_path, "wb") as f: pickle.dump(returns_cache, f)
    with open(weights_cache_path, "wb") as f: pickle.dump(weights_cache, f)

    # return all three
    var_df = pd.DataFrame(results)
    return var_df, returns_cache, weights_cache

