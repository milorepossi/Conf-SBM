import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

def ising_mle(samples, verbose=True, chunk_size=1 << 20,
              save_path="Exact_mech"):
    N, n = samples.shape
    n_configs = 1 << n
    triu_idx = np.triu_indices(n, k=1)
    n_chunks = (n_configs + chunk_size - 1) // chunk_size

    def unpack(params):
        h = params[:n]
        J = np.zeros((n, n))
        J[triu_idx] = params[n:]
        J += J.T
        return h, J

    def configs_chunk(start, end):
        idx = np.arange(start, end, dtype=np.int32)
        bits = (idx[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1
        return bits.astype(np.float32) * 2 - 1

    mu_data = samples.mean(0)
    C_data  = (samples.T @ samples) / N

    eval_bar  = tqdm(total=None, desc="Evaluations", unit=" evals",
                     position=0, disable=not verbose)
    chunk_bar = tqdm(total=None, desc="  Chunks   ", unit=" chunks",
                     position=1, disable=not verbose)      # accumulates forever

    def neg_loglik_and_grad(params):
        h, J = unpack(params)

        E_data    = samples @ h + (samples @ J * samples).sum(1) / 2
        data_term = E_data.mean()

        global_max = -np.inf
        chunk_maxes, chunk_sums = [], []
        acc_mu = np.zeros(n)
        acc_C  = np.zeros((n, n))

        for start in range(0, n_configs, chunk_size):
            end = min(start + chunk_size, n_configs)
            cfg = configs_chunk(start, end)
            E   = cfg @ h + (cfg @ J * cfg).sum(1) / 2

            cmax = E.max()
            w    = np.exp(E - cmax)
            chunk_maxes.append(cmax)
            chunk_sums.append(w.sum())
            global_max = max(global_max, cmax)

            scale   = np.exp(cmax - global_max)
            acc_mu += (w @ cfg) * scale
            acc_C  += (cfg.T * w) @ cfg * scale

            chunk_bar.update(1)

        log_Z    = np.log(sum(s * np.exp(m - global_max)
                              for s, m in zip(chunk_sums, chunk_maxes))) + global_max
        Z_unnorm = np.exp(log_Z - global_max)

        mu_model = acc_mu / Z_unnorm
        C_model  = acc_C  / Z_unnorm
        loss     = -(data_term - log_Z)

        grad_h = mu_model - mu_data
        grad_J = (C_model - C_data)[triu_idx]

        eval_bar.update(1)
        eval_bar.set_postfix(loss=f"{loss:.6f}")
        return loss, np.concatenate([grad_h, grad_J])

    res = minimize(
        neg_loglik_and_grad,
        np.zeros(n + n * (n - 1) // 2),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8},
    )

    chunk_bar.close()
    eval_bar.close()

    if verbose:
        status = "✓ converged" if res.success else "✗ did not converge"
        print(f"{status} in {eval_bar.n} evals | final loss: {res.fun:.6f}")

    h_opt, J_opt = unpack(res.x)

    np.save(f"{save_path}_h.npy", h_opt)
    np.save(f"{save_path}_J.npy", J_opt)
    if verbose:
        print(f"Saved → {save_path}_h.npy, {save_path}_J.npy")

    return h_opt, J_opt


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Exact Ising MLE on the minimal-model data.')
    parser.add_argument('--T', type=int, required=True, help='Temperature in K')
    parser.add_argument('--data_dir', type=str, default='data/MinimalModelData', help='Folder containing DHFR_<T>K')
    parser.add_argument('--out_dir', type=str, default='results/MinimalModelModels', help='Folder where DHFR_<T>K is created')
    args = parser.parse_args()

    samples = np.load(f"{args.data_dir}/DHFR_{args.T}K/discretized_array_min_{args.T}K.npy").astype(np.float64)
    assert set(np.unique(samples)) <= {-1.0, 1.0}, "samples must be in the ±1 convention"
    print(f"T = {args.T}K | samples shape = {samples.shape}", flush=True)

    out = f"{args.out_dir}/DHFR_{args.T}K"
    os.makedirs(out, exist_ok=True)
    ising_mle(samples, save_path=f"{out}/Exact")
