"""utilities to fit frequentist models"""

import pymc as pm

import numpy as np


__all__ = [
    "create_model_fixed",
    "softmax",
    "softmax_1",
    "fitted_values",
    "pred_values",
    "compute_overdispersion",
    "make_jacobian",
    "project_se",
    "make_ratediff_confints",
    "make_fitness_confints",
]


def create_model_fixed(
    ts_lst,
    ys_lst,
    n: float = 1.0,
    coords: dict | None = None,
) -> pm.Model:
    """Creates a fixed effect model with varying intercepts and one rate vector."""
    if n < 0:
        raise ValueError("n must be positive")

    if coords is None:
        coords = {
            "cities": [],
            "variants": [],
        }

    with pm.Model(coords=coords) as model:
        midpoint_var = pm.Normal(
            "midpoint", mu=0.0, sigma=1500.0, dims=["cities", "variants"]
        )
        rate_var = pm.Gamma("rate", mu=0.15, sigma=0.1, dims="variants")

        # Kaan's trick to avoid overflows
        def softmax_1(x, rates, midpoints):
            E = rates[:, None] * x + midpoints[:, None]
            E_max = E.max(axis=0)
            un_norm = pm.math.exp(E - E_max)
            return un_norm / (pm.math.exp(-E_max) + pm.math.sum(un_norm, axis=0))

        ys_smooth = [
            softmax_1(ts_lst[i], rate_var, midpoint_var[i, :])
            for i, city in enumerate(coords["cities"])
        ]

        # make Multinom/n likelihood
        def log_likelihood(y, p, n):
            # return n*pm.math.sum(y * pm.math.log(p), axis=0) + n*(1-pm.math.sum(y, axis=0))*pm.math.log(1-pm.math.sum(p, axis=0))
            return n * pm.math.sum(y * pm.math.log(p), axis=0)

        [
            pm.DensityDist(
                f"ys_noisy_{city}",
                ys_smooth[i],
                n,
                logp=log_likelihood,
                observed=ys_lst[i],
            )
            for i, city in enumerate(coords["cities"])
        ]

    return model


def softmax(E):
    """softmax with the trick to avoid overflows"""
    E_max = E.max(axis=0)
    un_norm = np.exp(E - E_max)
    return un_norm / (np.exp(-E_max) + np.sum(un_norm, axis=0))


def softmax_1(x, rates, midpoints):
    """softmax with the trick to avoid overflows, assuming an additional class with rate fixed to 0"""
    E = rates[:, None] * x + midpoints[:, None]
    E_max = E.max(axis=0)
    un_norm = np.exp(E - E_max)
    return un_norm / (np.exp(-E_max) + np.sum(un_norm, axis=0))


def fitted_values(ts_lst, model_map_fixed, cities):
    """function to make the fitted values of a model"""
    y_fit_lst = [
        softmax_1(ts_lst[i], model_map_fixed["rate"], model_map_fixed["midpoint"][i, :])
        for i, city in enumerate(cities)
    ]

    return y_fit_lst


def pred_values(ts_lst, model_map_fixed, cities, horizon=60):
    """function to make the fitted values of a model"""
    ts_pred_lst = [np.arange(horizon) + tt.max() + 1 for tt in ts_lst]
    y_pred_lst = [
        softmax_1(
            ts_pred_lst[i], model_map_fixed["rate"], model_map_fixed["midpoint"][i, :]
        )
        for i, city in enumerate(cities)
    ]

    return ts_pred_lst, y_pred_lst


def compute_overdispersion(ys_lst, y_fit_lst, cities):
    """compute overdispersion from a quasimultinom model"""
    pearson_r_lst = [
        (y_fit_lst[k] - ys_lst[k]) ** 2 / (y_fit_lst[k] * (1 - y_fit_lst[k]))
        for k, city in enumerate(cities)
    ]
    overdisp_list2 = [
        i.sum() / ((i.shape[0]) * (i.shape[1] - 1)) for i in pearson_r_lst
    ]
    overdisp_fixed = np.concatenate(pearson_r_lst, axis=1).sum() / np.sum(
        [((i.shape[0]) * (i.shape[1] - 1)) for i in pearson_r_lst]
    )

    return pearson_r_lst, overdisp_list2, overdisp_fixed


def make_jacobian(rate, midpoint, t):
    """function to make the jacobian of logit(y), with the +1"""
    linear_predict = rate * t + midpoint
    tiled_array = np.tile(linear_predict, (linear_predict.shape[0], 1))
    np.fill_diagonal(tiled_array, 0)
    softmax_array = np.apply_along_axis(softmax, 1, tiled_array) * (-1)
    np.fill_diagonal(softmax_array, 1)
    array_out = np.concatenate([softmax_array, softmax_array], axis=1)
    return array_out


def project_se(rate, midpoint, t, model_hessian_inv, overdisp=1.0):
    """function to project the standard errors on the logit(y) scale"""
    jacobian = make_jacobian(rate, midpoint, t)
    return np.sqrt(np.diag(jacobian.dot(model_hessian_inv).dot(jacobian.T))) * overdisp


def make_ratediff_confints(t_rate, model_hessian_fixed, overdisp_fixed=1.0, g=7.0):
    """project the standard errors on the rate difference scale, compute wald confints, transform to fitness scale"""
    p_variants = t_rate.shape[0]
    t_hess_inv = np.linalg.inv(model_hessian_fixed)[-p_variants:, -p_variants:]

    rate_diff = np.array([[j - i for i in t_rate] for j in t_rate])
    fitness_diff = np.exp(rate_diff * g) - 1

    rate_diff_se = np.sqrt(
        np.array(
            [
                [
                    t_hess_inv[[i, j], :][:, [i, j]].diagonal().sum()
                    - np.fliplr(t_hess_inv[[i, j], :][:, [i, j]]).diagonal().sum()
                    for i, _ in enumerate(t_rate)
                ]
                for j, _ in enumerate(t_rate)
            ]
        )
    )
    rate_diff_se = overdisp_fixed * rate_diff_se
    rate_diff_lower = rate_diff - 1.96 * rate_diff_se
    rate_diff_upper = rate_diff + 1.96 * rate_diff_se
    fitness_diff_lower = np.exp(rate_diff_lower * g) - 1
    fitness_diff_upper = np.exp(rate_diff_upper * g) - 1

    return fitness_diff, rate_diff_se, fitness_diff_lower, fitness_diff_upper


def make_fitness_confints(t_rate, model_hessian_fixed, overdisp_fixed=1.0, g=7.0):
    """project the standard errors on the fitness scale, compute wald confints"""
    p_variants = t_rate.shape[0]
    t_hess_inv = np.linalg.inv(model_hessian_fixed)[-p_variants:, -p_variants:]

    rate_diff = np.array([[j - i for i in t_rate] for j in t_rate])
    fitness_diff = np.exp(rate_diff * g) - 1

    fitness_diff_se = []
    for i, r1 in enumerate(t_rate):
        t_lst = []
        for j, r2 in enumerate(t_rate):
            t_jacobian = np.array(
                [g * np.exp(g * (r2 - r1)), -g * np.exp(g * (r2 - r1))]
            )
            se = np.sqrt(
                t_jacobian.dot(t_hess_inv[[i, j], :][:, [i, j]]).dot(t_jacobian.T)
            )
            t_lst.append(se)
        fitness_diff_se.append(t_lst)
    fitness_diff_se = np.array(fitness_diff_se)

    fitness_diff_se = overdisp_fixed * fitness_diff_se
    fitness_diff_lower = fitness_diff - 1.96 * fitness_diff_se
    fitness_diff_upper = fitness_diff + 1.96 * fitness_diff_se

    return fitness_diff, fitness_diff_se, fitness_diff_lower, fitness_diff_upper
