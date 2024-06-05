"""Bayesian regression using B-splines."""
import covvfit as cv
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


def create_model(
    xs: np.ndarray,
    ys: np.ndarray,
    n_coefs: int = 5,
    degree: int = 3,
) -> pm.Model:
    # Create the B-spline basis functions
    pts = cv.create_spline_matrix(xs, n_coefs=n_coefs)

    with pm.Model() as model:
        # Weights are the coefficients of the B-spline basis functions
        weights = pm.Normal("weights", 0, 10, size=(n_coefs, 1))

        # The function is a linear combination of the basis functions
        func = pm.Deterministic("func", (pts @ weights).ravel())

        # We add normal noise with unknown standard deviation
        sigma = pm.HalfCauchy("sigma", 1)
        pm.Normal("observed", mu=func, sigma=sigma, observed=ys)

    return model


def main() -> None:
    rng = np.random.default_rng(42)

    xs = np.linspace(0, 1, 151)
    ys_perfect = 2 * xs + 3 * np.sin(5 * xs)
    ys_obs = ys_perfect + 2 * rng.normal(size=xs.shape)

    model = create_model(xs=xs, ys=ys_obs, n_coefs=5)

    n_samples_per_chain = 500
    n_chains = 4
    thinning = 10
    with model:
        idata = pm.sample(tune=500, draws=n_samples_per_chain, chains=n_chains)
        thinned_idata = idata.sel(draw=slice(None, None, thinning))
        idata.extend(pm.sample_posterior_predictive(thinned_idata))

    fig, ax = plt.subplots()

    # Plot individual posterior predictive samples
    post_pred = idata.posterior_predictive["observed"]  # pyright: ignore

    for chain in range(n_chains):
        for sample in range(n_samples_per_chain // thinning):
            ys = post_pred[chain, sample]
            ax.plot(xs, ys, alpha=0.02, color="navy")

    # Plot mean of posterior predictive
    mean_predictive = post_pred.mean(axis=(0, 1))
    ax.plot(xs, mean_predictive, color="navy", label="Posterior predictive mean")

    # Plot "perfect data"
    ax.plot(xs, ys_perfect, color="maroon", label="Noiseless data")
    # Plot noisy data
    ax.scatter(xs, ys_obs, c="k", s=3, label="Observed data")

    ax.set_title("Bayesian regression using B-splines")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.legend(frameon=False)
    fig.savefig("splines_demonstration.pdf")


if __name__ == "__main__":
    main()
