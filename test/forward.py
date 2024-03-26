import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def simulate_forward(alpha, sigma, x0, k=1000):
    T = len(sigma)
    # xT = sp.stats.norm(alpha[-1]*x0, sigma[-1]).rvs(k)
    x = np.zeros([T, k])
    x[-1] = sp.stats.norm().rvs(k)
    for t in range(T-1, 0, -1):
        sigma_t_t_pre = sigma[t] - (alpha[t] / alpha[t-1])**2 * sigma[t-1]
        s_t = sigma_t_t_pre * sigma[t-1] / sigma[t]
        mu_t = alpha[t] / alpha[t-1] * sigma[t-1] / sigma[t] * x[t] + alpha[t-1] * sigma_t_t_pre / sigma[t] * x0
        # x[t-1] = sp.stats.norm(loc=mu_t, scale=np.sqrt(s_t)).rvs()
        delta_t = np.sqrt(12 * s_t)
        x[t-1] = sp.stats.uniform(loc=mu_t - delta_t / 2, scale=delta_t).rvs()

    # Estimate pdfs
    pdfs = [sp.stats.gaussian_kde(x_) for x_ in x]
    return pdfs


def visualize_gaussian(alpha, sigma, x0=0):
    T = len(sigma)
    # dists = [sp.stats.norm(a*x0, s) for a, s in zip(alpha, sigma)]
    dists = simulate_forward(alpha, sigma, x0)

    x_ = np.linspace(*sp.stats.norm.ppf([0.01, 0.99]), T)
    logpdf = np.array([d.logpdf(x_) for d in dists])
    # assume mean=0
    normpdf = np.array([d.pdf(x_) / d.pdf(0) for d in dists])

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    # Heatmap
    ax[0].imshow(logpdf, vmin=-1, vmax=1, cmap='plasma', aspect='auto', extent=(x_[0], x_[-1], 0, T-1))
    try:
        for q in [0.05, 0.2, 0.8, 0.95]:
            ql = ax[0].plot(
                np.array([d.ppf(q) for d in dists]), np.arange(T-1, -1, -1),
                'r--', alpha=0.5, label='5%/20%/80%/95% quantile'
            )
        ax[0].legend(handles=ql)
    except AttributeError:
        pass
    ax[1].set_title('log-pdfs')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    # 3D
    ax[1].remove()
    ax[1] = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(x_, np.arange(T))
    ax[1].plot_surface(X, Y, normpdf, cmap='plasma')
    ax[1].set_zlim(0, 1)
    ax[1].set_title('pdfs, normalized to range [0, 1]')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    plt.show()

    print()


if __name__ == '__main__':
    # sigma as in DDPM:
    T = 1000
    beta = np.linspace(0.0001, 0.02, T)
    sigma = 1 - np.cumprod(1 - beta)
    alpha = np.sqrt(1 - sigma)

    visualize_gaussian(alpha, sigma)
