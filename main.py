import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# 1. SIMULATION DES DONNEES
# ============================================================

def main_first_case():
    n = 1000

    alpha = 4
    beta = 1
    lambda_value = 10

    rng = np.random.default_rng(10)

    X_sample = rng.gamma(shape=alpha, scale=beta, size=n)
    C_sample = rng.exponential(scale=lambda_value, size=n)

    Z_sample = np.minimum(X_sample, C_sample)
    delta = (X_sample <= C_sample).astype(int)

    return Z_sample, delta


def main_second_case():
    n = 1000

    alpha_u = 2
    alpha_v = 16
    beta_u = 1/2
    beta_v = 1/4
    lambda_value = 4

    rng = np.random.default_rng(10)

    U = rng.gamma(shape=alpha_u, scale=beta_u, size=n)
    V = rng.gamma(shape=alpha_v, scale=beta_v, size=n)

    X_sample = 0.4 * U + 0.6 * V

    C_sample = rng.exponential(scale=lambda_value, size=n)

    Z_sample = np.minimum(X_sample, C_sample)
    delta = (X_sample <= C_sample).astype(int)

    return Z_sample, delta


# ============================================================
# 2. BASE TRIGONOMETRIQUE
# ============================================================

def base_trigo(D, a, b, x):

    x = np.asarray(x)
    phi = np.zeros((D, x.size))

    length = b - a

    phi[0] = 1 / np.sqrt(length)

    for j in range(1, D):

        k = (j + 1) // 2
        frequency = 2 * np.pi * k / length

        if j % 2 == 1:
            phi[j] = np.sqrt(2 / length) * np.cos(frequency * (x - a))
        else:
            phi[j] = np.sqrt(2 / length) * np.sin(frequency * (x - a))

    mask = (x >= a) & (x <= b)
    phi[:, ~mask] = 0

    return phi


# ============================================================
# 3. BASE HISTOGRAMME
# ============================================================

def base_histogramme(D, a, b, x):

    x = np.asarray(x)

    phi = np.zeros((D, x.size))

    h = (b - a) / D

    indices = np.floor((x - a) / h).astype(int)

    indices[x == b] = D - 1

    valid = (x >= a) & (x <= b)

    columns = np.where(valid)[0]

    phi[indices[valid], columns] = 1 / np.sqrt(h)

    return phi


# ============================================================
# 4. ESTIMATEUR PAR PROJECTION
# ============================================================

def projection_in_bases(Z, delta, D, a, b, base_function):

    phi_Z = base_function(D, a, b, Z)
    coefficients = (phi_Z @ delta) / len(Z)

    return coefficients


def evaluate_projection(x, coefficients, D, a, b, base_function):

    phi_x = base_function(D, a, b, x)

    psi = coefficients @ phi_x

    return psi


# ============================================================
# 5. AFFICHAGE DES PROJECTIONS
# ============================================================

def plot_projections(Z, delta, case_name):

    a = 0
    b = 10

    D_trigo = [3, 5, 7, 9]
    D_hist = [10, 15]

    x = np.linspace(a, b, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --------------------------------------------------------
    # Base trigonométrique
    # --------------------------------------------------------

    ax = axes[0]

    for D in D_trigo:

        coefficients = projection_in_bases(
            Z, delta, D, a, b, base_trigo
        )

        psi = evaluate_projection(
            x, coefficients, D, a, b, base_trigo
        )

        ax.plot(
            x,
            psi,
            label=f"D = {D}"
        )

    ax.set_title("Base trigonométrique")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\widehat{\psi}_D(x)$")
    ax.legend()
    ax.grid(alpha=0.3)


    # --------------------------------------------------------
    # Base histogramme
    # --------------------------------------------------------

    ax = axes[1]

    for D in D_hist:

        coefficients = projection_in_bases(
            Z, delta, D, a, b, base_histogramme
        )

        # Valeurs au centre de chaque intervalle
        h = (b - a) / D
        edges = np.linspace(a, b, D + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        psi = evaluate_projection(
            centers, coefficients, D, a, b,
            base_histogramme
        )

        ax.stairs(
            psi,
            edges,
            label=f"{D} cases"
        )

    ax.set_title("Base histogramme")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\widehat{\psi}_D(x)$")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Estimateur par projection - {case_name}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()



def plot_distributions(Z_first, Z_second, delta_first, delta_second):
    observed_first = np.mean(delta_first)
    observed_second = np.mean(delta_second)

    print("Premier cas :")
    print(f"Proportion observée : {observed_first:.2%}") # 69.80%
    print(f"Taux de censure : {1-observed_first:.2%}") # 30.20%

    print("\nDeuxième cas :")
    print(f"Proportion observée : {observed_second:.2%}") # 52.60%
    print(f"Taux de censure : {1-observed_second:.2%}") # 47.40%

    plt.figure(figsize=(10, 6))

    plt.hist(
        Z_first,
        bins=40,
        density=True,
        alpha=0.5,
        label="Premier cas : Gamma(4,1)"
    )

    plt.hist(
        Z_second,
        bins=40,
        density=True,
        alpha=0.5,
        label="Deuxième cas : Mélange de Gamma"
    )

    plt.xlabel("x")
    plt.ylabel("Densité")
    plt.title("Distributions empiriques des observations censurées")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def gamma_pdf(x, alpha, beta):
    return (
        x**(alpha-1) * np.exp(-x/beta)
        / (math.gamma(alpha) * beta**alpha)
    )


def exponential_pdf(x, lambda_value):
    return (1/lambda_value) * np.exp(-x/lambda_value)


def exponential_cdf(x, lambda_value):
    return 1 - np.exp(-x/lambda_value)


def gamma_cdf(x, alpha, beta, dx=0.001):
    grid = np.arange(0, np.max(x) + dx, dx)

    density = gamma_pdf(grid, alpha, beta)

    cdf = np.zeros_like(grid)
    cdf[1:] = np.cumsum(
        (density[:-1] + density[1:]) * dx / 2
    )

    return np.interp(x, grid, cdf)

def true_functions(alpha, beta, lambda_value, x_end, x_begin=0, linespace=1000):

    x = np.linspace(x_begin, x_end, linespace)

    f_X = gamma_pdf(x, alpha, beta)
    F_C = exponential_cdf(x, lambda_value)

    return f_X, F_C

def main():

    Z_first, delta_first = main_first_case()
    Z_second, delta_second = main_second_case()

    # plot_distributions(Z_first, Z_second, delta_first, delta_second)

    # # Projections du premier cas
    # plot_projections(Z_first, delta_first, "Gamma(4,1)")

    # # Projections du deuxième cas
    # plot_projections(Z_second, delta_second, "Mélange de Gamma")

    alpha_first = 4
    beta_first = 1
    lambda_first = 10

    f_first, C_first = true_functions(alpha_first, beta_first, lambda_first, 12)
    psi_first = f_first*C_first

    alpha_u = 2
    alpha_v = 16
    beta_u = 1/2
    beta_v = 1/4
    lambda_value = 4

    f_u, C_u = true_functions(alpha_u, beta_u, lambda_value, 12)
    f_v, C_v = true_functions(alpha_v, beta_v, lambda_value, 12)

    psi_second = (0.4*(f_u*C_u)+0.6*(f_v*C_v))

    plt.plot(psi_second)
    plt.show()





if __name__ == "__main__":
    # \psi = f*S_C
    main()