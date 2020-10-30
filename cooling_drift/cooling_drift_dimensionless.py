"""
This tests how different ways of drifting density and energy work.

"""

import numpy as np
from math import sqrt

from numba import jit
from scipy.optimize import newton
from math import pow

from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use("../../mnras_durham.mplstyle")


kernel_gamma = 1.936_492
kernel_constant = 21 / (2 * 3.14159)
gas_gamma = 5.0 / 3.0
one_over_gamma = 1.0 / gas_gamma
k_B = 1.381e-16  # cgs
eta = 1.2


def generate_cube(num_on_side, side_length=1.0):
    """
    Generates a cube of particles
    """

    values = np.linspace(0.0, side_length, num_on_side + 1)[:-1]

    positions = np.empty((num_on_side ** 3, 3), dtype=np.float64)

    for x in range(num_on_side):
        for y in range(num_on_side):
            for z in range(num_on_side):
                index = x * num_on_side + y * num_on_side ** 2 + z

                positions[index, 0] = values[x]
                positions[index, 1] = values[y]
                positions[index, 2] = values[z]

    return positions


def generate_two_cube(num_on_side, side_length=1.0):
    """
    Generates two cubes of particles overlaid to make a BCC lattice.
    """

    cube = generate_cube(num_on_side // 2, side_length)

    mips = side_length / num_on_side

    positions = np.concatenate([cube, cube + mips])

    return positions


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, np.float64], H: Union[float, np.float64]):
    """
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).
    Give it a radius and a kernel width (i.e. not a smoothing length, but the
    radius of compact support) and it returns the contribution to the
    density.
    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = np.float64(0.0)

    if ratio < 1.0:
        ratio_2 = ratio * ratio
        ratio_3 = ratio_2 * ratio

        if ratio < 0.5:
            kernel += 3.0 * ratio_3 - 3.0 * ratio_2 + 0.5

        else:
            kernel += -1.0 * ratio_3 + 3.0 * ratio_2 - 3.0 * ratio + 1.0

        kernel *= kernel_constant * inverse_H * inverse_H * inverse_H

    return kernel


@jit(nopython=True, fastmath=True)
def calculate_pressure(
    position: np.float64,
    smoothing_length: np.float64,
    positions: np.float64,
    energies: np.float64,
    particle_mass: np.float64,
) -> np.float64:

    kernel_width = smoothing_length * kernel_gamma

    P = 0.0

    # This hack is required to numba me.
    for i in range(len(energies)):
        pos = positions[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])
        this_energy = energies[i]

        if r <= kernel_width:
            P += this_energy * kernel(r, kernel_width)

    return P * (gas_gamma - 1.0) * particle_mass


@jit(nopython=True, fastmath=True)
def calculate_pressure_drift_equation(
    position: np.float64,
    smoothing_length: np.float64,
    positions: np.float64,
    d_energies_dt: np.float64,
    particle_mass: np.float64,
) -> np.float64:

    kernel_width = smoothing_length * kernel_gamma

    dP_dt = 0.0

    # This hack is required to numba me.
    for i in range(len(d_energies_dt)):
        pos = positions[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])
        this_dudt = d_energies_dt[i]

        if r <= kernel_width:
            dP_dt += this_dudt * kernel(r, kernel_width)

    return dP_dt * (gas_gamma - 1.0) * particle_mass


@jit(nopython=True, fastmath=True)
def calculate_pressure_without_specific_index(
    position: np.float64,
    smoothing_length: np.float64,
    positions: np.float64,
    energies: np.float64,
    particle_mass: np.float64,
    bad_index: int,
) -> np.float64:
    kernel_width = smoothing_length * kernel_gamma

    P = 0.0

    # This hack is required to numba me.
    for i in range(len(energies)):
        if i == bad_index:
            continue

        pos = positions[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])

        if r <= kernel_width:
            P += energies[i] * kernel(r, kernel_width)

    return P * (gas_gamma - 1.0) * particle_mass


def calculate_number_density(position, smoothing_length, positions):
    kernel_width = smoothing_length * kernel_gamma

    N = 0.0

    for pos in positions:
        dx = pos - position
        r = sqrt(np.sum(dx * dx))

        if r <= kernel_width:
            N += kernel(r, kernel_width)

    return N


def find_h(position, positions):
    """
    Find the smoothing length.
    """

    # Start off with a first guess
    initial_guess = 3.0 / positions.size ** (1 / 3)

    def to_root(h):
        h_from_number_density = eta * (
            1.0 / (calculate_number_density(position, h, positions))
        ) ** (1 / 3)

        return h - h_from_number_density

    return newton(to_root, initial_guess)


if __name__ == "__main__":
    # Internal units: cgs

    num_along_side = 32
    mips = 1.0 / num_along_side

    # First, generate positions:
    positions = generate_two_cube(num_along_side)

    # Now find the one closest to the centre (this will be the hot particle):
    sorted_distance = np.argsort(np.sum((positions - 0.5) ** 2, axis=1))
    arg_our_favourite = sorted_distance[0]
    arg_nearest_neighbour = sorted_distance[1]
    # Now actually transform to cm
    position = positions[arg_our_favourite]

    print(f"Central particle at {position}.")

    # Now have a gander at its smoothing length
    smoothing_length = find_h(position, positions)

    print(f"It has h={smoothing_length:e}, MIPS={mips:e}")

    # Calculate number of particles here
    volume = (4.0 * np.pi / 3.0) * (smoothing_length * kernel_gamma) ** 3
    number_density = calculate_number_density(position, smoothing_length, positions)

    print(f"This gives it N={number_density * volume}")
    print(f"Ratio: {mips**(-3) / number_density:e}, {mips**(3) * number_density:e}")

    # Now need to set everyone's u
    background_u = 10.0
    hot_u = 1000.0
    print(f"Background u: {background_u:e}, hot u: {hot_u:e}")
    background_dt = np.float64(
        2
        * kernel_gamma
        * smoothing_length
        / np.sqrt((gas_gamma - 1.0) * gas_gamma * background_u)
    )
    hot_dt = np.float64(
        2
        * kernel_gamma
        * smoothing_length
        / np.sqrt((gas_gamma - 1.0) * gas_gamma * hot_u)
    )
    cooling_rate = (background_u - hot_u) / hot_dt
    print(f"u ratio: {background_u / hot_u}")
    print(f"ratio of time-steps: {background_dt / hot_dt}")

    # Set these values
    internal_energy = np.ones(len(positions)) * background_u
    internal_energy_once_cooled = internal_energy.copy()
    internal_energy[arg_our_favourite] = hot_u
    internal_energy_dt = np.zeros_like(internal_energy)
    internal_energy_dt[arg_our_favourite] = cooling_rate

    dP_dt_neighbour = calculate_pressure_drift_equation(
        positions[arg_nearest_neighbour],
        smoothing_length,
        positions,
        internal_energy_dt,
        1.0,
    )

    # Calculate pressure field
    print("First pressure loop")
    pressures = np.array(
        [
            calculate_pressure(pos, smoothing_length, positions, internal_energy, 1.0)
            for pos in positions
        ]
    )
    print("Done")

    plt.scatter(
        positions[np.argsort(pressures), 0],
        positions[np.argsort(pressures), 1],
        c=pressures[np.argsort(pressures)],
        norm=LogNorm(),
        alpha=0.5,
    )
    plt.colorbar(pad=0, label="P")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.close()

    # Now need to drift everything to various times.
    n_steps = int(background_dt / hot_dt)
    t = np.arange(n_steps) * hot_dt

    central_particle_pressure_post_cooling = calculate_pressure(
        position, smoothing_length, positions, internal_energy_once_cooled, 1.0
    )
    central_particle_pressure = (
        np.ones(n_steps + 1) * central_particle_pressure_post_cooling
    )
    central_particle_pressure[0] = pressures[arg_our_favourite]

    # Drifted neighbour
    neighbour_particle_pressure = (
        np.ones(n_steps + 1) * pressures[arg_nearest_neighbour]
    )
    for i in range(len(neighbour_particle_pressure) - 1):
        previous_P = neighbour_particle_pressure[i]
        neighbour_particle_pressure[i + 1] = previous_P * np.exp(
            dP_dt_neighbour * hot_dt / previous_P
        )

    # Neighbour but re-calculated every step.
    neighbour_particle_pressure_recalc = (
        np.ones(n_steps + 1) * central_particle_pressure_post_cooling
    )
    neighbour_particle_pressure_recalc[0] = pressures[arg_nearest_neighbour]

    plt.plot(
        central_particle_pressure / central_particle_pressure[0],
        label="Central Particle",
        linestyle="solid",
        marker=".",
    )
    plt.plot(
        neighbour_particle_pressure / neighbour_particle_pressure[0],
        label="Neighbour (Smooth Drift, Eqn. 35)",
        linestyle="solid",
        marker=".",
    )
    plt.plot(
        [1.0] * len(neighbour_particle_pressure),
        label="Neighbour (Approximate Drift, Eqn. 36)",
        linestyle="dotted",
        marker=".",
    )
    plt.plot(
        neighbour_particle_pressure_recalc / neighbour_particle_pressure_recalc[0],
        label="Neighbour (Single-d$t$)",
        linestyle="dotted",
        marker=".",
    )

    plt.xlabel("Time $t$ [d$t_{\\rm hot}$]")
    plt.ylabel("$\\hat{P}(t) / \\hat{P}(t=0)$")
    plt.semilogy()
    plt.ylim(1e-5, 10 ** (0.5))

    plt.axvline(1, zorder=-10, color="k", linestyle="dashed", linewidth=1)
    plt.text(
        0.95,
        10
        ** (
            0.5
            * (np.log10(plt.gca().get_ylim()[0]) + np.log10(plt.gca().get_ylim()[1]))
        ),
        "d$t_{\\rm hot}$",
        rotation=90,
        ha="right",
        va="center",
    )

    plt.axvline(10, zorder=-10, color="k", linestyle="dashed", linewidth=1)
    plt.text(
        9.95,
        10
        ** (
            0.5
            * (np.log10(plt.gca().get_ylim()[0]) + np.log10(plt.gca().get_ylim()[1]))
        ),
        "d$t_{\\rm cold}$",
        rotation=90,
        ha="right",
        va="center",
    )

    plt.legend(loc=(0.32, 0.05), markerfirst=False, fontsize=6)

    plt.tight_layout()
    plt.savefig("cooling_pressure_ratio.pdf")
    plt.close()

    err = (
        lambda x: -(neighbour_particle_pressure_recalc - x)
        / neighbour_particle_pressure_recalc
    )

    plt.plot(
        err(neighbour_particle_pressure),
        label="Smooth Drift (Eqn. 35)",
        linestyle="solid",
        marker=".",
        color="C1",
    )
    plt.plot(
        err(
            neighbour_particle_pressure_recalc[0]
            * np.array([1.0] * len(neighbour_particle_pressure))
        ),
        label="Approximate Drift (Eqn. 36)",
        linestyle="dotted",
        marker=".",
        color="C2",
    )
    plt.plot(
        err(neighbour_particle_pressure_recalc),
        label="Single-d$t$",
        linestyle="dotted",
        marker=".",
        color="C3",
    )

    plt.legend(markerfirst=False, fontsize=6)
    plt.xlabel("Time $t$ [d$t_{\\rm hot}$]")
    plt.ylabel("Fractional error in $\\hat{P}$ of neighbour")

    plt.axvline(1, zorder=-10, color="k", linestyle="dashed", linewidth=1)
    plt.text(
        0.95,
        0.5 * (plt.gca().get_ylim()[0] + plt.gca().get_ylim()[1]),
        "d$t_{\\rm hot}$",
        rotation=90,
        ha="right",
        va="center",
    )

    plt.axvline(10, zorder=-10, color="k", linestyle="dashed", linewidth=1)
    plt.text(
        9.95,
        0.5 * (plt.gca().get_ylim()[0] + plt.gca().get_ylim()[1]),
        "d$t_{\\rm cold}$",
        rotation=90,
        ha="right",
        va="center",
    )

    plt.tight_layout()
    plt.savefig("cooling_error_ratio.pdf")
