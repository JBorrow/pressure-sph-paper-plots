"""
This function looks at energy injection into a medium in a very simple way.

We set up a 32^3 particle cubic grid on a BCC lattice. We select the central
particle. Then, we (using pressure entropy) try to inject some amount of energy
into it.

Now, this verison is the same as energy_injection.py but does things in the same
way as EAGLE, updating only the particle self-contribution to set the energy.
"""

import numpy as np
from math import sqrt

from numba import jit
from scipy.optimize import newton
from math import pow

from typing import Union

import matplotlib.pyplot as plt

plt.style.use("../../mnras_durham.mplstyle")


kernel_gamma = 1.936_492
kernel_constant = 21 / (2 * 3.14159)
gas_gamma = 5.0 / 3.0
one_over_gamma = 1.0 / gas_gamma
eta = 1.2


def generate_cube(num_on_side, side_length=1.0):
    """
    Generates a cube of particles
    """

    values = np.linspace(0.0, side_length, num_on_side + 1)[:-1]

    positions = np.empty((num_on_side ** 3, 3), dtype=float)

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

    positions = np.concatenate([cube, cube + mips * 0.5])

    return positions


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, np.float32], H: Union[float, np.float32]):
    """
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).
    Give it a radius and a kernel width (i.e. not a smoothing length, but the
    radius of compact support) and it returns the contribution to the
    density.
    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

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
    entropies: np.float64,
) -> np.float64:
    entropy_to_one_over_gamma = entropies ** (one_over_gamma)
    kernel_width = smoothing_length * kernel_gamma

    P = 0.0

    # This hack is required to numba me.
    for i in range(len(entropy_to_one_over_gamma)):
        pos = positions[i]
        A = entropy_to_one_over_gamma[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])

        if r <= kernel_width:
            P += A * kernel(r, kernel_width)

    return P ** gas_gamma


@jit(nopython=True, fastmath=True)
def calculate_pressure_without_specific_index(
    position: np.float64,
    smoothing_length: np.float64,
    positions: np.float64,
    entropies: np.float64,
    bad_index: int,
) -> np.float64:
    entropy_to_one_over_gamma = entropies ** (one_over_gamma)
    kernel_width = smoothing_length * kernel_gamma

    P = 0.0

    # This hack is required to numba me.
    for i in range(len(entropy_to_one_over_gamma)):
        if i == bad_index:
            continue

        pos = positions[i]
        A = entropy_to_one_over_gamma[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])

        if r <= kernel_width:
            P += A * kernel(r, kernel_width)

    return P


def calculate_number_density(position, smoothing_length, positions):
    kernel_width = smoothing_length * kernel_gamma

    N = 0.0

    for pos in positions:
        dx = pos - position
        r = sqrt(np.sum(dx * dx))

        if r <= kernel_width:
            N += kernel(r, kernel_width)

    return N


def calculate_u(P, A):
    return A ** (one_over_gamma) * P ** (1.0 - one_over_gamma) / ((gas_gamma - 1.0))


def calculate_A(P, u):
    return (P ** (1.0 - gas_gamma)) * (gas_gamma - 1.0) ** gas_gamma * u ** gas_gamma


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


def subcycle(positions, entropies, index, target_energy_addition, smoothing_length):
    """
    Gets the energy to inject by using the newton-raphson method using the equation
    that RGB suggested.
    """

    kernel_width = smoothing_length * kernel_gamma

    # First, let's find all particles we're going to interact with.
    # We need to calculate their initial internal energies as defined
    # by their smoothed pressure. Then, when we inject ENTROPY into one
    # particle, we can see how much this has changed the energy of the entire
    # system.
    position = positions[index]

    dx = positions - position
    r = np.sqrt(np.sum(dx * dx, axis=1))
    interact = r <= kernel_width
    # Find where we are in the interact array
    our_interact_part = np.where((r == 0)[interact])[0][0]

    interact_positions = positions[interact]
    interact_entropies = entropies[interact]
    interact_entropies_one_over_gamma = interact_entropies ** (1.0 / gas_gamma)
    r_interact = r[interact]

    # A couple of quick things to calculate the pressures of the relevant particles
    # (i.e. those that we interact with) and their energies
    def get_pressures_no_self_contrib():
        return np.array(
            [
                calculate_pressure_without_specific_index(
                    x, smoothing_length, positions, entropies, index
                )
                for x in interact_positions
            ]
        )

    def get_pressures():
        return np.array(
            [
                calculate_pressure(x, smoothing_length, positions, entropies)
                for x in interact_positions
            ]
        )

    def get_total_energy():
        pressures = get_pressures()
        interact_entropies = entropies[interact]

        return sum([calculate_u(p, A) for p, A in zip(pressures, interact_entropies)])

    initial_energy = get_total_energy()
    pressures_without_our_contrib = get_pressures_no_self_contrib()
    pressures_total = get_pressures()

    def get_energy_given_delta_A(delta_A: float):
        if delta_A <= 0.0:
            return 1e32

        front_fac = 1.0 / (gas_gamma - 1.0)
        our_entropy_plus_delta_one_over_gamma = pow(
            interact_entropies[our_interact_part] + delta_A, 1.0 / gas_gamma
        )

        # our particles!
        output_energy = 0.0

        for part in range(len(interact_positions)):
            rhs = pow(
                pressures_without_our_contrib[part]
                + kernel(np.float32(r_interact[our_interact_part]), kernel_width)
                * our_entropy_plus_delta_one_over_gamma,
                gas_gamma - 1,
            )

            if part == our_interact_part:
                output_energy += our_entropy_plus_delta_one_over_gamma * rhs
            else:
                output_energy += interact_entropies_one_over_gamma[part] * rhs

        output_energy = output_energy * front_fac - initial_energy

        return output_energy

    # Now we actually need to use a rootfinder to solve this:

    # First we need a 'best guess'
    central_particle_energy = calculate_u(
        pressures_total[our_interact_part], entropies[index]
    )

    central_particle_energy_new = central_particle_energy + target_energy_addition

    # Based on our new energy, calculate the entropy and set it as the
    # particle's real entropy
    first_delta_A = (
        calculate_A(pressures_total[our_interact_part], central_particle_energy_new)
        - entropies[index]
    ) * 0.1

    def to_root(delta_A: float):
        return get_energy_given_delta_A(delta_A) - target_energy_addition

    root_delta_A = newton(to_root, x0=first_delta_A)

    delta_A = np.logspace(2, 4.5, 100)
    delta_u = [
        (get_energy_given_delta_A(dA)) / target_energy_addition for dA in delta_A
    ]

    plt.axhline(1.0, color="grey", linestyle="dotted", lw=1.0)
    plt.axvline(root_delta_A, color="C1", linestyle="dashed", lw=1.0)
    plt.semilogx(delta_A, delta_u)
    plt.xlabel(r"Change in $A_i$")
    plt.ylabel(r"Change in $u/\Delta u$")
    plt.xlim(1e2, 10 ** (4.5))
    plt.ylim(0, None)
    plt.text(
        root_delta_A * 0.95,
        2.0,
        "Root found with Newton-Raphson",
        ha="right",
        va="center",
        rotation=90,
        color="C1",
        fontsize=6,
    )
    plt.tight_layout()
    plt.savefig("converge.pdf")

    return root_delta_A


if __name__ == "__main__":
    # First, generate positions:

    positions = generate_two_cube(32)

    # Now find the one closest to the centre:
    arg_our_favourite = np.argmin(np.sum((positions - 0.5) ** 2, axis=1))
    position = positions[arg_our_favourite]

    print(f"Central particle at {position}.")

    # Now have a gander at its smoothing length
    smoothing_length = find_h(position, positions)

    print(f"It has h={smoothing_length:e}, MIPS={1 / 32:e}")

    # Calculate number of particles here
    volume = (4.0 * np.pi / 3.0) * (smoothing_length * kernel_gamma) ** 3
    number_density = calculate_number_density(position, smoothing_length, positions)

    print(f"This gives it N={number_density * volume}")

    # Now generate entropies
    np.random.seed(1234)
    true_entropies = np.random.rand(positions.shape[0])
    entropies = true_entropies.copy()

    pressure = calculate_pressure(position, smoothing_length, positions, entropies)

    print(f"The particle has initial pressure P={pressure:e}")
    print(f"We would expect this to be P={number_density**gas_gamma:e}")
    print(f"The particle has energy u={calculate_u(pressure, 1.0):e}")

    # Now the fun begins.
    print(f"Let's try to multiply its energy by 10^3.5 like EAGLE!")

    target_injection = calculate_u(pressure, 1.0) * 10 ** (3.5)

    delta_A = subcycle(
        positions, entropies, arg_our_favourite, target_injection, smoothing_length
    )

    print(delta_A, delta_A + entropies[arg_our_favourite])
    exit(0)

    print(f"Attempting to inject {target_injection:e} energy")

    # print("Entropy as a function of time")
    # print(entropy_history)
    # print("Energy as a function of time")
    # print(energy_history)
    # print("Diff from target as a function of time")
    # print(energy_diff)
    # print("Ratio to expected")
    # print([x / target_injection for x in energy_diff])
    # print("Injected this step")
    # print(energy_injected)

    print("Creating plots")

    import matplotlib.pyplot as plt

    plt.style.use("../../mnras_durham.mplstyle")

    fig, ax = plt.subplots(2, 2, figsize=(6.974, 6.974), sharex=True)
    ax = ax.flatten()

    to_plot = {
        # Scale the values for plotting otherwise it messes up y-axis
        "Entropy of particle injected into": np.array(entropy_history) * 0.01,
        "Total energy of system": np.array(energy_history) * 0.0001,
        "Energy actually injected": np.array(energy_diff) * 0.0001,
        "Ratio to required injection": [x / target_injection for x in energy_diff],
    }

    hlines_at = [
        None,
        (energy_history[0] + target_injection) * 0.0001,
        target_injection * 0.0001,
        1.0,
    ]

    for axis, hline, (name, stat) in zip(ax, hlines_at, to_plot.items()):
        axis.plot(range(len(stat)), stat)
        axis.set_ylabel(name)

        if hline is not None:
            axis.axhline(hline, linestyle="dashed", linewidth=1.0, color="C3")

    for axis in ax[-2:]:
        axis.set_xlabel("Iterations")

    fig.tight_layout()

    fig.savefig("energy_injection_EAGLE.pdf")

    print("Saved plot to energy_injection_EAGLE.pdf")

