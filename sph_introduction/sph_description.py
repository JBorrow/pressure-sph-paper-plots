"""
SPH description plot. Somewhat complex, but just shows:

+ 1D Kernel with the correct number of neighbours
+ 1D Kernel with too many / too few neighbouurs
+ 2D distribution with kernels.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import durham_cmaps as dc

from numba import njit
from scipy.optimize import newton
from matplotlib.colors import LogNorm

from make_particles import get_positions, draw_circle

from swiftsimio.visualisation.projection import scatter

from typing import List

kernel_gamma_1D = 1.732_051
kernel_constant_1d = 8 / 3
kernel_gamma_2D = 1.778_002
kernel_constant_2d = 80 / (7 * np.pi)
eta = 1.1

plt.style.use("../../mnras_durham.mplstyle")


@njit(fastmath=True)
def kernel_1D(r: np.float32, H: np.float32):
    """
    The 1D kernel - a cubic spline
    """

    ratio = r / H
    kernel = np.float32(0.0)

    if ratio < 1.0:
        one_m_ratio = 1.0 - ratio
        one_m_ratio_cube = one_m_ratio * one_m_ratio * one_m_ratio
        kernel += one_m_ratio_cube

    if ratio < 0.5:
        half_m_ratio = 0.5 - ratio
        half_m_ratio_cube = half_m_ratio * half_m_ratio * half_m_ratio
        kernel -= 4.0 * half_m_ratio_cube

    return kernel * kernel_constant_1d / H


@njit(fastmath=True)
def kernel_2D(r: np.float32, H: np.float32):
    """
    The 2D kernel - a cubic spline
    """

    ratio = r / H
    kernel = np.float32(0.0)

    if ratio < 1.0:
        one_m_ratio = 1.0 - ratio
        one_m_ratio_cube = one_m_ratio * one_m_ratio * one_m_ratio
        kernel += one_m_ratio_cube

    if ratio < 0.5:
        half_m_ratio = 0.5 - ratio
        half_m_ratio_cube = half_m_ratio * half_m_ratio * half_m_ratio
        kernel -= 4.0 * half_m_ratio_cube

    return kernel * kernel_constant_2d / (H * H)


@njit(fastmath=True)
def calculate_number_density(
    position: np.float32, smoothing_length: np.float32, positions: List[np.float32]
):
    kernel_width = smoothing_length * kernel_gamma_1D

    N = 0.0

    for pos in positions:
        dx = pos - position
        r = math.fabs(dx)

        if r <= kernel_width:
            N += kernel_1D(r, kernel_width)

    return N


@njit(fastmath=True)
def calculate_number_density_2D(
    position: np.float32,
    smoothing_length: np.float32,
    x: List[np.float32],
    y: List[np.float32],
):
    kernel_width = smoothing_length * kernel_gamma_2D

    N = 0.0

    for this_x, this_y in zip(x, y):
        dx = this_x - position[0]
        dy = this_y - position[1]
        r = math.sqrt(dx * dx + dy * dy)

        if r <= kernel_width and kernel_width >= 0.0:
            N += kernel_2D(r, kernel_width)

    if N == 0.0:
        print(position, kernel_width)

    return N


def find_h(position, positions):
    """
    Find the smoothing length.
    """

    # Start off with a first guess
    initial_guess = 1 / (positions[1] - positions[0])

    def to_root(h):
        h_from_number_density = eta * (
            1.0 / (calculate_number_density(position, h, positions))
        )

        return h - h_from_number_density

    return newton(to_root, initial_guess)


def find_h_2D(position, positions):
    """
    Find the smoothing length (2D version).
    """

    # Start off with a first guess
    initial_guess = 15.0

    def to_root(h):
        if h < 0:
            return 1e33

        try:
            h_from_number_density = eta * (
                1.0
                / (
                    calculate_number_density_2D(
                        position, h, positions.T[0], positions.T[1]
                    )
                )
            ) ** (1 / 2)
        except ZeroDivisionError:
            return 1e33

        return h - h_from_number_density

    return newton(to_root, initial_guess)


def create_ax_left(ax_left):
    # Set up intial particle positions
    particle_spacing = 1.0
    particle_positions = np.arange(1.0, 20.0, particle_spacing)

    our_particle_x = 5.0
    our_particle_h = find_h(our_particle_x, particle_positions)
    our_particle_ngb = calculate_number_density(
        our_particle_x, our_particle_h, particle_positions
    )

    ax_left.plot(
        particle_positions, np.zeros_like(particle_positions), color="k", zorder=-10
    )

    ax_left.scatter(particle_positions, np.zeros_like(particle_positions))
    ax_left.scatter(our_particle_x, 0.0, color="C2")
    kernel_x = np.linspace(
        our_particle_x - our_particle_h * 5.0,
        our_particle_x + our_particle_h * 5.0,
        1000,
    )
    ax_left.fill_between(
        kernel_x,
        np.zeros_like(kernel_x),
        [
            kernel_1D(abs(x - our_particle_x), our_particle_h * kernel_gamma_1D)
            for x in kernel_x
        ],
        alpha=0.2,
        zorder=-100,
    )

    # Check what h should be
    h_from_number_density = eta * (
        1.0
        / (calculate_number_density(our_particle_x, our_particle_h, particle_positions))
    )
    chi_string = f"$h_i / (\eta / \hat{{n}}_i) = \chi_i = {our_particle_h / h_from_number_density:2.2f}$"

    total_string = (
        "Kernel is correctly sized ("
        + chi_string
        + ")\n$\hat{n}_i = \sum_j \; W_{ij} = "
    )
    for particle in particle_positions:
        distance_to_our_particle = abs(particle - our_particle_x)
        kernel_width = kernel_gamma_1D * our_particle_h
        if distance_to_our_particle < kernel_width:
            kernel_value = kernel_1D(distance_to_our_particle, kernel_width)

            ax_left.plot(
                [particle, particle], [0.0, kernel_value], color="C3", zorder=-50, lw=1
            )

            ax_left.scatter(particle, kernel_value, color="C3", s=5.0)

            total_string += f"{kernel_value:2.2f} +"

            ax_left.text(
                particle,
                kernel_value + 0.025,
                f"$W_{{ij}}={kernel_value:2.2f}$",
                ha="center",
                va="bottom",
            )

    # Remove additional +
    total_string = total_string[:-2] + f" = {our_particle_ngb:2.2f}$"

    ax_left.text(
        0.025, 0.975, total_string, ha="left", va="top", transform=ax_left.transAxes
    )

    # Add in bottom labels for particles
    for particle in np.arange(
        -2.0 * particle_spacing, 3.0 * particle_spacing, particle_spacing
    ):
        label = "$j=i"
        if particle:
            label += f"{int(particle):+d}"
        label += "$"

        ax_left.text(
            particle + our_particle_x, -0.05, label, ha="center", va="top", fontsize=6
        )

    # Add in h-overlay

    ax_left.arrow(
        our_particle_x + particle_spacing * 1.75 + 0.25,
        kernel_1D(0, our_particle_h * kernel_gamma_1D) * 0.9,
        0,
        # This ratio comes from x_width / y_width
        -our_particle_h * kernel_gamma_1D * (1.1 / 5),
        facecolor="C3",
        width=0.03,
        edgecolor="none",
        length_includes_head=True,
        head_length=0.02,
    )

    ax_left.arrow(
        our_particle_x + particle_spacing * 1.75,
        kernel_1D(0, our_particle_h * kernel_gamma_1D) * 0.9,
        0,
        -our_particle_h * (1.1 / 5),
        facecolor="C3",
        width=0.03,
        edgecolor="none",
        length_includes_head=True,
        head_length=0.02,
    )

    ax_left.text(
        our_particle_x + particle_spacing * 1.75,
        kernel_1D(0, our_particle_h * kernel_gamma_1D) * 0.9,
        "$h$",
        color="C3",
        ha="center",
        va="bottom",
    )

    ax_left.text(
        our_particle_x + particle_spacing * 1.75 + 0.25,
        kernel_1D(0, our_particle_h * kernel_gamma_1D) * 0.9,
        "$H$",
        color="C3",
        ha="center",
        va="bottom",
    )

    ax_left.text(
        our_particle_x + particle_spacing * 1.75 + 0.25 / 2,
        kernel_1D(0, our_particle_h * kernel_gamma_1D),
        f"$\eta={eta}$",
        ha="center",
        va="bottom",
    )

    ax_left.set_xlim(2.5, 7.5)
    ax_left.set_ylim(-0.1, 1.0)

    return


def create_ax_top(ax_top):
    # Set up intial particle positions
    particle_spacing = 1.0
    particle_positions = np.arange(1.0, 20.0, particle_spacing)

    our_particle_x = 5.0
    our_particle_h = find_h(our_particle_x, particle_positions) * 2.0
    our_particle_ngb = calculate_number_density(
        our_particle_x, our_particle_h, particle_positions
    )

    ax_top.plot(
        particle_positions, np.zeros_like(particle_positions), color="k", zorder=-10
    )

    ax_top.scatter(particle_positions, np.zeros_like(particle_positions))
    ax_top.scatter(our_particle_x, 0.0, color="C2")
    kernel_x = np.linspace(
        our_particle_x - our_particle_h * 5.0,
        our_particle_x + our_particle_h * 5.0,
        1000,
    )
    ax_top.fill_between(
        kernel_x,
        np.zeros_like(kernel_x),
        [
            kernel_1D(abs(x - our_particle_x), our_particle_h * kernel_gamma_1D)
            for x in kernel_x
        ],
        alpha=0.2,
        zorder=-100,
    )

    for particle in particle_positions:
        distance_to_our_particle = abs(particle - our_particle_x)
        kernel_width = kernel_gamma_1D * our_particle_h
        if distance_to_our_particle < kernel_width:
            kernel_value = kernel_1D(distance_to_our_particle, kernel_width)

            ax_top.plot(
                [particle, particle], [0.0, kernel_value], color="C3", zorder=-50, lw=1
            )

            ax_top.scatter(particle, kernel_value, color="C3", s=5.0)

    # Remove additional +
    total_string = f"$\hat{{n}}_i = {our_particle_ngb:2.2f}$"

    # Check what h should be
    h_from_number_density = eta * (
        1.0
        / (calculate_number_density(our_particle_x, our_particle_h, particle_positions))
    )
    total_string += f"\n$\chi_i = {our_particle_h / h_from_number_density:2.2f}$"

    ax_top.text(
        0.05, 0.95, total_string, ha="left", va="top", transform=ax_top.transAxes
    )

    ax_top.set_xlim(2.5, 7.5)
    ax_top.set_ylim(-0.1, 1.0)

    return


def create_ax_bottom(ax_bottom):
    # Set up intial particle positions
    particle_spacing = 1.0
    particle_positions = np.arange(1.0, 20.0, particle_spacing)

    our_particle_x = 5.0
    our_particle_h = find_h(our_particle_x, particle_positions) * 0.5
    our_particle_ngb = calculate_number_density(
        our_particle_x, our_particle_h, particle_positions
    )

    ax_bottom.plot(
        particle_positions, np.zeros_like(particle_positions), color="k", zorder=-10
    )

    ax_bottom.scatter(particle_positions, np.zeros_like(particle_positions))
    ax_bottom.scatter(our_particle_x, 0.0, color="C2")
    kernel_x = np.linspace(
        our_particle_x - our_particle_h * 5.0,
        our_particle_x + our_particle_h * 5.0,
        1000,
    )
    ax_bottom.fill_between(
        kernel_x,
        np.zeros_like(kernel_x),
        [
            kernel_1D(abs(x - our_particle_x), our_particle_h * kernel_gamma_1D)
            for x in kernel_x
        ],
        alpha=0.2,
        zorder=-100,
    )

    for particle in particle_positions:
        distance_to_our_particle = abs(particle - our_particle_x)
        kernel_width = kernel_gamma_1D * our_particle_h
        if distance_to_our_particle < kernel_width:
            kernel_value = kernel_1D(distance_to_our_particle, kernel_width)

            ax_bottom.plot(
                [particle, particle], [0.0, kernel_value], color="C3", zorder=-50, lw=1
            )

            ax_bottom.scatter(particle, kernel_value, color="C3", s=5.0)

    # Remove additional +
    total_string = f"$\hat{{n}}_i = {our_particle_ngb:2.2f}$"
    # Check what h should be
    h_from_number_density = eta * (
        1.0
        / (calculate_number_density(our_particle_x, our_particle_h, particle_positions))
    )
    total_string += f"\n$\chi_i = {our_particle_h / h_from_number_density:2.2f}$"

    ax_bottom.text(
        0.05, 0.95, total_string, ha="left", va="top", transform=ax_bottom.transAxes
    )

    ax_bottom.set_xlim(2.5, 7.5)
    ax_bottom.set_ylim(-0.1, 1.0)

    return


def create_ax_right(ax_right):
    positions = np.array(get_positions()).T
    hsml = np.array([find_h_2D(x, positions) for x in positions]) * 2.0

    grid = scatter(
        positions.T[0] / 100, positions.T[1] / 100, np.ones_like(hsml), hsml / 100, 1024
    )

    ax_right.imshow(
        np.log10(grid.T),
        extent=[0, 100, 0, 100],
        origin="lower",
        cmap=dc.dark_durham_cmap,
        vmin=np.log10(grid.min()),
        vmax=np.log10(grid.max()),
    )

    # Draw a few circles to indicate hsml
    for part in [50, 75, 100, 200, 250, 300, 2]:
        draw_circle(
            positions[part][0],
            positions[part][1],
            hsml[part],
            "black",
            ax_right,
            "dotted",
        )

        draw_circle(
            positions[part][0],
            positions[part][1],
            kernel_gamma_2D * hsml[part],
            "black",
            ax_right,
            "dashed",
        )

    # Display h and gamma * h for special particle
    special_part = 50

    arrow_length = hsml[special_part] / np.sqrt(2)

    # Annotate h
    ax_right.arrow(
        positions[special_part][0],
        positions[special_part][1],
        arrow_length,
        arrow_length,
        facecolor="black",
        width=0.3,
        edgecolor="none",
        length_includes_head=True,
    )

    ax_right.text(
        positions[special_part][0] + arrow_length / 2,
        positions[special_part][1] + arrow_length / 2,
        "$h$",
        color="black",
        ha="right",
        va="bottom",
    )

    # Annotate H
    ax_right.arrow(
        positions[special_part][0],
        positions[special_part][1],
        arrow_length * kernel_gamma_2D,
        -arrow_length * kernel_gamma_2D,
        facecolor="black",
        width=0.3,
        edgecolor="none",
        length_includes_head=True,
    )

    ax_right.text(
        positions[special_part][0] + arrow_length * kernel_gamma_2D / 2 - 2,
        positions[special_part][1] - arrow_length * kernel_gamma_2D / 2,
        "$H$",
        color="black",
        ha="right",
        va="top",
    )

    # Overlay sph particles
    ax_right.scatter(
        positions.T[0],
        positions.T[1],
        color="white",
        s=0.5,
        edgecolor="C3",
        linewidth=0.2,
        zorder=2,
    )

    ax_right.set_xlim(0, 100)
    ax_right.set_ylim(0, 100)


if __name__ == "__main__":
    # Actually make the plots

    # First we need to set up our image.
    fig = plt.figure(figsize=(6.974, 6.974 * (2 / 5) * 0.975), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_right = fig.add_subplot(gs[:, 3:])
    ax_top = fig.add_subplot(gs[0, 2])
    ax_bottom = fig.add_subplot(gs[1, 2])
    axes = [ax_left, ax_right, ax_top, ax_bottom]
    for ax in axes:
        ax.tick_params(
            axis="both",
            left=False,
            top=False,
            right=False,
            bottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
            labelbottom=False,
        )

    create_ax_left(ax_left)
    create_ax_top(ax_top)
    create_ax_bottom(ax_bottom)
    create_ax_right(ax_right)

    plt.savefig("sph_description.pdf")

