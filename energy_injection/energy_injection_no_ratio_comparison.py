"""
The same as energy_injection.py but this makes a single plot for multiple values of 
the energy injection.
"""

import energy_injection as original
import energy_injection_just_like_EAGLE as eagle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # First, generate positions:

    positions = original.generate_two_cube(32)

    # Now find the one closest to the centre:
    arg_our_favourite = np.argmin(np.sum((positions - 0.5) ** 2, axis=1))
    position = positions[arg_our_favourite]

    print(f"Central particle at {position}.")

    # Now have a gander at its smoothing length
    smoothing_length = original.find_h(position, positions)

    print(f"It has h={smoothing_length:e}, MIPS={1 / 32:e}")

    # Calculate number of particles here
    volume = (4.0 * np.pi / 3.0) * (smoothing_length * original.kernel_gamma) ** 3
    number_density = original.calculate_number_density(position, smoothing_length, positions)

    print(f"This gives it N={number_density * volume}")

    # Now generate entropies
    true_entropies = np.random.rand(positions.shape[0])
    entropies = true_entropies.copy()

    pressure = original.calculate_pressure(position, smoothing_length, positions, entropies)

    print(f"The particle has initial pressure P={pressure:e}")
    print(f"We would expect this to be P={number_density**original.gas_gamma:e}")
    print(f"The particle has energy u={original.calculate_u(pressure, 1.0):e}")

    target_injections = [1, 9, 99, 999, 9999]

    energy_errors_original = []
    energy_errors_eagle = []

    for target_injection in target_injections:
        target_injection = original.calculate_u(pressure, 1.0) * target_injection

        print(f"Attempting to inject {target_injection:e} energy")

        _, _, energy_diff, _ = original.subcycle(
            positions, entropies, arg_our_favourite, target_injection, smoothing_length, max_cycles=50
        )

        entropies = true_entropies.copy()

        energy_errors_original.append([abs(x - target_injection) for x in energy_diff])

        _, _, energy_diff, _ = eagle.subcycle(
            positions, entropies, arg_our_favourite, target_injection, smoothing_length, max_cycles=50
        )

        energy_errors_eagle.append([abs(x - target_injection) for x in energy_diff])

        entropies = true_entropies.copy()


    print("Creating plots")

    import matplotlib.pyplot as plt

    plt.style.use("../../mnras_durham.mplstyle")

    fig, ax = plt.subplots()
    ax.semilogy()

    for energy_error_eagle, energy_error_original, target_injection, color in zip(energy_errors_eagle, energy_errors_original, target_injections, range(len(target_injections))):
        ax.plot(
            range(len(energy_error_original))[1:],
            energy_error_original[1:],
            linestyle="dashed",
            alpha=0.5,
            color=f"C{color}",
        )

        ax.plot(
            range(len(energy_error_eagle))[1:],
            energy_error_eagle[1:],
            color=f"C{color}",
            label=f"$10^{int(np.log10(target_injection+1))}$",
        )

    energy_leg = ax.legend(title="Ratio $u_{\\rm new} / u_{\\rm old}$", fontsize=6, loc="upper right")

    ax.set_ylabel(r"Energy Injection Error")
    ax.set_xlabel("Iterations")

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="black", linestyle="solid"),
                    Line2D([0], [0], color="black", linestyle="dashed")]

    ax.legend(custom_lines, ["Original", "With neighbour loop"], markerfirst=True, loc="lower left", fontsize=6)

    ax.add_artist(energy_leg)

    ax.set_ylim(1e-3, 1e10)
    ax.set_xlim(0, 50)

    fig.tight_layout()

    fig.savefig("energy_compare.pdf")

    print("Saved plot to energy_compare.pdf")

