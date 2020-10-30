"""
The same as energy_injection.py but this makes a single plot for multiple values of 
the energy injection.
"""

from energy_injection import *

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
    true_entropies = np.random.rand(positions.shape[0])
    entropies = true_entropies.copy()

    pressure = calculate_pressure(position, smoothing_length, positions, entropies)

    print(f"The particle has initial pressure P={pressure:e}")
    print(f"We would expect this to be P={number_density**gas_gamma:e}")
    print(f"The particle has energy u={calculate_u(pressure, 1.0):e}")

    target_injections = [1, 9, 99, 999, 9999]

    energy_ratios = []

    for target_injection in target_injections:
        target_injection = calculate_u(pressure, 1.0) * target_injection

        print(f"Attempting to inject {target_injection:e} energy")

        _, _, energy_diff, _ = subcycle(
            positions, entropies, arg_our_favourite, target_injection, smoothing_length
        )

        entropies = true_entropies.copy()

        energy_ratios.append([x / target_injection for x in energy_diff])

    print("Creating plots")

    import matplotlib.pyplot as plt

    plt.style.use("../../mnras_durham.mplstyle")

    fig, ax = plt.subplots()
    ax.semilogy()

    for zorder, (energy_ratio, target_injection) in enumerate(zip(energy_ratios, target_injections)):
        ax.plot(
            range(len(energy_ratio))[1:],
            energy_ratio[1:],
            label=f"$10^{int(np.log10(target_injection+1))}$",
            zorder=15-zorder,
        )

    ax.axhline(1.0, linestyle="dashed", linewidth=1.0, color="C3")

    ax.legend(title="Ratio $u_{\\rm new} / u_{\\rm old}$", fontsize=6)

    ax.set_ylabel(r"$E_{\rm injected} / E_{\rm target}$")
    ax.set_xlabel("Iterations")

    ax.set_ylim(1e-1, 1e1)

    fig.tight_layout()

    fig.savefig("energy_ratio_improved.pdf")

    print("Saved plot to energy_ratio.pdf")

