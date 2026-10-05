import numpy as np
import matplotlib.pyplot as plt

def main():
    dimensions = [2**i for i in range(11)] # [1, 2, 4, 8, ... 1024]
    num_points = 100

    # Initialize lists to store the statistics for both metrics
    sq_eucl_means = []
    sq_eucl_stds = []
    manhattan_means = []
    manhattan_stds = []

    # get the upper triangle indices for the matrix (all unique combos)
    i, j = np.triu_indices(num_points, k=1)

    #
    rng = np.random.default_rng()

    for d in dimensions:
        # make samples of 100 points from a d dimensional unit cube (random)
        points = rng.uniform(0, 1, size=(num_points, d))

        # Calculate dimensional differences for all 4,950 unique pairs simultaneously
        diffs = points[i] - points[j]

        # Calculate Squared Euclidean Distances (sum of squared differences)
        sq_eucl_dists = np.sum(diffs ** 2, axis=1)
        sq_eucl_means.append(np.mean(sq_eucl_dists))
        sq_eucl_stds.append(np.std(sq_eucl_dists))
        
        # Calculate Manhattan Distances (sum of absolute differences)
        manhattan_dists = np.sum(np.abs(diffs), axis=1)
        manhattan_means.append(np.mean(manhattan_dists))
        manhattan_stds.append(np.std(manhattan_dists))

    # Generate the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean and Manhattan and Squared Euclidean Distances
    ax1.plot(dimensions, manhattan_means, label='Manhattan', marker='o', color='red')
    ax1.plot(dimensions, sq_eucl_means, label='Squared Euclidean', marker='o', color='blue')
    ax1.set_xscale('log', base=2)
    ax1.set_title('Mean Distance vs. Dimension (d)')
    ax1.set_xlabel('Dimension (d)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Standard Deviation of Manhattan and Squared Euclidean Distances
    ax2.plot(dimensions, manhattan_stds, label='Manhattan', marker='s', color='red')
    ax2.plot(dimensions, sq_eucl_stds, label='Squared Euclidean', marker='s', color='blue')
    ax2.set_xscale('log', base=2)
    ax2.set_title('Standard Deviation vs. Dimension (d)')
    ax2.set_xlabel('Dimension (d)')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()