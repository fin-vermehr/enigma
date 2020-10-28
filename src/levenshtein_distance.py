import numpy as np


def levenshtein_distance(reference: str, target: str):
    num_rows = len(reference) + 1
    num_cols = len(target) + 1
    distances = np.zeros((num_rows, num_cols), dtype=int)

    for i in range(1, num_rows):
        for k in range(1, num_cols):
            distances[i][0] = i
            distances[0][k] = k

    for col in range(1, num_cols):
        for row in range(1, num_rows):
            # If the characters match, then the cost is 0
            if reference[row-1] == target[col-1]:
                substitution_cost = 0
            else:
                substitution_cost = 2

            distances[row][col] = min(distances[row-1][col] + 1,      # Deletion
                                      distances[row][col-1] + 1,          # Insertions
                                      distances[row-1][col-1] + substitution_cost)     # Substitutions

        return ((len(reference)+len(target)) - distances[row][col]) / (len(reference)+len(target))
