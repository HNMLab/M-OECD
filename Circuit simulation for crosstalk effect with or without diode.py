# basic_code5.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ====================================
# User-Defined Parameters
# ====================================
applied_voltage = 0.6               # Voltage applied to the active (fixed) row (V)
resistance_not_pressed = 6000.0     # Resistance for a non-pressed pixel (Ω)
resistance_pressed = resistance_not_pressed * (1-0.05)         # Resistance for a pressed pixel (Ω)
array_sizes = [5, 6, 8, 10, 12, 14, 16, 18, 20]  # List of array sizes (n x n)
rect_ratio = 500                    # Diode rectification ratio

# ====================================
# Helper Function: Determine Pressed Pixel Positions
# ====================================
def get_pressed_positions(n):
    """
    Returns three pressed pixel positions along the main diagonal near the center with one-cell spacing.
    For n < 7, returns: [(0,0), (n//2, n//2), (n-1, n-1)] (0-indexed).
    For n >= 7, returns: [(center-3, center-3), (center-1, center-1), (center+1, center+1)]
    where center = n // 2.
    (For example, for n=10, center=5, positions are (2,2), (4,4), (6,6) in 0-indexed,
    i.e. (3,3), (5,5), (7,7) in 1-indexed.)
    """
    if n < 7:
        return [(n // 2 - 2, n // 2 - 2), (n // 2, n // 2), (n // 2 + 2, n // 2 + 2)]
    else:
        center = n // 2
        return [(center - 3, center - 3), (center - 1, center - 1), (center + 1, center + 1)]

# ====================================
# Function Definitions
# ====================================
def simulate_measurement(fixed_row, fixed_col, R_matrix):
    """
    Compute the effective resistance, total current, direct current, and crosstalk current
    via nodal analysis for a given fixed row (set to applied_voltage) and fixed column (set to 0V).
    (Passive network: all pixels are set to resistance_not_pressed except the pressed pixels are set to resistance_pressed)
    """
    n_rows, n_cols = R_matrix.shape
    # Define floating nodes.
    floating_rows = [r for r in range(n_rows) if r != fixed_row]
    floating_cols = [c for c in range(n_cols) if c != fixed_col]
    n_unknown = len(floating_rows) + len(floating_cols)
    
    # Create mappings for floating nodes.
    row_to_idx = {r: i for i, r in enumerate(floating_rows)}
    col_to_idx = {c: i + len(floating_rows) for i, c in enumerate(floating_cols)}
    
    A = np.zeros((n_unknown, n_unknown))
    b = np.zeros(n_unknown)
    
    # Build equations for floating row nodes (with fixed column = 0V)
    for r in floating_rows:
        eq_idx = row_to_idx[r]
        sum_inv = 0.0
        for c in range(n_cols):
            invR = 1.0 / R_matrix[r, c]
            sum_inv += invR
            if c == fixed_col:
                continue
            else:
                A[eq_idx, col_to_idx[c]] -= invR
        A[eq_idx, eq_idx] += sum_inv
    
    # Build equations for floating column nodes (with fixed row = applied_voltage)
    for c in floating_cols:
        eq_idx = col_to_idx[c]
        sum_inv = 0.0
        for r in range(n_rows):
            invR = 1.0 / R_matrix[r, c]
            sum_inv += invR
            if r == fixed_row:
                # Multiply by applied_voltage to reflect the fixed node's value
                b[eq_idx] -= applied_voltage * invR
            elif r in floating_rows:
                A[eq_idx, row_to_idx[r]] += invR
        A[eq_idx, eq_idx] -= sum_inv
    
    x = np.linalg.solve(A, b)
    
    # Compute total current from fixed row (applied_voltage) to each column.
    I_total = 0.0
    for c in range(n_cols):
        V_c = 0.0 if c == fixed_col else x[col_to_idx[c]]
        I_total += (applied_voltage - V_c) / R_matrix[fixed_row, c]
    
    # Effective resistance is defined by Ohm's law: applied_voltage / I_total.
    effective_R = applied_voltage / I_total
    I_direct = applied_voltage / R_matrix[fixed_row, fixed_col]
    I_crosstalk = I_total - I_direct
    
    return effective_R, I_total, I_direct, I_crosstalk

def compute_unknown_node_voltages(fixed_row, fixed_col, R_matrix):
    """
    Compute the voltages of the floating nodes (excluding the fixed row (applied_voltage) and fixed column (0V)).
    Returns the list of floating rows, floating columns, and the voltage vector x.
    """
    n_rows, n_cols = R_matrix.shape
    floating_rows = [r for r in range(n_rows) if r != fixed_row]
    floating_cols = [c for c in range(n_cols) if c != fixed_col]
    n_unknown = len(floating_rows) + len(floating_cols)
    
    row_to_idx = {r: i for i, r in enumerate(floating_rows)}
    col_to_idx = {c: i + len(floating_rows) for i, c in enumerate(floating_cols)}
    
    A = np.zeros((n_unknown, n_unknown))
    b = np.zeros(n_unknown)
    
    for r in floating_rows:
        eq_idx = row_to_idx[r]
        sum_inv = 0.0
        for c in range(n_cols):
            invR = 1.0 / R_matrix[r, c]
            sum_inv += invR
            if c == fixed_col:
                continue
            else:
                A[eq_idx, col_to_idx[c]] -= invR
        A[eq_idx, eq_idx] += sum_inv
    
    for c in floating_cols:
        eq_idx = col_to_idx[c]
        sum_inv = 0.0
        for r in range(n_rows):
            invR = 1.0 / R_matrix[r, c]
            sum_inv += invR
            if r == fixed_row:
                b[eq_idx] -= applied_voltage * invR
            elif r in floating_rows:
                A[eq_idx, row_to_idx[r]] += invR
        A[eq_idx, eq_idx] -= sum_inv
    
    x = np.linalg.solve(A, b)
    return floating_rows, floating_cols, x

def simulate_measurement_diode(fixed_row, fixed_col, R_matrix, tol=1e-6, max_iter=100, epsilon=1e-3):
    """
    Compute the effective resistance for the network with diodes.
    For each connection, based on (V_row - V_col):
      - if (V_row - V_col) >= -1e-6, use forward conductance (G = 1/R);
      - otherwise, use reverse conductance (G = 1/(rect_ratio * R)).
    Fixed nodes: fixed_row = applied_voltage, fixed_col = 0V.
    (All pixels are set to resistance_not_pressed except the pressed pixel is set to resistance_pressed.)
    Uses the global variable 'rect_ratio'.
    """
    n_rows, n_cols = R_matrix.shape
    floating_rows = [r for r in range(n_rows) if r != fixed_row]
    floating_cols = [c for c in range(n_cols) if c != fixed_col]
    n_unknown = len(floating_rows) + len(floating_cols)
    
    row_to_idx = {r: i for i, r in enumerate(floating_rows)}
    col_to_idx = {c: i + len(floating_rows) for i, c in enumerate(floating_cols)}
    
    # Use the passive network solution as the initial guess.
    _, _, x0 = compute_unknown_node_voltages(fixed_row, fixed_col, R_matrix)
    x_old = x0.copy()
    
    for it in range(max_iter):
        A = np.zeros((n_unknown, n_unknown))
        b = np.zeros(n_unknown)
        
        for r in floating_rows:
            eq_idx = row_to_idx[r]
            sum_G = 0.0
            for c in range(n_cols):
                V_r = x_old[row_to_idx[r]]
                V_c = 0.0 if c == fixed_col else x_old[col_to_idx[c]]
                diff = V_r - V_c
                if diff >= -1e-6:
                    G = 1.0 / R_matrix[r, c]
                else:
                    G = 1.0 / (rect_ratio * R_matrix[r, c])
                sum_G += G
                if c == fixed_col:
                    continue
                else:
                    A[eq_idx, col_to_idx[c]] -= G
            A[eq_idx, eq_idx] += sum_G
        
        for c in floating_cols:
            eq_idx = col_to_idx[c]
            sum_G = 0.0
            for r in range(n_rows):
                if r == fixed_row:
                    V_r = applied_voltage
                else:
                    if r in floating_rows:
                        V_r = x_old[row_to_idx[r]]
                    else:
                        continue
                V_c = x_old[col_to_idx[c]]
                diff = V_r - V_c
                if diff >= -1e-6:
                    G = 1.0 / R_matrix[r, c]
                else:
                    G = 1.0 / (rect_ratio * R_matrix[r, c])
                sum_G += G
                if r == fixed_row:
                    b[eq_idx] -= applied_voltage * G
                else:
                    A[eq_idx, row_to_idx[r]] += G
            A[eq_idx, eq_idx] -= sum_G
        
        x_new = np.linalg.solve(A, b)
        if np.max(np.abs(x_new - x_old)) < tol:
            x_old = x_new
            break
        x_old = x_new
    
    I_total = 0.0
    for c in range(n_cols):
        V_c = 0.0 if c == fixed_col else x_old[col_to_idx[c]]
        diff = applied_voltage - V_c
        if diff >= -1e-6:
            G = 1.0 / R_matrix[fixed_row, c]
        else:
            G = 1.0 / (rect_ratio * R_matrix[fixed_row, c])
        I_total += diff * G
    
    effective_R = applied_voltage / I_total
    I_direct = applied_voltage / R_matrix[fixed_row, fixed_col]
    I_crosstalk = I_total - I_direct
    
    return effective_R, I_total, I_direct, I_crosstalk

# ====================================
# Main Simulation: Basic Code 5
# ====================================
# In this simulation, we compare the average effective resistance vs. array size
# for both Passive and Diode networks with three pressed pixels placed along a diagonal
# near the center (with one-cell spacing).
avg_eff_R_passive = []
avg_eff_R_diode = []
all_data_passive = {}
all_data_diode = {}

for n in array_sizes:
    # Create an n x n matrix for the Passive network.
    R_matrix_passive = np.full((n, n), resistance_not_pressed)
    pressed_positions = get_pressed_positions(n)
    for pos in pressed_positions:
        R_matrix_passive[pos] = resistance_pressed
    effective_R_mat_passive = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            eff_R, _, _, _ = simulate_measurement(r, c, R_matrix_passive)
            effective_R_mat_passive[r, c] = eff_R
    avg_passive = np.mean(effective_R_mat_passive)
    avg_eff_R_passive.append(avg_passive)
    all_data_passive[f"Passive_n{n}"] = effective_R_mat_passive
    
    # Create an n x n matrix for the Diode network.
    R_matrix_diode = np.full((n, n), resistance_not_pressed)
    for pos in pressed_positions:
        R_matrix_diode[pos] = resistance_pressed
    effective_R_mat_diode = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            eff_R_d, _, _, _ = simulate_measurement_diode(r, c, R_matrix_diode, tol=1e-6, max_iter=100, epsilon=1e-3)
            effective_R_mat_diode[r, c] = eff_R_d
    avg_diode = np.mean(effective_R_mat_diode)
    avg_eff_R_diode.append(avg_diode)
    all_data_diode[f"Diode_n{n}"] = effective_R_mat_diode

# Plot average effective resistance vs. array size.
plt.figure(figsize=(10,6))
plt.plot(array_sizes, avg_eff_R_passive, marker='o', label="Passive Matrix without diode")
plt.plot(array_sizes, avg_eff_R_diode, marker='s', label="Passive Matrix with diode")
plt.xlabel("Array Size (n x n)")
plt.ylabel("Average Effective Resistance (Ω)")
plt.title("Average Effective Resistance vs. Array Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize all array data as subplots.
num_sizes = len(array_sizes)
cols = 5
rows = (num_sizes + cols - 1) // cols

# Passive network visualization.
fig_passive, axs_passive = plt.subplots(rows, cols, figsize=(15, 6))
axs_passive = axs_passive.flatten()
for i, n in enumerate(array_sizes):
    data = all_data_passive[f"Passive_n{n}"]
    im = axs_passive[i].imshow(data, cmap='viridis', origin='lower')
    axs_passive[i].set_title(f"Passive n={n}")
    axs_passive[i].set_xticks([])
    axs_passive[i].set_yticks([])
for j in range(i+1, len(axs_passive)):
    axs_passive[j].axis('off')
fig_passive.suptitle("Passive Matrix without diode", fontsize=16)
fig_passive.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Diode network visualization.
fig_diode, axs_diode = plt.subplots(rows, cols, figsize=(15, 6))
axs_diode = axs_diode.flatten()
for i, n in enumerate(array_sizes):
    data = all_data_diode[f"Diode_n{n}"]
    im = axs_diode[i].imshow(data, cmap='viridis', origin='lower')
    axs_diode[i].set_title(f"Diode n={n}")
    axs_diode[i].set_xticks([])
    axs_diode[i].set_yticks([])
for j in range(i+1, len(axs_diode)):
    axs_diode[j].axis('off')
fig_diode.suptitle("Passive Matrix with diode", fontsize=16)
fig_diode.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Export all results to an Excel file.
results = pd.DataFrame({
    "Array Size": array_sizes,
    "Passive Avg Effective R (Ω)": avg_eff_R_passive,
    "Diode Avg Effective R (Ω)": avg_eff_R_diode
})
save_path = input("Please enter the directory path to save the Excel file: ").strip()
if not os.path.isdir(save_path):
    os.makedirs(save_path)
excel_file_path = os.path.join(save_path, "basic_code5_results.xlsx")
with pd.ExcelWriter(excel_file_path) as writer:
    for sheet_name, data in all_data_passive.items():
        pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)
    for sheet_name, data in all_data_diode.items():
        pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)
    results.to_excel(writer, sheet_name="Summary", index=False)
print(f"Results exported to '{excel_file_path}'.")
