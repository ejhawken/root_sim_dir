# import matplotlib.pyplot as plt
import tol_colors as tc
color =  tc.tol_cset('muted')
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True

# termaal_caple_data = np.load("thermalcaple_temp_data.npy")
# termaal_caple_time_data = np.load("thermalcaple_time_data.npy")
# print(termaal_caple_data[0])
# expent_data = np.load("y_data.npy")-22.5
# time_data = np.load("x_data.npy")
# std_data = np.load("std.npy")

# def Model(t, a, b):
#    t = t
#    return a * np.exp(-b *t)

# color = tc.tol_cset("muted")


# temp = Model(time_data, 33.8, 0.03)
# import matplotlib.pyplot as plt
# import tol_colors as tc
# import json

# phase = np.load("simulations/sim_temp_motion/output/temp_decay/TE_70_80_3DGRE-Tref_22.5-T_56.4res_0.001_rad_0.02_bc_script_time_decay_T1_1.27_T2_0.171_bc_time_unit_ms_bc_sigma_10000000000.0_shape_rectangle_phase_data.npy")
# uncetanty = np.load("simulations/sim_temp_motion/output/temp_decay/TE_70_80_3DGRE-Tref_22.5-T_56.4res_0.001_rad_0.02_bc_script_time_decay_T1_1.27_T2_0.171_bc_time_unit_ms_bc_sigma_10000000000.0_shape_rectangle_unceranty_data.npy")

# # phase = np.load("simulations/sim_temp_motion/output/temp_decay/Delay_10_20_SE_matching_simulation_copy-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_1.27_T2_0.171_bc_time_unit_ms_bc_sigma_0.00e+00_shape_rectangle_-1.0300000000000001e-08_exp_No_phase_data.npy")
# # uncetanty = np.load("simulations/sim_temp_motion/output/temp_decay/Delay_10_20_SE_matching_simulation_copy-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_1.27_T2_0.171_bc_time_unit_ms_bc_sigma_0.00e+00_shape_rectangle_-1.0300000000000001e-08_exp_No_uncertanty_data.npy")
# def Model(t, a, b):
#    t = t
#    return a * np.exp(-b *t)

# color = tc.tol_cset("muted")
# time = np.linspace(0,140, 101)

# temp = Model(time, 33.8, 0.03)



# plt.figure(figsize=(4, 3))

# # # Plot the data
# # plt.scatter(termaal_caple_time_data, termaal_caple_data, s=4, color=color.green,  label='Thermocouple', marker="^", linewidths=1)
# plt.plot(time, temp, '--', c = color.rose, label = "Eaxt", linewidth = 0.5)
# # plt.errorbar(x=time, y=phase, yerr=uncetanty*3, fmt=".", mfc = color.green, mec = color.green,  ecolor = color.green,  ms = 3, elinewidth=0.1, label = "Simulated",
#             #  capsize=2, markeredgewidth=0.1, alpha = 0.5)
# plt.scatter(time, phase, s=3, color=color.green,  label='Simulated', marker="4", linewidths=0.5)
# plt.fill_between(time, phase + uncetanty*3, phase - uncetanty*3, alpha=0.1, color=color.green)

# # # # Add labels and title
# plt.grid(alpha = 0.5)
# plt.legend()
# plt.xlabel('Time (min)')
# plt.ylabel('Temperture change ($^\circ C$)')
# plt.savefig("simulations/sim_temp_motion/output/Thesis_graphs/Exp_model_2_graphs_non_experiemnta_data_GRE.pdf")


# plt.figure(figsize=(4, 3))
# plt.scatter(time, -phase+temp, s=3, color=color.green,  label='Simulated', marker="4", linewidths=0.5)
# # plt.errorbar(x=time, y=-phase+temp, yerr=uncetanty*3, fmt=".", mfc = color.green, mec = color.green,  ecolor = color.green,  ms = 3, elinewidth=0.5, label = "MRI data",
# #              capsize=4, markeredgewidth=0.5)
# plt.fill_between(time, (-phase+temp) + uncetanty*3, (-phase+temp) - uncetanty*3, alpha=0.1, color=color.green)

# plt.xlabel('Time (min)')
# plt.ylabel('Temperture Difference ($^\circ C$)')
# plt.grid(alpha = 0.5)
# # # Plot the data
# # plt.scatter(termaal_caple_time_data, temp, s=4, color=color.green,  label='Model', marker="^", linewidths=1)
# # plt.scatter(termaal_caple_time_data, (termaal_caple_data)-temp, s=4, color=color.green, marker="4", linewidths=0.1)
# plt.savefig("simulations/sim_temp_motion/output/Thesis_graphs/Exponemtal_model_error_non_experiemnta_data_GRE.pdf")
# # plt.scatter(termaal_caple_time_data, temp-termaal_caple_data, s=4, color=color.green,  label='Model', marker="^", linewidths=1)

# # # Add labels and titl


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# cmps = tc.tol_cmap("sunset")
# def goussen_point(k, sigma, Temp, x, y, z, t):
#     r = np.sqrt(x * x + y * y + z * z)
#     numerator = np.sqrt(np.pi * (2 * sigma)) * Temp
#     denominator = np.sqrt(np.pi * (2 * sigma + 4 * t * k))
#     exponent = -(r * r) / (2 * sigma + 4 * t * k)
#     u = numerator / denominator * np.exp(exponent)
#     return u

# # Parameters
# k = 1
# sigma = 15
# Temp = 10
# t = 0

# # Define the grid
# x = np.linspace(-10, 10, 101)
# y = np.linspace(-10, 10, 101)
# x, y = np.meshgrid(x, y)
# z = np.zeros_like(x)  # Assuming z=0 for a 2D surface plot

# # Compute the function values
# u = goussen_point(k, sigma, Temp, x, y, z, t)

# # Plotting
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, u, cmap=cmps)

# # Add a color bar
# color_bar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# color_bar.set_label('Temperature')

# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_zlabel('z-axis')
# plt.savefig("simulations/sim_temp_motion/output/Thesis_graphs/Gaussen_3d.pdf")
# plt.show()



data1 = "Delay_10_20_SE_matching_simulation_copy-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_2.01744_T2_0.071524_bc_time_unit_ms_bc_sigma_1.56e+00_shape_rectangle_-1.0300000000000001e-08_50"

data2 = 'Delay_10_20_SE_matching_simulation_copy-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_2.01744_T2_0.071524_bc_time_unit_ms_bc_sigma_1.56e+00_shape_rectangle_-1.0300000000000001e-08_100'

data3 = 'Delay_10_20_SE_matching_simulation_copy-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_1.27_T2_0.171_bc_time_unit_ms_bc_sigma_1.56e+00_shape_rectangle_-1.0300000000000001e-08_50'

data4 = 'Delay_10_20_3DSE-Tref_22.5-T_56.3res_0.001_rad_0.02_bc_script_time_decay_T1_2.01744_T2_0.071524_bc_time_unit_ms_bc_sigma_1.56e+00_shape_rectangle_-1.0300000000000001e-08_50'

phase1 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data1}_phase_data.npy")
uncetanty1 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data1}_uncertanty_data.npy")

phase2 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data2}_phase_data.npy")
uncetanty2 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data2}_uncertanty_data.npy")

phase3 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data3}_phase_data.npy")
uncetanty3 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data3}_uncertanty_data.npy")

phase4 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data4}_phase_data.npy")
uncetanty4 = np.load(f"simulations/sim_temp_motion/output/temp_decay/{data4}_uncertanty_data.npy")


diff_Te =  -phase1+phase2
diff_TE_uncert = uncetanty1 + uncetanty2
diff_relax =  -phase1+phase3
diff_relax_uncert = uncetanty1 + uncetanty3
diff_sque =  -phase1+phase4
diff_sque_uncert = uncetanty1 + uncetanty4
time = np.linspace(0,140, 101)
print(len(phase1))


plt.figure(figsize=(4, 3))
plt.scatter(time, diff_Te, c= color.rose, marker="^", s =3, label = 'Different TE',  linewidths= 0.1)  
plt.fill_between(time, diff_Te + diff_TE_uncert, diff_Te - diff_TE_uncert, alpha=0.5, color = color.rose)
plt.scatter(time, diff_sque, c= color.green, marker="4", s =6, label = 'Different sequences', linewidths=0.2)  
plt.fill_between(time, diff_sque + diff_relax_uncert, diff_relax - diff_relax_uncert, alpha=0.09, color = color.green)
plt.scatter(time, diff_relax, c= color.wine, marker="*", s =2, label = 'Different relaxation times', linewidths= 0.1)  
plt.fill_between(time, diff_relax + diff_relax_uncert, diff_relax - diff_relax_uncert, alpha=0.2, color = color.wine)
plt.xlabel('Time (min)')
plt.ylabel('Temperature Difference $(\circ C)$')

plt.legend()
plt.savefig("simulations/sim_temp_motion/output/Thesis_graphs/Comparing_diff_parame.pdf")