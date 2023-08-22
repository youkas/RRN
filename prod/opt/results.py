import numpy as np
import matplotlib.pyplot as plt

def print_results(results):
    print('RRN Results:')
    print(f" {54 * '_'} "
          f"|{35 * chr(8254)}||{35 * chr(8254)}||{92 * chr(8254)}|")
    print(f"|{'Category':54s}|"
          f"|{'Fitness (J)':35s}||{'Ratio (rho_Cd/rho_Cl)':35s}||{'Coefficients (rho_Cl, rho_Cd)':92s}|")
    print(f"|{'Configuration':13}|{'Dimension':9s}|{'Radius':6}|{'Optimization':12}|{'Functional':9}|"
          f"|{r' Z':11s}|{' X':11s}|{' Effective':11s}|"
          f"|{' Z':11s}|{' X':11s}|{' Effective':11s}|"
          f"|{' Z':30s}|{' X':30s}|{' Effective':30s}|")
    print(f"|{54 * '_'}|"
          f"|{35 * '_'}||{35 * '_'}||{92 * '_'}|")
    for k, v in results.items():
        M, C, d, r, O, f = k.split(sep='_')

        print(f"|{C:13}|{d:9s}|{r:6}|{O:12}|{f:10}|"
              f"|{str(v['J_z']) if 'J_z' in v else '-':10s} |{v['J_x']:10f} |{v['effective']['J']:10f} |"
              f"|{str(v['ratio_z']) if 'ratio_z' in v else '-':10s} |{v['ratio_x']:10f} |{v['effective']['ratio']:10f} |"
              f"|{str(v['Y_z'][0]) if 'Y_z' in v else '-':10s}    - {str(v['Y_z'][1]) if 'Y_z' in v else '-':10s}    "
              f"|{v['Y_x'][0]:10f}    - {v['Y_x'][1]:10f}    "
              f"|{v['effective']['Y'][0]:10f}    - {v['effective']['Y'][1]:10f}    |")
        print(f"|{54 * '_'}||{35 * '_'}||{35*'_'}||{92*'_'}|")

def print_robust_results(res, selection):
    for rs in selection:
        results = res[rs]['robust analysis']
        selection = results["selection"]
        plt.scatter(results['J_z' if 'J_z' in results else 'J_x'][:, 0],
                  results['J_z' if 'J_z' in results else 'J_x'][:, 1], s=1, label='X')

        plt.scatter(results['J_z' if 'J_z' in results else 'J_x'][selection, 0],
                  results['J_z' if 'J_z' in results else 'J_x'][selection, 1], s=20, label='X', c='red')
        plt.annotate('Selection', (results['J_z' if 'J_z' in results else 'J_x'][selection, 0],
                                  results['J_z' if 'J_z' in results else 'J_x'][selection, 1]))
        plt.scatter(results['J_det'][0, 0],
                    results['J_det'][0, 1], s=20, label='X', c='red')
        plt.annotate('Determinist', (results['J_det'][0, 0],
                                     results['J_det'][0, 1]))

        plt.xlabel('Mean fitness')
        plt.ylabel('Fitness standard deviation')
        plt.show()

        plt.scatter(results['master_points_mach'][:, 0], results['Y_det_mach'][0], s=1, label='Determinist', c='b')
        plt.scatter(results['master_points_mach'][:, 0], results['Y_selected_mach'][0], s=1, label='Selected', c='r')
        plt.xlabel('Mach number')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        plt.scatter(results['master_points_alpha'][:, 1], results['Y_det_alpha'][0], s=1, label='Determinist', c='b')
        plt.scatter(results['master_points_alpha'][:, 1], results['Y_selected_alpha'][0], s=1, label='Selected', c='r')
        plt.xlabel('Incidence')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

database = np.load(f"../database/results.npz", allow_pickle=True)['item'].item()
print('Configurations:')
for k, v in database['data configuration'].items():
    print(f"{5 * ' '}{k}:")
    for vk, vv in v.items():
        print(f"{10 * ' '}{vk}")
        for vvk, vvv in v.items():
            print(f"{10 * ' '}{vvk}:  {vvv}")

print('Parametrization dimension:')
for k, v in database['parametrization dimension'].items():
    print(f"{5 * ' '}{k}: {v}")

print('Design space radius:')
for k, v in database['design space radius'].items():
    print(f"{5 * ' '}{k}: {v}")

print('Optimization:')
for k, v in database['optimization'].items():
    print(f"{5 * ' '}{k}: {v}")

print('Functional:')
for k, v in database['objectives'].items():
    print(f"{5 * ' '}{k}: {v}")

_results = database["results"]
robuste_selection = ['RRN_D5_2_m_Rob3_J1', 'RRNLike_D5_12_m_Rob3_J1', 'Surrogate_D5_12_m_Rob3_J1']

print_results(_results)
print_robust_results(_results, robuste_selection)