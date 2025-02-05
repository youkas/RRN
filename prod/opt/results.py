import numpy as np
import matplotlib.pyplot as plt

def print_results(results, selection=None):
    print('Results:')
    print(f" {68 * '_'} "
          f"|{35 * chr(8254)}||{35 * chr(8254)}||{92 * chr(8254)}|")
    print(f"|{'Category':68s}|"
          f"|{'J_i':35s}||{' rho_Cd/rho_Cl ':35s}||{'(rho_Cl, rho_Cd)':92s}|")
    print(f"|{'Model':13}|{'Configuration':13}|{'Dimension':9s}|{'Radius':6}|{'Optimization':12}|{'Functional':9}|"
          f"|{r'at Z hat':11s}|{'at  X hat':11s}|{' Effective':11s}|"
          f"|{'at Z hat':11s}|{'at  X hat':11s}|{' Effective':11s}|"
          f"|{'at Z hat':30s}|{'at  X hat':30s}|{' Effective':30s}|")
    print(f"|{68 * '_'}|"
          f"|{35 * '_'}||{35 * '_'}||{92 * '_'}|")
    ids = selection if selection is not None else results.keys()
    for k in ids:
        v = results[k]
        M, C, d, r, O, f = k.split(sep='_')
        print(f"|{M:13}|{C:13}|{d:9s}|{r:6}|{O:12}|{f:10}|"
              f"|{str(v['J_z']) if 'J_z' in v else '-':10s} |{v['J_x']:10f} |{v['effective']['J']:10f} |"
              f"|{str(v['ratio_z']) if 'ratio_z' in v else '-':10s} |{v['ratio_x']:10f} |{v['effective']['ratio']:10f} |"
              f"|({str(v['Y_z'][0]) if 'Y_z' in v else '-':10s}    , {str(v['Y_z'][1]) if 'Y_z' in v else '-':10s})  "
              f"|({v['Y_x'][0]:10f}    , {v['Y_x'][1]:10f})  "
              f"|({v['effective']['Y'][0]:10f}    , {v['effective']['Y'][1]:10f})  |")
        print(f"|{68 * '_'}||{35 * '_'}||{35*'_'}||{92*'_'}|")

def plot_pareto_frontiers(res, ids, selections=None):
    if selections is None:
        selections = ids
    for rs in ids:
        results = res[rs]['robust analysis']

        plt.scatter(results['J_z' if 'J_z' in results else 'J_x'][:, 0],
                    results['J_z' if 'J_z' in results else 'J_x'][:, 1], s=1, label='X', color='b')

        selection = results["selection"]
        if rs in selections:
            plt.scatter(results['J_z' if 'J_z' in results else 'J_x'][selection, 0],
                      results['J_z' if 'J_z' in results else 'J_x'][selection, 1], s=100, marker='*', label='X', c='k')
            plt.annotate('  Random \n   Selection', (results['J_z' if 'J_z' in results else 'J_x'][selection, 0],
                                      results['J_z' if 'J_z' in results else 'J_x'][selection, 1]))

        plt.scatter(results['J_det'][0, 0],
                    results['J_det'][0, 1], s=20, label='X', c='red')
        plt.annotate('  Determinist', (results['J_det'][0, 0],
                                     results['J_det'][0, 1]))

        plt.xlabel('Mean fitness')
        plt.ylabel('Fitness standard deviation')
        plt.show()

def plot_robust_performance(res, selection):
    for rs in selection:
        results = res[rs]['robust analysis']
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
DETERMINIST = ['RRN_D5_2_s_Det_J1', 'kRRN_D5_2_s_Det_J1',
               'RRN_D5_2_m_Det_J1', 'kRRN_D5_2_m_Det_J1',
               'RRN_D5_2_l_Det_J1', 'kRRN_D5_2_l_Det_J1',
               'RRN_D5_2_s_Det_J2', 'kRRN_D5_2_s_Det_J2',
               'RRN_D5_2_m_Det_J2', 'kRRN_D5_2_m_Det_J2',
               'RRN_D5_2_l_Det_J2', 'kRRN_D5_2_l_Det_J2']
ROBUST = ['RRN_D5_2_l_Rob1_J1','kRRN_D5_2_l_Rob1_J1',
          'RRN_D5_2_l_Rob2_J1','kRRN_D5_2_l_Rob2_J1',
          'RRN_D5_2_l_Rob3_J1', 'kRRN_D5_2_l_Rob3_J1']
ROBUST_SHAPE = ['RRN_D5_2_l_Rob3_J1', 'kRRN_D5_2_l_Rob3_J1']

print_results(_results, DETERMINIST)
plot_pareto_frontiers(_results, ROBUST, ROBUST_SHAPE)
print_results(_results, ROBUST_SHAPE)
plot_robust_performance(_results, ROBUST_SHAPE)