import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis import rms, distances
# Настройки
system_with_ss = {
    'topology': 'ss_enabled.gro',
    'trajectory': 'ss_enabled.xtc',
    'label': 'With SS bonds'
}

system_without_ss = {
    'topology': 'ss_disabled.gro',
    'trajectory': 'ss_disabled.xtc',
    'label': 'Without SS bonds'
}

cys_selection = 'resname CYS and name SG' 
def analyze_disulfide_bonds(ag, cutoff=2.2):
    pairs = ag.atoms.groupby('resids')
    if len(pairs) < 2: return 0
    dist = distances.distance_array(ag.positions, ag.positions)
    return np.sum(dist < cutoff) // 2

def analyze_system(system):
    u = mda.Universe(system['topology'], system['trajectory'])
    protein = u.select_atoms('protein')
    
    # Анализ дисульфидных связей
    cys = u.select_atoms(cys_selection)
    n_ss = [analyze_disulfide_bonds(cys) for _ in u.trajectory]
    
    # Исправленный RMSD анализ
    aligner = rms.RMSD(protein, protein, select='backbone').run()
    rmsd = aligner.results.rmsd.T[2]  # <-- Ключевое исправление
    
    # Исправленный RMSF анализ
    backbone = protein.select_atoms('backbone')
    rmsfer = rms.RMSF(backbone).run()
    
    return {
        'time': aligner.results.rmsd.T[0],
        'rmsd': rmsd,
        'rmsf': rmsfer.results.rmsf,  # <-- Ключевое исправление
        'resids': backbone.resids,
        'n_ss': n_ss,
        'label': system['label']
    }

# Остальная часть скрипта без изменений
results_with = analyze_system(system_with_ss)
results_without = analyze_system(system_without_ss)

# Визуализация результатов
plt.figure(figsize=(15, 10))

# График дисульфидных связей
plt.subplot(2, 2, 1)
plt.hist(results_with['n_ss'], alpha=0.7, label=system_with_ss['label'])
plt.hist(results_without['n_ss'], alpha=0.7, label=system_without_ss['label'])
plt.title('Disulfide Bonds Distribution')
plt.xlabel('Number of SS bonds')
plt.ylabel('Frequency')
plt.legend()

# График RMSD
plt.subplot(2, 2, 2)
plt.plot(results_with['time'], results_with['rmsd'], label=system_with_ss['label'])
plt.plot(results_with['time'], results_without['rmsd'], label=system_without_ss['label'])
plt.title('RMSD Analysis')
plt.xlabel('Time (ps)')
plt.ylabel('RMSD (Å)')
plt.legend()

# График RMSF
plt.subplot(2, 2, 3)
plt.plot(results_with['resids'], results_with['rmsf'], label=system_with_ss['label'])
plt.plot(results_without['resids'], results_without['rmsf'], label=system_without_ss['label'])
plt.title('RMSF Analysis')
plt.xlabel('Residue ID')
plt.ylabel('RMSF (Å)')
plt.legend()

plt.tight_layout()
plt.savefig('comparison1.png')
plt.show()