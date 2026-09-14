import matplotlib.pyplot as plt
import numpy as np
import damask
import scipy
import pyvista as pv
from matplotlib.ticker import AutoMinorLocator
import matflow as mf
import sklearn
import skimage
import pyvoro2
from tqdm import tqdm
import pandas as pd

### Plot settings ###
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

### 1. Microstructure construction ###

def generate_seed_locations(n_grains, VE_size):
       
    seed_locations = (np.random.random_sample(size=(n_grains, len(VE_size)))*VE_size).tolist()

    return seed_locations

def generate_orientations(uvw, hkl, lattice, n_grains, sigma):

    center = damask.Orientation.from_directions(uvw=uvw, hkl=hkl, lattice=lattice)
    orientations = damask.Orientation.from_spherical_component(center=center, shape=n_grains, degrees=True, sigma=sigma, lattice=lattice).as_quaternion()

    return orientations

def save_microstructure_seeds(seed_locations,orientations,fname):

    lines = []

    for i in range(len(seed_locations)):

        lines.append([*seed_locations[i],*orientations[i]])

    lines = np.array(lines)

    np.savetxt(X=lines,fname=fname,delimiter=',')

### 2. Changing values ###

def extract_material_IDs(workflow):

    wk = mf.Workflow(workflow)
    material_IDs = wk.tasks.generate_volume_element_from_voronoi.elements[0].outputs.volume_element.value['element_material_idx']
    material_IDs_array = np.array(material_IDs)

    return material_IDs_array

def grain_averaged_values(val_array, material_IDs, no_grains):

    grain_averaged_vals = []
    for i in range(0,no_grains):
        material_IDs_grain = np.where(material_IDs.flatten()== i)
        grain_vals = []
        for j in range(len(material_IDs_grain)):
            grain_vals.append(val_array.flatten()[material_IDs_grain[j]])

        grain_averaged_vals.append(np.mean(grain_vals))

    return grain_averaged_vals

def max_val_from_grains(val_array, material_IDs, no_grains):

    max_vals_from_grains = []

    for i in range(0,no_grains):
        material_IDs_grain = np.where(material_IDs.flatten()==i)
        grain_vals = []
        for j in range(len(material_IDs_grain)):
            grain_vals.append(val_array.flatten()[material_IDs_grain[j]])

        max_vals_from_grains.append(np.max(grain_vals))

    return max_vals_from_grains

def kernel_smoothed_3x3x3_values(val_array):
    val_array_3x3x3_avg = scipy.ndimage.convolve(val_array, weights = [[[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]],
                                                                       [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]],
                                                                       [[1/27,1/27,1/27], [1/27,1/27,1/27], [1/27,1/27,1/27]]])
    return val_array_3x3x3_avg

### 3. Extract data from CP simulations ###

# Load experimental data from CSV file #

def load_exp_stress_strain_data(exp_data_file, true_data = True):
    """
    Column headers in csv must be 'strain' and 'stress' to be extractable
    """

    if true_data == False:
        exp_data = pd.read_csv(exp_data_file)
        exp_eng_strain = exp_data['strain']
        exp_eng_stress = exp_data['stress']

        exp_true_strain = np.log(1 + exp_eng_strain)
        exp_true_stress = exp_eng_stress*(1 + exp_eng_strain)

    elif true_data == True:
        exp_data = pd.read_csv(exp_data_file)
        exp_true_strain = exp_data['strain']
        exp_true_stress = exp_data['stress']
    
    return exp_true_strain, exp_true_stress

# 3a. Stress-strain data from MatFlow #

def extract_stress_strain(workflow):

    wk = mf.Workflow(workflow)

    elements = wk.tasks.simulate_VE_loading_damask.elements
    true_stress_tensor = np.array(elements[0].get("outputs.VE_response.phase_data.vol_avg_stress.data"))
    true_strain_tensor = np.array(elements[0].get("outputs.VE_response.phase_data.vol_avg_strain.data"))

    true_stress = true_stress_tensor[:,0,0]*1e-6
    true_strain = true_strain_tensor[:,0,0]

    return true_strain, true_stress 

# 3b. Extract data from DAMASK output file #

class extractDAMASKdata:

    def extract_strain(damask_file, rve_shape, direction = 'xx'):
        """
        Extract strain component from crystal plasticity simulation

        direction = {'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'}

        """

        result = damask.Result(damask_file)

        x = result.view(increments = result.increments[-1])
        strain = x.get('epsilon_V^0(F)')
        if direction == 'xx':
            strain_comp = strain[:,0,0]
        elif direction == 'xy':
            strain_comp = strain[:,0,1]
        elif direction == 'xz':
            strain_comp = strain[:,0,2]
        elif direction == 'yx':
            strain_comp = strain[:,1,0]
        elif direction == 'yy':
            strain_comp = strain[:,1,1]
        elif direction == 'yz':
            strain_comp = strain[:,1,2]
        elif direction == 'zx':
            strain_comp = strain[:,2,0]
        elif direction == 'zy':
            strain_comp = strain[:,2,1]
        elif direction == 'zz':
            strain_comp = strain[:,2,2]

        strain_map = strain_comp.reshape(rve_shape,order='F')

        return strain_map

    def extract_vM_strain(damask_file, rve_shape):

        result = damask.Result(damask_file)

        x = result.view(increments = result.increments[-1])
        strain = x.get('epsilon_V^0(F)')
        equiv_vM_strain = damask.mechanics.equivalent_strain_Mises(strain)
        equiv_vM_strain_map = equiv_vM_strain.reshape(rve_shape,order='F')

        return equiv_vM_strain_map

    def extract_stress_triaxiality(damask_file, rve_shape):

        result = damask.Result(damask_file)

        x = result.view(increments = result.increments[-1])

        try:
            x.add_spherical('sigma')
            x.add_equivalent_Mises('sigma')
        
        except:
            pass

        hydrostatic_stress = x.place('p_sigma')
        vM_stress = x.place('sigma_vM')

        hydrostatic_stress_map = hydrostatic_stress.reshape(rve_shape,order='F')
        vM_stress_map = vM_stress.reshape(rve_shape,order='F')

        triaxiality = hydrostatic_stress_map/vM_stress_map

        return triaxiality

    def extract_r_values(damask_file, number_of_incs):
        result = damask.Result(damask_file)

        r_vals = []
        equiv_vM_strains = []

        for increment in tqdm.tqdm(result.increments[::(len(result.increments)/number_of_incs)]):

            x = result.view(increments = increment)

            try:
                x.add_strain('F_p','U')
                x.add_stretch_tensor('F_p',t='U')
            except:
                pass

            plastic_strain_tensor = x.get('epsilon_U^0.0(F_p)')

            # Calculating r-value
            r_value = (np.average(plastic_strain_tensor[:,2,2]))/(np.average(plastic_strain_tensor[:,1,1]))

            # Calculating plastic vM strain
            epsilon_F_p_vM = damask.mechanics.equivalent_strain_Mises(plastic_strain_tensor)
            epsilon_F_p_vM_avg = np.average(epsilon_F_p_vM)

            r_vals.append(r_value)
            equiv_vM_strains.append(epsilon_F_p_vM_avg)

        return equiv_vM_strains, r_vals

    def extract_slip_systems(self, material_file):
        mat = damask.ConfigMaterial.load(material_file)
        crystal_structure = damask.Crystal(lattice = mat['phase']['Cu']['lattice'])
        slip_dirs = crystal_structure.kinematics('slip')['direction'][0]
        slip_norms = crystal_structure.kinematics('slip')['plane'][0]

        return slip_dirs, slip_norms

    def update_slip_systems(self, damask_view, slip_dirs, slip_norms, rve_shape):

        rot_matrices = damask.Rotation(damask_view.get('O')).reshape(rve_shape, order = 'F').as_matrix()

        slip_dirs = np.broadcast_to(slip_dirs, [128,128,128,12,3])
        slip_norms = np.broadcast_to(slip_norms, [128,128,128,12,3])

        new_slip_dirs = np.einsum('...ij, ...ki -> ...kj', rot_matrices, slip_dirs)
        new_slip_norms = np.einsum('...ij, ...ki -> ...kj', rot_matrices, slip_norms)

        #rot_matrices = damask.Rotation(damask_view.get('O')).reshape(rve_shape, order = 'F').as_matrix()

        #all_new_slip_dirs = []
        #all_new_slip_norms = []

        #for i in range(len(rot_matrices)):
        #    new_slip_dirs = []
        #    new_slip_norms = []
        #    for j in range(len(slip_dirs)):
        #        new_slip_dir = rot_matrices[i] @ slip_dirs[j]
        #        new_slip_norm = rot_matrices[i] @ slip_norms[j]
        #        new_slip_dirs.append(new_slip_dir)
        #        new_slip_norms.append(new_slip_norm)
            
        #    all_new_slip_dirs.append(new_slip_dirs)
        #    all_new_slip_norms.append(new_slip_norms)

        return new_slip_dirs, new_slip_norms

    def extract_accum_plastic_strain_energy_density(self, damask_file, mat_file, rve_shape):
        result = damask.Result(damask_file)

        slip_dirs, slip_norms = self.extract_slip_systems(mat_file)

        total_plastic_work_arrs = []

        for i in tqdm(range(len(result.increments))):
            x = result.view(increments = result.increments[i])
            gamma = x.get('gamma_sl').reshape([rve_shape[0], rve_shape[1], rve_shape[2], 12], order = 'F')#rve_shape.append(12), order = 'F')

            if i == 0:
                #new_slip_dirs, new_slip_norms = slip_dirs, slip_norms
                new_slip_dirs = np.broadcast_to(slip_dirs, [128,128,128,12,3])
                new_slip_norms = np.broadcast_to(slip_norms, [128,128,128,12,3])

            else:
                new_slip_dirs, new_slip_norms = self.update_slip_systems(x, slip_dirs, slip_norms, rve_shape)

            sigma = x.get('sigma').reshape([rve_shape[0], rve_shape[1], rve_shape[2], 3, 3], order = 'F')

            # Calculate resolved shear stresses
            #tau = np.einsum('vij,vsi,vsj->vs', sigma, new_slip_dirs, new_slip_norms)
            tau = np.einsum('...ij, ...si, ...sj -> ...s', sigma, new_slip_dirs, new_slip_norms)

            plastic_work = np.abs(np.multiply(tau, gamma))

            total_plastic_work_per_inc = np.sum(plastic_work, axis=3)

            if i == 0:
                total_plastic_work_arrs.append(total_plastic_work_per_inc)
            else:
                total_plastic_work_current_time_inc = total_plastic_work_per_inc - total_plastic_work_arrs[i-1]
                total_plastic_work_arrs.append(total_plastic_work_current_time_inc)

        total_plastic_work_arrs = np.array(total_plastic_work_arrs) # Should be shape [12,128,128,128]  

        accum_plastic_work = np.sum(total_plastic_work_arrs, axis = 0)

        return total_plastic_work_arrs, accum_plastic_work

### 4. Extracting grain boundaries ###

#def extract_voxelised_GBs():

#    return

def extract_triple_points_from_voxelised_coords(material_IDs, voxel_size):
    triple_point_locations = []

    for i in range(material_IDs.shape[0]-1):
        for j in range(material_IDs.shape[1]-1):
            for k in range(material_IDs.shape[2]-1):
                region = [material_IDs[i+1][j+1][k], material_IDs[i+1][j+1][k+1], material_IDs[i+1][j+1][k-1],
                          material_IDs[i+1][j][k], material_IDs[i+1][j][k+1], material_IDs[i+1][j][k-1],
                          material_IDs[i+1][j-1][k], material_IDs[i+1][j-1][k+1], material_IDs[i+1][j-1][k-1], 

                          material_IDs[i][j+1][k], material_IDs[i][j+1][k+1], material_IDs[i][j+1][k-1],
                          material_IDs[i][j][k], material_IDs[i][j][k+1], material_IDs[i][j][k-1],
                          material_IDs[i][j-1][k], material_IDs[i][j-1][k+1], material_IDs[i][j-1][k-1],

                          material_IDs[i-1][j+1][k], material_IDs[i-1][j+1][k+1], material_IDs[i-1][j+1][k-1],
                          material_IDs[i-1][j][k], material_IDs[i-1][j][k+1], material_IDs[i-1][j][k-1],
                          material_IDs[i-1][j-1][k], material_IDs[i-1][j-1][k+1], material_IDs[i-1][j-1][k-1], 
                          ]
            if len(np.unique(region)) == 3 and all([list(region).count(n) for n in set(region)][i] > 6 for i in range(3)):#(np.unique(region)[i] > 5 for i in range(3)):
                triple_point_locations.append([i,j,k])

    triple_point_locations = np.array(triple_point_locations)

    triple_point_coords = np.empty(triple_point_locations.shape)
    for i in range(len(triple_point_locations)):
        triple_point_coords[i] = [(triple_point_locations[i][0] + 0.5)*voxel_size, 
                                  (triple_point_locations[i][1] + 0.5)*voxel_size, 
                                  (triple_point_locations[i][2] + 0.5)*voxel_size
                                 ]
    
    return triple_point_coords

def extract_voxel_coords(origin = [0.0,0.0,0.0], size = [1.0e-3, 1.0e-3, 1.0e-3], voxels = [128,128,128]):
    start = np.array(origin)         + (np.array(size)/np.array(voxels))*0.5 # Add voxel centre offset
    end = np.array(origin) + np.array(size)   - (np.array(size)/np.array(voxels))*0.5 # Add voxel offset
    
    voxel_coords = np.stack(np.meshgrid(np.linspace(start[0],end[0],voxels[0]),
                                           np.linspace(start[1],end[1],voxels[1]),
                                           np.linspace(start[2],end[2],voxels[2]),
                                           indexing='ij'),
                                           axis=-1)
    
    return voxel_coords

def min_distance_from_voxels_to_nearest_GB_continuous(voronoi_seeds, voxel_coords):
# seeds: (N, 3) array of Voronoi cell generator centers
# voxel_coords: (2097152, 3) array of voxel centers for 128^3 VE
    tree = scipy.spatial.cKDTree(voronoi_seeds)

# Find the 1st nearest seed (cell owner) and k-1 nearest neighbor seeds
# k=20 covers virtually all 3D Voronoi neighbor adjacencies
    k = 20
    distances, indices = tree.query(voxel_coords, k=k)

# r1: distance to the voxel's own cell seed (2097152, 1)
    r1 = distances[:, 0:1]
    s1_coords = voronoi_seeds[indices[:, 0]]  # (2097152, 3)

# r_j: distances to neighboring seeds (2097152, k-1)
    rj = distances[:, 1:]
    sj_coords = voronoi_seeds[indices[:, 1:]]  # (2097152, k-1, 3)

# Distance between seed 1 and neighbor seeds S_j
    seed_distances = np.linalg.norm(sj_coords - s1_coords[:, np.newaxis, :], axis=2)

# Distance to each bisecting plane: d = (r_j^2 - r_1^2) / (2 * ||S_j - S_1||)
    plane_distances = (rj**2 - r1**2) / (2.0 * seed_distances)

# The shortest distance across all neighbor planes is the exact boundary distance
    min_distance_from_GBs = np.min(plane_distances, axis=1)

    return min_distance_from_GBs


### 5. Statistical analysis ###

def statistical_comparison_metrics(vals_1, vals_2):
    max_val = max(max(vals_1.flatten(order = 'F')), max(vals_2.flatten(order = 'F')))
    min_val = min(min(vals_1.flatten(order = 'F')), min(vals_2.flatten(order = 'F')))

    nrmse_pheno_norm = np.sqrt(np.mean((vals_1 - vals_2)**2))/np.mean(np.abs(vals_1))
    nrmse_dislo_norm = np.sqrt(np.mean((vals_1 - vals_2)**2))/np.mean(np.abs(vals_2))
    ssim_score = skimage.metrics.structural_similarity(vals_1, vals_2, data_range = max_val - min_val)
    corr_coeff = np.corrcoef(vals_1.ravel(), vals_2.ravel())[0,1]
    #mean_err = np.abs(np.mean(vals_1.flatten()) - np.mean(vals_2.flatten()))/(np.mean(vals_1.flatten()))
    print(f'NRMSE (val 1 normalised): {nrmse_pheno_norm:.3f} \nNRMSE (val 2 normalised): {nrmse_dislo_norm:.3f} \nSSIM: {ssim_score:.3f} \nCorrelation coefficient: {corr_coeff:.3f}')# \nMean error: {mean_err:.5f}')


### 6. Plots ### 

# 6a. Stress-strain #

def plot_stress_strain(true_strain, true_stress, model_type, savefig = False, fig_name = None):

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(true_strain, true_stress, label = model_type)
    ax.legend(fontsize = 20)
    ax.set_ylabel('True Stress (MPa)', fontsize=16)
    ax.set_xlabel('True Strain', fontsize=16)
    ax.tick_params(labelsize = 16)
    ax.set_ylim(0)
    ax.set_xlim(0, max(true_strain))

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

def plot_LHS_sample_stress_strain_curves(strain_incs, outputs, model_type, savefig=False, fig_name=None, plot_exp_data=False, exp_data_file=None, true_data=True, title_fontsize=24, label_fontsize=24, tick_label_fontsize=20):

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(outputs)):
        stress_vals = outputs[i]
        ax.plot(strain_incs, stress_vals, alpha=0.2)

    #plt.plot(exp_true_strain, exp_true_stress, color = 'k', alpha=1)
    ax.set_title(f'{model_type}', fontsize = title_fontsize)
    ax.set_ylabel('True stress, $\sigma_{xx}$ (MPa)', fontsize=label_fontsize)
    ax.set_xlabel('True strain, $\epsilon_{xx}$', fontsize=label_fontsize)
    ax.tick_params(labelsize = tick_label_fontsize, direction = 'in', bottom = True, top = True, left = True, right = True)
    ax.set_ylim(0)
    ax.set_xlim(0, max(strain_incs))

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

    if plot_exp_data==True:
        if true_data == True:
            exp_true_strain, exp_true_stress = load_exp_stress_strain_data(exp_data_file, true_data=True)

        elif true_data == False:
            exp_true_strain, exp_true_stress = load_exp_stress_strain_data(exp_data_file, true_data=False)
        
        ax.plot(exp_true_strain, exp_true_stress, label = 'Experimental', color = 'k', alpha = 1)
    
    return None

# 6b. Comparison 2D maps #

def comparison_map(data1, data2, data1_label, data2_label, color_bar_label, savefig = False, fig_name = None):    
    fig, ax = plt.subplots(1,2, figsize = (10,6), sharey = True, sharex=True, constrained_layout=True)
    im_1 = ax[0].imshow(data1, vmin = min(min(value) for value in (data1.flatten(), data2.flatten())), vmax = max(max(value) for value in(data1.flatten(), data2.flatten())))
    im_2 = ax[1].imshow(data2, vmin = min(min(value) for value in (data1.flatten(), data2.flatten())), vmax = max(max(value) for value in (data1.flatten(), data2.flatten())))
    cbar = fig.colorbar(im_2, orientation = 'vertical', shrink = 0.73)
    cbar.set_label(color_bar_label, rotation = 90, fontsize = 16, math_fontfamily = 'cm')
    ax[0].set_title(f'{data1_label}', fontsize = 16)
    ax[1].set_title(f'{data2_label}', fontsize = 16)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

def triple_comparison_map(data1, data2, data3, data1_label, data2_label, data3_label, color_bar_label, savefig = False, fig_name = None):    
    fig, ax = plt.subplots(1,3, figsize = (10,6), sharey = True, sharex=True, constrained_layout=True)
    im_1 = ax[0].imshow(data1, vmin = min(min(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())), vmax = max(max(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())))
    im_2 = ax[1].imshow(data2, vmin = min(min(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())), vmax = max(max(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())))
    im_3 = ax[2].imshow(data2, vmin = min(min(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())), vmax = max(max(value) for value in (data1.flatten(), data2.flatten(), data3.flatten())))
    cbar = fig.colorbar(im_3, orientation = 'vertical', shrink = 0.5)
    cbar.set_label(color_bar_label, rotation = 90, fontsize = 16, math_fontfamily = 'cm')
    ax[0].set_title(f'{data1_label}', fontsize = 16)
    ax[1].set_title(f'{data2_label}', fontsize = 16)
    ax[2].set_title(f'{data3_label}', fontsize = 16)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

# 6c. Comparison plots between CP model results #

def comparison_plot(data_1, data_2, xlabel, ylabel, 
                    diag_line_start = None,
                    min_interval = None, 
                    max_interval = None, 
                    add_r2_score = False, r2_score = None, annotation_pos = None,
                    savefig = False, fig_name = None):


    fig, ax = plt.subplots(1,1, figsize = (10,10))

    if diag_line_start==None:
        diag_line_start = [min(min(data_1), min(data_2)) - 0.1*np.abs(min(min(data_1), min(data_2))), 
                           min(min(data_1), min(data_2)) - 0.1*np.abs(min(min(data_1), min(data_2)))]
        
    if min_interval==None:
        min_interval =  min(min(data_1), min(data_2)) - 0.1*np.abs(min(min(data_1), min(data_2)))

    if max_interval==None:
        max_interval =  max(max(data_1), max(data_2)) + 0.05*np.abs(max(max(data_1), max(data_2)))

    ax.plot(data_1, data_2, 'ko', ls = 'None', label = 'CP model predictions', alpha = 0.1)
    ax.axline(diag_line_start, slope = 1, color = 'r', label = 'Equal predictions')
  
    ax.set_xlabel(fr'{xlabel}', fontsize = 24, math_fontfamily = 'cm')
    ax.set_ylabel(fr'{ylabel}', fontsize = 24, math_fontfamily = 'cm')
    ax.set_xlim(min_interval, max_interval)
    ax.set_ylim(min_interval, max_interval)
    ax.tick_params(which = 'both', direction = 'in', top = True, right = True, labelsize = 20)
    ax.tick_params(which = 'minor', length = 4)
    ax.tick_params(which = 'major', length = 8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    if add_r2_score == True:
        ax.annotate(f'$r^2$ = {r2_score:.3f}', xy = annotation_pos, fontsize = 16, math_fontfamily = 'cm')
    ax.legend(shadow = True, fontsize  = 16)

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

def comparison_violin_plot(data_1, data_2, label_1, label_2, title, qoi, savefig = False, fig_name = None):
    all_data = [data_1, data_2]

    fig_violin, ax_violin = plt.subplots(1,1,figsize = (8,8))

    vplot = ax_violin.violinplot(all_data, showmedians = True)

    colors = ["mediumorchid", "green"]
    
    for i, vp in enumerate(vplot['bodies']):    
        vp.set_facecolor(colors[i])    
        vp.set_edgecolor(colors[i])
    
    for key in ["cmaxes", "cmins", "cmedians", "cbars"]: 
        vplot[key].set_color(colors)

    ax_violin.set_title(f'{title}', fontsize = 24)
    ax_violin.set_xticks(ticks = [y+1 for y in range(len(all_data))], labels = [label_1, label_2])
    ax_violin.tick_params(axis = 'both', labelsize = 20)
    ax_violin.set_xlabel('CP Model', fontsize = 24)
    ax_violin.set_ylabel(fr'{qoi}', math_fontfamily = 'cm', fontsize = 24)

    if savefig==True:
        plt.savefig(fig_name, dpi=800)
    

def comparison_plot_distance_to_GB(min_distance_from_foi, normalisation, qoi_array_1, qoi_array_2, qoi_label, title_1 = "Phenomenological CP", title_2 = "Dislocation Density CP", savefig = False, fig_name = None):

    fig, ax = plt.subplots(1, 2, figsize=(20,6), sharey= True, sharex = True, constrained_layout = True)
    ax[0].scatter(min_distance_from_foi/(normalisation), qoi_array_1, color = 'mediumorchid', alpha = 0.1)
    ax[1].scatter(min_distance_from_foi/(normalisation), qoi_array_2, color = 'green', alpha = 0.1)

    ax[0].set_title(title_1, fontsize = 24)
    ax[1].set_title(title_2, fontsize = 24)
    ax[0].set_xlabel("Distance to nearest grain boundary (number of voxels)", fontsize = 20)
    ax[1].set_xlabel("Distance to nearest grain boundary (number of voxels)", fontsize = 20)
    ax[0].set_ylabel(qoi_label, fontsize = 20, math_fontfamily = 'cm')

    ax[0].tick_params(which = 'both', direction = 'in', top = True, right = True, labelsize = 18)
    ax[0].tick_params(which = 'minor', length = 4)
    ax[0].tick_params(which = 'major', length = 8)
    ax[1].tick_params(which = 'both', direction = 'in', top = True, right = True, labelsize = 18)
    ax[1].tick_params(which = 'minor', length = 4)
    ax[1].tick_params(which = 'major', length = 8)

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))

    if savefig==True:
        plt.savefig(fig_name, dpi=800)

# 6d. Surrogate model plots

#def comparison_graph():