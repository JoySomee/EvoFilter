from tmm_core import coh_tmm
import numpy as np
import pickle
from initial import FilterInitializer
import matplotlib.pyplot as plt

def get_n_list(material_nk_low, material_nk_high, material_nk_substrat, layer_num=20, output_path='nk_map.pkl'):
    nk_map = {}
    nk_list = []
    wavelength_list = np.linspace(400, 700, 301)
    for wavelength in wavelength_list:
        for i in range(layer_num):
            if i % 2 == 0:
                nk_list.append(material_nk_low[int(wavelength) - 400, 1])
            else:
                nk_list.append(material_nk_high[int(wavelength) - 400, 1])
        nk_list.append(material_nk_substrat[int(wavelength) - 400, 1])
        nk_map[wavelength] = nk_list.copy()
    with open(output_path, 'wb') as f:
        pickle.dump(nk_map, f)

def cal_response(d_list, nk_path='nk_map.pkl'):
    with open(nk_path, 'rb') as f:
        nk_list_map = pickle.load(f)
    lambda_list = np.linspace(400, 700, 301) #in nm 
    T_list = []
    for lambda_vac in lambda_list:
        n_list = nk_list_map[lambda_vac].copy()
        n_list.append(1)
        n_list.insert(0, 1)
        T_list.append(coh_tmm('s', n_list, d_list, 0, lambda_vac)['T'])
    return lambda_list, T_list

def cal_response_batch(thick_designs, nk_path='nk_map.pkl'):
    num_designs = thick_designs.shape[0]
    inf_col = np.full((num_designs, 1), np.inf)
    substrate_col = np.full((num_designs, 1), 500)
    d_lists = np.hstack([inf_col, thick_designs, substrate_col, inf_col])
    T_lists = []
    for _, d_list_numpy in enumerate(d_lists):
        d_list = d_list_numpy.tolist()
        _, T_list = cal_response(d_list, nk_path)
        T_lists.append(T_list)
    return np.array(T_lists)

if __name__ == '__main__':
    initializer = FilterInitializer()
    # Generate a 2D array of thickness designs
    thick_designs, labels = initializer.generate_individual_4x4()
    # Prepare the full layer structure for each design
    num_designs = thick_designs.shape[0]
    inf_col = np.full((num_designs, 1), np.inf)
    substrate_col = np.full((num_designs, 1), 500)
    # d_lists is now a 2D array where each row is a full layer stack design
    d_lists = np.hstack([inf_col, thick_designs, substrate_col, inf_col])
    # Create a 4x4 grid of subplots for the 16 designs
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    # Calculate and plot the response for each design
    for i, d_list_numpy in enumerate(d_lists):
        if i >= len(axes):
            print(f"Skipping design {i+1} as it exceeds the number of subplots.")
            continue
        print(f"Calculating for design {i+1}/{num_designs}...")
        d_list = d_list_numpy.tolist() # cal_response expects a list
        lambda_list, T_list = cal_response(d_list)
        
        ax = axes[i]
        ax.plot(lambda_list, T_list)
        ax.set_title(f'{labels[i//4, i%4]}')
        ax.grid(True)

    # Hide any unused subplots if there are fewer than 16 designs
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add common labels and a main title
    fig.supxlabel("Wavelength (nm)")
    fig.supylabel("Transmittance")
    fig.suptitle("Transmittance vs. Wavelength for Multiple Designs", fontsize=16)

    # Adjust layout to prevent titles/labels overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("transmittance_vs_wavelength.png")
    print(labels)
    