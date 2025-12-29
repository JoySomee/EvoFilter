import numpy as np
import pickle

def get_n_list(material_nk_low, material_nk_high, material_nk_substrat, wavelength_list, layer_num=20, output_path='nk_map_special.pkl'):
    nk_map = {}
    for wavelength in wavelength_list:
        nk_list = []
        for i in range(layer_num):
            if i % 2 == 0:
                nk_list.append(material_nk_low(wavelength))
            else:
                nk_list.append(material_nk_high(wavelength))
        nk_list.append(material_nk_substrat(wavelength))
        nk_map[wavelength] = nk_list.copy()
        print(len(nk_list))
    with open(output_path, 'wb') as f:
        pickle.dump(nk_map, f)

def cal_nk_SiO2(wavelength_nm):
    """
    使用 Sellmeier 公式计算 SiO2 在给定波长（nm）下的折射率。
    
    参数:
        wavelength_nm : float 或 array-like，波长，单位为纳米 (nm)
    
    返回:
        n : 对应波长下的折射率（实数，假设无吸收）
    """
    # 将波长从 nm 转换为 μm（Sellmeier 公式中 λ 单位为微米）
    lam = np.array(wavelength_nm) / 1000.0  # nm → μm

    # Sellmeier 系数（来源：中国光学玻璃手册 / refractiveindex.info）
    B1 = 1.07044083
    B2 = 1.10202242
    C1 = 1.00585997e-2   # μm²
    C2 = 100   # μm²

    # Sellmeier 公式：n^2 = 1 + Σ [Bi * λ^2 / (λ^2 - Ci)]
    n_squared = (
        1 +
        0.28604141 + 
        B1 * lam**2 / (lam**2 - C1) +
        B2 * lam**2 / (lam**2 - C2)
    )

    # 避免因数值误差导致负值（理论上 n² > 1）
    n = np.sqrt(np.clip(n_squared, a_min=1.0, a_max=None))
    return n

def cal_nk_TiO2(wavelength_nm):
    """
    使用 Sellmeier 公式计算 K9 玻璃在给定波长（nm）下的折射率。
    
    参数:
        wavelength_nm : float 或 array-like，波长，单位为纳米 (nm)
    
    返回:
        n : 对应波长下的折射率（实数，假设无吸收）
    """
    # 将波长从 nm 转换为 μm（Sellmeier 公式中 λ 单位为微米）
    lam = np.array(wavelength_nm) / 1000.0  # nm → μm

    # Sellmeier 系数（来源：中国光学玻璃手册 / refractiveindex.info）
    B1 = 0.2441
    C1 = 0.0803   # μm²

    # Sellmeier 公式：n^2 = 1 + Σ [Bi * λ^2 / (λ^2 - Ci)]
    n_squared = (
        5.913 + 
        B1 / (lam**2 - C1)
    )

    # 避免因数值误差导致负值（理论上 n² > 1）
    n = np.sqrt(np.clip(n_squared, a_min=1.0, a_max=None))
    return n

def cal_nk_K9(wavelength_nm):
    """
    使用 Sellmeier 公式计算 K9 玻璃在给定波长（nm）下的折射率。
    
    参数:
        wavelength_nm : float 或 array-like，波长，单位为纳米 (nm)
    
    返回:
        n : 对应波长下的折射率（实数，假设无吸收）
    """
    # 将波长从 nm 转换为 μm（Sellmeier 公式中 λ 单位为微米）
    lam = np.array(wavelength_nm) / 1000.0  # nm → μm

    # K9 玻璃的 Sellmeier 系数（来源：中国光学玻璃手册 / refractiveindex.info）
    B1 = 1.183185030
    B2 = 0.087175643
    B3 = 1.031337010
    C1 = 0.007221420
    C2 = 0.0268216805
    C3  =101.70236200
    # Sellmeier 公式：n^2 = 1 + Σ [Bi * λ^2 / (λ^2 - Ci)]
    n_squared = (
        1 +
        B1 * lam**2 / (lam**2 - C1) +
        B2 * lam**2 / (lam**2 - C2) +
        B3 * lam**2 / (lam**2 - C3)
    )

    # 避免因数值误差导致负值（理论上 n² > 1）
    n = np.sqrt(np.clip(n_squared, a_min=1.0, a_max=None))
    return n

if __name__ == '__main__':
    wavelength_list = np.round(np.linspace(453.3, 648.1, 1949), 1)
    # get_n_list(cal_nk_SiO2, cal_nk_TiO2, cal_nk_K9, wavelength_list)
    with open('./nk_map.pkl', 'rb') as f:
        nk_list_map = pickle.load(f)
    print(len(nk_list_map[648]))