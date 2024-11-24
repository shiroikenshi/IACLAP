# extract_unique_craters.py

"""Extração de Crateras Únicas

Este script contém funções para extrair crateras de predições de modelos
em imagens da Lua, convertendo coordenadas de pixels para longitude/latitude
e filtrando duplicatas.
"""

import numpy as np
import h5py
import utils.crater_detection as cd
import utils.pre_processing as prep
import utils.spatial_transform as st

def load_or_generate_predictions(CP):
    """Carrega ou gera as predições do modelo.

    Parâmetros
    ----------
    CP : dict
        Parâmetros de configuração para os caminhos de dados e modelo.

    Retorna
    -------
    numpy.ndarray
        Predições geradas pelo modelo.
    """
    try:
        with h5py.File(CP['dir_preds'], 'r') as h5f:
            preds = h5f[CP['datatype']][:]
            print("Predições carregadas com sucesso.")
            return preds
    except (OSError, KeyError):
        print("Predições não encontradas. Gerando predições...")
        raise NotImplementedError("Geração de predições não implementada.")

def add_unique_craters(new_craters, unique_craters, longlat_thresh2, rad_thresh):
    """Filtra crateras duplicadas e adiciona crateras únicas à lista mestre.

    Parâmetros
    ----------
    new_craters : numpy.ndarray
        Novas crateras para avaliar e verificar duplicação.
    unique_craters : numpy.ndarray
        Lista existente de crateras únicas.
    longlat_thresh2 : float
        Limite para diferença quadrática de longitude/latitude.
    rad_thresh : float
        Limite para diferença de raio.

    Retorna
    -------
    numpy.ndarray
        Lista atualizada de crateras únicas.
    """
    k2d = 180. / (np.pi * 1737.4) # Conversão de km para graus.
    Long, Lat, Rad = unique_craters.T

    for lo, la, r in new_craters:
        la_mean = (la + Lat) / 2.0
        min_r = np.minimum(r, Rad)

        # Filtragem baseada em distância
        dL = (((Long - lo) / (min_r * k2d / np.cos(np.pi * la_mean / 180.)))**2
              + ((Lat - la) / (min_r * k2d))**2)
        dR = np.abs(Rad - r) / min_r
        is_unique = not np.any((dL < longlat_thresh2) & (dR < rad_thresh))

        if is_unique:
            unique_craters = np.vstack((unique_craters, [lo, la, r]))

    return unique_craters

def estimate_real_world_coordinates(dim, llbd, dist_coeff, coords):
    """Converte coordenadas de pixels para longitude/latitude e raio (em km).

    Parâmetros
    ----------
    dim : tuple
        Dimensões da imagem de entrada (largura, altura).
    llbd : tuple
        Limites de longitude e latitude da imagem.
    dist_coeff : float
        Coeficiente de distorção da projeção.
    coords : numpy.ndarray
        Coordenadas de pixels e raios das crateras.

    Retorna
    -------
    numpy.ndarray
        Coordenadas convertidas em longitude, latitude e raio (em km).
    """
    x_pix, y_pix, r_pix = coords.T
    km_per_pix = 1.0 / st.km_to_pixel(dim[1], llbd[3] - llbd[2], dc=dist_coeff)
    r_km = r_pix * km_per_pix

    deg_per_pix = km_per_pix * 180.0 / (np.pi * 1737.4)
    long_central = np.mean(llbd[:2])
    lat_central = np.mean(llbd[2:])

    lat_diff = (lat_central - deg_per_pix * (y_pix - dim[1] / 2.0))
    lat_deg = lat_central - deg_per_pix * (y_pix - dim[1] / 2.0) * (np.pi * np.abs(lat_diff) / 180.0) / np.sin(np.pi * np.abs(lat_diff) / 180.0)
    long_deg = long_central + deg_per_pix * (x_pix - dim[0] / 2.0) / np.cos(np.pi * lat_deg / 180.0)

    return np.column_stack((long_deg, lat_deg, r_km))

def extract_unique_craters(CP, unique_craters):
    """Extrai crateras únicas de predições do modelo.

    Parâmetros
    ----------
    CP : dict
        Parâmetros de configuração.
    unique_craters : numpy.ndarray
        Array para armazenar crateras únicas.

    Retorna
    -------
    numpy.ndarray
        Array atualizado com crateras únicas.
    """
    preds = load_or_generate_predictions(CP)
    with h5py.File(CP['dir_data'], 'r') as data:
        llbd, distcoeff = data['longlat_bounds'], data['pix_distortion_coefficient']

        for i in range(CP['n_imgs']):
            img_id = prep.get_hdf5_id(i)
            coords = cd.detect_craters_with_template(preds[i])

            if coords.size > 0:
                real_coords = estimate_real_world_coordinates(
                    (CP['dim'], CP['dim']), llbd[img_id], distcoeff[img_id][0], coords)
                unique_craters = add_unique_craters(real_coords, unique_craters, CP['llt2'], CP['rt'])

    np.save(CP['dir_result'], unique_craters)
    return unique_craters