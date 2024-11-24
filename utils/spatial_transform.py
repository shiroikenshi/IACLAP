# utils/spatial_transform.py

"""Funções de Transformação de Coordenadas

Utilizado em: funções de dataset_generator.py.
"""

########## Importações ##########

import numpy as np

########## Funções de Projeção de Coordenadas para Pixels ##########

def geocoord_to_pixel(cx, cy, cdim, imgdim, origin="upper"):
    """Converte coordenadas geográficas em posições de pixel na imagem.

    Parâmetros
    ----------
    cx : float ou ndarray
        Coordenada x.
    cy : float ou ndarray
        Coordenada y.
    cdim : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da imagem.
    imgdim : list, tuple ou ndarray
        Comprimento e altura da imagem, em pixels.
    origin : 'upper' ou 'lower', opcional
        Baseado na convenção do imshow para exibir o eixo y da imagem. 'upper' significa
        que [0, 0] é o canto superior esquerdo da imagem; 'lower' significa que é
        o canto inferior esquerdo.

    Retornos
    -------
    x : float ou ndarray
        Posições de pixel x.
    y : float ou ndarray
        Posições de pixel y.
    """

    x = imgdim[0] * (cx - cdim[0]) / (cdim[1] - cdim[0])

    if origin == "lower":
        y = imgdim[1] * (cy - cdim[2]) / (cdim[3] - cdim[2])
    else:
        y = imgdim[1] * (cdim[3] - cy) / (cdim[3] - cdim[2])

    return x, y


def pixel_to_geocoord(x, y, cdim, imgdim, origin="upper"):
    """Converte posições de pixel na imagem em coordenadas geográficas.

    Assume que o meridiano central está em 0 (portanto, long em [-180, 180)).

    Parâmetros
    ----------
    x : float ou ndarray
        Posições de pixel x.
    y : float ou ndarray
        Posições de pixel y.
    cdim : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da imagem.
    imgdim : list, tuple ou ndarray
        Comprimento e altura da imagem, em pixels.
    origin : 'upper' ou 'lower', opcional
        Baseado na convenção do imshow para exibir o eixo y da imagem. 'upper' significa
        que [0, 0] é o canto superior esquerdo da imagem; 'lower' significa que é
        o canto inferior esquerdo.

    Retornos
    -------
    cx : float ou ndarray
        Coordenada x.
    cy : float ou ndarray
        Coordenada y.
    """

    cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]

    if origin == "lower":
        cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])

    return cx, cy

########## Função de Conversão de Quilômetros para Pixels ##########

def km_to_pixel(imgheight, latextent, dc=1., a=1737.4):
    """Calcula o fator de conversão de quilômetros para pixels (pix/km).

    Parâmetros
    ----------
    imgheight : float
        Altura da imagem em pixels.
    latextent : float
        Extensão da latitude da imagem em graus.
    dc : float de 0 a 1, opcional
        Fator de escala para distorções.
    a : float, opcional
        Raio do mundo em km. O padrão é a Lua (1737.4 km).

    Retornos
    -------
    km2pix : float
        Fator de conversão pix/km
    """

    km2pix = (180.0 / np.pi) * imgheight * dc / (latextent * a)

    return km2pix