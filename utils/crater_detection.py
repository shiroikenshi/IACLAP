# utils/crater_detection.py

"""Funções de Detecção de Crateras em Imagens com Template Matching

Este script contém funções para detecção de crateras em imagens de elevação lunar (DEM) utilizando 
a técnica de template matching. As funções extraem coordenadas de crateras de imagens preditas por redes 
neuronais e comparam as detecções com dados de crateras anotados manualmente. Utilizado para análise 
e validação de resultados em etapas posteriores de avaliação de desempenho.

Utilizado em: scripts de avaliação de acurácia e de análise de detecção de crateras.
"""

########## Importações ##########

import numpy as np
from skimage.feature import match_template
import cv2

########## Hiperparâmetros de Detecção de Crateras ##########

minrad_ = 5            # Raio mínimo para busca de crateras
maxrad_ = 40           # Raio máximo para busca de crateras
longlat_thresh2_ = 1.8 # Limiar de distância ao quadrado entre crateras para distinguir detecções
rad_thresh_ = 1.0      # Limiar de diferença de raio para distinguir crateras
template_thresh_ = 0.5 # Limiar de correlação para considerar uma correspondência como cratera detectada
target_thresh_ = 0.1   # Limiar para binarizar a imagem-alvo

########## Função de Detecção de Crateras ##########

def detect_craters_with_template(target, minrad=minrad_, maxrad=maxrad_,
                                 longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_,
                                 template_thresh=template_thresh_, target_thresh=target_thresh_):
    """
    Detecta crateras em uma imagem-alvo através de template matching usando templates em formato de anel.

    Parâmetros
    ----------
    target : array
        Imagem predita pelo modelo que representa crateras.
    minrad : int, opcional
        Raio mínimo dos anéis de template.
    maxrad : int, opcional
        Raio máximo dos anéis de template.
    longlat_thresh2 : float, opcional
        Diferença mínima em distância para distinguir crateras distintas.
    rad_thresh : float, opcional
        Diferença mínima em raio para distinguir crateras distintas.
    template_thresh : float, opcional
        Limiar de correlação para contar como uma cratera detectada.
    target_thresh : float, opcional
        Limiar para binarizar a imagem-alvo.

    Retornos
    -------
    coords : array
        Coordenadas (x, y, raio) das crateras detectadas.
    """
    
    # Define a espessura dos anéis usados para template matching
    rw = 2

    # Binariza a imagem-alvo para destacar regiões que podem conter crateras
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0

    # Gera uma lista de raios de anel entre minrad e maxrad
    radii = np.arange(minrad, maxrad + 1, 1, dtype=int)
    coords = []  # Lista para armazenar coordenadas (x, y, raio) das crateras detectadas
    corr = []    # Lista para armazenar o coeficiente de correlação das crateras detectadas

    # Percorre cada raio possível para criar e aplicar o template de anel
    for r in radii:
        # Cria um template circular com raio r e espessura rw
        n = 2 * (r + rw + 1)
        template = np.zeros((n, n))
        cv2.circle(template, (r + rw + 1, r + rw + 1), r, 1, rw)

        # Aplica o template matching para encontrar crateras com o raio atual
        result = match_template(target, template, pad_input=True)
        index_r = np.where(result > template_thresh)  # Índices onde a correlação excede o limiar
        coords_r = np.asarray(list(zip(*index_r)))    # Converte índices para coordenadas
        corr_r = np.asarray(result[index_r])          # Salva o valor de correlação

        # Armazena as coordenadas e os valores de correlação dos pontos detectados
        if len(coords_r) > 0:
            for c in coords_r:
                coords.append([c[1], c[0], r])
            for l in corr_r:
                corr.append(np.abs(l))

    # Remove crateras duplicadas detectadas em diferentes raios/posições
    coords, corr = np.asarray(coords), np.asarray(corr)
    i, N = 0, len(coords)
    while i < N:
        Long, Lat, Rad = coords.T
        lo, la, r = coords[i]
        minr = np.minimum(r, Rad)

        # Calcula distância entre crateras para verificar duplicatas
        dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
        dR = abs(Rad - r) / minr
        index = (dR < rad_thresh) & (dL < longlat_thresh2)

        if len(np.where(index == True)[0]) > 1:
            # Mantém apenas a coordenada com maior correlação entre as duplicatas
            coords_i = coords[np.where(index == True)]
            corr_i = corr[np.where(index == True)]
            coords[i] = coords_i[corr_i == np.max(corr_i)][0]
            index[i] = False
            coords = coords[np.where(index == False)]
        N, i = len(coords), i + 1

    return coords

def compare_crater_detections_with_csv(target, csv_coords, minrad=minrad_, maxrad=maxrad_,
                                       longlat_thresh2=longlat_thresh2_,
                                       rad_thresh=rad_thresh_, template_thresh=template_thresh_,
                                       target_thresh=target_thresh_, rmv_oor_csvs=0):
    """Compara crateras detectadas com coordenadas anotadas em um conjunto de dados CSV.

    Parâmetros
    ----------
    target : array
        Imagem predita com crateras pelo modelo.
    csv_coords : array
        Coordenadas anotadas das crateras.
    minrad : int, opcional
        Raio mínimo dos templates de anéis.
    maxrad : int, opcional
        Raio máximo dos templates de anéis.
    longlat_thresh2 : float, opcional
        Distância mínima entre crateras para serem consideradas distintas.
    rad_thresh : float, opcional
        Diferença mínima de raio para crateras serem distintas.
    template_thresh : float, opcional
        Correlação mínima para considerar uma cratera detectada.
    target_thresh : float, opcional
        Limiar para binarização da imagem.
    rmv_oor_csvs : int, opcional
        Se 1, remove crateras fora do alcance detectável.

    Retornos
    -------
    N_match : int
        Número de crateras correspondentes entre detecções e CSV.
    N_csv : int
        Número de crateras no CSV.
    N_detect : int
        Número total de crateras detectadas.
    maxr : int
        Raio da maior cratera detectada.
    err_lo : float
        Erro médio de longitude.
    err_la : float
        Erro médio de latitude.
    err_r : float
        Erro médio de raio.
    frac_dupes : float
        Fração de crateras com múltiplas correspondências no CSV.
    """

    # Obtenha as coordenadas das crateras detectadas usando template matching
    templ_coords = detect_craters_with_template(target, minrad, maxrad, longlat_thresh2,
                                                rad_thresh, template_thresh, target_thresh)

    # Inicializa o valor máximo do raio
    maxr = 0
    if len(templ_coords) > 0:
        maxr = np.max(templ_coords.T[2]) # Define maxr como o raio da maior cratera detectada

    # Inicializa variáveis para contagem de correspondências e duplicatas, além de variáveis para erros.
    N_match = 0
    frac_dupes = 0
    err_lo, err_la, err_r = 0, 0, 0 
    N_csv, N_detect = len(csv_coords), len(templ_coords)

    # Itera sobre cada cratera detectada para calcular correspondências com o conjunto de dados anotado
    for lo, la, r in templ_coords:
        Long, Lat, Rad = csv_coords.T
        minr = np.minimum(r, Rad)

        # Calcula a distância normalizada entre crateras para verificar se são correspondentes
        dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
        dR = abs(Rad - r) / minr
        index = (dR < rad_thresh) & (dL < longlat_thresh2)

        # Localiza as crateras que atendem aos critérios de correspondência e conta quantas são
        index_True = np.where(index == True)[0]
        N = len(index_True)

        if N >= 1:
            # Se houver uma correspondência, calcula os erros de longitude, latitude e raio
            Lo, La, R = csv_coords[index_True[0]].T
            meanr = (R + r) / 2.0 
            err_lo += abs(Lo - lo) / meanr
            err_la += abs(La - la) / meanr
            err_r += abs(R - r) / meanr

            # Se houver mais de uma correspondência, incrementa a fração de duplicatas
            if N > 1:
                frac_dupes += (N-1) / float(len(templ_coords))
        
        # Incrementa N_match com o valor mínimo entre 1 e N, garantindo que só conte uma vez
        N_match += min(1, N)
        
        # Remove as crateras do CSV que foram correspondidas, para evitar múltiplas correspondências
        csv_coords = csv_coords[np.where(index == False)]
        # Interrompe o loop se todas as crateras do CSV já foram correspondidas
        if len(csv_coords) == 0:
            break

    # Se rmv_oor_csvs estiver ativado, remove crateras fora do alcance detectável do CSV
    if rmv_oor_csvs == 1:
        upper = 15
        lower = minrad
        N_large_unmatched = len(np.where((csv_coords.T[2] > upper) |
                                         (csv_coords.T[2] < lower))[0])
        # Diminui N_csv pelo número de crateras que estão fora do alcance
        if N_large_unmatched < N_csv:
            N_csv -= N_large_unmatched

    # Calcula a média dos erros de longitude, latitude e raio caso existam correspondências
    if N_match >= 1:
        err_lo /= N_match
        err_la /= N_match
        err_r /= N_match

    return N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes