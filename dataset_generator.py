# dataset_generator.py

"""Funções do Gerador de Conjuntos de Dados de Imagem de Entrada

Funções para gerar conjuntos de dados de imagens de entrada e alvo a partir de mapas de elevação digital lunar e catálogos de crateras.
"""

########## Importações ##########

import numpy as np
import pandas as pd
from PIL import Image
import cartopy.crs as ccrs
import utils.img_transform as cimg
import collections
import cv2
import h5py
import utils.spatial_transform as st
import os

########## Leitura e Combinação de Catálogos de Crateras ##########

def load_lroc_crater_catalog(filename: str = "catalogues/LROCCraters.csv", sortlat: bool = True) -> pd.DataFrame:
    """Lê o catálogo de crateras LROC de 5 a 20 km em formato CSV.

    Parâmetros
    ----------
    filename : str, opcional
        Caminho e nome do arquivo CSV da LROC. Padrão é o que está na pasta atual.
    sortlat : bool, opcional
        Se `True` (padrão), ordena o catálogo pela latitude.

    Retornos
    -------
    craters : pandas.DataFrame
        DataFrame com os dados das crateras.
    """

    try:
        # Lê o arquivo CSV especificando as colunas de interesse
        craters = pd.read_csv(filename, header=0, usecols=list(range(2, 6)))
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {filename} não encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Nenhum dado no arquivo {filename}.")

    # Ordena o DataFrame pela latitude, se solicitado
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(drop=True, inplace=True)

    return craters

def load_head_crater_catalog(filename: str = "catalogues/HeadCraters.csv", sortlat: bool = True) -> pd.DataFrame:
    """Lê o catálogo de crateras de Head et al. 2010 com diâmetro >= 20 km em formato CSV.

    Parâmetros
    ----------
    filename : str, opcional
        Caminho e nome do arquivo CSV de Head et al. Padrão é o que está na pasta atual.
    sortlat : bool, opcional
        Se `True` (padrão), ordena o catálogo pela latitude.

    Retornos
    -------
    craters : pandas.DataFrame
        DataFrame com os dados das crateras.
    """

    try:
        # Lê o arquivo CSV e define os nomes das colunas
        craters = pd.read_csv(filename, header=0, names=['Long', 'Lat', 'Diameter (km)'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {filename} não encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Nenhum dado no arquivo {filename}.")

    # Ordena o DataFrame pela latitude, se solicitado
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)
        craters.reset_index(drop=True, inplace=True)

    return craters

def combine_crater_catalogs(filelroc: str = "catalogues/LROCCraters.csv",
                            filehead: str = "catalogues/HeadCraters.csv",
                            sortlat: bool = True) -> pd.DataFrame:
    """Combina o conjunto de dados de crateras LROC de 5 a 20 km com o conjunto de dados Head com diâmetro >= 20 km.

    Parâmetros
    ----------
    filelroc : str, opcional
        Localização do arquivo de crateras LROC. Padrão é o que está na pasta atual.
    filehead : str, opcional
        Localização do arquivo de crateras de Head et al. Padrão é o que está na pasta atual.
    sortlat : bool, opcional
        Se `True` (padrão), ordena o catálogo pela latitude.

    Retornos
    -------
    craters : pandas.DataFrame
        DataFrame com os dados das crateras.
    """

    # Carrega os dados de crateras de Head e valida
    ctrs_head = load_head_crater_catalog(filename=filehead, sortlat=False)
    assert ctrs_head.shape == ctrs_head[ctrs_head["Diameter (km)"] > 20].shape, "Validação de crateras de Head falhou."
    
    # Carrega os dados de crateras LROC e remove a coluna desnecessária
    ctrs_lroc = load_lroc_crater_catalog(filename=filelroc, sortlat=False)
    ctrs_lroc.drop(["tag"], axis=1, inplace=True)

    # Combina os dois DataFrames e ordena pela latitude, se solicitado
    craters = pd.concat([ctrs_lroc, ctrs_head], axis=0, ignore_index=True, copy=True)
    
    if sortlat:
        craters.sort_values(by='Lat', inplace=True)

    # Reinicia os índices do DataFrame
    craters.reset_index(drop=True, inplace=True)

    return craters

########## Transformação de Projeções de Imagens e Coordenadas ##########

def adjust_aspect_ratio(regrid_shape, target_extent):
    """Função auxiliar copiada de cartopy.img_transform para redimensionar uma imagem
    sem alterar sua proporção.

    Parâmetros
    ----------
    regrid_shape : int ou float
        Comprimento alvo do eixo mais curto (em unidades de pixels).
    target_extent : algum
        Largura e altura da imagem alvo (geralmente não em unidades de
        pixels).

    Retornos
    -------
    regrid_shape : tupla
        Largura e altura da imagem alvo em pixels.
    """

    # Se regrid_shape não for uma sequência, calcula a nova proporção
    if not isinstance(regrid_shape, collections.abc.Sequence):
        target_size = int(regrid_shape)
        x_range, y_range = np.diff(target_extent)[::2]
        desired_aspect = x_range / y_range

        # Ajusta a nova forma de regrid com base na proporção desejada
        if x_range >= y_range:
            regrid_shape = (target_size * desired_aspect, target_size)
        else:
            regrid_shape = (target_size, target_size / desired_aspect)
    
    return regrid_shape

def transform_image_projection(img, iproj, iextent, oproj, oextent,
              origin="upper", rgcoeff=1.2):
    """Transforma imagens de uma projeção para outra usando cartopy.

    Parâmetros
    ----------
    img : numpy.ndarray
        Imagem como um array 2D.
    iproj : cartopy.crs.Projection instance
        Sistema de coordenadas de entrada.
    iextent : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da entrada.
    oproj : cartopy.crs.Projection instance
        Sistema de coordenadas de saída.
    oextent : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da saída.
    origin : "lower" ou "upper", opcional
        Baseado na convenção imshow para exibir o eixo y da imagem. "upper" significa
        [0,0] está no canto superior esquerdo da imagem; "lower" significa que está no
        canto inferior esquerdo.
    rgcoeff : float, opcional
        Aumento de tamanho fracionário da altura da imagem transformada. Definido genericamente
        para 1.2 para evitar perda de fidelidade durante a transformação (embora parte dela
        seja inevitavelmente perdida devido ao redimensionamento).
    """

    # Verifica se as projeções de entrada e saída são idênticas
    if iproj == oproj:
        raise Warning("As transformações de entrada e saída são idênticas! Retornando a entrada!")
    
    # Ajusta a imagem se a origem for 'upper'
    if origin == 'upper':
        img = img[::-1]

    # Define a nova forma de regrid com base no fator de aumento
    regrid_shape = adjust_aspect_ratio(rgcoeff * min(img.shape), oextent)

    # Cria limites de extensão sem zeros para evitar problemas na projeção
    iextent_nozeros = np.where(iextent == 0, 1e-8, iextent).tolist()

    # Realiza a transformação da imagem em uma única chamada
    imgout, extent = cimg.warp_array(
        img,
        source_proj=iproj,
        source_extent=iextent_nozeros,
        target_proj=oproj,
        target_res=regrid_shape,
        target_extent=oextent,
        mask_extrapolated=True
    )

    # Ajusta a imagem de volta se a origem era 'upper'
    if origin == 'upper':
        imgout = imgout[::-1]

    return imgout

def transform_image_with_padding(img, iproj, iextent, oproj, oextent, origin="upper",
                 rgcoeff=1.2, fillbg="black"):
    """Adiciona preenchimento à imagem transformada para mantê-la do mesmo tamanho que a original.

    Parâmetros
    ----------
    img : numpy.ndarray
        Imagem como um array 2D.
    iproj : cartopy.crs.Projection instance
        Sistema de coordenadas de entrada.
    iextent : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da entrada.
    oproj : cartopy.crs.Projection instance
        Sistema de coordenadas de saída.
    oextent : list-like
        Limites de coordenadas (x_min, x_max, y_min, y_max) da saída.
    origin : "lower" ou "upper", opcional
        Baseado na convenção imshow para exibir o eixo y da imagem. "upper" significa
        [0,0] está no canto superior esquerdo da imagem; "lower" significa que está no
        canto inferior esquerdo.
    rgcoeff : float, opcional
        Aumento de tamanho fracionário da altura da imagem transformada. Definido genericamente
        para 1.2 para evitar perda de fidelidade durante a transformação (embora parte dela
        seja inevitavelmente perdida devido ao redimensionamento).
    fillbg : 'black' ou 'white', opcional.
        Preenche o preenchimento com valores pretos (0) ou brancos (255). O padrão é
        preto.

    Retornos
    -------
    imgo : PIL.Image.Image
        Imagem transformada com preenchimento
    imgw.size : tupla
        Largura, altura da imagem sem preenchimento
    offset : tupla
        Largura em pixels do preenchimento (esquerda, cima)
    """

    # Converte a imagem para um array se necessário
    img = np.asanyarray(img) if isinstance(img, Image.Image) else img

    # Verifica se a imagem não está em branco
    assert img.sum() > 0, "A imagem de entrada para WarpImagePad está em branco!"

    # Define a cor de fundo conforme especificado
    bgval = 255 if fillbg == "white" else 0

    # Transforma a imagem
    imgw = transform_image_projection(img, iproj, iextent, oproj, oextent,
                                       origin=origin, rgcoeff=rgcoeff)

    # Remove a máscara da imagem transformada e converte para PIL.Image
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw = Image.fromarray(imgw, mode="L")

    # Mantém a proporção da imagem ao redimensionar
    imgw_loh = imgw.size[0] / imgw.size[1]

    if imgw_loh > (img.shape[1] / img.shape[0]):
        imgw = imgw.resize([img.shape[0], int(np.round(img.shape[0] / imgw_loh))],
                            resample=Image.NEAREST)
    else:
        imgw = imgw.resize([int(np.round(imgw_loh * img.shape[0])), img.shape[0]],
                           resample=Image.NEAREST)

    # Cria a imagem de fundo e cola a imagem transformada
    imgo = Image.new('L', (img.shape[1], img.shape[0]), bgval)
    offset = ((imgo.size[0] - imgw.size[0]) // 2, (imgo.size[1] - imgw.size[1]) // 2)
    imgo.paste(imgw, offset)

    return imgo, imgw.size, offset

def transform_crater_coordinates(craters, geoproj, oproj, oextent, imgdim, llbd=None,
                  origin="upper"):
    """Transforma as coordenadas das crateras para o sistema de coordenadas da imagem.

    Parâmetros
    ----------
    craters : pandas.DataFrame
        Informações sobre crateras
    geoproj : instância de cartopy.crs.Geodetic
        Sistema de coordenadas de entrada lat/long
    oproj : instância de cartopy.crs.Projection
        Sistema de coordenadas de saída
    oextent : lista-like
        Limites de coordenadas (x_min, x_max, y_min, y_max)
        da saída
    imgdim : lista, tupla ou ndarray
        Largura e altura da imagem, em pixels
    llbd : lista-like
        Limites de long/lat (long_min, long_max,
        lat_min, lat_max) da imagem
    origin : "lower" ou "upper"
        Baseado na convenção de imshow para exibir o eixo y da imagem.
        "upper" significa que [0,0] é o canto superior esquerdo da imagem;
        "lower" significa que é o canto inferior esquerdo.

    Retornos
    -------
    ctr_wrp : pandas.DataFrame
        DataFrame que inclui posições em pixels x, y
    """

    # Obtém um subconjunto das crateras dentro dos limites llbd, se fornecido
    ctr_wrp = craters.copy() if llbd is None else craters[
        (craters["Long"].between(llbd[0], llbd[1])) & 
        (craters["Lat"].between(llbd[2], llbd[3]))
    ].copy()

    # Verifica se existem crateras a serem transformadas
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].values
        ilat = ctr_wrp["Lat"].values
        # Transforma as coordenadas de lat/long para a nova projeção
        res = oproj.transform_points(x=ilong, y=ilat, src_crs=geoproj)[:, :2]

        # Converte as coordenadas para pixels usando os limites da saída
        ctr_wrp["x"], ctr_wrp["y"] = st.geocoord_to_pixel(res[:, 0], res[:, 1],
                                                            oextent, imgdim,
                                                            origin=origin)
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp

########## Conversão de Plate Carree para Projeção Ortográfica ##########

def convert_platecarree_to_orthographic(img, llbd, craters, iglobe=None,
                                ctr_sub=False, arad=1737.4, origin="upper",
                                rgcoeff=1.2, slivercut=0.):
    """Transforma a imagem Plate Carree e o arquivo csv associado em Orthographic.

    Parâmetros
    ----------
    img : PIL.Image.image ou str
        Arquivo ou nome do arquivo.
    llbd : lista-like
        Limites de long/lat (long_min, long_max, lat_min, lat_max) da imagem.
    craters : pandas.DataFrame
        Catálogo de crateras.
    iglobe : instância de cartopy.crs.Geodetic
        Globo para imagens. Se False, usa-se o modelo esférico da Lua.
    ctr_sub : bool, opcional
        Se `True`, assume que o dataframe de crateras inclui apenas crateras dentro
        da imagem. Se `False` (padrão), llbd é usado para cortar crateras de fora
        da imagem da (cópia do) dataframe.
    arad : float
        Raio do mundo em km. O padrão é a Lua (1737.4 km).
    origin : "lower" ou "upper", opcional
        Baseado na convenção de imshow para exibir o eixo y da imagem. "upper"
        (padrão) significa que [0,0] é o canto superior esquerdo da imagem; "lower" significa
        que é o canto inferior esquerdo.
    rgcoeff : float, opcional
        Aumento de tamanho fracionário da altura da imagem transformada. Por padrão, é definido
        como 1.2 para evitar perda de fidelidade durante a transformação (embora a deformação possa
        ser tão extrema que isso pode ser irrelevante).
    slivercut : float de 0 a 1, opcional
        Se a proporção da imagem transformada for muito estreita (e levar a um
        grande preenchimento, retorna imagens nulas).

    Retornos
    -------
    imgo : PIL.Image.image
        Imagem transformada e preenchida no formato PIL.Image.
    ctr_xy : pandas.DataFrame
        Crateras com posições x, y transformadas em pixels e raios em pixels.
    distortion_coefficient : float
        Razão entre as alturas centrais da imagem transformada e da imagem original.
    centrallonglat_xy : pandas.DataFrame
        Posição xy da longitude e latitude centrais.
    """

    # Se o usuário não fornecer propriedades do globo da Lua
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad * 1000.,
                            semiminor_axis=arad * 1000., ellipse=None)

    # Configura projeções
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]), central_latitude=np.mean(llbd[2:]), globe=iglobe)

    # Criando e transformando as coordenadas dos cantos da imagem
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()

    # Transformação das coordenadas
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [res[:, 0].min(), res[:, 0].max(), res[:, 1].min(), res[:, 1].max()]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [res[:, 0].min(), res[:, 0].max(), res[:, 1].min(), res[:, 1].max()]

    # Verificação de sanidade para imagens estreitas
    if (oextent[1] - oextent[0]) / (oextent[3] - oextent[2]) < slivercut:
        return [None, None]

    # Carregando a imagem
    img = Image.open(img).convert("L") if not isinstance(img, Image.Image) else img

    # Deformando a imagem
    imgo, imgwshp, offset = transform_image_with_padding(img, iproj, iextent, oproj, oextent,
                                                         origin=origin, rgcoeff=rgcoeff, fillbg="black")

    # Transformando as coordenadas das crateras
    llbd_in = None if ctr_sub else llbd
    ctr_xy = transform_crater_coordinates(craters, geoproj, oproj, oextent, imgwshp, llbd=llbd_in, origin=origin)

    # Ajustando as posições das crateras
    ctr_xy[["x", "y"]] += offset

    # Cálculo do coeficiente de distorção
    distortion_coefficient = (res[7, 1] - res[1, 1]) / (oextent[3] - oextent[2])
    if distortion_coefficient < 0.7:
        raise ValueError(f"O Coeficiente de Distorção não pode ser {distortion_coefficient:.2f}!")

    pixperkm = st.km_to_pixel(imgo.size[1], llbd[3] - llbd[2], dc=distortion_coefficient, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm

    # Calculando a posição xy da lat/long central
    centrallonglat = pd.DataFrame({"Long": [xll[4]], "Lat": [yll[4]]})
    centrallonglat_xy = transform_crater_coordinates(centrallonglat, geoproj, oproj, oextent, imgwshp, llbd=llbd_in, origin=origin)

    # Ajustando a posição central
    centrallonglat_xy[["x", "y"]] += offset

    return [imgo, ctr_xy, distortion_coefficient, centrallonglat_xy]

########## Criação de Máscaras de Crateras para Conjunto de Dados Alvo (e funções auxiliares) ##########

def create_circle_mask(r=10.):
    """Cria uma máscara circular com raio r.
    
    Parâmetros
    ----------
    r : float
        Raio do círculo
    
    Retorna
    -------
    circle : numpy.ndarray 
        Máscara circular em formato de array
    """

    # Definir a grade para capturar a máscara com extensão suficiente (+1 para garantir que capturamos o raio).
    rhext = int(r) + 1
    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]  # Grade 2D centrada no círculo
    
    # Criar a máscara com base na equação de um círculo
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)

def create_ring_mask(r=10., dr=1):
    """Cria um anel com raio r e espessura dr.

    Parâmetros
    ----------
    r : float
        Raio do anel
    dr : int
        Espessura do anel (cv2.circle requer um inteiro)
    
    Retorna
    -------
    ring : numpy.ndarray
        Máscara do anel em formato de array
    """

    # Definir a extensão da grade com base no raio e na espessura do anel
    rhext = int(np.ceil(r + dr / 2.)) + 1
    mask = np.zeros([2 * rhext + 1, 2 * rhext + 1], np.uint8)  # Inicializa a máscara com zeros
    
    # Criar o anel com a função cv2.circle
    ring = cv2.circle(mask, (rhext, rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)

def compute_merge_indices(cen, imglen, ks_h, ker_shp):
    """Função auxiliar que retorna índices para mesclar o estêncil com a imagem base,
    incluindo o tratamento de casos extremos. x e y são idênticos, então o código é
    neutro em relação ao eixo. Presume valores INTEIROS para todas as entradas!
    
    Parâmetros
    ----------
    cen : int
        Coordenada central (x ou y) da cratera
    imglen : int
        Dimensão da imagem (largura ou altura)
    ks_h : int
        Metade do suporte do kernel
    ker_shp : int
        Tamanho do kernel
    
    Retorna
    -------
    List[int] - Índices de início e fim para a imagem e o kernel.
    """

    # Define os limites esquerdo e direito para o kernel na imagem
    left = cen - ks_h
    right = cen + ks_h + 1

    # Ajusta para evitar transbordamento do kernel fora da imagem
    img_l, g_l = max(left, 0), max(-left, 0)  # Ajuste para o lado esquerdo
    img_r, g_r = min(right, imglen), ker_shp - max(right - imglen, 0)  # Ajuste para o lado direito

    return [img_l, img_r, g_l, g_r]

def generate_crater_mask(craters, img, binary=True, rings=False, ringwidth=1,
                         truncate=True):
    """Faz uma máscara binária da imagem das crateras, utilizando cache para otimizar o desempenho.

    Parâmetros
    ----------
    craters : pandas.DataFrame
        Catálogo de crateras que inclui colunas de pixel x e y.
    img : numpy.ndarray
        Imagem original; assume que o canal de cor está no último eixo (padrão tf).
    binary : bool, opcional
        Se True, retorna uma imagem binária das máscaras de crateras.
    rings : bool, opcional
        Se True, a máscara usa anéis ocos em vez de círculos preenchidos.
    ringwidth : int, opcional
        Se rings for True, a largura do anel define a largura (dr) do anel.
    truncate : bool, opcional
        Se True, corta a máscara onde a imagem é truncada.

    Retorna
    -------
    mask : numpy.ndarray
        Imagem da máscara alvo.
    """

    # Inicializa a máscara com zeros, com o mesmo tamanho da imagem
    imgshape = img.shape[:2]
    mask = np.zeros(imgshape)

    # Extrai as coordenadas x, y e o raio das crateras
    cx = craters["x"].values.astype('int')
    cy = craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.

    # Inicializa um dicionário para armazenar máscaras já criadas (cache)
    mask_cache = {}

    # Itera sobre cada cratera para criar e adicionar sua máscara
    for i in range(craters.shape[0]):
        r = radius[i]
        r_int = int(np.round(r))  # Arredonda o raio para o inteiro mais próximo

        # Define a chave para o cache (considera ringwidth se 'rings' for True)
        if rings:
            key = (r_int, ringwidth)
        else:
            key = r_int

        # Verifica se a máscara para este raio já foi criada
        if key in mask_cache:
            # Se já existe, reutiliza a máscara do cache
            kernel = mask_cache[key]
        else:
            # Caso contrário, cria a máscara e adiciona ao cache
            if rings:
                kernel = create_ring_mask(r=r, dr=ringwidth)
            else:
                kernel = create_circle_mask(r=r)
            mask_cache[key] = kernel  # Armazena no cache para uso futuro

        # Calcula o suporte do kernel e sua metade
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calcula os índices na imagem onde o kernel será mesclado
        [imxl, imxr, gxl, gxr] = compute_merge_indices(cx[i], imgshape[1],
                                                       ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = compute_merge_indices(cy[i], imgshape[0],
                                                       ks_half, kernel_support)

        # Adiciona o kernel à máscara na posição correta
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    # Converte a máscara para binária se necessário
    if binary:
        mask = (mask > 0).astype(float)

    # Trunca a máscara onde a imagem original é zero (por exemplo, preenchimento)
    if truncate:
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0

    return mask

########## Geração de Conjunto de Dados de Imagens e Máscaras (e funções auxiliares) ##########

def add_pixel_coordinates(craters, imgdim, cdim=[-180., 180., -90., 90.], 
                      origin="upper"):
    """Adiciona localizações de pixels x e y ao dataframe de crateras.

    Parâmetros
    ----------
    craters : pandas.DataFrame
        Informações sobre crateras
    imgdim : list, tuple ou ndarray
        Comprimento e altura da imagem, em pixels
    cdim : lista, opcional
        Limites de coordenadas (x_min, x_max, y_min, y_max) da imagem. O padrão é
        [-180., 180., -90., 90.].
    origin : "upper" ou "lower", opcional
        Baseado na convenção do imshow para exibir o eixo y da imagem.
        "upper" significa que [0,0] é o canto superior esquerdo da imagem;
        "lower" significa que é o canto inferior esquerdo.
    """
    
    # Converte coordenadas geográficas (longitude e latitude) para coordenadas de pixel (x, y)
    craters["x"], craters["y"] = st.geocoord_to_pixel(
        craters["Long"].to_numpy(), # Converte longitudes para um array NumPy
        craters["Lat"].to_numpy(),  # Converte latitudes para um array NumPy
        cdim,                       # Limites de coordenadas geográficas
        imgdim,                     # Dimensões da imagem (largura e altura)
        origin=origin               # Definição da origem ("upper" ou "lower")
    )

def filter_craters_by_size_and_bounds(craters, llbd, imgheight, arad=1737.4, minpix=0):
    """Corta o arquivo de crateras e remove crateras menores que um determinado valor mínimo.

    Parâmetros
    ----------
    craters : pandas.DataFrame
        DataFrame de crateras.
    llbd : lista
        Limites de long/lat (long_min, long_max, lat_min, lat_max) da imagem.
    imgheight : int
        Altura em pixels da imagem.
    arad : float, opcional
        Raio do mundo em km. Padrão é o raio da Lua (1737.4 km).
    minpix : int, opcional
        Tamanho mínimo do pixel da cratera a ser incluído na saída. O padrão é 0
        (equivalente a nenhum corte).

    Retorna
    -------
    ctr_sub : pandas.DataFrame
        DataFrame cortado e filtrado.
    """

    # Filtra crateras que estão dentro dos limites de longitude e latitude fornecidos
    ctr_sub = craters[
        (craters["Long"].between(llbd[0], llbd[1])) & # Filtra por longitude
        (craters["Lat"].between(llbd[2], llbd[3]))    # Filtra por latitude
    ].copy()

    # Verifica se há um tamanho mínimo de pixel para filtrar crateras
    if minpix > 0:
        # Calcula a conversão de pixels por km (baseado na altura da imagem e nos limites de latitude)
        pixperkm = st.km_to_pixel(imgheight, llbd[3] - llbd[2], dc=1., a=arad)
        minkm = minpix / pixperkm # Converte o valor mínimo de pixel para quilômetros
        
        # Filtra crateras com diâmetro maior ou igual ao valor mínimo em quilômetros
        ctr_sub = ctr_sub[ctr_sub["Diameter (km)"] >= minkm]

    # Reseta o índice do DataFrame após o filtro
    ctr_sub.reset_index(drop=True, inplace=True)
    
    return ctr_sub

def crop_image_to_bounds(img, cdim, newcdim):
    """Corta a imagem, de modo que a saída do corte possa ser usada em GenDataset.

    Parâmetros
    ----------
    img : PIL.Image.Image
        Imagem
    cdim : lista
        Limites de coordenadas (x_min, x_max, y_min, y_max) da imagem.
    newcdim : lista
        Limites de corte (x_min, x_max, y_min, y_max). Atualmente NÃO HÁ VERIFICAÇÃO
        de que newcdim está dentro de cdim!

    Retorna
    -------
    img : PIL.Image.Image
        Imagem cortada
    """

    # Converte os novos limites geográficos para coordenadas de pixel (x, y)
    x, y = st.geocoord_to_pixel(
        np.array(newcdim[:2]), # Novos limites de longitude (x_min, x_max)
        np.array(newcdim[2:]), # Novos limites de latitude (y_min, y_max)
        cdim,                  # Limites de coordenadas geográficas da imagem original
        img.size,              # Dimensões da imagem (largura e altura)
        origin="upper"         # Define a origem no canto superior esquerdo
    )
    
    # Define a caixa de corte (x_min, y_min, x_max, y_max) - y é invertido por causa da origem "upper"
    img = img.crop([x[0], y[1], x[1], y[0]])
    
    # Carrega a imagem cortada (otimização para garantir que o crop foi aplicado corretamente)
    img.load()
    
    return img

def gen_dataset(img, craters, outhead, rawlen_range=[1000, 2000],
               rawlen_dist='log', ilen=256, cdim=[-180., 180., -60., 60.],
               arad=1737.4, minpix=0, tglen=256, binary=True, rings=True,
               ringwidth=1, truncate=True, amt=100, istart=0, seed=None,
               verbose=False):
    """Gera um conjunto de dados aleatórios a partir de um DEM global e catálogo de crateras.

    A função amostra aleatoriamente pequenas imagens de um mapa de elevação digital
    global (DEM) que utiliza uma projeção de Plate Carree, e converte as pequenas
    imagens para a projeção ortográfica. As coordenadas de pixel e os raios das crateras
    do catálogo que caem dentro de cada imagem são colocados em um dataframe do Pandas
    correspondente. As imagens e dataframes são salvos no disco em formato hdf5.

    Parâmetros
    ----------
    img : PIL.Image.Image
        Imagem fonte.
    craters : pandas.DataFrame
        Catálogo de crateras em formato .csv.
    outhead : str
        Caminho e prefixo do arquivo das imagens e da tabela de crateras em hdf5.
    rawlen_range : list-like, opcional
        Limites inferior e superior das larguras de imagens brutas, em pixels, a serem
        recortadas da fonte. Para sempre recortar a mesma imagem, defina o limite inferior
        com o mesmo valor que o superior. O padrão é [300, 4000].
    rawlen_dist : 'uniforme' ou 'log'
        Distribuição da qual amostrar aleatoriamente as larguras das imagens. 'uniforme'
        é amostragem uniforme, e 'log' é amostragem loguniforme.
    ilen : int, opcional
        Largura da imagem de entrada, em pixels. Imagens recortadas serão reduzidas para
        esse tamanho. O padrão é 256.
    cdim : list-like, opcional
        Limites de coordenadas (x_min, x_max, y_min, y_max) da imagem. O padrão é
        [-180., 180., -60., 60.] da LRO-Kaguya.
    arad : float, opcional
        Raio do mundo em km. O padrão é o raio da Lua (1737.4 km).
    minpix : int, opcional
        Diâmetro mínimo da cratera em pixels a ser incluído na lista de crateras.
        Útil quando as menores crateras no catálogo são menores que 1 pixel de diâmetro.
    tglen : int, opcional
        Largura da imagem alvo, em pixels.
    binary : bool, opcional
        Se True, retorna uma imagem binária de máscaras de crateras.
    rings : bool, opcional
        Se True, a máscara usa anéis ocos em vez de círculos preenchidos.
    ringwidth : int, opcional
        Se rings for True, ringwidth define a largura (dr) do anel.
    truncate : bool
        Se True, trunca a máscara onde a imagem é truncada.
    amt : int, opcional
        Número de imagens a serem produzidas. O padrão é 100.
    istart : int
        Número inicial do arquivo de saída, ao criar conjuntos de dados que se estendem por
        vários arquivos.
    seed : int ou None
        Entrada para np.random.seed (para fins de teste).
    verbose : bool
        Se True, imprime o número da imagem que está sendo gerada.
    """

    # Semeia o gerador de números aleatórios
    np.random.seed(seed)

    # Adiciona coordenadas de pixel às crateras
    add_pixel_coordinates(craters, list(img.size), cdim=cdim, origin="upper")

    iglobe = ccrs.Globe(semimajor_axis=arad * 1000, semiminor_axis=arad * 1000, ellipse=None)

    # Cria amostrador aleatório
    random_sampler = (lambda: int(10**np.random.uniform(np.log10(rawlen_range[0]), np.log10(rawlen_range[1])))) if rawlen_dist == 'log' else (lambda: np.random.randint(rawlen_range[0], rawlen_range[1] + 1))
    
    # Inicializa os hdf5s de saída
    output_folder_name = os.path.basename(os.path.normpath(outhead))
    imgs_h5 = h5py.File(f"{outhead}{output_folder_name}_images.hdf5", 'w')

    # Cria os datasets
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen), dtype='uint8')
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen), dtype='float32')

    # Define os atributos após a criação dos datasets
    imgs_h5_inputs.attrs['definition'] = "Conjunto de dados da imagem de entrada."
    imgs_h5_tgts.attrs['definition'] = "Conjunto de dados da máscara alvo."

    # Ajuste aqui para grupos
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = "(long min, long max, lat min, lat max) da imagem recortada."

    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = "Limites de pixel da região do DEM Global que foi recortada para a imagem."

    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = "Coeficiente de distorção devido à transformação de projeção."

    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = "Coordenadas de pixel (x, y) do centro da longitude/latitude."

    craters_h5 = pd.HDFStore(f"{outhead}{output_folder_name}_craters.hdf5", 'w')

    zeropad = int(np.log10(amt)) + 1

    for i in range(amt):
        img_number = f"img_{istart + i:0{zeropad}d}"
        if verbose:
            print(f"Gerando {img_number}")

        # Recorte aleatório da imagem
        rawlen = random_sampler()
        xc, yc = np.random.randint(0, img.size[0] - rawlen), np.random.randint(0, img.size[1] - rawlen)
        box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')

        im = img.crop(box).resize([ilen, ilen], resample=Image.NEAREST)

        # Limites de longitude/latitude
        ix, iy = box[::2], box[1::2]
        llong, llat = st.pixel_to_geocoord(ix, iy, cdim, list(img.size), origin="upper")
        llbd = np.r_[llong, llat[::-1]]

        # Filtra crateras pequenas
        ctr_sub = filter_craters_by_size_and_bounds(craters, llbd, im.size[1], arad=arad, minpix=minpix)

        # Converte a imagem para a projeção ortográfica
        imgo, ctr_xy, distortion_coefficient, clonglat_xy = convert_platecarree_to_orthographic(
            im, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True, arad=arad, origin="upper", rgcoeff=1.2, slivercut=0.5)

        if imgo is None:
            print("Descartando imagem estreita")
            continue

        # Cria máscara de crateras
        tgt = np.asanyarray(imgo.resize((tglen, tglen), resample=Image.BILINEAR))
        mask = generate_crater_mask(ctr_xy, tgt, binary=binary, rings=rings, ringwidth=ringwidth, truncate=truncate)

        # Salva os dados
        imgs_h5_inputs[i] = np.asanyarray(imgo)
        imgs_h5_tgts[i] = mask
        imgs_h5_box.create_dataset(img_number, data=box)
        imgs_h5_llbd.create_dataset(img_number, data=llbd)
        imgs_h5_dc.create_dataset(img_number, data=[distortion_coefficient])
        imgs_h5_cll.create_dataset(img_number, data=clonglat_xy.loc[:, ['x', 'y']].to_numpy().ravel())
        craters_h5[img_number] = ctr_xy

        # Flush nos dados
        imgs_h5.flush()
        craters_h5.flush()

    # Fecha os arquivos
    imgs_h5.close()
    craters_h5.close()