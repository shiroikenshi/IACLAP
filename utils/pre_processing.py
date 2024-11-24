# utils/pre_processing.py

"""Funções de Pré-processamento de Imagens

Funções para normalizar e redimensionar imagens, utilizadas na preparação de conjuntos de dados de imagens lunares

Utilizado em: funções de test_plot_hdf5.py e pelo modelo.
"""

########## Importações ##########

import numpy as np

########## Funções de Pré-processamento ##########

def normalize_and_resize_images(Data, dim=256, low=0.1, hi=1.0):
    """Normaliza e redimensiona (e opcionalmente inverte) as imagens.

    Parâmetros
    ----------
    Data : hdf5
        Array de dados.
    dim : inteiro, opcional
        Dimensões das imagens, assume-se que são quadradas.
    low : float, opcional
        Valor mínimo para reescalonamento. O padrão é 0.1, pois os pixels de fundo são 0.
    hi : float, opcional
        Valor máximo para reescalonamento.
    """

    for key in Data:
        images = Data[key][0]
        # Redimensiona as imagens para as dimensões especificadas e adiciona um canal (para compatibilidade com modelos que esperam 4D)
        images = images.reshape(len(images), dim, dim, 1)
        for i, img in enumerate(images):
            img = img / 255.0 # Normaliza os pixels para o intervalo [0, 1]
            # Opcionalmente, inverte as cores (descomentando a linha abaixo)
            # img[img > 0.] = 1.0 - img[img > 0.]

            # Normaliza os valores não nulos entre 'low' e 'hi'
            non_zero_pixels = img > 0
            if np.any(non_zero_pixels):
                min_val = np.min(img[non_zero_pixels])
                max_val = np.max(img[non_zero_pixels])
                img[non_zero_pixels] = low + (img[non_zero_pixels] - min_val) * (hi - low) / (max_val - min_val)
                images[i] = img # Atualiza a imagem processada no array
        Data[key][0] = images   # Atualiza o conjunto de imagens no dicionário

def generate_hdf5_id(i, zeropad=5):
    """Gera uma string de identificação para indexar corretamente os arquivos HDF5.

    Parâmetros
    ----------
    i : int
        Número da imagem a ser indexado.
    zeropad : inteiro, opcional
        Número de zeros para preencher a string.

    Retornos
    -------
    String do índice hdf5.
    """
    
    return f'img_{i:0{zeropad}d}'