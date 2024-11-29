# run_extract_unique_craters.py

"""Execução da Extração de Crateras Únicas

Este script executa a extração de crateras únicas a partir de predições de um modelo.
As crateras duplicadas são filtradas e os resultados são salvos em arquivo.
"""

import extract_unique_craters as euc
import sys
import numpy as np

def main():
    """Fluxo principal de execução para extração de crateras únicas."""
    # Parâmetros de configuração
    CP = {
        'dim': 256,
        'datatype': 'test',
        'n_imgs': 30000,
        'llt2': float(sys.argv[1]), # Limite para diferença quadrática de longitude/latitude
        'rt': float(sys.argv[2]), # Limite para diferença de raio
        'dir_model': 'models/model.pth',
        'dir_data': 'catalogues/test_images.hdf5',
        'dir_preds': 'catalogues/test_preds.hdf5',
        'dir_result': 'catalogues/test_craterdist.npy',
    }

    # Inicializa array vazio para crateras únicas
    crateras_unicas = np.empty((0, 3))

    # Executa a extração de crateras únicas
    crateras_unicas = euc.extract_unique_craters(CP, crateras_unicas)
    print(f"Extração concluída. Crateras únicas salvas em {CP['dir_result']}")


if __name__ == '__main__':
    main()
