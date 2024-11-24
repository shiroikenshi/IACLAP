# run_unet_training_pipeline.py

"""Treinamento de Rede Neural Convolucional (UNET) para Detecção de Crateras

Este script executa o treinamento de um modelo UNET em imagens da Lua com alvos em formato de anel binário.
Utiliza os parâmetros definidos na seção de configurações globais.

Talvez seja interessante fazer uma cópia deste script cada vez que for treinar um modelo específico.
"""

########## Importações ##########

import unet_training_pipeline as utp

########## Configurações Globais ##########

# Parâmetros do Modelo
MP = {}

# Diretório dos arquivos hdf5 de imagens e crateras de treino/validação/teste
MP['dir'] = 'input_data/'

# Checkpoint caso já exista um modelo pré-treinado
MP['checkpoint_path'] = 'models/checkpoint.pth'

# Dimensão das imagens (assumindo imagens quadradas)
MP['dim'] = 256

# Tamanho do batch de treinamento: valores menores = menos memória mas gradiente menos preciso
MP['bs'] = 8

# Tamanho do batch de validação
MP['val_batch_size'] = 8

# Número de épocas de treinamento
MP['epochs'] = 10

# Número de amostras de treino/validação/teste (deve ser múltiplo do tamanho do batch)
MP['num_train'] = 5000
MP['num_val'] = 5000
MP['num_test'] = 5000

# Salvar o modelo (1 para salvar, 0 para não salvar) e diretório
MP['save_models'] = 1
MP['save_dir'] = 'models/model.pth' # Extensão .pth, padrão do PyTorch

# Parâmetros do Modelo (para iterar, mantenha em listas)
MP['N_runs'] = 1          # Número de execuções
MP['filter_length'] = [3] # Tamanho do filtro (mantido para compatibilidade)
MP['lr'] = [0.0001]       # Taxa de aprendizado
MP['n_filters'] = [64]   # Número de filtros
MP['lambda'] = [1e-6]     # Regularização (Weight Decay)
MP['dropout'] = [0.15]    # Fração de dropout

# Exemplo de iteração sobre parâmetros
# MP['N_runs'] = 2
# MP['lambda'] = [1e-4, 1e-4]

########## Execução do Script ##########

if __name__ == '__main__':
    # Executa o treinamento do modelo com os parâmetros especificados
    utp.get_models(MP)