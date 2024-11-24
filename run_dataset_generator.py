# run_dataset_generator.py

"""Gerador de Conjunto de Dados de Imagens de Entrada

Script para gerar conjuntos de dados de entrada a partir de mapas digitais de elevação lunar 
(DEM) e catálogos de crateras.

Este script foi projetado para usar o DEM LRO-Kaguya e uma combinação dos 
catálogos de crateras LOLA-LROC de 5 a 20 km e Head et al. 2010 >=20 km. 
Ele gera um conjunto aleatório de pequenas imagens (corretas em projeção) e 
crateras-alvo correspondentes. Os conjuntos de imagens de entrada e alvo 
são armazenados como arquivos hdf5. Os limites de longitude e latitude de 
cada imagem são incluídos no arquivo do conjunto de entrada, e tabelas das 
crateras em cada imagem são armazenadas em um arquivo separado do Pandas HDFStore hdf5.

Os parâmetros do script estão localizados sob as Configurações Globais.

Talvez seja interessante fazer uma cópia desse script cada vez que for gerar um conjunto
de dados específico.

O MPI4py pode ser usado para gerar vários arquivos hdf5 simultaneamente - cada thread 
escreve um número 'amt' de imagens em seu próprio arquivo.
"""

########## Importações ##########

from PIL import Image
import tables
import dataset_generator as igen
import time
import os

# Remove a limitação de pixels para evitar o erro "DecompressionBombError"
Image.MAX_IMAGE_PIXELS = None

# Define um novo limite para "MAX_GROUP_WIDTH" no PyTables
tables.parameters.MAX_GROUP_WIDTH = 65536

########## Configurações Globais ##########

# Configurações para uso opcional do MPI4py para processamento paralelo
mpi_settings = {
    'use_mpi4py': False
}

# Caminhos dos arquivos
paths = {
    'source_image_path': "./img/LunarLROLrocKaguya_118mperpix.png",
    'lroc_csv_path': "./catalogues/LROCCraters.csv",
    'head_csv_path': "./catalogues/HeadCraters.csv",
    'robbins_csv_path': "./catalogues/RobbinsCraters.csv", # Não está sendo utilizado pelo script no momento
    'outhead': "./input_data/test/"                       # train/val/test
}

# Verifique se a pasta de saída existe e, se não, cria-a
if not os.path.exists(paths['outhead']):
    os.makedirs(paths['outhead'])

# Parâmetros de geração de dados
generation_params = {
    'amt': 30000,                # Número de imagens a serem geradas
    'rawlen_range': [500, 6500], # Faixa de larguras de imagem a serem cortadas da imagem fonte
    'rawlen_dist': 'log',        # Distribuição para amostragem da rawlen_range ('uniforme' ou 'log')
    'ilen': 256,                 # Tamanho das imagens de entrada
    'tglen': 256,                # Tamanho das imagens-alvo
    'minpix': 1.0,               # Diâmetro mínimo em pixels das crateras a serem incluídas na máscara de alvo
    'verbose': True              # Se True, o script imprime a imagem na qual está trabalhando atualmente
}

# Parâmetros geográficos
coordinates = {
    'source_cdim': [-180., 180., -60., 60.], # Dimensões da imagem fonte [Min long, max long, min lat, max lat]
    # 'sub_cdim': [-180., 180., -60., 60.],    # Abrange toda a extensão do DEM (Dimensões da região da fonte a serem usadas ao cortar aleatoriamente)
    # 'sub_cdim': [-180., -60., -60., 60.],    # Abrange longitude -180 até -60 (Treino)
    # 'sub_cdim': [-60., 60., -60., 60.],      # Abrange longitude -60 até 60 (Validação)
    'sub_cdim': [60., 180., -60., 60.],      # Abrange longitude 60 até 180 (Teste)
    'R_km': 1737.4                           # Raio do Corpo Celeste em km (1737.4 para a Lua)
}

# Parâmetros da máscara de alvo
mask_params = {
    'truncate': True, # Se True, truncar máscara onde a imagem tem preenchimento
    'rings': True,    # Se True, usar anéis em vez de círculos preenchidos
    'ringwidth': 1    # Espessura do anel em pixels
}

########## Funções Auxiliares ##########

def init_mpi(use_mpi4py):
    """Inicializa MPI4py se configurado para uso, possibilitando processamento paralelo."""
    if use_mpi4py:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Thread {rank} de {size}")
    else:
        rank = 0
        size = 1
    return rank, size

def prepare_craters(lroc_csv_path, head_csv_path, source_cdim, sub_cdim, R_km):
    """Carrega e prepara o catálogo de crateras."""
    # Ler catálogos de crateras
    craters = igen.combine_crater_catalogs(filelroc=lroc_csv_path, filehead=head_csv_path)

    # Se a sub_cdim for diferente da source_cdim, recorta a imagem e filtra crateras
    if sub_cdim != source_cdim:
        craters = igen.filter_craters_by_size_and_bounds(craters, sub_cdim, None, arad=R_km)

    return craters

def generate_dataset(rank, amt, img, craters):
    """Gera o conjunto de dados de imagens e máscaras."""
    # Calcula o índice inicial baseado no rank
    istart = rank * amt

    # Chama a função de geração de dataset
    igen.gen_dataset(
        img, craters, paths['outhead'],
        rawlen_range=generation_params['rawlen_range'],
        rawlen_dist=generation_params['rawlen_dist'],
        ilen=generation_params['ilen'],
        cdim=coordinates['sub_cdim'],
        arad=coordinates['R_km'],
        minpix=generation_params['minpix'],
        tglen=generation_params['tglen'],
        binary=True,
        rings=mask_params['rings'],
        ringwidth=mask_params['ringwidth'],
        truncate=mask_params['truncate'],
        amt=amt,
        istart=istart,
        verbose=generation_params['verbose']
    )

########## Execução do Script ##########

if __name__ == '__main__':

    # Marca o tempo de início do script
    start_time = time.time()

    # Inicializa MPI, se configurado
    rank, size = init_mpi(mpi_settings['use_mpi4py'])

    # Carrega a imagem fonte
    img = Image.open(paths['source_image_path']).convert("L")

    # Prepara o catálogo de crateras
    craters = prepare_craters(
        paths['lroc_csv_path'],
        paths['head_csv_path'],
        coordinates['source_cdim'],
        coordinates['sub_cdim'],
        coordinates['R_km']
    )

    # Recorta a imagem se necessário
    if coordinates['sub_cdim'] != coordinates['source_cdim']:
        img = igen.crop_image_to_bounds(img, coordinates['source_cdim'], coordinates['sub_cdim'])

    # Gera o conjunto de dados
    generate_dataset(rank, generation_params['amt'], img, craters)

    # Calcula o tempo total de execução do script
    elapsed_time = time.time() - start_time

    # Converte o tempo decorrido para minutos e segundos
    minutes, seconds = divmod(elapsed_time, 60)

    if generation_params['verbose']:
        # Exibe o tempo total decorrido em minutos e segundos
        print(f"Tempo decorrido: {int(minutes)} min e {int(seconds)} seg")