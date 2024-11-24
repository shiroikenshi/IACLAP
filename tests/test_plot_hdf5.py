# tests/test_plot_hdf5.py

"""Script para Visualização de Imagens e Máscaras Geradas com Interface Gráfica Tkinter

Este script permite carregar, pré-processar e visualizar imagens de entrada e
máscaras geradas a partir dos conjuntos de dados de elevação digital lunar (DEM) 
e catálogos de crateras. Ele carrega os dados a partir de arquivos HDF5 criados
por scripts anteriores, aplica pré-processamento (se necessário) e exibe as
imagens e suas máscaras correspondentes para verificação.

Este script utiliza o Tkinter para criar uma interface gráfica que inclui a figura
do Matplotlib e um widget de texto rolável para exibir os dados das crateras.

Os parâmetros do script estão localizados sob as Variáveis Globais.
"""

########## Importações ##########

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import os

# Importa funções de processamento do módulo utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.pre_processing as proc

########## Variáveis Globais ##########

# Caminhos dos arquivos HDF5
images_hdf5_path = './input_data/train/train_images.hdf5'
craters_hdf5_path = './input_data/train/train_craters.hdf5'

# Inicializa o índice da imagem
image_index = 0

########## Funções Auxiliares ##########

def carregar_dados_hdf5():
    """Carrega os dados de imagens e máscaras dos arquivos HDF5."""
    try:
        gen_imgs = h5py.File(images_hdf5_path, 'r')
        sample_data = {
            'imgs': [
                gen_imgs['input_images'][...].astype('float32'),
                gen_imgs['target_masks'][...].astype('float32')
            ]
        }
        total_images = len(sample_data['imgs'][0])
        return sample_data, total_images, gen_imgs
    except (FileNotFoundError, KeyError) as e:
        print(f"Erro ao carregar dados: {e}")
        return None, 0, None

def carregar_dados_crateras():
    """Carrega o arquivo HDF5 de crateras."""
    try:
        craters_h5 = pd.HDFStore(craters_hdf5_path, 'r')
        return craters_h5
    except FileNotFoundError:
        print("O arquivo HDF5 de crateras não foi encontrado.")
        return None

def fechar_arquivos_hdf5():
    """Fecha os arquivos HDF5 ao sair da aplicação."""
    if craters_h5 is not None:
        craters_h5.close()
    if gen_imgs is not None:
        gen_imgs.close()

def on_closing():
    """Função que fecha a aplicação ao clicar no botão de fechar."""
    fechar_arquivos_hdf5()
    root.destroy()

########## Interface Gráfica ##########

def configurar_janela():
    """Configura a janela principal do Tkinter."""
    root = tk.Tk()
    root.title("Visualização de Imagens e Máscaras Geradas")
    root.minsize(800, 600)
    return root

def configurar_layout(root):
    """Configura o layout da interface."""
    frame_main = tk.Frame(root)
    frame_main.pack(fill=tk.BOTH, expand=True)
    
    # Frame para a figura Matplotlib
    frame_plot = tk.Frame(frame_main)
    frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Frame para os botões de navegação
    frame_buttons = tk.Frame(frame_main)
    frame_buttons.pack(side=tk.TOP, fill=tk.X, pady=5)
    
    # Frame para o widget de texto com scrollbar
    frame_text = tk.Frame(frame_main)
    frame_text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
    
    return frame_plot, frame_buttons, frame_text

def configurar_canvas(frame_plot):
    """Configura o canvas do Matplotlib."""
    fig = Figure(figsize=(8, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Subplots para as imagens
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Remove os eixos para otimizar o espaço
    ax1.axis('off')
    ax2.axis('off')
    
    return fig, canvas, ax1, ax2

def configurar_botoes(frame_buttons):
    """Configura os botões de navegação."""
    btn_prev = tk.Button(frame_buttons, text='Anterior', command=lambda: navegar(-1))
    btn_prev.pack(side=tk.LEFT, padx=10, pady=5)
    
    btn_next = tk.Button(frame_buttons, text='Próximo', command=lambda: navegar(1))
    btn_next.pack(side=tk.RIGHT, padx=10, pady=5)

def configurar_text_widget(frame_text):
    """Configura o widget de texto para exibir dados das crateras."""
    text_scrollbar = tk.Scrollbar(frame_text)
    text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_widget = tk.Text(frame_text, yscrollcommand=text_scrollbar.set, height=8)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    text_scrollbar.config(command=text_widget.yview)
    
    return text_widget

def configurar_label_coordenadas(frame_buttons):
    """Configura o label para exibir as coordenadas do mouse."""
    coord_label = tk.Label(frame_buttons, text='Coordenadas: (X, Y)')
    coord_label.pack(side=tk.TOP, pady=5)
    return coord_label

########## Funções de Visualização ##########

def update_plot():
    """Atualiza o plot com a imagem e a máscara atual."""
    global image_index
    if image_index < 0:
        image_index = 0
    elif image_index >= total_images:
        image_index = total_images - 1
    
    # Limpa e atualiza os subplots
    ax1.clear()
    ax2.clear()
    ax1.axis('off')
    ax2.axis('off')

    ax1.imshow(sd_input_images[image_index].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
    ax2.imshow(sd_target_masks[image_index].squeeze(), origin='upper', cmap='Greys_r')
    ax1.set_title(f'Imagem DEM da Lua - Índice {image_index}', fontsize=10)
    ax2.set_title(f'Máscara Verdadeira - Índice {image_index}', fontsize=10)
    
    # Atualiza o canvas
    canvas.draw()

    # Atualiza os dados das crateras
    update_text_widget()

def update_text_widget():
    """Atualiza o widget de texto com os dados das crateras."""
    if craters_h5 is not None:
        craters_index = proc.generate_hdf5_id(image_index, zeropad=zeropad)
        try:
            crater_data = craters_h5[craters_index]
            text_widget.delete('1.0', tk.END)
            crater_str = crater_data.to_string(index=True)
            text_widget.insert(tk.END, f"Dados das crateras do DEM '{craters_index}':\n{crater_str}")
        except KeyError:
            text_widget.delete('1.0', tk.END)
            text_widget.insert(tk.END, f"A chave '{craters_index}' não foi encontrada no arquivo HDF5.")
    else:
        text_widget.delete('1.0', tk.END)
        text_widget.insert(tk.END, "Arquivo HDF5 de crateras não está disponível.")

def navegar(step):
    """Navega entre as imagens."""
    global image_index
    image_index += step
    update_plot()

def on_move(event):
    """Atualiza as coordenadas do mouse no label."""
    if event.inaxes == ax1 or event.inaxes == ax2:
        x, y = int(event.xdata), int(event.ydata)
        coord_label.config(text=f'Coordenadas: (X: {x}, Y: {y})')
    else:
        coord_label.config(text='Coordenadas: (X, Y)')

########## Execução do Script ##########

if __name__ == '__main__':
    # Carrega os dados HDF5
    sample_data, total_images, gen_imgs = carregar_dados_hdf5()
    craters_h5 = carregar_dados_crateras()

    if sample_data is not None:
        try:
            proc.normalize_and_resize_images(sample_data)
            sd_input_images = sample_data['imgs'][0]
            sd_target_masks = sample_data['imgs'][1]
        except Exception as e:
            print(f"Ocorreu um erro durante o pré-processamento: {e}")

    # Calcula o zeropad com base no número de imagens
    zeropad = len(str(total_images))

    # Configura a interface
    root = configurar_janela()
    frame_plot, frame_buttons, frame_text = configurar_layout(root)
    fig, canvas, ax1, ax2 = configurar_canvas(frame_plot)
    configurar_botoes(frame_buttons)
    text_widget = configurar_text_widget(frame_text)
    coord_label = configurar_label_coordenadas(frame_buttons)

    # Conecta o evento de movimento do mouse
    canvas.mpl_connect('motion_notify_event', on_move)

    # Inicializa a visualização
    if sample_data is not None:
        update_plot()

    # Define ação ao fechar a janela
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Inicia o loop principal do Tkinter
    root.mainloop()