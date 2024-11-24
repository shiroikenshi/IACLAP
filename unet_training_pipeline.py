# unet_training_pipeline.py

"""Funções de Treinamento de Rede Neural Convolucional

Este script contém funções para construir e treinar uma Rede Neural Convolucional (UNET)
em imagens da Lua e alvos binários em forma de anéis, utilizando PyTorch.
"""

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os

import utils.crater_detection as cd
import utils.pre_processing as prep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## Funções para Construção do Modelo UNet ##########

# Função para criar duas camadas convolucionais seguidas de ReLU
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# Função para criar uma camada de "downsampling"
def down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels)
    )

# Classe para criar uma camada de "upsampling" com concatenação
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# Definição da arquitetura UNet completa
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_filters=64):
        super(UNet, self).__init__()
        self.inc = double_conv(n_channels, n_filters)
        self.down1 = down(n_filters, n_filters * 2)
        self.down2 = down(n_filters * 2, n_filters * 4)
        self.down3 = down(n_filters * 4, n_filters * 8)
        self.bottleneck = double_conv(n_filters * 8, n_filters * 16)
        self.up1 = Up(n_filters * 16, n_filters * 8)
        self.up2 = Up(n_filters * 8, n_filters * 4)
        self.up3 = Up(n_filters * 4, n_filters * 2)
        self.up4 = Up(n_filters * 2, n_filters)
        self.outc = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_bottleneck = self.bottleneck(x4)
        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

########## Funções de Pré-processamento e Data Augmentation ##########

# Função para aumentar os dados com flips, rotações, etc.
def augment_data(images, masks):
    augmented_images = []
    augmented_masks = []
    for img, mask in zip(images, masks):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # Aplicação de transformações aleatórias
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if np.random.rand() > 0.5:
            angle = np.random.choice([0, 90, 180, 270])
            img = np.rot90(img, k=angle // 90)
            mask = np.rot90(mask, k=angle // 90)

        img = img[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]
        augmented_images.append(img)
        augmented_masks.append(mask)

    augmented_images = np.stack(augmented_images, axis=0)
    augmented_masks = np.stack(augmented_masks, axis=0)
    return augmented_images, augmented_masks

# Preparação dos dados para entrada no modelo (formatação para tensores)
def prepare_data(images, masks):
    images = torch.tensor(images, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.float32)

    # Ajuste de dimensões se necessário
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[3] == 1:
        images = images.permute(0, 3, 1, 2)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    elif masks.ndim == 4 and masks.shape[3] == 1:
        masks = masks.permute(0, 3, 1, 2)

    return images, masks

########## Função para Treinar e Avaliar o Modelo ##########

def train_and_evaluate_model(Data, Craters, MP, i_MP):
    # Configuração dos hiperparâmetros
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Inicialização do modelo e otimização
    model = UNet(n_channels=1, n_classes=1, n_filters=n_filters)
    model = model.to(MP['device'])
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=lmbda)
    criterion = nn.BCEWithLogitsLoss()

    # Carregamento dos dados de treino, validação e teste
    train_images, train_masks = Data['train']
    val_images, val_masks = Data['val']
    n_samples = train_images.shape[0]

    val_images_tensor, val_masks_tensor = prepare_data(val_images, val_masks)
    val_dataset = TensorDataset(val_images_tensor, val_masks_tensor)
    val_loader = DataLoader(val_dataset, batch_size=MP['val_batch_size'], shuffle=False)

    test_images, test_masks = Data['test']
    test_images_tensor, test_masks_tensor = prepare_data(test_images, test_masks)
    test_dataset = TensorDataset(test_images_tensor, test_masks_tensor)
    test_loader = DataLoader(test_dataset, batch_size=MP['val_batch_size'], shuffle=False)

    # Carregar checkpoint se existir
    start_epoch = 0
    if MP['checkpoint_path'] and os.path.exists(MP['checkpoint_path']):
        start_epoch = load_checkpoint(model, optimizer, MP['checkpoint_path'])

    for epoch in range(start_epoch, nb_epoch):
        print(f"Época {epoch+1}/{nb_epoch}")
        model.train()
        running_loss = 0.0

        # Embaralhamento dos dados de treino
        perm = np.random.permutation(n_samples)
        train_images_shuffled = train_images[perm]
        train_masks_shuffled = train_masks[perm]

        # Loop sobre batches
        for i in range(0, n_samples, bs):
            images_batch = train_images_shuffled[i:i+bs]
            masks_batch = train_masks_shuffled[i:i+bs]

            images_batch_aug, masks_batch_aug = augment_data(images_batch, masks_batch)
            images_batch_aug, masks_batch_aug = prepare_data(images_batch_aug, masks_batch_aug)

            images_batch_aug = images_batch_aug.to(MP['device'])
            masks_batch_aug = masks_batch_aug.to(MP['device'])

            # Forward e cálculo da perda
            outputs = model(images_batch_aug)
            loss = criterion(outputs, masks_batch_aug)

            print(f"Batch {i // bs + 1}/{n_samples // bs}, Perda do Batch: {loss.item():.4f}")

            # Backward e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_batch_aug.size(0)

        # Perda de treino por época
        epoch_loss = running_loss / n_samples
        print(f"Perda de treinamento: {epoch_loss:.4f}")

        # Avaliação do modelo no conjunto de validação
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for val_images_batch, val_masks_batch in val_loader:
                val_images_batch = val_images_batch.to(MP['device'])
                val_masks_batch = val_masks_batch.to(MP['device'])

                outputs = model(val_images_batch)
                loss = criterion(outputs, val_masks_batch)

                val_loss += loss.item() * val_images_batch.size(0)
                val_samples += val_images_batch.size(0)

                preds = torch.sigmoid(outputs).cpu().numpy()
                masks = val_masks_batch.cpu().numpy()

                all_preds.append(preds)
                all_masks.append(masks)

        val_loss /= val_samples
        print(f"Perda de validação: {val_loss:.4f}")

        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        get_metrics(all_preds, all_masks, Craters['val'], dim, device=MP['device'])

        # Salvando checkpoints
        if MP['save_models'] == 1:
            save_checkpoint(model, optimizer, epoch, MP['checkpoint_path'])

    # Após todas as épocas, salva o modelo completo
    if MP['save_models'] == 1:
        torch.save(model.state_dict(), MP['save_dir'])
        print(f"Modelo completo salvo em {MP['save_dir']}")

    # Avaliação do modelo no conjunto de teste
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for test_images_batch, test_masks_batch in test_loader:
            test_images_batch = test_images_batch.to(MP['device'])
            test_masks_batch = test_masks_batch.to(MP['device'])

            outputs = model(test_images_batch)

            preds = torch.sigmoid(outputs).cpu().numpy()
            masks = test_masks_batch.cpu().numpy()

            all_preds.append(preds)
            all_masks.append(masks)

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    get_metrics(all_preds, all_masks, Craters['test'], dim, device=MP['device'])

########## Função para Calcular Métricas Personalizadas ##########

# Função que calcula métricas de recall, precisão, erro médio e outras estatísticas
def get_metrics(preds, masks, craters, dim, beta=1, device='cpu'):
    n_samples = preds.shape[0]
    csvs = []
    minrad, maxrad, cutrad = 3, 50, 0.8
    diam = 'Diameter (pix)'
    for i in range(n_samples):
        csv = craters[prep.generate_hdf5_id(i)]
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T if len(csv) >= 3 else [-1]
        csvs.append(csv_coords)

    recall_list, precision_list, fscore_list = [], [], []
    err_lo_list, err_la_list, err_r_list, frac_duplicates_list = [], [], [], []

    for i in range(n_samples):
        if len(csvs[i]) < 3:
            continue
        pred = preds[i][0]
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = cd.compare_crater_detections_with_csv(pred, csvs[i], rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / (N_match + (N_detect - N_match))
            r = float(N_match) / N_csv
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            recall_list.append(r)
            precision_list.append(p)
            fscore_list.append(f)
            err_lo_list.append(elo)
            err_la_list.append(ela)
            err_r_list.append(er)
            frac_duplicates_list.append(frac_dupes)

    if len(recall_list) > 3:
        print(f"Recall médio e desvio padrão: {np.mean(recall_list):.4f}, {np.std(recall_list):.4f}")
        print(f"Precisão média e desvio padrão: {np.mean(precision_list):.4f}, {np.std(precision_list):.4f}")
        print(f"F{beta}-score médio e desvio padrão: {np.mean(fscore_list):.4f}, {np.std(fscore_list):.4f}")
        print(f"Erro médio de longitude: {np.mean(err_lo_list):.4f}")
        print(f"Erro médio de latitude: {np.mean(err_la_list):.4f}")
        print(f"Erro médio de raio: {np.mean(err_r_list):.4f}")
        print(f"Fração média de duplicatas: {np.mean(frac_duplicates_list):.4f}")

########## Função para Obter Hiperparâmetros Iteráveis ##########

# Função para obter um hiperparâmetro específico
def get_param_i(param, i):
    return param[i] if len(param) > i else param[0]

########## Funções de Checkpoint ##########

# Função para salvar checkpoints do modelo
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint salvo na época {epoch+1}")

# Função para carregar checkpoints do modelo
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint carregado. Retomando da época {start_epoch}")
    return start_epoch

########## Função Principal para Carregar Dados e Iniciar Treinamento ##########

# Função principal para carregar dados e iniciar o treinamento
def get_models(MP):
    dir = MP['dir']
    num_train, num_val, num_test = MP['num_train'], MP['num_val'], MP['num_test']

    train = h5py.File(f'{dir}train/train_images.hdf5', 'r')
    val = h5py.File(f'{dir}val/val_images.hdf5', 'r')
    test = h5py.File(f'{dir}test/test_images.hdf5', 'r')

    Data = {
        'train': [train['input_images'][:num_train].astype('float32'),
                  train['target_masks'][:num_train].astype('float32')],
        'val': [val['input_images'][:num_val].astype('float32'),
                val['target_masks'][:num_val].astype('float32')],
        'test': [test['input_images'][:num_test].astype('float32'),
                 test['target_masks'][:num_test].astype('float32')]
    }

    train.close()
    val.close()
    test.close()

    prep.normalize_and_resize_images(Data)

    Craters = {
        'train': pd.HDFStore(f'{dir}train/train_craters.hdf5', 'r'),
        'val': pd.HDFStore(f'{dir}val/val_craters.hdf5', 'r'),
        'test': pd.HDFStore(f'{dir}test/test_craters.hdf5', 'r')
    }

    MP['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {MP["device"]}')

    for i in range(MP['N_runs']):
        train_and_evaluate_model(Data, Craters, MP, i)