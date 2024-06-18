from data_transformation import preprocessing_data
from monai.utils import set_determinism
import numpy as np
import os
import time
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
set_determinism(seed = 0)
import torch

data_folder = "/mnt/master/netapp/1010167/Arnau/"

train_loader, test_loader= preprocessing_data(data_folder)

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True, include_background=True)
    value = 1 - dice_value(predicted, target).item()
    return value

#Creamos el modelo UNet 3D, los dos canales de salida uno es para el fondo de la imagen y el otro para la segmentación de los riñones
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


#Creamos la función de pérdida y el optimizador para la fase de backpropagation del entrenamiento, la métrica elegida es la pérdida DICE. 
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, include_background=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)


def train(model, train_loader, test_loader, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []

    start_time = time.time()    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            train_step += 1

            volume = batch_data["image"]
            if volume.shape == (1, 2, 256, 256, 32):
                volume = volume[:, :1, :, :, :] #Seleccionamos solo 1 de los canales en el caso que haya 2 canales en la RM
            label = batch_data["label"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad() #Establecemos el gradiente a 0
            outputs = model(volume) #Entrenamiento del modelo con los volúmenes de las RM
            
            train_loss = loss(outputs, label) #Función de pérdida DICE entre los valores predichos de la segmentación y la segmentación real
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train_UNet+new_norm.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train_UNet+new_norm.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader: #Evaluamos el modelo con el testing dataset

                    test_step += 1

                    test_volume = test_data["image"]
                    if test_volume.shape == (1, 2, 256, 256, 32):
                        test_volume = test_volume[:, :1, :, :, :]
                    test_label = test_data["label"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device))
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
                
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test_UNet+new_norm.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test_UNet+new_norm.npy'), save_metric_test)

                if epoch_metric_test > best_metric: #Guardamos el modelo con los weights actualizados si el valor DICE mejora
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model_UNet+new_norm.pth"))
                    patience = 60 #Reiniciamos el valor de patience
                else:
                    patience -= 1 #Reducimos el valor de patience
                    if patience == 0:
                        break #Finalizamos con el entrenamiento porque el modelo tarda mucho en mejorar
                
                print(
                    f"Epoch actual: {epoch + 1} valor dice medio: {test_metric:.4f}"
                    f"\nMejor valor dice medio: {best_metric:.4f} "
                    f"a epoch: {best_metric_epoch}"
                )

    total_time_training = time.time() - start_time
    print(
        f"Training completado, mejor métrica: {best_metric:.4f} "
        f"a epoch: {best_metric_epoch}"
        f"Tiempo total de training: {total_time_training}")

train(model, train_loader, test_loader, loss_function, optimizer, 600, data_folder)
