
import numpy as np
import os
import matplotlib.pyplot as plt
import torch


model_dir = "/mnt/master/netapp/1010167/Arnau/"

train_loss = np.load(os.path.join(model_dir, 'loss_train_UNet+new_norm.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train_UNet+new_norm.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test_UNet+new_norm.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test_UNet+new_norm.npy'))

plt.figure("Resultados 12 de mayo", (12, 6))
plt.suptitle("U-Net model", size=16)
plt.subplot(2, 2, 1)
plt.title("Train DICE loss") #Gráfica con la evolución de la función de pérdida Dice media del training
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.ylim(0,1)
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Train DICE coeficient") #Gráfica con la evolución del coeficiente Dice medio del training
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.ylim(0,1)
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title("Test DICE loss") #Gráfica con la evolución de la función de pérdida Dice media de la evaluación del modelo
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.ylim(0, 1)
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title("Test DICE coeficient") #Gráfica con la evolución del coeficiente Dice medio de la evaluación del modelo
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.ylim(0,1)
plt.xlabel("epoch")
plt.plot(x, y)

plt.show()

from glob import glob
from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    Spacingd,
    ScaleIntensityd,
    RandAffined,
    Rand3DElasticd,
    ToTensord,
    Activations
)
from monai.data import DataLoader, CacheDataset

path_train_volumes = glob(os.path.join(model_dir, "Volumenes", "train", "*.nii"))
path_train_segmentation = glob(os.path.join(model_dir, "Segmentaciones", "train", "*.nii"))

path_test_volumes = glob(os.path.join(model_dir, "Volumenes", "val", "*.nii"))
path_test_segmentation = glob(os.path.join(model_dir, "Segmentaciones", "val", "*.nii"))

train_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_train_volumes, path_train_segmentation)]
test_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_test_volumes, path_test_segmentation)]

#Aplicamos el data augmentation al testing dataset
val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(keys = ["image", "label"],
                mode = ("bilinear", "nearest"),
                spatial_size = (256, 256, 32)),
            Spacingd(keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
            ScaleIntensityd(keys = ["image"], minv=0, maxv=1),
            RandAffined(keys = ["image", "label"],
                mode = ("bilinear", "nearest"),
                prob = 1.0,
                spatial_size = (256, 256, 32),
                translate_range = (40, 40, 2),
                rotate_range = (np.pi/36, np.pi/36, np.pi/4),
                scale_range = (0.15, 0.15, 0.15),
                padding_mode = 'border'),
            Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                spatial_size = (256, 256, 32),
                translate_range=(50, 50, 2),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border"),
            ToTensord(keys=["image", "label"])
            
    ])

test_ds = CacheDataset(data=test_nifties, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size = 1)

device_UNet = torch.device("cuda:2")
model_UNet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device_UNet)

device_segresnet = torch.device("cuda:1")
model_segresnet = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=16,   
).to(device_segresnet)

#Cargamos el modelo 3D U-Net
model_UNet.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_UNet+new_norm.pth"), map_location="cuda:2"))
model_UNet.eval()

#Cargamos el modelo 3D SegResNet
model_segresnet.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_Segresnet+new_norm.pth"), map_location="cuda:1"))
model_segresnet.eval()
count = 0
with torch.no_grad():
    for i, test_patient in enumerate(test_loader):
        sw_batch_size = 4
        roi_size = (256, 256, 32)
        t_volume = test_patient['image']
        test_outputs_UNet = sliding_window_inference(t_volume.to(device_UNet), roi_size, sw_batch_size, model_UNet)
        test_outputs_segresnet = sliding_window_inference(t_volume.to(device_segresnet), roi_size, sw_batch_size, model_segresnet)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs_UNet = sigmoid_activation(test_outputs_UNet)
        test_outputs_segresnet = sigmoid_activation(test_outputs_segresnet)
        test_outputs_UNet = test_outputs_UNet > 0.55
        test_outputs_segresnet = test_outputs_segresnet > 0.55
        
        for element in range(3):

            sl_position = 0
            if element == 0:
                sl_position = 5 #Seleccionamos el corte 5 de la RM
            if element == 1:
                sl_position = 15 #Seleccionamos el corte 15 de la RM
            if element == 2:
                sl_position = 28 #Seleccionamos el corte 28 de la RM
            
            plt.figure("check", figsize=(18, 6))
            plt.subplot(1, 4, 1)
            plt.title(f"image {i}, slice: {sl_position}")
            plt.imshow(test_patient["image"][0, 0, :, :, sl_position], cmap="gray") #Ponemos el corte X de la RM en el plot
            plt.subplot(1, 4, 2)
            plt.title(f"segmentation {i}, slice: {sl_position}")
            plt.imshow(test_patient["label"][0, 0, :, :, sl_position] != 0) #Ponemos la máscara de segmentación manual del corte X de la RM en el plot
            plt.subplot(1, 4, 3)
            plt.title(f"prediction U-Net {i}, slice: {sl_position}")
            plt.imshow(test_outputs_UNet.detach().cpu()[0, 1, :, :, sl_position]) #Ponemos la máscara de segmentación predicha por el modelo 3D U-Net del corte X de la RM en el plot
            plt.subplot(1, 4, 4)
            plt.title(f"prediction SegResNet {i}, slice: {sl_position}")
            plt.imshow(test_outputs_segresnet.detach().cpu()[0, 1, :, :, sl_position]) #Ponemos la máscara de segmentación predicha por el modelo 3D SegResNet del corte X de la RM en el plot
            plt.show()
        
        count += 1

        if count == 3:
            break