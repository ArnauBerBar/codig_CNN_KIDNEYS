
import nibabel as nib
import numpy as np
import os
from glob import glob
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
from torch.utils.data import SequentialSampler
from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet, SegResNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    RandAffined,
    ToTensord,
    Activations)
#Función para el cálculo de los volúmenes renales
def calculate_volume_nifti(nifti_file):
    nii = nib.load(nifti_file)
    img = nii.get_fdata() 

    voxel_dims = (nii.header["pixdim"])[1:4] #Seleccionamos el tamaño de los voxels en los diferentes ejes
    nonzero_voxel_count = np.count_nonzero(img) #Número de voxels de la segmentación
    voxel_volume = np.prod(voxel_dims) #Volumen total del espacio de la RM
    nonzero_voxel_volume = nonzero_voxel_count * voxel_volume #Volumen de los voxels de la segmentación

    return nonzero_voxel_volume/1000
    
def main():
    model_dir = "/mnt/master/netapp/1010167/Arnau/"
    
    path_test_volumes = glob(os.path.join(model_dir, "Volumenes", "val", "*.nii"))
    path_test_segmentation = glob(os.path.join(model_dir, "Segmentaciones", "val", "*.nii"))

    test_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_test_volumes, path_test_segmentation)]

#Transformamos los datos de validación, en este caso no aplicamos la deformación 3D para no alterar los volúmenes
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
            ScaleIntensityd(keys = ["image"], minv=0, maxv=1),
            RandAffined(keys = ["image", "label"],
                mode = ("bilinear", "nearest"),
                prob = 1.0,
                padding_mode = 'border'),
            ToTensord(keys=["image", "label"])
    ])

    test_ds = Dataset(data=test_nifties, transform=val_transform)
    test_loader = DataLoader(test_ds, sampler = SequentialSampler(test_ds), batch_size = 1)
    
#Usamos el modelo UNet 3D ya entrenado y optimizado para predecir los valores del testing dataset
    device_UNet = torch.device("cuda:1")
    model_UNet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device_UNet)
    
#Usamos el modelo SegResNet 3D ya entrenado y optimizado para predecir los valores del testing dataset    
    device_segresnet = torch.device("cuda:2")
    model_segresnet = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=16,   
    ).to(device_segresnet)

    model_UNet.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_UNet+new_norm.pth"), map_location="cuda:1"))
    model_UNet.eval()
    model_segresnet.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_Segresnet+new_norm.pth"), map_location="cuda:2"))
    model_segresnet.eval()

    count_pred = 0
    sw_batch_size = 4
    roi_size = (256, 256, 32)
    path_pred_nifti_UNet = os.path.join(model_dir, "pred_seg_UNet")
    path_pred_nifti_segresnet = os.path.join(model_dir, "pred_seg_segresnet")
    with torch.no_grad():
        for test_patient in test_loader:
            t_volume = test_patient['image']
            seg_info = test_nifties[count_pred]["label"]
            t_segmentation_file = nib.load(path_test_segmentation[0]) #Cargamos la segmentación real para obtener los datos de header 
            test_outputs_UNet = sliding_window_inference(t_volume.to(device_UNet), roi_size, sw_batch_size, model_UNet)
            test_outputs_segresnet = sliding_window_inference(t_volume.to(device_segresnet), roi_size, sw_batch_size, model_segresnet)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs_UNet = sigmoid_activation(test_outputs_UNet)
            test_outputs_segresnet = sigmoid_activation(test_outputs_segresnet)
            test_outputs_UNet = test_outputs_UNet > 0.55
            test_outputs_segresnet = test_outputs_segresnet > 0.55
            test_outputs_UNet = test_outputs_UNet.detach().cpu().numpy() #Transformamos las predicciones de metatensor a un array numpy
            test_outputs_segresnet = test_outputs_segresnet.detach().cpu().numpy()
            test_outputs_UNet = test_outputs_UNet[0,1,:,:,:] #Seleccionamos solo el canal de la segmentación de los riñones
            test_outputs_segresnet = test_outputs_segresnet[0,1,:,:,:]
            test_outputs_UNet = nib.Nifti1Image(test_outputs_UNet, affine = np.eye(4), header=t_segmentation_file.header) #Convertimos el array numpy de la predicción a archivo Nifti con los valores de header y affine de su segmentación real
            nib.save(test_outputs_UNet, os.path.join(path_pred_nifti_UNet, f"{os.path.basename(seg_info)}")) #Guardamos los archivos Nifti en la carpeta de predicciones
            test_outputs_segresnet = nib.Nifti1Image(test_outputs_segresnet, affine = np.eye(4), header=t_segmentation_file.header)
            nib.save(test_outputs_segresnet, os.path.join(path_pred_nifti_segresnet, f"{os.path.basename(seg_info)}"))
            count_pred += 1


    folder = "/mnt/master/netapp/1010167/Arnau/Segmentaciones/val" #Carpeta con los archivos de segmentación Nifti del testing dataset
    seg_files = glob(os.path.join(folder, "*.nii"))
    seg_files_pred_UNet = glob(os.path.join(path_pred_nifti_UNet, "*.nii"))
    seg_files_pred_segresnet = glob(os.path.join(path_pred_nifti_segresnet, "*.nii"))
    rm = []
    volums = []
    volums_pred_UNet = []
    volums_pred_segresnet = []
    

    for file in os.listdir(folder):
        rm.append(file[13:-4]) #Seleccionamos el nombre del paciente y la RM

    for volumn in seg_files:
        volums.append(calculate_volume_nifti(volumn)) #Calculamos el volumen renal total y lo añadimos a la lista de volums
    
    for volum_pred in seg_files_pred_UNet:
        volums_pred_UNet.append(calculate_volume_nifti(volum_pred)) #Calculamos el volumen renal total de las predicciones U-Net
    
    for volum_pred in seg_files_pred_segresnet:
        volums_pred_segresnet.append(calculate_volume_nifti(volum_pred)) #Calculamos el volumen renal total de las predicciones SegResNet

    volumenes_reales = pd.DataFrame({"Resonancia magnética": rm, "Volumen ml": volums}) #Convertimos en dataframe ambas listas con el nombre de la RM y su volumen renal total
    volumenes_pred_UNet = pd.DataFrame({"Resonancia magnética": rm, "Volumen pred ml": volums_pred_UNet})
    volumenes_pred_segresnet = pd.DataFrame({"Resonancia magnética": rm, "Volumen pred ml": volums_pred_segresnet})
    dataframe_volums_UNet = pd.concat([volumenes_reales, volumenes_pred_UNet], axis=1) #Dataframe con los VRT reales y los predichos por el modelo U-Net para compararlos
    dataframe_volums_segresnet = pd.concat([volumenes_reales, volumenes_pred_segresnet], axis=1) #Dataframe con los VRT reales y los predichos por el modelo SegResNet para compararlos
    volumenes_reales_ordenados = volumenes_reales.sort_values(by = "Volumen ml", ascending=False) #Ordenamos en orden decreciente el dataframe de los volúmenes renales totales
    volumenes_UNet_ordenados = volumenes_pred_UNet.sort_values(by="Volumen pred ml", ascending=False)
    volumenes_segresnet_ordenados = volumenes_pred_segresnet.sort_values(by="Volumen pred ml", ascending=False)
    
    print(dataframe_volums_UNet)
    print(dataframe_volums_segresnet)
    print(volumenes_reales_ordenados.to_string())
    print(volumenes_UNet_ordenados.to_string())
    print(volumenes_segresnet_ordenados.to_string())

    correlation_UNet, p_value_UNet = pearsonr(volumenes_reales["Volumen ml"], volumenes_pred_UNet["Volumen pred ml"]) #Obtenemos el coeficiente de correlación de Pearson entre los VRT reales y los predichos por U-Net
    correlation_segresnet, p_value_segresnet = pearsonr(volumenes_reales["Volumen ml"], volumenes_pred_segresnet["Volumen pred ml"]) #Obtenemos el coeficiente de correlación de Pearson entre los VRT reales y los predichos por SegResNet
    slope_UNet, intercept_UNet, r_value, p_value_UN, std_err_UNet = stats.linregress(pd.to_numeric(dataframe_volums_UNet["Volumen ml"]), pd.to_numeric(dataframe_volums_UNet["Volumen pred ml"])) #Obtenemos la pendiente y la ordenada en el origen de la función de regresión entre los VRT Reales y los predichos por U-Net
    slope_segresnet, intercept_segresnet, r_val, p_val, std_err_seg = stats.linregress(pd.to_numeric(dataframe_volums_segresnet["Volumen ml"]), pd.to_numeric(dataframe_volums_segresnet["Volumen pred ml"])) #Obtenemos la pendiente y la ordenada en el origen de la función de regresión entre los VRT Reales y los predichos por SegResNet

    #Gráfica con la recta de regresión entre los VRT Reales y los del modelo 3D U-Net
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
    x=np.asarray(dataframe_volums_UNet["Volumen ml"], dtype="float64"),
    y=np.asarray(dataframe_volums_UNet["Volumen pred ml"], dtype="float64"),
    line_kws={"color": "r"},
    ax=ax
    )
    ax.set_title("Regresión Volúmenes Renales Totales reales vs predichos modelo 3D U-Net")
    ax.set(xlabel = "VRT Real (mL)", ylabel = "VRT predicho (mL)")
    ax.set_xlim(0,5000)
    ax.set_ylim(0,5000)
    plt.text(1000, 4000, "y = {0:.3f}x+{1:.3f}".format(slope_UNet, intercept_UNet))
    plt.text(1000, 3500, "R2 = {:.3f}".format(correlation_UNet))
    plt.show()

    #Gráfica con la recta de regresión entre los VRT Reales y los del modelo 3D SegResNet
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
    x=np.asarray(dataframe_volums_segresnet["Volumen ml"], dtype="float64"),
    y=np.asarray(dataframe_volums_segresnet["Volumen pred ml"], dtype="float64"),
    line_kws={"label": "y = {0:.3f}x+{1:.3f}".format(slope_segresnet, intercept_segresnet),"color": "r"},
    ax=ax
    )
    ax.set_title("Regresión Volúmenes Renales Totales reales vs predichos modelo 3D SegResNet")
    ax.set(xlabel = "VRT Real (mL)", ylabel = "VRT predicho (mL)")
    ax.set_xlim(0,4000)
    ax.set_ylim(0,6000)
    plt.text(1000, 5000, "y = {0:.3f}x+{1:.3f}".format(slope_segresnet, intercept_segresnet))
    plt.text(1000, 4500, "R2 = {:.3f}".format(correlation_segresnet))
    plt.show()


if __name__ == "__main__":
    main()
