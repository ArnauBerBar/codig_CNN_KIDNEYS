import numpy as np
import shutil
import random
import os
from glob import glob
from monai.data import DataLoader, Dataset
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    Spacingd,
    ScaleIntensityd,
    RandAffined,
    Rand3DElasticd,
    ToTensord,
    Orientationd)
from monai.utils import set_determinism

#Creamos una función para hacer el data augmentation de las imágenes disponibles
def preprocessing_data(input_folder):

#Establecemos la semilla a 0 y así poder tener resultados reproducibles en el caso de repetir el proceso
    set_determinism(seed = 0)

#Creamos una variable lista con todas las rutas de los archivos Nifti del training
    path_train_volumes = glob(os.path.join(input_folder, "Volumenes", "train", "*.nii"))
    path_train_segmentation = glob(os.path.join(input_folder,"Segmentaciones", "train", "*.nii"))

#Creamos una variable lista con todas las rutas de los archivos Nifti del testing
    path_test_volumes = glob(os.path.join(input_folder, "Volumenes", "val", "*.nii"))
    path_test_segmentation = glob(os.path.join(input_folder, "Segmentaciones", "val", "*.nii"))


    train_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_train_volumes, path_train_segmentation)]
    test_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_test_volumes, path_test_segmentation)]

#Modificación de las imágenes para data augmentation de ambos datasets

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]), #Cargar imagen
            EnsureChannelFirstd(keys=["image", "label"]), #Asegurar que el canal esté el primero en el tensor
            Orientationd(keys=["image", "label"], axcodes="RAS"), #Orientación de los ejes 3D
            Resized(keys = ["image", "label"],
                mode = ("bilinear", "nearest"),
                spatial_size = (256, 256, 32)),
            Spacingd(keys=["image", "label"], #Dimensión de los píxeles
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
            ScaleIntensityd(keys = ["image"], minv=0, maxv=1), #Normalización de la intensidad de la señal de las RM
            RandAffined(keys = ["image", "label"],
                mode = ("bilinear", "nearest"),
                prob = 1.0,
                spatial_size = (256, 256, 32),
                translate_range = (40, 40, 2),
                rotate_range = (np.pi/36, np.pi/36, np.pi/4),
                scale_range = (0.15, 0.15, 0.15),
                padding_mode = 'border'),
            Rand3DElasticd( #Deformación del volumen, recorte del volumen a 256X256X32
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size = (256, 256, 32),
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                translate_range=(50, 50, 2),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border"),
            ToTensord(keys=["image", "label"]) #Transformación a tensor
        ]
    )
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

    train_ds = Dataset(data=train_nifties, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1)

    test_ds = Dataset(data=test_nifties, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader

#Creamos una función para hacer las carpetas train, val sin que haya volumenes o segmentaciones de un paciente en ambas carpetas
def custom_splitfolders(input_folder, output_folder, patient_position = 0, split_ratio=(0.8, 0.2), seed=1337):
    
    #Crear carpetas según split_ratio
    if len(split_ratio)== 3:
        os.makedirs(os.path.join(output_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "val"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "test"), exist_ok=True)
    elif len(split_ratio) == 2:
        os.makedirs(os.path.join(output_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "val"), exist_ok=True)
    else:
        print("len of split_ratio must be 2 or 3")
        exit()
    #Crear un diccionario con los archivos de cada paciente
    dict_patients = {}

    #Se pone como llave el nombre del paciente y los valores cada archivo Nifti del paciente
    for file in glob(os.path.join(input_folder, "*.nii")):
        patient_name = (file.split("/")[-1].split("_")[patient_position])
        if patient_name not in dict_patients.keys():
            dict_patients[patient_name] = [file]
        else:
            dict_patients[patient_name].append(file)
    
    #Cantidad de archivos en la carpeta train, val y test respectivamente
    if len(split_ratio)== 2:
        prop_in_train = int(split_ratio[0]*len(dict_patients.values())) 
        
        random.seed(seed) #Establecemos la semilla para tener resultados reproducibles
        patients_unique = list(sorted(dict_patients.keys()))

        #Barajamos los pacientes
        random.shuffle(patients_unique)
        
        files_in_train = []
        files_in_val = []
        
        for patient in patients_unique:
            #Añadimos archivos de paciente a la lista de train
            if len(files_in_train) < prop_in_train:
                files_in_train.append(dict_patients[patient])
            #Añadimos archivos de paciente a la lista de val
            else:
                files_in_val.append(dict_patients[patient])
        
        #Copiamos archivo de la lista de train a la carpeta train
        for files in files_in_train:
            for nifti in files:
                shutil.copy(nifti, os.path.join(output_folder, "train"))
        #Copiamos archivo de la lista de train a la carpeta val
        for files in files_in_val:
            for nifti in files:
                shutil.copy(nifti, os.path.join(output_folder, "val"))

    elif len(split_ratio)== 3:
        prop_in_train = int(split_ratio[0]*len(dict_patients.values())) 
        prop_in_val = int(split_ratio[1]*len(dict_patients.values())) 

        random.seed(seed) #Establecemos la semilla para tener resultados reproducibles
        patients_unique = list(dict_patients.keys())

        #Barajamos los pacientes
        random.shuffle(patients_unique)
        
        files_in_train = []
        files_in_val = []
        files_in_test = []

        for patient in patients_unique:
            if len(files_in_train) < prop_in_train:
                files_in_train.append(dict_patients[patient])
            elif len(files_in_val) < prop_in_val:
                files_in_val.append(dict_patients[patient])
            else:
                files_in_test.append(dict_patients[patient])
        
        for files in files_in_train:
            for nifti in files:
                shutil.copy(nifti, os.path.join(output_folder, "train"))

        for files in files_in_val:
            for nifti in files:
                shutil.copy(nifti, os.path.join(output_folder, "val"))

        for files in files_in_test:
            for nifti in files:
                shutil.copy(nifti, os.path.join(output_folder, "test"))
        

def main():

    volumenes = "/mnt/master/netapp/1010167/Arnau/Volumenes"
    segmentaciones = "/mnt/master/netapp/1010167/Arnau/Segmentaciones"
    
    custom_splitfolders(input_folder=os.path.join(volumenes, "data"), patient_position=0, output_folder=volumenes)
    custom_splitfolders(input_folder=os.path.join(segmentaciones, "data"), patient_position=1, output_folder=segmentaciones) 

if __name__ == "__main__":
    main()