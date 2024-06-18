def preprocessing_data(input_folder):

    import numpy as np
    import os
    from glob import glob
    from monai.data import DataLoader, CacheDataset
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
        Orientationd
        )
    from monai.utils import set_determinism
    from  torch.utils.data import SequentialSampler
    

    set_determinism(seed = 0)
    
    path_test_volumes = glob(os.path.join(input_folder, "Volumenes", "val", "*.nii"))
    path_test_segmentation = glob(os.path.join(input_folder, "Segmentaciones", "val", "*.nii"))
    folder_volumes_val = os.listdir(os.path.join(input_folder, "Volumenes", "val"))
    folder_seg_val = os.listdir(os.path.join(input_folder, "Segmentaciones", "val"))

    test_nifties = [{"image": vol_name, "label": seg_name} for vol_name, seg_name in zip(path_test_volumes, path_test_segmentation)]

#Preprocesado de las imágenes 
    original_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"])
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
    original_ds = CacheDataset(data=test_nifties, transform=original_transform, cache_rate=1.0)
    original_loader = DataLoader(original_ds, sampler = SequentialSampler(original_ds), batch_size = 1)

    test_ds = CacheDataset(data=test_nifties, transform=val_transform, cache_rate=1.0)
    test_loader = DataLoader(test_ds, sampler = SequentialSampler(test_ds), batch_size = 1)

    return original_loader, test_loader, folder_volumes_val, folder_seg_val

def view_transform(original_loader, test_loader, folder_volumes_val, folder_seg_val, slice = 15):    
    
    from monai.utils import first
    import matplotlib.pyplot as plt
    view_test_patient = first(test_loader)
    view_original_patient = first(original_loader)  
    plt.figure(f"Visualización original/transformado")
    plt.subplot(2,2,1)
    plt.rc("figure", titlesize = 4)
    plt.title(f"Volumen paciente {folder_volumes_val[0][:-4]} tras transformación, slice {slice}")
    plt.imshow(view_test_patient["image"][0, 0, :, :, slice], cmap="gray")
    plt.subplot(2,2,3)
    plt.title(f"Volumen paciente {folder_volumes_val[0][:-4]} original, slice {slice}")
    plt.imshow(view_original_patient["image"][0, 0, :, :, slice], cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title(f"Segmentación paciente {folder_seg_val[0][13:-4]} tras transformación, slice {slice}")
    plt.imshow(view_test_patient["label"][0, 0, :, :,slice])
    plt.subplot(2,2,4)
    plt.title(f"Segmentación paciente {folder_seg_val[0][13:-4]} original, slice {slice}")
    plt.imshow(view_original_patient["label"][0, 0, :, :,slice])

    plt.show()

def main():

    original_loader, test_loader, folder_volumes_val, folder_seg_val = preprocessing_data("/mnt/master/netapp/1010167/Arnau/")

    view_transform(original_loader, test_loader, folder_volumes_val, folder_seg_val)

if __name__ == "__main__":
    main()