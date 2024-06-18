#Información datos RM
def get_patients(path):

    import os

    patients = os.listdir(path) #Lista con todos los directorios de la ruta indicada

    patients_dict = {}
    
    for data in patients:
        patient = data.split("_")[0]
        if patient not in patients_dict.keys():
            patients_dict[patient] = 1
        else:
            patients_dict[patient] += 1
    
    number_patients = len(patients_dict.keys()) #Número total de pacientes
    number_rm = len(patients) #Número total de RM

    print("Paciente: Número de resonancias magnéticas",  "\n", f"{patients_dict}", "\n", f"Número de pacientes en este estudio {number_patients}", "\n", f"Número total de resonancias magnéticas {number_rm}")
    print("-"*20)

#Obtener los archivos DICOM

def get_DICOM(path):

    import os
    from glob import glob

    patients = os.listdir(path)

    for patient in patients:
        patient_folder = os.path.join(path, patient)
        if len(glob(os.path.join(patient_folder,"*.dcm"))) < 20:
            print(f"Paciente con resonancia magnética {patient}, con {len(glob(os.path.join(patient_folder,'*.dcm')))} slices, eliminado por tener menos de 20 slices")
        


#create Nifti files in folder volumenes


def move_dicom_to_nifti_folder(in_path, out_path):
    
    import dicom2nifti
    import os
    from pathlib import Path

    in_path = Path(in_path)
    for rm in in_path.iterdir():
        if rm.is_dir():
            rm_name = rm.name
            dicom2nifti.dicom_series_to_nifti(rm, os.path.join(out_path, rm_name + ".nii.gz"))


def main():
    get_patients("/mnt/master/netapp/1010167/Arnau/Poliquistosis_imagenes_DICOM_SEGMENTADAS/imagenes_DICOM")
    get_DICOM("/mnt/master/netapp/1010167/Arnau/Poliquistosis_imagenes_DICOM_SEGMENTADAS/imagenes_DICOM")

   

if __name__ == "__main__":
    main()