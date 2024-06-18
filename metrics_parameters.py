from pytorch_model_summary import summary
import torch
from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm

device_segresnet = torch.device("cuda:2")

model_segresnet = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=16,   
).to(device_segresnet)

device_UNet = torch.device("cuda:0")
model_UNet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device_UNet)

input_tensor_segresnet = torch.zeros(1,1,256,256,32).to(device_segresnet) #Creamos un tensor de 0 con el shape de input del trainig del modelo SegResNet
input_tensor_UNet = torch.zeros(1,1,256,256,32).to(device_UNet) #Creamos un tensor de 0 con el shape de input del trainig del modelo U-Net
model_segresnet.load_state_dict(torch.load("/mnt/master/netapp/1010167/Arnau/best_metric_model_Segresnet+new_norm.pth", map_location="cuda:2")) #Cargamos el modelo SegResNet
model_segresnet.eval()
model_UNet.load_state_dict(torch.load("/mnt/master/netapp/1010167/Arnau/best_metric_model_UNet+new_norm.pth", map_location="cuda:0")) #Cargamos el modelo U-Net
model_UNet.eval()
print(summary(model_segresnet, input_tensor_segresnet))
print(summary(model_UNet, input_tensor_UNet))