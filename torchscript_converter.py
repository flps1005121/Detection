import torch
import torch.nn as nn
from model_train import SimCLRNet, device

def convert_model_to_torchscript():
    # 加載預訓練模型
    # from your_model_file import YourModel
    model = SimCLRNet()
    model.load_state_dict(torch.load('output/simclr_mobilenetv3.pth', map_location=device))

    model.classifier = nn.Identity()
    
    # 設置模型為評估模式
    model.eval()
    
    # 轉換為 TorchScript 並保存
    scripted_model = torch.jit.script(model)
    scripted_model.save("output/simclr_mobilenetv3.pt")
    print("Model has been saved as simclr_mobilenetv3.pt")

if __name__ == "__main__":
    convert_model_to_torchscript()