import os
import torch
from PIL import Image
import pandas as pd
from model import CustomTNT
from config import Config
from utils.transforms import val_transform

def predict_and_save(model, img_folder, transform, device, output_folder):
    model.eval()
    with torch.no_grad():
        for img_name in os.listdir(img_folder):
            prefix = img_name.split('.')[0]
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            output = model(img)
            output_matrix = output.cpu().numpy().reshape(*Config.OUTPUT_SIZE)  # 使用配置中的输出大小
            output_csv_path = os.path.join(output_folder, prefix + ".csv")
            pd.DataFrame(output_matrix).to_csv(output_csv_path, index=False, header=False)

if __name__ == '__main__':
    device = torch.device(Config.DEVICE)
    model = CustomTNT().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_SAVE_PATH))
    predict_and_save(model, Config.TEST_IMG_FOLDER, val_transform, device, Config.OUTPUT_FOLDER)