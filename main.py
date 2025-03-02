import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Dataset import CustomDataset
from model import CustomTNT
from config import Config
from utils.transforms import train_transform, val_transform
from utils.losses import custom_loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = custom_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = custom_loss(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

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
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': Config.LEARNING_RATE_BACKBONE},
        {'params': model.up.parameters(), 'lr': Config.LEARNING_RATE_UP}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    writer = SummaryWriter(Config.TENSORBOARD_LOG_DIR)

    best_val_loss = float('inf')
    num_epochs = Config.NUM_EPOCHS

    train_dataset = CustomDataset(Config.TRAIN_IMG_FOLDER, Config.TRAIN_CSV_FOLDER, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataset = CustomDataset(Config.TEST_IMG_FOLDER, Config.TEST_CSV_FOLDER, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        avg_val_loss = evaluate(model, val_dataloader, device)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_SAVE_PATH)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    writer.close()
    torch.save(model.state_dict(), Config.FINAL_MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(Config.BEST_MODEL_SAVE_PATH))
    predict_and_save(model, Config.TEST_IMG_FOLDER, val_transform, device, Config.OUTPUT_FOLDER)