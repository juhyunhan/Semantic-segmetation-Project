from data.build_loader import build_dataloader
from models.u_net import UNet
from models.loss import UNet_metric
from utils.utils import save_model
import torch

def train_one_epoch(dataloaders, model, optimizer, criterion, device):
    losses = {}
    dice_coefficients = {}
    
    for phase in ["train", "val"]:
        running_loss = 0.0
        running_dice_coeff = 0.0
        
        if phase == "train":
            model.train()
        else:
            model.eval()
        
        for index, batch in enumerate(dataloaders[phase]):
            images = batch[0].to(device)
            targets = batch[1].to(device)
            
            with torch.set_grad_enabled(phase == "train"):
                predictions = model(images)
                loss, dice_coefficient = criterion(predictions, targets)
                
                if phase == "train":
                    optimizer.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item()
            running_dice_coeff += dice_coefficient.item()
            
            if phase == "train":
                if index % 2 == 0:
                    text = f"{index}/{len(dataloaders[phase])}" + \
                            f" - Running Loss: {loss.item():.4f}" + \
                            f" - Running Dice: {dice_coefficient.item():.4f}" 
                    print(text)

        losses[phase] = running_loss / len(dataloaders[phase])
        dice_coefficients[phase] = running_dice_coeff / len(dataloaders[phase])
    return losses, dice_coefficients

def main():
    DEVICE = torch.device('cpu')

    #! logger 선언
    
    #! dataloader 선언
    data_dir = "../cv-project/MEDICAL/MEDICAL-DATASET-001/Segmentation/"
    dataloaders = build_dataloader(data_dir)
    
    #! Model 선언
    num_classes = 4
    model = UNet(num_classes=num_classes)
    model = model.to(DEVICE)
    
    #! Loss 선언
    criterion = UNet_metric(num_classes=num_classes)
    
    #! optimizer 선언
    optimizer = torch.optim.SGD(model.parameters(), lr= 1E-3, momentum=0.9)
    
    #! scheduler 선언 
    #안함
    
    #! training & evalutae 
    num_epochs = 10
    best_epoch = 0
    best_score = 0.0
    train_loss, train_dice_coefficient = [], []
    val_loss, val_dice_coefficient = [], []

    for epoch in range(num_epochs):
        losses, dice_coefficients = train_one_epoch(dataloaders, model, optimizer, criterion, DEVICE)
        train_loss.append(losses["train"])
        val_loss.append(losses["val"])
        train_dice_coefficient.append(dice_coefficients["train"])
        val_dice_coefficient.append(dice_coefficients["val"])
        
        print(f"{epoch}/{num_epochs} - Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
        print(f"{epoch}/{num_epochs} - Train Dice Coeff: {dice_coefficients['train']:.4f}, Val Dice Coeff: {dice_coefficients['val']:.4f}")
        
        if (epoch > 3) and (dice_coefficients["val"] > best_score):
            best_epoch = epoch
            best_score = dice_coefficients["val"]
            save_model(best_epoch, best_score, model.state_dict(), f"model_{epoch:02d}.pth")
            
    print(f"Best epoch: {best_epoch} -> Best Dice Coeffient: {best_score:.4f}")
   
    

if __name__ == '__main__':
    main()