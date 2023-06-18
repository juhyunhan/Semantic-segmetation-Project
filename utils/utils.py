import torch
import os

def save_model(best_epoch, best_score, model_params, name):
    base_dir = "../logs/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    PATH = os.path.join(base_dir, name)
    torch.save(model_params, PATH)
    print(f"Model save at {PATH}, score : {best_score}, epoch : {best_epoch}")