import os
import yaml
import torch
import math
from torch.utils.data import DataLoader
from transformers import DeformableDetrImageProcessor, DeformableDetrConfig
from tqdm import tqdm
from src.dataset import FashionDataset
from src.model import FashionDeformableDETR
from src.utils import collate_fn
from src.eval import Evaluator

def train_model():
    if not os.path.exists("config/config.yaml"): raise FileNotFoundError("Missing config")
    with open("config/config.yaml", "r") as f: config_dict = yaml.safe_load(f)
    
    cfg_train, cfg_model, cfg_sys = config_dict['training'], config_dict['model'], config_dict['system']
    device = torch.device(cfg_sys['device'] if torch.cuda.is_available() else "cpu")
    print(f"--- TRAINING ON {device} | CLASSES: 46 ---")
    os.makedirs(cfg_train['output_dir'], exist_ok=True)

    # 1. PROCESSOR
    processor = DeformableDetrImageProcessor.from_pretrained(
        cfg_model['base_model'],
        size={"shortest_edge": 800, "longest_edge": 1333} 
    )

    # 2. DATASETS (HuggingFace)
    DATASET_NAME = "detection-datasets/fashionpedia"
    train_dataset = FashionDataset(DATASET_NAME, "train", processor, train=True, num_attributes=cfg_model['num_attributes'])
    val_dataset = FashionDataset(DATASET_NAME, "val", processor, train=False, num_attributes=cfg_model['num_attributes'])

    # 3. MODEL
    # 46 Object + 1 Background = 47 Labels
    config = DeformableDetrConfig.from_pretrained(cfg_model['base_model'])
    config.num_labels = 47 
    config.num_attributes = cfg_model['num_attributes']
    
    model = FashionDeformableDETR.from_pretrained(cfg_model['base_model'], config=config, ignore_mismatched_sizes=True)
    model.to(device)

    # 4. DATALOADER
    # Dùng Weighted Sampler vì EDA báo động về Imbalance
    train_sampler = train_dataset.get_weighted_sampler()
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg_train['batch_size'],
        sampler=train_sampler, shuffle=False, 
        collate_fn=collate_fn, num_workers=cfg_sys['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg_train['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=cfg_sys['num_workers']
    )

    # 5. OPTIMIZER
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": cfg_train['lr']},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg_train['lr'] * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])
    evaluator = Evaluator(device)
    best_map = 0.0

    # 6. LOOP
    for epoch in range(cfg_train['epochs']):
        print(f"\n{'='*10} Epoch {epoch + 1}/{cfg_train['epochs']} {'='*10}")
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss / cfg_train.get('gradient_accumulation_steps', 1)
            loss.backward()
            
            if (i + 1) % cfg_train.get('gradient_accumulation_steps', 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['clip_max_norm'])
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * cfg_train.get('gradient_accumulation_steps', 1)

        print(f"Avg Train Loss: {train_loss / len(train_loader):.4f}")

        # VAL
        print("Running Validation...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                
                target_sizes = torch.tensor([img.shape[1:] for img in pixel_values]).to(device)
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)
                
                clean_targets = []
                for t in batch["labels"]:
                    clean_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                    clean_targets.append(clean_t)
                evaluator.update(results, clean_targets, target_sizes)

        metrics = evaluator.compute()
        if metrics['map'] >= best_map:
            best_map = metrics['map']
            print(f"*** Saving Best Model (mAP: {best_map:.4f}) ***")
            model.save_pretrained(os.path.join(cfg_train['output_dir'], "best_model"))
            processor.save_pretrained(os.path.join(cfg_train['output_dir'], "best_model"))

if __name__ == "__main__":
    train_model()