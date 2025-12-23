import os
import yaml
import torch
import math
from torch.utils.data import DataLoader
from transformers import YolosImageProcessor, YolosConfig
from tqdm import tqdm
import time

# Import các module local đã xây dựng
from src.dataset import FashionDataset
from src.model import FashionYolos
from src.utils import collate_fn
from src.eval import Evaluator

def train_model():
    # ---------------------------------------------------------
    # 1. SETUP & CONFIGURATION
    # ---------------------------------------------------------
    # Load cấu hình
    if not os.path.exists("config/config.yaml"):
        raise FileNotFoundError("Không tìm thấy file config/config.yaml")

    with open("config/config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    cfg_train = config_dict['training']
    cfg_model = config_dict['model']
    cfg_sys = config_dict['system']

    # Thiết lập thiết bị (GPU/CPU)
    device = torch.device(cfg_sys['device'] if torch.cuda.is_available() else "cpu")
    print(f"--- F&A MODEL TRAINING STARTED ON {device} ---")

    # Tạo thư mục output
    os.makedirs(cfg_train['output_dir'], exist_ok=True)

    # ---------------------------------------------------------
    # 2. MODEL & PROCESSOR INITIALIZATION
    # ---------------------------------------------------------
    print(f"Loading base model: {cfg_model['base_model']}...")
    
    # Processor (Dùng để resize/normalize ảnh)
    # size={"shortest_edge": 512, "longest_edge": 800} là setting chuẩn cho YOLOS Small
    processor = YolosImageProcessor.from_pretrained(
        cfg_model['base_model'],
        size={"shortest_edge": 512, "longest_edge": 800} 
    )
    
    # Config Model
    config = YolosConfig.from_pretrained(cfg_model['base_model'])
    config.num_labels = cfg_model['num_classes']
    config.num_attributes = cfg_model['num_attributes']
    
    # Khởi tạo Custom Model (FashionYolos)
    model = FashionYolos.from_pretrained(
        cfg_model['base_model'], 
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # ---------------------------------------------------------
    # 3. DATASET & DATALOADER SETUP
    # ---------------------------------------------------------
    print("Initializing Datasets...")
    
    # --- Train Dataset ---
    # train=True để kích hoạt Augmentation (Crop, Flip, ColorJitter...)
    train_dataset = FashionDataset(
        img_dir=cfg_train['train_img_dir'],
        ann_file=cfg_train['train_ann'],
        feature_extractor=processor,
        train=True, 
        num_attributes=cfg_model['num_attributes']
    )
    
    # Imbalance Handling: Lấy WeightedSampler
    # Giúp model nhìn thấy các ảnh "hiếm" (nhẫn, đồng hồ) nhiều hơn
    sampler = train_dataset.get_weighted_sampler()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_train['batch_size'],
        sampler=sampler, # Đã dùng sampler thì KHÔNG dùng shuffle=True
        collate_fn=collate_fn,
        num_workers=cfg_sys['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # --- Validation Dataset ---
    # train=False để chỉ Resize ảnh chuẩn, không Augmentation
    val_dataset = FashionDataset(
        img_dir=cfg_train['val_img_dir'],
        ann_file=cfg_train['val_ann'],
        feature_extractor=processor,
        train=False,
        num_attributes=cfg_model['num_attributes']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_train['batch_size'],
        shuffle=False, # Val set không cần shuffle
        collate_fn=collate_fn,
        num_workers=cfg_sys['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---------------------------------------------------------
    # 4. OPTIMIZER & EVALUATOR
    # ---------------------------------------------------------
    # Tách LR: Backbone train chậm hơn (1e-5), Head train nhanh hơn (1e-4)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": cfg_train['lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg_train['lr'] * 0.1, 
        },
    ]
    
    optimizer = torch.optim.AdamW(
        param_dicts, 
        lr=cfg_train['lr'], 
        weight_decay=cfg_train['weight_decay']
    )

    # Công cụ đánh giá (Tính mAP)
    evaluator = Evaluator(device)
    
    # Biến lưu trữ
    best_map = 0.0
    start_time = time.time()

    # ---------------------------------------------------------
    # 5. TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(cfg_train['epochs']):
        print(f"\n{'='*20} Epoch {epoch + 1}/{cfg_train['epochs']} {'='*20}")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Di chuyển dữ liệu sang GPU
            pixel_values = batch["pixel_values"].to(device)
            # labels là list of dicts, cần chuyển từng tensor trong dict sang device
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            # Loss Calculation
            # outputs.loss là Detection Loss (Hungarian Loss từ YOLOS)
            # Để đơn giản trong v1.0, ta optimize dựa trên detection loss trước.
            # (Attribute learning sẽ diễn ra ngầm qua shared backbone + attribute head gradients nếu được thêm vào total loss)
            loss = outputs.loss
            
            # Kiểm tra NaN Loss (tránh crash)
            if math.isnan(loss.item()):
                print("Warning: Loss is NaN, skipping batch.")
                continue

            # Backward Pass
            loss.backward()
            
            # Gradient Clipping (Tránh bùng nổ gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['clip_max_norm'])
            
            optimizer.step()
            
            # Logging
            train_loss += loss.item()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION PHASE ---
        # Chạy Validation mỗi epoch (hoặc có thể chỉnh thành mỗi 5 epoch để tiết kiệm thời gian)
        print("Running Validation & mAP Calculation...")
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                # Forward (Không truyền labels để lấy prediction)
                outputs = model(pixel_values=pixel_values)
                
                # Lấy kích thước tensor hiện tại làm target size tham chiếu
                target_sizes = [(img.shape[1], img.shape[2]) for img in batch["pixel_values"]]
                
                # Cập nhật metrics
                evaluator.update(outputs, labels, target_sizes)

        # Tính toán kết quả
        metrics = evaluator.compute()
        map_score = metrics['map'].item()
        map_50 = metrics['map_50'].item()
        
        print(f"Validation Results -> mAP: {map_score:.4f} | mAP@50: {map_50:.4f}")

        # --- CHECKPOINT SAVING ---
        # 1. Lưu Best Model (Dựa trên mAP)
        if map_score > best_map:
            best_map = map_score
            save_path = os.path.join(cfg_train['output_dir'], "best_model")
            
            print(f"*** NEW BEST MODEL found (mAP: {best_map:.4f}) -> Saving to {save_path} ***")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
        
        # 2. Lưu Last Checkpoint (Để resume nếu cần)
        last_path = os.path.join(cfg_train['output_dir'], "last_checkpoint")
        model.save_pretrained(last_path)
        processor.save_pretrained(last_path)

    # End Training
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours.")
    print(f"Best mAP achieved: {best_map:.4f}")

if __name__ == "__main__":
    train_model()