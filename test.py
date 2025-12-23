import torch
from torch.utils.data import DataLoader
from transformers import YolosImageProcessor, YolosConfig
from tqdm import tqdm
import json

from src.dataset import FashionDataset
from src.model import FashionYolos
from src.utils import collate_fn
from src.eval import Evaluator

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./checkpoints/best_model" # Đường dẫn model tốt nhất
    test_ann_file = "test_annotations.json" # File test
    
    print(f"Loading model from {model_path}...")
    processor = YolosImageProcessor.from_pretrained(model_path)
    model = FashionYolos.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load Test Dataset
    test_dataset = FashionDataset(
        root="./data/fashionpedia", 
        ann_file=test_ann_file, 
        feature_extractor=processor, 
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    
    evaluator = Evaluator(device)
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values)
            target_sizes = [(img.shape[1], img.shape[2]) for img in batch["pixel_values"]]
            
            evaluator.update(outputs, labels, target_sizes)
            
    metrics = evaluator.compute()
    
    # --- REPORT KPI ---
    print("\n" + "="*30)
    print("BÁO CÁO KẾT QUẢ TEST (FINAL REPORT)")
    print("="*30)
    print(f"mAP (Overall): {metrics['map']:.4f}")
    print(f"mAP@50 (Detection): {metrics['map_50']:.4f}")
    
    # Chỉ số quan trọng cho Small Objects (Nhẫn, Đồng hồ)
    print(f"mAP (Small Objects): {metrics['map_small']:.4f}") 
    print(f"mAP (Large Objects): {metrics['map_large']:.4f}")
    
    # Lưu kết quả ra file
    with open("test_results.json", "w") as f:
        # Convert tensor to float for JSON serializable
        json_metrics = {k: v.item() for k, v in metrics.items()}
        json.dump(json_metrics, f, indent=4)
    print("Kết quả chi tiết đã lưu vào test_results.json")

if __name__ == "__main__":
    test_model()