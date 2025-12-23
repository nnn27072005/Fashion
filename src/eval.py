import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

class Evaluator:
    def __init__(self, device):
        self.device = device
        # Cấu hình tính mAP theo chuẩn COCO
        self.map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    def update(self, outputs, target, target_sizes):
        """
        outputs: Output từ model (logits, pred_boxes, attr_logits)
        target: Ground truth từ dataloader
        target_sizes: Kích thước ảnh gốc (để scale box về đúng pixel)
        """
        # 1. Xử lý Output của Model (Post-processing)
        # YOLOS trả về box [0, 1], cần nhân với kích thước ảnh để ra pixel [x, y, w, h]
        # Tuy nhiên, torchmetrics cần [x_min, y_min, x_max, y_max]
        
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        attr_logits = outputs.attr_logits # [Batch, Queries, Num_Attributes]
        
        preds = []
        
        for i in range(len(logits)):
            # Lấy kích thước ảnh gốc
            h, w = target_sizes[i]
            scale_fct = torch.tensor([w, h, w, h]).to(self.device)
            
            # Lọc bớt các box có confidence thấp để tính cho nhanh
            prob = logits[i].softmax(-1)
            scores, labels = prob[..., :-1].max(-1) # Bỏ class 'background' cuối cùng
            
            # Chỉ lấy những box có điểm > 0.5 (hoặc thấp hơn tùy strategy)
            keep = scores > 0.1 
            
            # Scale box về pixel
            boxes = pred_boxes[i][keep] * scale_fct
            
            preds.append({
                "boxes": boxes,
                "scores": scores[keep],
                "labels": labels[keep],
                "attr_logits": attr_logits[i][keep] # Lưu attribute tương ứng box
            })

        # 2. Xử lý Target (Ground Truth)
        targets_formatted = []
        for t in target:
            targets_formatted.append({
                "boxes": t["boxes"].to(self.device),
                "labels": t["class_labels"].to(self.device),
                "attributes": t.get("attribute_labels", None).to(self.device)
            })

        # 3. Cập nhật metric mAP
        # Lưu ý: torchmetrics tự lo phần matching IoU
        self.map_metric.update(preds, targets_formatted)
        
        # 4. Tính Attribute Accuracy (Thủ công)
        # Logic: Với mỗi box dự đoán đúng (IoU > 0.5), so khớp attribute
        # Phần này khá phức tạp vì cần matching box. 
        # Để đơn giản hóa trong training log, ta tính Accuracy dạng multi-label cơ bản
        # (Sẽ hoàn thiện kỹ hơn ở bước Test)
        return 0.0 # Placeholder cho attr acc

    def compute(self):
        # Trả về dictionary kết quả mAP
        result = self.map_metric.compute()
        # Reset cho epoch sau
        self.map_metric.reset()
        return result