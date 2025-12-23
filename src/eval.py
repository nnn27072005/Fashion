import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Evaluator:
    def __init__(self, device):
        self.device = device
        # Cấu hình tính mAP theo chuẩn COCO (xyxy: xmin, ymin, xmax, ymax)
        self.map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    def box_cxcywh_to_xyxy(self, x):
        """
        Hàm hỗ trợ chuyển đổi format box từ (cx, cy, w, h) sang (x1, y1, x2, y2)
        x: Tensor shape (N, 4)
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def update(self, outputs, target, target_sizes):
        """
        outputs: Output từ model (logits, pred_boxes)
        target: Ground truth từ dataloader (List of Dicts)
        target_sizes: Kích thước ảnh gốc (tensor [Batch, 2] -> (h, w))
        """
        
        # --- 1. Xử lý Output của Model (Predictions) ---
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes # Đang là (cx, cy, w, h) NORMALIZED [0-1]
        
        preds = []
        
        for i in range(len(logits)):
            # Lấy kích thước ảnh gốc (h, w)
            h_img, w_img = target_sizes[i]
            
            # Tạo vector scale (w, h, w, h) để nhân vào box
            scale_fct = torch.tensor([w_img, h_img, w_img, h_img]).to(self.device)
            
            # Softmax & Filter score
            prob = logits[i].softmax(-1)
            scores, labels = prob[..., :-1].max(-1)
            
            # Lấy các box có score > 0 (Lấy hết để tính mAP cho chuẩn, hoặc > 0.05 để nhẹ máy)
            keep = scores > 0.0 
            
            # A. Scale từ 0-1 ra Pixel (vẫn là cx, cy, w, h)
            scaled_boxes = pred_boxes[i][keep] * scale_fct
            
            # B. Convert từ (cx, cy, w, h) -> (x1, y1, x2, y2) <-- BƯỚC QUAN TRỌNG ĐÃ THÊM
            final_boxes = self.box_cxcywh_to_xyxy(scaled_boxes)
            
            preds.append({
                "boxes": final_boxes,
                "scores": scores[keep],
                "labels": labels[keep]
            })

        # --- 2. Xử lý Target (Ground Truth) ---
        targets_formatted = []
        
        for i, t in enumerate(target):
            # Lấy kích thước ảnh tương ứng
            h_img, w_img = target_sizes[i]
            scale_fct = torch.tensor([w_img, h_img, w_img, h_img]).to(self.device)

            # Target box từ Dataset cũng đang là (cx, cy, w, h) NORMALIZED
            tgt_boxes = t["boxes"].to(self.device)
            
            # A. Scale ra Pixel
            tgt_boxes_pixel = tgt_boxes * scale_fct
            
            # B. Convert sang (x1, y1, x2, y2)
            tgt_boxes_xyxy = self.box_cxcywh_to_xyxy(tgt_boxes_pixel)

            targets_formatted.append({
                "boxes": tgt_boxes_xyxy,
                "labels": t["class_labels"].to(self.device)
            })

        # --- 3. Cập nhật metric ---
        # Lúc này cả Preds và Targets đều đã là: Pixel tuyệt đối & Format XYXY
        self.map_metric.update(preds, targets_formatted)
        
        return 0.0

    def compute(self):
        # 1. Tính toán
        result = self.map_metric.compute()
        
        # 2. Reset cho epoch sau
        self.map_metric.reset()
        
        # 3. TRÍCH XUẤT CHỈ SỐ QUAN TRỌNG
        # result là một dict chứa rất nhiều key: map, map_50, map_75, map_small...
        
        print("\n" + "="*40)
        print("📊 CHI TIẾT HIỆU NĂNG (COCO METRICS)")
        print("="*40)
        
        # --- A. Tổng quan ---
        print(f"⭐ mAP (0.50:0.95): {result['map'].item():.4f}  (Mục tiêu: >0.5)")
        print(f"   mAP@50          : {result['map_50'].item():.4f}")
        print(f"   mAP@75          : {result['map_75'].item():.4f}")
        
        # --- B. Phân theo kích thước (Quan trọng cho đồ án của bạn) ---
        print("-" * 20)
        print(f"🐜 AP_small (Đồ nhỏ): {result['map_small'].item():.4f}")
        print(f"Medium AP_medium     : {result['map_medium'].item():.4f}")
        print(f"🐘 AP_large (Đồ to) : {result['map_large'].item():.4f}")
        
        # --- C. Recall (Độ nhạy - KPI của bạn) ---
        print("-" * 20)
        print(f"📡 Recall_small      : {result['mar_small'].item():.4f} (Mục tiêu: >0.4 - 0.7)")
        print(f"   Recall_large      : {result['mar_100'].item():.4f}")
        
        # --- D. Trả về dict để main log hoặc lưu model ---
        # Bạn có thể chọn map_small làm tiêu chí lưu model nếu muốn ưu tiên đồ nhỏ
        return result