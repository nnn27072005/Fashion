import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import YolosForObjectDetection, YolosConfig
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

class FashionYolos(YolosForObjectDetection):
    def __init__(self, config: YolosConfig):
        super().__init__(config)
        
        # 1. Attribute Head (Nhánh mới)
        self.num_attributes = getattr(config, "num_attributes", 294)
        self.attribute_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, self.num_attributes)
        )
        
        # 2. Khởi tạo Weights cho Attribute Head
        for m in self.attribute_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 3. Định nghĩa các hàm Loss
        # Hàm loss cho Attributes
        self.attr_criterion = nn.BCEWithLogitsLoss()
        
        # Hàm loss cho Classification (Bao gồm cả class 'background')
        # config.num_labels là số class vật thể (46). Model sẽ output 47 class (0-45 là vật thể, 46 là background)
        # Chúng ta đặt weight cho background thấp hơn một chút (0.1) để model tập trung học vật thể
        empty_weight = torch.ones(self.config.num_labels + 1)
        empty_weight[-1] = 0.1 
        self.class_criterion = nn.CrossEntropyLoss(weight=empty_weight)

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- BƯỚC 1: CHẠY BASE MODEL ---
        # QUAN TRỌNG: Luôn truyền labels=None để chặn thư viện dùng sai hàm Loss gốc
        outputs = super().forward(
            pixel_values=pixel_values,
            labels=None, 
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Lấy Hidden States để tính Attribute
        sequence_output = outputs.last_hidden_state
        attr_logits = self.attribute_classifier(sequence_output)
        outputs.attr_logits = attr_logits # Gán vào output để dùng sau này

        # --- BƯỚC 2: TỰ TÍNH LOSS (NẾU ĐANG TRAIN) ---
        if labels is not None:
            # Tự gọi hàm tính loss thủ công của chúng ta
            loss_dict = self.compute_loss_manual(outputs, labels)
            
            # Tổng hợp loss: Box + Class + GIoU + Attribute
            # Trọng số chuẩn của DETR/YOLOS: box=5, class=1, giou=2
            total_loss = (
                loss_dict['loss_ce'] * 1.0 +
                loss_dict['loss_bbox'] * 5.0 +
                loss_dict['loss_giou'] * 2.0 +
                loss_dict['loss_attr'] * 1.0
            )
            outputs.loss = total_loss
        
        return outputs

    def compute_loss_manual(self, outputs, targets):
        """
        Tự thực hiện Hungarian Matching và tính toán Loss cho:
        1. Classification (CrossEntropy)
        2. Bounding Box (L1 + GIoU)
        3. Attributes (BCE)
        """
        device = outputs.logits.device
        pred_logits = outputs.logits        # [Batch, 100, num_classes+1]
        pred_boxes = outputs.pred_boxes     # [Batch, 100, 4] (cx, cy, w, h) normalized
        pred_attrs = outputs.attr_logits    # [Batch, 100, num_attributes]
        
        batch_size = len(targets)
        
        # Các biến để cộng dồn loss
        loss_ce = 0.0
        loss_bbox = 0.0
        loss_giou = 0.0
        loss_attr = 0.0
        
        # Index của class "Background" (là class cuối cùng)
        background_class_idx = self.config.num_labels 

        # Duyệt qua từng ảnh trong batch để Matching
        for i in range(batch_size):
            tgt_ids = targets[i]["class_labels"]
            tgt_bbox = targets[i]["boxes"] # (cx, cy, w, h)
            tgt_attrs = targets[i].get("attribute_labels", None)
            
            # Nếu ảnh không có object nào (chỉ có background)
            if len(tgt_ids) == 0:
                # Tất cả query đều phải là background
                target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
                loss_ce += self.class_criterion(pred_logits[i], target_classes)
                continue

            # --- A. HUNGARIAN MATCHING ---
            with torch.no_grad():
                # 1. Cost Class: Lấy xác suất của class đúng
                # pred_logits[i]: [100, 47] -> softmax -> lấy cột tương ứng tgt_ids
                out_prob = pred_logits[i].softmax(-1)
                cost_class = -out_prob[:, tgt_ids]

                # 2. Cost Box: L1 distance
                cost_bbox = torch.cdist(pred_boxes[i], tgt_bbox, p=1)

                # 3. Cost GIoU: Generalized IoU
                # Cần đổi sang xyxy để tính GIoU
                src_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
                tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
                cost_giou = -generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)

                # Tổng hợp Cost matrix
                C = 2.0 * cost_bbox + 1.0 * cost_class + 2.0 * cost_giou
                C = C.cpu().numpy()

                # Tìm cặp ghép nối tối ưu
                src_idx, tgt_idx = linear_sum_assignment(C)
                # src_idx: Index của Query (Dự đoán)
                # tgt_idx: Index của Ground Truth

            # --- B. TÍNH LOSS ---
            
            # 1. Classification Loss
            # Tạo target vector: Mặc định tất cả là Background
            target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
            # Gán class thật cho những query đã match
            target_classes[src_idx] = tgt_ids[tgt_idx]
            
            loss_ce += self.class_criterion(pred_logits[i], target_classes)
            
            # 2. Box Loss (Chỉ tính cho các cặp đã match)
            src_boxes = pred_boxes[i][src_idx]
            target_boxes = tgt_bbox[tgt_idx]
            
            loss_bbox += F.l1_loss(src_boxes, target_boxes, reduction='mean')
            
            # 3. GIoU Loss (Chỉ tính cho các cặp đã match)
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            loss_giou += (1 - torch.diag(generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy))).mean()
            
            # 4. Attribute Loss (Chỉ tính cho các cặp đã match)
            if tgt_attrs is not None:
                src_attrs = pred_attrs[i][src_idx]
                target_attrs = tgt_attrs[tgt_idx]
                loss_attr += self.attr_criterion(src_attrs, target_attrs)

        # Chia trung bình cho batch size
        return {
            'loss_ce': loss_ce / batch_size,
            'loss_bbox': loss_bbox / batch_size,
            'loss_giou': loss_giou / batch_size,
            'loss_attr': loss_attr / batch_size
        }

# Hàm Helper chuyển đổi tọa độ
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)