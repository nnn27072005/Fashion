# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import YolosForObjectDetection, YolosConfig
# from scipy.optimize import linear_sum_assignment
# from torchvision.ops import generalized_box_iou, box_convert

# class FashionYolos(YolosForObjectDetection):
#     def __init__(self, config: YolosConfig):
#         super().__init__(config)
        
#         # 1. Attribute Head
#         self.num_attributes = getattr(config, "num_attributes", 294)
#         self.attribute_classifier = nn.Sequential(
#             nn.Linear(config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(config.hidden_size, self.num_attributes)
#         )
        
#         # 2. Weights Init
#         for m in self.attribute_classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#         # 3. Loss Functions
#         self.attr_criterion = nn.BCEWithLogitsLoss()
        
#         # Weight cho background class thấp hơn chút
#         empty_weight = torch.ones(self.config.num_labels + 1)
#         empty_weight[-1] = 0.1 
#         self.class_criterion = nn.CrossEntropyLoss(weight=empty_weight)

#     def forward(
#         self,
#         pixel_values=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # --- BƯỚC 1: CHẠY BASE MODEL ---
#         outputs = super().forward(
#             pixel_values=pixel_values,
#             labels=None, # Luôn để None để tự tính loss
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=return_dict,
#         )

#         # --- BƯỚC 2: XỬ LÝ ATTRIBUTE (ĐÃ FIX LỖI TẠI ĐÂY) ---
#         sequence_output = outputs.last_hidden_state
        
#         # [QUAN TRỌNG] Chỉ lấy 100 token cuối cùng (Detection Tokens)
#         # Sequence output có dạng [Batch, Seq_Len, Hidden].
#         # YOLOS đặt 100 detection tokens ở cuối cùng.
#         object_queries = sequence_output[:, -100:, :] 
        
#         # Đưa qua classifier
#         attr_logits = self.attribute_classifier(object_queries)
#         outputs.attr_logits = attr_logits

#         # --- BƯỚC 3: TỰ TÍNH LOSS ---
#         if labels is not None:
#             loss_dict = self.compute_loss_manual(outputs, labels)
            
#             # Tổng hợp loss
#             total_loss = (
#                 loss_dict['loss_ce'] * 1.0 +
#                 loss_dict['loss_bbox'] * 5.0 +
#                 loss_dict['loss_giou'] * 2.0 +
#                 loss_dict['loss_attr'] * 1.0
#             )
#             outputs.loss = total_loss
        
#         return outputs

#     def compute_loss_manual(self, outputs, targets):
#         """
#         Tự thực hiện Hungarian Matching và tính toán Loss
#         """
#         device = outputs.logits.device
#         pred_logits = outputs.logits        # [Batch, 100, num_classes+1]
#         pred_boxes = outputs.pred_boxes     # [Batch, 100, 4]
#         pred_attrs = outputs.attr_logits    # [Batch, 100, num_attributes] (Đã fix shape)
        
#         batch_size = len(targets)
        
#         loss_ce = 0.0
#         loss_bbox = 0.0
#         loss_giou = 0.0
#         loss_attr = 0.0
        
#         background_class_idx = self.config.num_labels 

#         for i in range(batch_size):
#             tgt_ids = targets[i]["class_labels"]
#             tgt_bbox = targets[i]["boxes"]
#             tgt_attrs = targets[i].get("attribute_labels", None)
            
#             if len(tgt_ids) == 0:
#                 target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
#                 loss_ce += self.class_criterion(pred_logits[i], target_classes)
#                 continue

#             with torch.no_grad():
#                 out_prob = pred_logits[i].softmax(-1)
#                 cost_class = -out_prob[:, tgt_ids]
#                 cost_bbox = torch.cdist(pred_boxes[i], tgt_bbox, p=1)
                
#                 src_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
#                 tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
#                 cost_giou = -generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)

#                 C = 2.0 * cost_bbox + 1.0 * cost_class + 2.0 * cost_giou
#                 C = C.cpu().numpy()
#                 src_idx, tgt_idx = linear_sum_assignment(C)

#             # 1. Classification Loss
#             target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
#             target_classes[src_idx] = tgt_ids[tgt_idx]
#             loss_ce += self.class_criterion(pred_logits[i], target_classes)
            
#             # 2. Box Loss
#             src_boxes = pred_boxes[i][src_idx]
#             target_boxes = tgt_bbox[tgt_idx]
#             loss_bbox += F.l1_loss(src_boxes, target_boxes, reduction='mean')
            
#             # 3. GIoU Loss
#             src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
#             tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
#             loss_giou += (1 - torch.diag(generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy))).mean()
            
#             # 4. Attribute Loss
#             if tgt_attrs is not None:
#                 src_attrs = pred_attrs[i][src_idx]
#                 target_attrs = tgt_attrs[tgt_idx]
#                 loss_attr += self.attr_criterion(src_attrs, target_attrs)

#         return {
#             'loss_ce': loss_ce / batch_size,
#             'loss_bbox': loss_bbox / batch_size,
#             'loss_giou': loss_giou / batch_size,
#             'loss_attr': loss_attr / batch_size
#         }

# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeformableDetrForObjectDetection, DeformableDetrConfig
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

class FashionDeformableDETR(DeformableDetrForObjectDetection):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        
        # 1. Attribute Head
        # Deformable DETR dùng d_model (thường là 256) thay vì hidden_size
        self.num_attributes = getattr(config, "num_attributes", 294)
        input_dim = config.d_model 
        
        self.attribute_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, self.num_attributes)
        )
        
        # Init weights cho head mới
        for m in self.attribute_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. Loss Functions
        self.attr_criterion = nn.BCEWithLogitsLoss()
        
        # Weight cho background class (Deformable DETR thường có 300 queries)
        empty_weight = torch.ones(self.config.num_labels + 1)
        empty_weight[-1] = 0.1 
        self.class_criterion = nn.CrossEntropyLoss(weight=empty_weight)

    def forward(
        self,
        pixel_values=None,
        pixel_mask=None, # Deformable DETR BẮT BUỘC phải có pixel_mask
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- BƯỚC 1: CHẠY BASE MODEL ---
        outputs = super().forward(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask, # Truyền mask vào
            labels=None, 
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # --- BƯỚC 2: XỬ LÝ ATTRIBUTE ---
        # Lấy hidden state cuối cùng của Decoder (Object Queries)
        # outputs.last_hidden_state shape: [Batch, Num_Queries (300), d_model]
        sequence_output = outputs.last_hidden_state
        
        # Đưa qua classifier
        attr_logits = self.attribute_classifier(sequence_output)
        outputs.attr_logits = attr_logits

        # --- BƯỚC 3: TỰ TÍNH LOSS ---
        if labels is not None:
            loss_dict = self.compute_loss_manual(outputs, labels)
            
            # Tổng hợp loss
            total_loss = (
                loss_dict['loss_ce'] * 2.0 +      # Class loss quan trọng
                loss_dict['loss_bbox'] * 5.0 +    # Box loss giữ nguyên
                loss_dict['loss_giou'] * 2.0 +    # GIoU
                loss_dict['loss_attr'] * 1.0
            )
            outputs.loss = total_loss
        
        return outputs

    def compute_loss_manual(self, outputs, targets):
        """
        Hungarian Matching và tính toán Loss
        """
        device = outputs.logits.device
        pred_logits = outputs.logits        # [Batch, 300, num_classes+1]
        pred_boxes = outputs.pred_boxes     # [Batch, 300, 4]
        pred_attrs = outputs.attr_logits    # [Batch, 300, num_attributes]
        
        batch_size = len(targets)
        
        loss_ce = 0.0
        loss_bbox = 0.0
        loss_giou = 0.0
        loss_attr = 0.0
        
        background_class_idx = self.config.num_labels 

        for i in range(batch_size):
            tgt_ids = targets[i]["class_labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_attrs = targets[i].get("attribute_labels", None)
            
            if len(tgt_ids) == 0:
                # Nếu ảnh không có object nào, gán tất cả là background
                target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
                loss_ce += self.class_criterion(pred_logits[i], target_classes)
                continue

            with torch.no_grad():
                # Matching Strategy
                out_prob = pred_logits[i].softmax(-1)
                cost_class = -out_prob[:, tgt_ids]
                cost_bbox = torch.cdist(pred_boxes[i], tgt_bbox, p=1)
                
                src_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
                tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
                cost_giou = -generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)

                # Hungarian Cost Matrix
                C = 2.0 * cost_bbox + 2.0 * cost_class + 5.0 * cost_giou
                C = C.cpu().numpy()
                src_idx, tgt_idx = linear_sum_assignment(C)

            # 1. Classification Loss
            target_classes = torch.full((pred_logits.shape[1],), background_class_idx, dtype=torch.long, device=device)
            target_classes[src_idx] = tgt_ids[tgt_idx]
            loss_ce += self.class_criterion(pred_logits[i], target_classes)
            
            # 2. Box Loss
            src_boxes = pred_boxes[i][src_idx]
            target_boxes = tgt_bbox[tgt_idx]
            loss_bbox += F.l1_loss(src_boxes, target_boxes, reduction='mean')
            
            # 3. GIoU Loss
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            loss_giou += (1 - torch.diag(generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy))).mean()
            
            # 4. Attribute Loss
            if tgt_attrs is not None:
                src_attrs = pred_attrs[i][src_idx]
                target_attrs = tgt_attrs[tgt_idx]
                loss_attr += self.attr_criterion(src_attrs, target_attrs)

        return {
            'loss_ce': loss_ce / batch_size,
            'loss_bbox': loss_bbox / batch_size,
            'loss_giou': loss_giou / batch_size,
            'loss_attr': loss_attr / batch_size
        }

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)