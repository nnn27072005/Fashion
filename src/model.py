import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeformableDetrForObjectDetection, DeformableDetrConfig
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
import numpy as np

class FashionDeformableDETR(DeformableDetrForObjectDetection):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.num_attributes = getattr(config, "num_attributes", 294)
        input_dim = config.d_model 
        self.attribute_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(input_dim, self.num_attributes)
        )
        for m in self.attribute_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        self.attr_criterion = nn.BCEWithLogitsLoss()
        # 46 Object + 1 Background = 47 Classes
        empty_weight = torch.ones(self.config.num_labels)
        empty_weight[-1] = 0.1 
        self.class_criterion = nn.CrossEntropyLoss(weight=empty_weight)

    def forward(self, pixel_values=None, pixel_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = super().forward(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=None, 
            output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict
        )
        sequence_output = outputs.last_hidden_state
        outputs.attr_logits = self.attribute_classifier(sequence_output)

        if labels is not None:
            loss_dict = self.compute_loss_manual(outputs, labels)
            outputs.loss = (loss_dict['loss_ce']*2.0 + loss_dict['loss_bbox']*5.0 + 
                            loss_dict['loss_giou']*2.0 + loss_dict['loss_attr']*1.0)
        return outputs

    def compute_loss_manual(self, outputs, targets):
        device = outputs.logits.device
        pred_logits = outputs.logits        
        pred_boxes = outputs.pred_boxes     
        pred_attrs = outputs.attr_logits    
        batch_size = len(targets)
        
        loss_ce, loss_bbox, loss_giou, loss_attr = 0.0, 0.0, 0.0, 0.0
        bg_idx = self.config.num_labels - 1 

        for i in range(batch_size):
            tgt_ids = targets[i]["class_labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_attrs = targets[i].get("attribute_labels", None)

            # [SAFETY CLAMP] Ngăn chặn Crash CUDA
            if (tgt_ids >= self.config.num_labels).any():
                tgt_ids = torch.clamp(tgt_ids, max=bg_idx)

            if len(tgt_ids) == 0:
                target_classes = torch.full((pred_logits.shape[1],), bg_idx, dtype=torch.long, device=device)
                loss_ce += self.class_criterion(pred_logits[i], target_classes)
                continue

            with torch.no_grad():
                out_prob = pred_logits[i].softmax(-1)
                pred_boxes_safe = pred_boxes[i].clamp(min=1e-6, max=1.0-1e-6)
                cost_class = -out_prob[:, tgt_ids]
                cost_bbox = torch.cdist(pred_boxes_safe, tgt_bbox, p=1)
                src_xyxy = box_cxcywh_to_xyxy(pred_boxes_safe)
                tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
                giou = torch.nan_to_num(generalized_box_iou(src_xyxy, tgt_xyxy), nan=-1.0)
                cost_giou = -giou
                
                C = 2.0*cost_bbox + 2.0*cost_class + 5.0*cost_giou
                C = np.nan_to_num(C.cpu().numpy(), nan=1e6)
                src_idx, tgt_idx = linear_sum_assignment(C)

            # Classification
            target_classes = torch.full((pred_logits.shape[1],), bg_idx, dtype=torch.long, device=device)
            target_classes[src_idx] = tgt_ids[tgt_idx]
            loss_ce += self.class_criterion(pred_logits[i], target_classes)
            
            # Box Regression
            src_boxes = pred_boxes[i][src_idx]
            target_boxes = tgt_bbox[tgt_idx]
            loss_bbox += F.l1_loss(src_boxes, target_boxes, reduction='mean')
            
            # GIoU
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
            loss_giou += (1 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).mean()
            
            # Attribute
            if tgt_attrs is not None:
                loss_attr += self.attr_criterion(pred_attrs[i][src_idx], tgt_attrs[tgt_idx])

        return {'loss_ce': loss_ce/batch_size, 'loss_bbox': loss_bbox/batch_size, 
                'loss_giou': loss_giou/batch_size, 'loss_attr': loss_attr/batch_size}

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5*w), (y_c - 0.5*h), (x_c + 0.5*w), (y_c + 0.5*h)]
    return torch.stack(b, dim=-1)