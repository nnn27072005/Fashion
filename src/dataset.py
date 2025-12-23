import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from pycocotools.coco import COCO
import albumentations as A

class FashionDataset(Dataset):
    def __init__(self, img_dir, ann_file, feature_extractor, train=False, num_attributes=294):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.feature_extractor = feature_extractor
        self.num_attributes = num_attributes
        self.train = train
        self.transforms = self.get_transforms(train)

    def get_transforms(self, train=False):
        if train:
            return A.Compose([
                A.OneOf([
                    # Compose để bọc Pad + Crop an toàn
                    A.Compose([
                        # Quay lại dùng 'value' vì phiên bản của bạn báo lỗi với 'pad_cval'
                        A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104]),
                        A.RandomCrop(width=800, height=800),
                    ]),
                    A.RandomResizedCrop(size=(800, 800), scale=(0.5, 1.0)),
                ], p=0.5),
                
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                
                A.LongestMaxSize(max_size=1333),
                A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104])
            ], 
            bbox_params=A.BboxParams(format='coco', label_fields=['indices'], min_visibility=0.3))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=1333),
                A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104])
            ], 
            bbox_params=A.BboxParams(format='coco', label_fields=['indices']))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, path)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot load image at {img_path}")
            return self.__getitem__((index + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Prepare raw data
        boxes = []
        category_ids = []
        attribute_ids_list = []
        area = []
        iscrowd = []

        for ann in coco_target:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            
            # Kẹp box vào trong ảnh
            h_img, w_img, _ = image.shape
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            boxes.append([x, y, w, h])
            category_ids.append(ann['category_id'])
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
            attribute_ids_list.append(ann.get('attribute_ids', []))

        # 2. Augmentation
        indices = list(range(len(boxes)))

        if self.transforms:
            try:
                transformed = self.transforms(
                    image=image, 
                    bboxes=boxes, 
                    indices=indices 
                )
                image = transformed['image']
                final_boxes_aug = transformed['bboxes']
                surviving_indices = transformed['indices']
                
                final_boxes = []
                final_categories = []
                final_attributes_list = []
                final_area = []
                final_iscrowd = []
                
                for i, original_idx in enumerate(surviving_indices):
                    idx = int(original_idx) # Fix lỗi float index
                    final_boxes.append(final_boxes_aug[i])
                    final_categories.append(category_ids[idx])
                    final_attributes_list.append(attribute_ids_list[idx])
                    final_area.append(area[idx])
                    final_iscrowd.append(iscrowd[idx])
                    
            except ValueError as e:
                print(f"Aug Error {img_path}: {e}")
                final_boxes = boxes
                final_categories = category_ids
                final_attributes_list = attribute_ids_list
                final_area = area
                final_iscrowd = iscrowd
        else:
            final_boxes = boxes
            final_categories = category_ids
            final_attributes_list = attribute_ids_list
            final_area = area
            final_iscrowd = iscrowd

        # 3. Format Output & NORMALIZE (Quan trọng)
        # YOLOS/DETR yêu cầu box dạng: [cx, cy, w, h] đã chuẩn hóa về 0-1
        # Chúng ta phải tự làm bước này vì không dùng Processor để xử lý annotation nữa
        out_boxes = []
        out_attributes = []
        img_h, img_w, _ = image.shape # Kích thước ảnh sau khi augment
        
        for i, box in enumerate(final_boxes):
            x, y, w, h = box
            
            # Chuyển đổi: (x, y, w, h) Top-Left Absolute -> (cx, cy, w, h) Center Normalized
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            
            # Clamp về 0-1 cho chắc chắn
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            
            out_boxes.append([cx, cy, nw, nh])
            
            # Attributes
            attr_vec = torch.zeros(self.num_attributes, dtype=torch.float32)
            valid_ids = [aid for aid in final_attributes_list[i] if aid < self.num_attributes]
            if valid_ids:
                attr_vec[valid_ids] = 1.0
            out_attributes.append(attr_vec)

        target = {}
        target["boxes"] = torch.as_tensor(out_boxes, dtype=torch.float32)
        target["class_labels"] = torch.as_tensor(final_categories, dtype=torch.long)
        target["image_id"] = torch.tensor([img_id])
        
        if len(out_attributes) > 0:
            target["attribute_labels"] = torch.stack(out_attributes)
            target["area"] = torch.as_tensor(final_area, dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(final_iscrowd, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["class_labels"] = torch.zeros((0,), dtype=torch.long)
            target["attribute_labels"] = torch.zeros((0, self.num_attributes), dtype=torch.float32)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # QUAN TRỌNG: Chỉ đưa ảnh vào Processor để chuẩn hóa pixel
        # KHÔNG đưa annotations vào nữa để tránh lỗi ValueError
        encoding = self.feature_extractor(
            images=image, 
            return_tensors="pt"
        )
        
        pixel_values = encoding["pixel_values"].squeeze()
        
        # Trả về pixel_values (đã norm) và target (đã tự tính toán)
        return pixel_values, target

    def __len__(self):
        return len(self.ids)

    def get_weighted_sampler(self):
        print("Đang tính toán Class Weights...")
        class_counts = {}
        img_class_map = {}
        
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            classes = [ann['category_id'] for ann in anns]
            img_class_map[img_id] = classes
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1
        
        total_samples = sum(class_counts.values()) if class_counts else 1
        class_weights = {c: total_samples / (cnt + 1e-6) for c, cnt in class_counts.items()}
        
        sample_weights = []
        for img_id in self.ids:
            classes = img_class_map.get(img_id, [])
            if not classes:
                weight = 0.0
            else:
                weight = max([class_weights.get(c, 0) for c in classes])
            sample_weights.append(weight)
            
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)