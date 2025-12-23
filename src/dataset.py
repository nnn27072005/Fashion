import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from pycocotools.coco import COCO
import albumentations as A
import random
import warnings
warnings.filterwarnings('ignore')


class ImageAugmentation:
    """
    Augmentation thủ công cho Object Detection (Image + Bounding Boxes).
    Thay thế cho Albumentations.
    """
    def __init__(self, 
                 target_size=(800, 800),
                 p_flip=0.5,
                 brightness=0.2, 
                 contrast=0.2,
                 train=True):
        
        self.target_size = target_size # (H, W)
        self.p_flip = p_flip
        self.brightness = brightness
        self.contrast = contrast
        self.train = train

    def __call__(self, image, boxes, category_ids):
        """
        Args:
            image: numpy array (H, W, 3) - RGB
            boxes: list of [x, y, w, h]
            category_ids: list of labels
        Returns:
            image: augmented image
            boxes: augmented boxes
            category_ids: valid labels (sau khi lọc box lỗi)
        """
        # Convert boxes to numpy for easier math
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        if self.train:
            # 1. Color Jitter (Chỉ đổi màu, không đổi vị trí box)
            image = self._color_jitter(image)

            # 2. Horizontal Flip (Cần đổi tọa độ box)
            if random.random() < self.p_flip:
                image, boxes = self._horizontal_flip(image, boxes)

        # 3. Resize & Pad (Letterbox) - Quan trọng để đưa về size chuẩn (800x800)
        # Bước này làm cho cả Train và Val
        image, boxes = self._resize_and_pad(image, boxes)

        return image, boxes, category_ids

    def _color_jitter(self, image):
        """Thay đổi độ sáng và tương phản"""
        img = image.astype(np.float32)
        
        # Brightness
        beta = random.uniform(-self.brightness, self.brightness) * 255
        img = img + beta
        
        # Contrast
        alpha = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * alpha + mean
        
        # Clip về 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _horizontal_flip(self, image, boxes):
        """Lật ảnh ngang và tính lại tọa độ box"""
        h, w = image.shape[:2]
        
        # Flip ảnh
        image = cv2.flip(image, 1) # 1 là horizontal flip

        # Flip boxes: [x, y, w, h]
        # x_new = width - (x_old + w_old)
        if len(boxes) > 0:
            boxes[:, 0] = w - (boxes[:, 0] + boxes[:, 2])
            
        return image, boxes

    def _resize_and_pad(self, image, boxes):
        """
        Resize giữ tỷ lệ (Letterbox) và thêm viền (Pad) để đạt target_size.
        Đây là kỹ thuật chuẩn của YOLO/DETR.
        """
        target_h, target_w = self.target_size
        h, w = image.shape[:2]
        
        # 1. Tính tỉ lệ scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 2. Resize ảnh
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # 3. Tạo canvas màu xám (124, 116, 104) để paste ảnh vào giữa
        canvas = np.full((target_h, target_w, 3), (124, 116, 104), dtype=np.uint8)
        
        # Tính offset để đặt ảnh vào giữa
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = image_resized
        
        # 4. Transform Boxes
        if len(boxes) > 0:
            # Scale
            boxes[:, :4] *= scale
            # Shift (Cộng thêm phần padding)
            boxes[:, 0] += pad_x
            boxes[:, 1] += pad_y
            
        return canvas, boxes

class FashionDataset(Dataset):
    def __init__(self, img_dir, ann_file, feature_extractor, train=False, num_attributes=294):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        # train with 100 images for quick testing
        all_ids = list(sorted(self.coco.imgs.keys()))
        
        if train:
            # Lấy 100 ảnh để debug
            self.ids = all_ids[:100]  
            print(f"⚠️ DEBUG MODE: Đã cắt ngắn Dataset còn {len(self.ids)} ảnh!", flush=True)
        else:
            # Lấy 50 ảnh để val
            self.ids = all_ids[:50]
            print(f"ℹ️ VAL MODE: Sử dụng {len(self.ids)} ảnh validation.", flush=True)
        self.feature_extractor = feature_extractor
        self.num_attributes = num_attributes
        self.train = train
        self.augmentor = ImageAugmentation(
            target_size=(800, 800),
            train=train,
            p_flip=0.5,
            brightness=0.2,
            contrast=0.2
        )

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
        image, final_boxes, final_categories = self.augmentor(image, boxes, category_ids)
        final_attributes_list = attribute_ids_list # Augment màu/flip không làm thay đổi thuộc tính
        final_area = area 
        final_iscrowd = iscrowd

        # 3. Format Output & NORMALIZE (Quan trọng)
        # YOLOS/DETR yêu cầu box dạng: [cx, cy, w, h] đã chuẩn hóa về 0-1
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