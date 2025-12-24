import numpy as np
import torch
import cv2
import random
from torch.utils.data import Dataset, WeightedRandomSampler
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# --- AUGMENTATION (Giữ nguyên logic ổn định) ---
class ImageAugmentation:
    def __init__(self, target_size=(800, 800), p_flip=0.5, brightness=0.2, contrast=0.2, train=True):
        self.target_size = target_size
        self.p_flip = p_flip
        self.brightness = brightness
        self.contrast = contrast
        self.train = train

    def __call__(self, image, boxes, category_ids):
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        if self.train:
            image = self._color_jitter(image)
            if random.random() < self.p_flip:
                image, boxes = self._horizontal_flip(image, boxes)

        image, boxes = self._resize_and_pad(image, boxes)
        return image, boxes, category_ids

    def _color_jitter(self, image):
        img = image.astype(np.float32)
        beta = random.uniform(-self.brightness, self.brightness) * 255
        img = img + beta
        alpha = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * alpha + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _horizontal_flip(self, image, boxes):
        h, w = image.shape[:2]
        image = cv2.flip(image, 1)
        if len(boxes) > 0:
            boxes[:, 0] = w - (boxes[:, 0] + boxes[:, 2])
        return image, boxes

    def _resize_and_pad(self, image, boxes):
        target_h, target_w = self.target_size
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), (124, 116, 104), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = image_resized
        if len(boxes) > 0:
            boxes[:, :4] *= scale
            boxes[:, 0] += pad_x
            boxes[:, 1] += pad_y
        return canvas, boxes

# --- DATASET CHÍNH ---
class FashionDataset(Dataset):
    def __init__(self, dataset_path, split, feature_extractor, train=False, num_attributes=294):
        print(f"🔄 Đang tải dataset '{dataset_path}' split='{split}'...")
        # Load không cần trust_remote_code
        self.dataset = load_dataset(dataset_path, split=split)
        
        # [CHỐT TỪ EDA]
        self.num_classes = 46
        print(f"✅ Cấu hình Dataset: {self.num_classes} classes (Theo EDA).")

        self.feature_extractor = feature_extractor
        self.num_attributes = num_attributes
        self.train = train
        self.augmentor = ImageAugmentation(target_size=(800, 800), train=train)

    def __getitem__(self, index):
        item = self.dataset[index]
        
        # 1. Ảnh
        image_pil = item['image']
        image = np.array(image_pil)
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 2. Objects
        objects = item['objects']
        raw_boxes = objects['bbox']
        raw_categories = objects['category']
        raw_attributes = objects.get('attributes', objects.get('attribute', []))

        boxes = []
        category_ids = []
        attribute_ids_list = []
        area = []
        iscrowd = []

        for i in range(len(raw_boxes)):
            x, y, w, h = raw_boxes[i]
            if w <= 0 or h <= 0: continue
            
            cls_id = raw_categories[i]
            # [SAFETY] Lọc bỏ ID rác nếu vượt quá 45
            if cls_id >= self.num_classes: continue

            boxes.append([x, y, w, h])
            category_ids.append(cls_id)
            
            if i < len(raw_attributes): attribute_ids_list.append(raw_attributes[i])
            else: attribute_ids_list.append([])
            
            area.append(w * h)
            iscrowd.append(0)

        # 3. Augmentation
        image, final_boxes, final_categories = self.augmentor(image, boxes, category_ids)

        # 4. Format Output
        out_boxes = []
        out_attributes = []
        img_h, img_w, _ = image.shape 
        
        for box, attr_ids in zip(final_boxes, attribute_ids_list):
            x, y, w, h = box
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))
            out_boxes.append([cx, cy, nw, nh])
            
            attr_vec = torch.zeros(self.num_attributes, dtype=torch.float32)
            valid_ids = [aid for aid in attr_ids if aid < self.num_attributes]
            if valid_ids: attr_vec[valid_ids] = 1.0
            out_attributes.append(attr_vec)

        target = {}
        target["boxes"] = torch.as_tensor(out_boxes, dtype=torch.float32)
        target["class_labels"] = torch.as_tensor(final_categories[:len(out_boxes)], dtype=torch.long)
        try: tid = int(item['image_id'])
        except: tid = index
        target["image_id"] = torch.tensor([tid])
        
        if len(out_boxes) > 0:
            target["attribute_labels"] = torch.stack(out_attributes)
            target["area"] = torch.as_tensor(area[:len(out_boxes)], dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(iscrowd[:len(out_boxes)], dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["class_labels"] = torch.zeros((0,), dtype=torch.long)
            target["attribute_labels"] = torch.zeros((0, self.num_attributes), dtype=torch.float32)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        encoding = self.feature_extractor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "pixel_mask": encoding.get("pixel_mask", None),
            "target": target
        }

    def __len__(self):
        return len(self.dataset)

    def get_weighted_sampler(self):
        """
        [QUAN TRỌNG] Xử lý Class Imbalance (72.3% vs 27.7%)
        """
        print("⚖️ Đang tính toán Class Weights để cân bằng dữ liệu...")
        class_counts = {}
        img_weights = []
        all_objects = self.dataset['objects']
        
        # Đếm tần suất
        for obj in all_objects:
            for c in obj['category']:
                if c < self.num_classes:
                    class_counts[c] = class_counts.get(c, 0) + 1
        
        # Tính weight (nghịch đảo tần suất)
        total_samples = sum(class_counts.values()) if class_counts else 1
        class_weights = {c: total_samples / (cnt + 1e-6) for c, cnt in class_counts.items()}
        
        # Gán weight cho từng ảnh
        for obj in all_objects:
            cats = [c for c in obj['category'] if c < self.num_classes]
            if not cats: weight = 0.0
            else: weight = max([class_weights.get(c, 0) for c in cats])
            img_weights.append(weight)
            
        return WeightedRandomSampler(img_weights, len(img_weights), replacement=True)