import os
import json
import yaml
import cv2
from tqdm import tqdm
from glob import glob

# --- CẤU HÌNH ĐƯỜNG DẪN GỐC ---
ROOT_DIR = "dataset"  # Thư mục chứa images và labels
YAML_PATH = os.path.join(ROOT_DIR, "data.yaml")
OUTPUT_DIR = "dataset" # Nơi sẽ lưu 3 file json

def convert_subset(subset_name):
    """
    Hàm chuyển đổi cho từng tập: train, val, test
    subset_name: 'train', 'val' hoặc 'test'
    """
    print(f"\n--- ĐANG XỬ LÝ TẬP: {subset_name.upper()} ---")

    images_dir = os.path.join(ROOT_DIR, "images", subset_name)
    labels_dir = os.path.join(ROOT_DIR, "labels", subset_name)
    output_json = os.path.join(OUTPUT_DIR, f"{subset_name}_annotations.json")

    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(images_dir):
        print(f"Bỏ qua {subset_name}: Không tìm thấy thư mục {images_dir}")
        return

    # 2. Đọc thông tin Class từ YAML (Chỉ cần đọc 1 lần nhưng để đây cho gọn)
    with open(YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
    names = data_yaml.get('names', [])
    categories = [{"id": i, "name": name} for i, name in enumerate(names)]

    # 3. Quét file ảnh
    img_files = glob(os.path.join(images_dir, "*.jpg")) + \
                glob(os.path.join(images_dir, "*.png")) + \
                glob(os.path.join(images_dir, "*.jpeg"))
    
    if len(img_files) == 0:
        print(f"Cảnh báo: Không tìm thấy ảnh nào trong {images_dir}")
        return

    images = []
    annotations = []
    ann_id_cnt = 0
    
    print(f"Tìm thấy {len(img_files)} ảnh. Đang chuyển đổi...")
    
    for img_path in tqdm(img_files):
        # --- Xử lý thông tin ảnh ---
        file_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: 
            print(f"Lỗi đọc ảnh: {file_name}")
            continue
            
        height, width, _ = img.shape
        
        # Tạo Image ID (Đảm bảo duy nhất)
        # Sử dụng hash filename để tránh trùng lặp giữa các tập nếu gộp chung sau này
        image_id = int(hash(file_name) % 1_000_000_000)
        
        images.append({
            "id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })
        
        # --- Xử lý Label ---
        label_file = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # Parse tọa độ YOLO
                cls_id = int(parts[0])
                x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                
                # Chuyển sang COCO [x_min, y_min, w, h] absolute
                w_abs = w_norm * width
                h_abs = h_norm * height
                x_min = (x_center * width) - (w_abs / 2)
                y_min = (y_center * height) - (h_abs / 2)
                
                annotations.append({
                    "id": ann_id_cnt,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0,
                    "attribute_ids": [] # Dummy attributes
                })
                ann_id_cnt += 1

    # 4. Lưu file JSON
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "attributes": [{"id": i, "name": f"attr_{i}"} for i in range(294)]
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)
        
    print(f"-> Đã lưu file: {output_json}")
    print(f"-> Thống kê: {len(images)} ảnh, {len(annotations)} boxes.")

if __name__ == "__main__":
    # Chạy lần lượt cho 3 tập
    subsets = ['train', 'val', 'test']
    
    print("BẮT ĐẦU CHUYỂN ĐỔI DATASET YOLO -> COCO JSON")
    for subset in subsets:
        convert_subset(subset)
    print("\nHOÀN TẤT TOÀN BỘ!")