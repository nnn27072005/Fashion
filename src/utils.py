import torch

def collate_fn(batch):
    """
    Hàm gom data thành batch.
    Xử lý padding để các ảnh có kích thước khác nhau có thể nằm chung một batch.
    """
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 1. Tìm kích thước lớn nhất trong batch hiện tại (Max Height, Max Width)
    max_h = max([img.shape[1] for img in pixel_values])
    max_w = max([img.shape[2] for img in pixel_values])
    
    # 2. Tạo tensor rỗng (Batch_size, 3, MaxH, MaxW)
    # Giá trị 0.0 tương ứng với padding màu đen/xám tùy normalization
    batch_size = len(pixel_values)
    padded_pixel_values = torch.zeros((batch_size, 3, max_h, max_w), dtype=torch.float32)
    
    # 3. Chép từng ảnh vào góc trên-trái của tensor rỗng
    for i, img in enumerate(pixel_values):
        h, w = img.shape[1], img.shape[2]
        padded_pixel_values[i, :, :h, :w] = img
        
    return {
        "pixel_values": padded_pixel_values,
        "labels": labels
    }