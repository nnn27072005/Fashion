import torch

def collate_fn(batch):
    """
    Collate function cho Deformable DETR.
    Tự động Padding ảnh và Tạo Pixel Mask.
    """
    # batch là list các dict trả về từ __getitem__
    # item: {"pixel_values": tensor, "pixel_mask": None/Tensor, "target": dict}
    
    pixel_values = [item["pixel_values"] for item in batch]
    targets = [item["target"] for item in batch]
    
    # 1. Padding Pixel Values
    # Tìm kích thước lớn nhất trong batch
    max_h = max([img.shape[1] for img in pixel_values])
    max_w = max([img.shape[2] for img in pixel_values])
    
    batch_size = len(pixel_values)
    
    # Tạo tensor tổng (Batch, 3, H, W) và Mask (Batch, H, W)
    padded_imgs = torch.zeros((batch_size, 3, max_h, max_w), dtype=torch.float32)
    pixel_masks = torch.zeros((batch_size, max_h, max_w), dtype=torch.long) # 0 là padding
    
    for i, img in enumerate(pixel_values):
        h, w = img.shape[1], img.shape[2]
        padded_imgs[i, :, :h, :w] = img
        pixel_masks[i, :h, :w] = 1 # Đánh dấu vùng có ảnh là 1
        
    return {
        "pixel_values": padded_imgs,
        "pixel_mask": pixel_masks, # QUAN TRỌNG
        "labels": targets
    }