import torch

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["target"] for item in batch]
    
    max_h = max([img.shape[1] for img in pixel_values])
    max_w = max([img.shape[2] for img in pixel_values])
    
    batch_size = len(pixel_values)
    padded_imgs = torch.zeros((batch_size, 3, max_h, max_w), dtype=torch.float32)
    pixel_masks = torch.zeros((batch_size, max_h, max_w), dtype=torch.long) # 0 là padding
    
    for i, img in enumerate(pixel_values):
        h, w = img.shape[1], img.shape[2]
        padded_imgs[i, :, :h, :w] = img
        pixel_masks[i, :h, :w] = 1 
        
    return {
        "pixel_values": padded_imgs,
        "pixel_mask": pixel_masks, 
        "labels": labels
    }