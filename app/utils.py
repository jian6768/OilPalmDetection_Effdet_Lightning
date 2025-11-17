import torch
from effdet.data.transforms import RandomFlip, RandomResizePad, resolve_fill_color, ImageToNumpy, Compose, ResizePad
from app.config.config import config


def custom_collate(batch):
    """
    batch: list of (img, target)
        img:    np.ndarray, shape (C, H, W), dtype=uint8
        target: dict with keys 'img_idx', 'img_size', 'bbox', 'cls', 'img_scale'
    """

    batch_size = len(batch)

    # ---- 1. Stack images: (B, C, H, W) ----
    # all images same size (3, 512, 512)
    imgs = [torch.from_numpy(sample[0]) for sample in batch]  # each (C, H, W)
    imgs = torch.stack(imgs, dim=0)                           # (B, C, H, W)
    # imgs = imgs.float() / 255.0  
    imgs = imgs.float()  

    # ---- 2. Simple fields: img_idx, img_size, img_scale ----
    img_idx = torch.tensor([sample[1]["img_idx"] for sample in batch], dtype=torch.int64)

    img_size = torch.tensor(
        [sample[1]["img_size"] for sample in batch], dtype=torch.int32
    )  # (B, 2)

    img_scale = torch.tensor(
        [sample[1]["img_scale"] for sample in batch], dtype=torch.float32
    )  # (B,)

    # ---- 3. Variable-length bboxes & cls -> pad to max_det ----
    # Number of boxes per image
    num_boxes = [sample[1]["bbox"].shape[0] for sample in batch]
    max_det = max(num_boxes)

    # Initialize with -1 as padding (to match what you printed)
    bbox = torch.full(
        (batch_size, max_det, 4), -1.0, dtype=torch.float32
    )
    cls = torch.full(
        (batch_size, max_det), -1, dtype=torch.int32
    )

    for i, (_, target) in enumerate(batch):
        n = target["bbox"].shape[0]
        if n == 0:
            continue

        bbox[i, :n] = torch.from_numpy(target["bbox"]).to(torch.float32)
        cls[i, :n] = torch.from_numpy(target["cls"]).to(torch.int32)

    targets = {
        "img_idx": img_idx,
        "img_size": img_size,
        "bbox": bbox,
        "cls": cls,
        "img_scale": img_scale,
    }
    # print("mydetectioncollate used.")
    return imgs, targets

def create_custom_data_loader(
        dataset, 
        input_size,
        batch_size = 8, 
        is_training = False, 
        num_workers=1,
        mean=config.IMAGENET_DEFAULT_MEAN,
        std=config.IMAGENET_DEFAULT_STD,
        fill_color='mean',
        transform_fn=None,
        collate_fn=None,
        device="cuda"
        
        ):

    #Set up transform that will be used for dataset which will be fed into data loader. 
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if transform_fn is not None:
        # transform_fn should accept inputs (img, annotations) from the dataset and return a tuple
        # of img, annotations for the data loader collate function.
        # The valid types of img and annotations depend on the dataset and collate abstractions used.
        # The default dataset outputs PIL Image and dict of numpy ndarrays or python scalar annotations.
        # The fast collate fn accepts ONLY numpy uint8 images and annotations dicts of ndarrays and python scalars
        transform = transform_fn
    else:

        fill_color = resolve_fill_color(fill_color, mean)

        if is_training:
            image_tfl = [
                RandomFlip(horizontal=True, prob=0.5),
                RandomResizePad(
                    target_size=img_size, interpolation="random", fill_color=fill_color),
                ImageToNumpy(),
                
                
            ]
            transform = Compose(image_tfl)

            
        else:
            image_tfl = [
                ResizePad(
                    target_size=img_size, interpolation="bilinear", fill_color=fill_color),
                ImageToNumpy(),
            ]

            transform = Compose(image_tfl)
    
    #Detection transform is 
    dataset.transform = transform

    # collate_fn = collate_fn or DetectionFastCollate(anchor_labeler=None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        
    )

    return loader