import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.utils.data import DataLoader
from torchvision import transforms, ops
from PIL import Image
import numpy as np
from monai.losses import DiceLoss
from LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from statistics import mean
from monai.metrics import DiceMetric, GeneralizedDiceScore
from time import time
import psutil
from torch.utils.tensorboard import SummaryWriter
import json
import os
import argparse

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainingname", help="Training identifier",
                        type=str, required=False)
    parser.add_argument("-e", "--epochs", help="Number of epochs",
                        type=int, required=False)
    parser.add_argument("-m", "--masks", help="Number of masks to use for training",
                        type=int, required=False)
    args = parser.parse_args()
    return args

class Logging:
    def __init__(self, logdir):
        self.writer = SummaryWriter(log_dir=logdir)
        self.gpu = 'cuda:0'

    def log_usage(self):
        """
        Gets current RAM, CPU, and GPU usage (if available)
        """
        # Get RAM usage
        virtual_mem = psutil.virtual_memory()
        ram_usage = virtual_mem.percent

        # Get CPU usage
        cpu_usage = psutil.cpu_percent()

        # Get GPU usage 
        vram_usage = 0  # Initialize to 0 (no GPU by default)
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_reserved(device=self.gpu) / 1e9

        self.writer.add_scalar('Resources/RAM', ram_usage, epoch)
        self.writer.add_scalar('Resources/CPU', cpu_usage, epoch)
        self.writer.add_scalar('Resources/VRAM', vram_usage, epoch)
    
    def log_metrics(self, scalar, name):
        self.writer.add_scalar(f'Metrics/{name}', scalar, epoch)

    def log_figure(self, figure, name):
        self.writer.add_figure(name, figure)

    def log_img(self, name, img):
        self.writer.add_image(name, img)

    def close(self):
        self.writer.close()

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, imgp_list, maskp_list, maskn, imgtransform=None, masktransform=None, val=False):
        self.imgps = imgp_list
        self.maskps = maskp_list
        self.imgtransform = imgtransform
        self.masktransform = masktransform
        self.val = val
        self.maskn = maskn

    def __len__(self):
        return len(self.imgps)

    def __getitem__(self, idx):
        image_path, mask_paths = self.imgps[idx], self.maskps[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        img_size = image.size

        # Load masks
        masks = [Image.open(mask_path) for mask_path in mask_paths]
        resized_masks = None

        if self.maskn and not self.val:
            if len(masks) < self.maskn:
                randsize = len(masks)
            else:
                randsize = self.maskn
            relevant_idcs = np.random.choice(len(masks), size=randsize, replace=False)
        else:
            relevant_idcs = np.arange(len(masks))

        if self.imgtransform:
            image = self.imgtransform(image)
            resized_masks = [self.imgtransform(masks[ri]) for ri in relevant_idcs]

        if self.masktransform:
            masks = [self.masktransform(masks[ri]) for ri in relevant_idcs]

        return {'image': image, 'masks': masks, "resized masks": resized_masks, "img size": img_size}

def collate_fn(batch):
    images = [item['image'] for item in batch]
    img_sizes = [item['img size'] for item in batch]
    masks = [item["masks"] for item in batch]
    resized_masks = [item["resized masks"] for item in batch]

    # Stack images
    images = torch.stack(images, dim=0)  
    
    # Pad masks to have the same number of instances
    batchsize = len(batch)
    max_instances = max(len(m) for m in masks)

    padded_resized_masks = torch.zeros((batchsize, max_instances, 1024, 1024), dtype=torch.uint8)
    padded_bboxes = torch.zeros((batchsize, max_instances, 4), dtype=torch.float32)

    for si, rms in enumerate(resized_masks):
        for mi, resized_mask in enumerate(rms):          
            padded_resized_masks[si][mi] = resized_mask
            bbox = ops.masks_to_boxes(torch.reshape(resized_mask, (1, 1024, 1024)))
            padded_bboxes[si][mi] = bbox

    output = {'images': images, 'masks': masks, "resized masks": padded_resized_masks, 'bboxes': padded_bboxes, "img sizes": img_sizes}
    return output

def imgtransform_func():
    return transforms.Compose([
        transforms.Resize((1024, 1024)), # Resize to the size SAM expects
        transforms.ToTensor()
    ])

def masktransform_func():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def illustrate_masks(img, masks, gt=False):
    g2 = None
    annotated_image = img.copy()
    sizex, sizey, _ = img.shape
    for m in masks:
        b, g, r = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        if gt:
            kernelmask = np.array(m, dtype=np.uint8)
        else:
            if m['area'] > 0.25 * sizex * sizey: # Don't display large masks that would lead to occlusion
                continue
            kernelmask = np.array(m["segmentation"], dtype=np.uint8) * 255
            x, y = np.array(m["point_coords"][0], dtype=np.int32)
            g2 = g + 30 if g + 30 <= 255 else g - 30
            r2 = r + 30 if r + 30 <= 255 else r - 30
        if len(np.unique(kernelmask)) > 2:
            raise ValueError('Invalid value in binary mask')
        
        annotated_image[np.where(kernelmask == 255)] = b, g, r
        if g2:
            annotated_image[y-3:y+3, x-3:x+3] = 0, 0, 0
            annotated_image[y-2:y+2, x-2:x+2] = b, g2, r2
    return annotated_image

def example_sam(sam, dataset, logger, size=5):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=40, points_per_batch=128, pred_iou_thresh=0.88, box_nms_thresh=0.4)
    
    if len(dataset) < size:
        size = len(dataset)

    for ri in range(size):
        img0 = dataset[ri]["image"]
        gt_masks = dataset[ri]["masks"]
        img = np.array(img0, dtype=np.uint8)
        
        result_masks = mask_generator.generate(img)
    
        res_annots = illustrate_masks(img, result_masks)
        merged = np.hstack((img, res_annots))
        gt_annots = illustrate_masks(img, gt_masks, True)
        merged = np.hstack((merged, gt_annots))
        merged = np.rot90(merged)
        
        logger.log_img(f'Images/img{ri}', merged.transpose((2, 0, 1)))

    return logger

if __name__ == "__main__":
    args = parsing()

    TRAINING_NAME = args.trainingname
    MASKN = args.masks if args.masks else 10 # Max number of masks to use per image for training
    NUM_EPOCHS = args.epochs if args.epochs else 1
    BATCHSIZE = 8

    TRAINING_DICT_PATH = r"C:\Users\KI-Lab2\Documents\Finetune\Images\dataset.json"
    SAM_PATH = r"C:\Users\KI-Lab2\Documents\Finetune\sam_vit_h_4b8939.pth"
    SAM_TYPE = 'vit_h'

    if TRAINING_NAME:
        destdir = os.path.join(r"C:\Users\KI-Lab2\Documents\Finetune\Dice", TRAINING_NAME)
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        else:
            while True:
                continue_inp = input("Should folder be overwritten? (y/n)")
                if continue_inp == "y":
                    break
                elif continue_inp == "n":
                    raise FileExistsError("Error: File already exists.")
                else:
                    print("Input not recognized")
    else:
        destdir = ""

    # Loading JSON dictionary containing lists of paths with x being RGB images and y being binary masks
    # x shape: (n images), y shape (n images, n masks), n masks can vary
    with open(TRAINING_DICT_PATH, "r") as f:
        pathdict = json.load(f)

    x_train, x_val, y_train, y_val = pathdict["x_train"], pathdict["x_val"], pathdict["y_train"], pathdict["y_val"]
    trainsize = len(x_train)
    valsize = len(x_val)
    print("TRAIN: ", trainsize, "VAL: ", valsize)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        raise FileNotFoundError("No GPU was found")

    logger = Logging(logdir=destdir)

    # Load the SAM model
    sam_model = sam_model_registry[SAM_TYPE](checkpoint=SAM_PATH)
    sam_model.to(device)

    imgtransform = imgtransform_func()
    masktransform = masktransform_func()
    # Load custom dataset
    train_dataset = SegmentationDataset(x_train, y_train, MASKN, imgtransform=imgtransform, masktransform=masktransform)

    # Create a DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, collate_fn=collate_fn, pin_memory=False, num_workers=2)

    # Load custom dataset
    val_dataset = SegmentationDataset(x_val, y_val, MASKN, imgtransform=imgtransform, masktransform=masktransform, val=True)

    # Create a DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=collate_fn, pin_memory=False, num_workers=2)

    lr = 4e-6
    wd = 1e-4
    torch.backends.cudnn.benchmark = True
    parameters = list(sam_model.mask_decoder.parameters()) + list(sam_model.image_encoder.parameters()) + list(sam_model.prompt_encoder.parameters())
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=30, max_epochs=NUM_EPOCHS, warmup_start_lr=5e-7, eta_min=1e-6)
    loss_fn = DiceLoss(to_onehot_y=False)

    losses = []
    dice_score = []
    gd_score = []

    best_dice = -1
    best_gd = -1
    best_score = -1

    for epoch in range(NUM_EPOCHS):
        epochstart = time()
        epoch_losses = []

        # TRAINING
        for epochi, batch in enumerate(train_dataloader):
            print("Batch ", epochi + 1, " out of ", trainsize / BATCHSIZE)
            logger.log_usage()
            input_images, gt_binary_masks, gt_resized, bboxes, imgsizes = batch["images"], batch["masks"], batch["resized masks"], batch['bboxes'], batch['img sizes']

            batch_loss = 0
            batchlength = len(gt_binary_masks)
            
            for batchi in range(batchlength):
                maskn = len(gt_binary_masks[batchi])
                if epoch == 0 and epochi == 0 and batchi == 0:
                    print("Training Masks: ", maskn)

                with torch.cuda.amp.autocast():
                    image_embedding = sam_model.image_encoder(input_images[batchi].unsqueeze(0).to(device))
                    xsize, ysize = imgsizes[batchi]
                    
                    imageloss = 0
                    for maski in range(maskn):
                        box_torch = torch.unsqueeze(bboxes[batchi][maski].to(device), dim=0)

                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )

                        low_res_masks, iou_predictions = sam_model.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=sam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )

                        upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (ysize, xsize)).to(device)
                        binary_mask = torch.sigmoid(upscaled_masks)

                        gt_binary_mask = gt_binary_masks[batchi][maski].to(torch.float16).to(device)

                        if binary_mask.size()[0] > 1:
                            binary_mask = torch.unsqueeze(torch.sum(binary_mask, 0) / binary_mask.size()[0], 0)

                        loss = loss_fn(binary_mask[0], gt_binary_mask) / batchlength / maskn
                        imageloss += loss
                        batch_loss += loss.item()

                    scaler.scale(imageloss).backward()

            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(batch_loss)
            optimizer.zero_grad()
        scheduler.step()
        losses.append(mean(epoch_losses))
        logger.log_metrics(mean(epoch_losses), 'Training loss')
        logger.log_metrics(scheduler.get_last_lr()[0], 'Learning rate')
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

        # VALIDATION
        with torch.no_grad():
            batch_dice = []
            batch_gd = []

            for epochi, batch in enumerate(val_dataloader):
                input_images, gt_binary_masks, gt_resized, bboxes, imgsizes = batch["images"], batch["masks"], batch["resized masks"], batch['bboxes'], batch['img sizes']

                dice = DiceMetric()
                gd = GeneralizedDiceScore()

                batchlength = len(gt_binary_masks)
            
                for batchi in range(batchlength):
                    masknval = len(gt_binary_masks[batchi])
                    if epoch == 0 and epochi == 0 and batchi == 0:
                        print("Validation Masks: ", masknval)

                    image_embedding = sam_model.image_encoder(input_images[batchi].unsqueeze(0).to(device))
                    xsize, ysize = imgsizes[batchi]
                    
                    for maski in range(masknval):
                        box_torch = torch.unsqueeze(bboxes[batchi][maski].to(device), dim=0)

                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )

                        low_res_masks, iou_predictions = sam_model.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=sam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )

                        upscaled_masks = sam_model.postprocess_masks(low_res_masks, (1024, 1024), (ysize, xsize))
                        binary_mask = torch.sigmoid(upscaled_masks.detach().cpu())
                        binary_mask = (binary_mask > 0.5).float()

                        gt_binary_mask = gt_binary_masks[batchi][maski].detach().cpu()

                        if binary_mask.size()[0] > 1:
                            binary_mask = torch.unsqueeze(torch.sum(binary_mask, 0) / binary_mask.size()[0], 0)

                        dice.reset()
                        gd.reset()

                        dice(binary_mask[0, :], gt_binary_mask)
                        gd(binary_mask[0, :], gt_binary_mask)

                        final_dice = dice.aggregate().numpy()[0]
                        final_gd = gd.aggregate().numpy()[0]
                        batch_dice.append(final_dice)
                        batch_gd.append(final_gd)

            if (sum(batch_dice) / len(batch_dice)) > best_dice:
                best_dice = sum(batch_dice) / len(batch_dice)
                if TRAINING_NAME:
                    torch.save(sam_model.state_dict(), os.path.join(destdir, "BestDice.pth"))
                print("saved new best dice model")
            
            if (sum(batch_gd) / len(batch_gd)) > best_gd:
                best_gd = sum(batch_gd) / len(batch_gd)
                if TRAINING_NAME:
                    torch.save(sam_model.state_dict(), os.path.join(destdir, "BestGD.pth"))
                print("saved new best GD model")

            dice_score.append(sum(batch_dice) / len(batch_dice))
            gd_score.append(sum(batch_gd) / len(batch_gd))
        logger.log_metrics(dice_score[-1], 'Final val dice')
        logger.log_metrics(gd_score[-1], 'Final val gd')

        print(f'Mean val dice: {dice_score[-1]}')
        print(f'Mean val gd: {gd_score[-1]}')
        print(f"Epoch took {time() - epochstart} s")

    exp_dataset = SegmentationDataset(x_val, y_val, MASKN)
    logger = example_sam(sam_model, exp_dataset, logger)
    logger.close()
