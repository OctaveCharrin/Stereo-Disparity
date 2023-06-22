import torch
import hydra
import torchvision
import cv2
import numpy as np
from model.dilnet import DilNetLRDisp
from dataset.datamodule import KittiDataset

import matplotlib.pyplot as plt


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    pre_filter_cap = 0
    sad_window_size = 3
    p1 = sad_window_size*sad_window_size*4
    p2 = sad_window_size*sad_window_size*32
    min_disparity = 0
    num_disparities = 32
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    disp_max_diff = 1
    full_dp = 1

    sgbm = cv2.StereoSGBM_create(min_disparity, num_disparities, sad_window_size,p1,p2,disp_max_diff, pre_filter_cap,
                    uniqueness_ratio, speckle_window_size, speckle_range, full_dp)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stereo_model = DilNetLRDisp(6, 1).to(device)

    if cfg.resume:
        stereo_model.load_state_dict(torch.load(cfg.resume_checkpoint))
    
    val_transform = torchvision.transforms.ToTensor()
    val_dataset = KittiDataset(stereo_path=cfg.test_stereo_path, gtdisp_path=cfg.test_gtdisp_path, co_transforms=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    stereo_model.eval()
    with torch.no_grad():
        total_loss = 0.
        num_sample = 0
        for i, (imgs, targets) in enumerate(val_loader):
            left_inputs, right_inputs = imgs
            left_inputs = left_inputs.to(device)
            right_inputs = right_inputs.to(device)
            targets = targets.to(device)

            outputs = stereo_model((left_inputs, right_inputs))

            masks = targets > 0
            outputs[~(masks)] = 0
            targets[~(masks)] = 0
            loss = torch.nn.functional.l1_loss(outputs, targets)

            total_loss += loss.detach().cpu().numpy() * len(left_inputs)
            num_sample += len(left_inputs)

            if i%10==0:
                output = torch.squeeze(outputs.cpu()).numpy()
                target = torch.squeeze(targets.cpu()).numpy()

                l,r = imgs
                l = 255*torch.squeeze(l).numpy().transpose(1,2,0).astype(np.uint8)
                r = 255*torch.squeeze(r).numpy().transpose(1,2,0).astype(np.uint8)
                sgbm_output = sgbm.compute(l, r)
                plot_res(l,r,output, sgbm_output, target)
        
        print("evaluation loss: ", total_loss/num_sample)


def plot_res(l,r,model_output, sgbm_output, target):
    fig, axes = plt.subplots(2, 3)
    im0 = axes[0][0].imshow(model_output)
    axes[0][0].axis('off')
    im1 = axes[0][1].imshow(sgbm_output)
    axes[0][1].axis('off')
    im2 = axes[0][2].imshow(target)
    axes[0][2].axis('off')

    axes[1][0].imshow(l)
    axes[1][0].axis('off')
    axes[1][1].imshow(r)
    axes[1][1].axis('off')
    axes[1][2].imshow(target)
    axes[1][2].axis('off')
    
    # Add colorbars
    cbar0 = fig.colorbar(im0, ax=axes[0][0])
    cbar0.set_label('Pixel Value')
    cbar1 = fig.colorbar(im1, ax=axes[0][1])
    cbar1.set_label('Pixel Value')
    cbar2 = fig.colorbar(im2, ax=axes[0][2])
    cbar2.set_label('Pixel Value')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()