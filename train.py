import torch
import hydra
import torchvision
from tqdm import tqdm
from model.dilnet import DilNetLRDisp
from dataset.datamodule import KittiDataset
import dataset.transforms as tf


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stereo_model = DilNetLRDisp(6, 1).to(device)

    if cfg.resume:
        stereo_model.load_state_dict(torch.load(cfg.resume_checkpoint))

    optimizer = torch.optim.AdamW(stereo_model.parameters(), lr=cfg.lr)

    input_transform = tf.RandomColorJitter(0.5,0.5,0.5,0.35,0.5)
    co_transform = tf.Compose([tf.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
    val_transform = torchvision.transforms.ToTensor()

    train_dataset = KittiDataset(stereo_path=cfg.train_stereo_path, gtdisp_path=cfg.train_gtdisp_path, co_transforms=co_transform, input_transforms=input_transform)
    val_dataset = KittiDataset(stereo_path=cfg.test_stereo_path, gtdisp_path=cfg.test_gtdisp_path, co_transforms=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Training
    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0.
        num_sample = 0
        for i, (imgs, targets) in enumerate(train_loader):
            left_inputs, right_inputs = imgs
            left_inputs = left_inputs.to(device)
            right_inputs = right_inputs.to(device)
            targets = targets.to(device)

            outputs = stereo_model((left_inputs, right_inputs))

            masks = targets > 0

            outputs[~(masks)] = 0
            targets[~(masks)] = 0
            loss = torch.nn.functional.l1_loss(outputs, targets)

            epoch_loss += loss.detach().cpu().numpy() * len(left_inputs)
            num_sample += len(left_inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch     : ", epoch)
        print("epoch_loss: ", epoch_loss/num_sample)

    # Evaluation
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
        print("validation loss: ", total_loss/num_sample)
    
    if cfg.checpoint:
        torch.save(stereo_model.state_dict(), f"checkpoints/stereo_model_LR_only_{cfg.epochs}epochs.pth")


if __name__ == "__main__":
    main()