import torch
import hydra
import os
import torchvision
from tqdm import tqdm
from model.dilnet import DilNetLRDisp
from dataset.transforms import TupleTransform


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stereo_model = DilNetLRDisp(6, 1).to(device)
    optimizer = torch.optim.AdamW(stereo_model.parameters(), lr=cfg.lr)

    transform = TupleTransform(torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((375, 1242))]))

    train_dataset = torchvision.datasets.Kitti2012Stereo(root=cfg.root, transforms=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Training
    for epoch in tqdm(range(cfg.epochs)):

        for i, (left_inputs, right_inputs, targets) in enumerate(train_loader):
            left_inputs = left_inputs.to(device)
            right_inputs = right_inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = stereo_model((left_inputs, right_inputs))

            optimizer.zero_grad()

            outputs[~(masks)] = 0
            targets[~(masks)] = 0
            loss = torch.nn.functional.l1_loss(outputs, targets)

            epoch_loss += loss.detach().cpu().numpy() * len(left_inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch     : ", epoch)
            print("epoch_loss: ", loss)


    # Evaluation
    if False:
        stereo_model.eval()
        with torch.no_grad():
            for left_inputs, right_inputs, targets in enumerate(test_loader):
                print("Testing will be programmed later... Please try later")


if __name__ == "__main__":
    main()