import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from models import ResNet, Classifier
from data import QuarkGluonClassificationDataset
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    
    train_files = [
        (f"/kaggle/input/ml4sci2data2/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_X_chunk_{i+1}.npy", f"/kaggle/input/ml4sci2data2/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_Y_chunk_{i+1}.npy")
        for i in range(4)
    ]

    test_files = [
        ("/kaggle/input/ml4sci2data2/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_X_chunk_5.npy", "/kaggle/input/ml4sci2data2/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_Y_chunk_5.npy")
    ]

    train = ConcatDataset([
        QuarkGluonClassificationDataset(
            x_file=f"{f[0]}",
            y_file=f"{f[1]}"
        )
        for f in train_files
    ])

    test = ConcatDataset([
        QuarkGluonClassificationDataset(
            x_file=f"{f[0]}",
            y_file=f"{f[1]}"
        )
        for f in test_files
    ])

    train = DataLoader(
        train,
        batch_size=64,
        shuffle=True,
    )

    test = DataLoader(
        test,
        batch_size=64,
        shuffle=False
    )

    net = Classifier(
        ResNet(
            in_channels=3, 
            dropout=.2, 
            depth=8
        ),
        hidden_size=128,
        num_classes=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net.to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-5)

    lossfn = nn.CrossEntropyLoss()

    train_accuracy_over_time = []
    test_accuracy_over_time = []
    train_loss_over_time = []
    test_loss_over_time = []

    best_test_loss = float("inf")

    for epoch in range(50):
        running_train_loss = []
        running_train_accuracy = []

        running_test_loss = []
        running_test_accuracy = []

        net.train()
        for x, y in tqdm(train):
            net.zero_grad()

            x = x.to(device)
            y = y.to(device)

            p = net(x)

            loss = lossfn(p, y)
            loss.backward()

            optimiser.step()

            accuracy = (p.argmax(-1) == y).float().mean()

            running_train_loss.append(loss.item())
            running_train_accuracy.append(accuracy.item())

        net.eval()
        with torch.no_grad():
            for x, y in tqdm(test):
                x = x.to(device)
                y = y.to(device)

                p = net(x)

                loss = lossfn(p, y)

                accuracy = (p.argmax(-1) == y).float().mean()

                running_test_loss.append(loss.item())
                running_test_accuracy.append(accuracy.item())

        train_loss = sum(running_train_loss)/len(running_train_loss)
        train_accuracy = sum(running_train_accuracy)/len(running_train_accuracy)

        test_loss = sum(running_test_loss)/len(running_test_loss)
        test_accuracy = sum(running_test_accuracy)/len(running_test_accuracy)

        train_accuracy_over_time.append(train_accuracy)
        test_accuracy_over_time.append(test_accuracy)

        train_loss_over_time.append(train_loss)
        test_loss_over_time.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            checkpoint = {
                "model": net.state_dict(),
                "epoch": epoch+1,
                "optimiser": optimiser.state_dict(),
            }


            torch.save(checkpoint, f"resnet_classifier_checkpoint.pt")

        performance_checkpoint = {
            "train_loss_over_time": train_loss_over_time,
            "test_loss_over_time": test_loss_over_time,
            "train_accuracy_over_time": train_accuracy_over_time,
            "test_accuracy_over_time": test_accuracy_over_time
        }

        torch.save(performance_checkpoint, f"resnet_classifier_performance_checkpoint.pt")

        print(f"Epch ({epoch + 1}) Trn Loss ({train_loss:.3f}) Tst Loss ({test_loss:.3f}) Trn Acc ({train_accuracy:.3f}) Tst Loss ({test_accuracy:.3f})")