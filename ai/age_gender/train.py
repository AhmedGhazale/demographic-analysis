import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import AgeGenderModel
from dataset import UTKFaceDataset, get_train_transforms, get_val_transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, targets = self.subset[idx]
        age = targets['age']
        gender = targets['gender']

        # Convert PIL Image to numpy array for albumentations
        if self.transform:
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            image = transformed['image']

        # Convert targets to tensors
        age = torch.tensor(age / 116.0, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.float32)

        return image, {'age': age, 'gender': gender}

    def __len__(self):
        return len(self.subset)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    base_dataset = UTKFaceDataset(args.data_dir)
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset = TransformSubset(train_subset, transform=get_train_transforms())
    val_dataset = TransformSubset(val_subset, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = AgeGenderModel(backbone_name=args.backbone).to(device)

    # Loss functions
    age_criterion = torch.nn.L1Loss()
    gender_criterion = torch.nn.BCELoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_age_loss = 0.0
        train_gender_loss = 0.0
        train_gender_acc = 0.0

        for images, targets in tqdm(train_loader):
            images = images.to(device)
            age_labels = targets['age'].to(device)
            gender_labels = targets['gender'].to(device)

            optimizer.zero_grad()

            age_pred, gender_pred = model(images)

            # Calculate losses
            age_loss = age_criterion(age_pred.squeeze() / 116, age_labels)
            gender_loss = gender_criterion(gender_pred.squeeze(), gender_labels)
            total_loss = age_loss + gender_loss

            total_loss.backward()
            optimizer.step()

            # Calculate metrics
            train_age_loss += age_loss.item()
            train_gender_loss += gender_loss.item()
            train_gender_acc += ((gender_pred.squeeze() > 0.5).float() == gender_labels).float().mean().item()

        scheduler.step()

        # Validation
        model.eval()
        val_age_loss = 0.0
        val_gender_loss = 0.0
        val_gender_acc = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                age_labels = targets['age'].to(device)
                gender_labels = targets['gender'].to(device)

                age_pred, gender_pred = model(images)

                val_age_loss += age_criterion(age_pred.squeeze() / 116, age_labels).item()
                val_gender_loss += gender_criterion(gender_pred.squeeze(), gender_labels).item()
                val_gender_acc += ((gender_pred.squeeze() > 0.5).float() == gender_labels).float().mean().item()

        # Calculate averages
        train_age_loss /= len(train_loader)
        train_gender_loss /= len(train_loader)
        train_gender_acc /= len(train_loader)

        val_age_loss /= len(val_loader)
        val_gender_loss /= len(val_loader)
        val_gender_acc /= len(val_loader)

        # Logging
        writer.add_scalars('Loss/Age', {'train': train_age_loss, 'val': val_age_loss}, epoch)
        writer.add_scalars('Loss/Gender', {'train': train_gender_loss, 'val': val_gender_loss}, epoch)
        writer.add_scalars('Accuracy/Gender', {'train': train_gender_acc, 'val': val_gender_acc}, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Save best model
        current_val_loss = val_age_loss + val_gender_loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), os.path.join(args.log_dir,f'best_model_{args.backbone}.pth'))

        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'Train Age Loss: {train_age_loss * 116:.2f} | Gender Acc: {train_gender_acc:.4f}')
        print(f'Val Age Loss: {val_age_loss * 116:.2f} | Gender Acc: {val_gender_acc:.4f}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/UTKFace')
    parser.add_argument('--log_dir', type=str, default='exp2')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=20)

    args = parser.parse_args()
    main(args)