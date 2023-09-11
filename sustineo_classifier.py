import pytorch_lightning as pl
from lib.classification.cnn import PlotCNN

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
    
def main():
    image_shape = (3, 200, 200)
    
    transform = transforms.Compose([
        transforms.Resize(image_shape[1::]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data_cifar', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    
    val_dataset = datasets.CIFAR10(root='./data_cifar', train=False, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8)
    
    model = PlotCNN(image_shape, num_classes=10)
    
    trainer = pl.Trainer(accelerator="auto", max_epochs=10)  # Puoi specificare il numero di epoche e la GPU
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()