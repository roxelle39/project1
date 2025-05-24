import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Transformation standard
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Mapping multi-classes vers binaire
# 0 = cerveau sain (notumor), 1 = cerveau malade (glioma, meningioma, pituitary)
multi_to_binary_map = {
    0: 1,  # ici 0 est à remplacer selon l'ordre des dossiers dans ImageFolder
    1: 0,
    2: 1,
    3: 1
}

class BinaryLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        binary_label = multi_to_binary_map[label]
        return image, binary_label

def get_data_loaders(data_dir='breast_cancer', batch_size=32):
    train_dataset_multi = datasets.ImageFolder(root=f"/home/rokhayadiop/Téléchargements/project/breast_cancer/training", transform=transform)
    test_dataset_multi = datasets.ImageFolder(root=f"/home/rokhayadiop/Téléchargements/project/breast_cancer/testing", transform=transform)

    # Vérifie les classes et leur ordre (important pour adapter multi_to_binary_map)
    print(f"Classes détectées: {train_dataset_multi.classes}")

    train_dataset = BinaryLabelDataset(train_dataset_multi)
    test_dataset = BinaryLabelDataset(test_dataset_multi)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

