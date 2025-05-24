import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_prep import get_data_loaders  # Ton script personnalisé
import os

# Utilise le GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Chargement du modèle ResNet18 pré-entraîné
def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(num_epochs=10, learning_rate=0.001, batch_size=32):
    # Charge les données binaires (sain vs malade)
    train_loader, test_loader = get_data_loaders("brain_tumor_binary", batch_size=batch_size)

    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_brain_binary.pth")
    print("Modèle sauvegardé dans models/resnet18_brain_binary.pth")

if __name__ == "__main__":
    train()
