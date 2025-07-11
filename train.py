import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

def resize_image(image, size=(32, 32)):
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil = image_pil.resize(size, Image.LANCZOS)
    image_pil = image_pil.convert('L')
    return np.array(image_pil) / 255.0

def get_4bit_quant():
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-8, quant_max=7,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False
    )

class FaceClassifier(nn.Module):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        self.quant_input = get_4bit_quant()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=4, stride=4, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=4, bias=False)

        self.quant_w1 = get_4bit_quant()
        self.quant_w2 = get_4bit_quant()
        self.quant_w3 = get_4bit_quant()
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        x = self.quant_input(x)

        self.conv1.weight.data = torch.clamp(self.conv1.weight.data, -1, 1)
        self.quant_w1(self.conv1.weight)
        x = self.conv1(x)
        x = self.relu(x)

        self.conv2.weight.data = torch.clamp(self.conv2.weight.data, -1, 1)
        self.quant_w2(self.conv2.weight)
        x = self.conv2(x)
        x = self.relu(x)
        
        self.conv3.weight.data = torch.clamp(self.conv3.weight.data, -1, 1)
        self.quant_w3(self.conv3.weight)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        return x.squeeze(1)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray, vmin=0, vmax=1)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

print("Loading LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=0.4, color=False)
X_face = np.array([resize_image(img) for img in lfw_people.images])
X_face = (X_face * 2) - 1 
X_face_flat = X_face.reshape(X_face.shape[0], -1)
y_face = np.ones(X_face.shape[0])

print("Loading CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1) 
])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
dog_indices = [i for i, (_, label) in enumerate(cifar10_dataset) if label == 5]
non_face_images = [cifar10_dataset[i][0][0].numpy() for i in dog_indices]

X_not_face = np.array(non_face_images)
X_not_face_flat = X_not_face.reshape(len(non_face_images), -1)
y_not_face = np.zeros(len(non_face_images))

min_samples = min(len(X_face_flat), len(X_not_face_flat))
np.random.seed(42)
face_indices = np.random.choice(len(X_face_flat), min_samples, replace=False)
non_face_indices = np.random.choice(len(X_not_face_flat), min_samples, replace=False)

X = np.vstack((X_face_flat[face_indices], X_not_face_flat[non_face_indices]))
y = np.hstack((y_face[face_indices], y_not_face[non_face_indices]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_test_original = X_test.copy()

X_train = torch.FloatTensor(X_train.reshape(-1, 1, 32, 32))
X_test = torch.FloatTensor(X_test.reshape(-1, 1, 32, 32))
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = FaceClassifier()
model.train()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
for epoch in range(200):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {total_loss / len(train_loader):.4f}")

print("\nEvaluating the model...")
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    relu_outputs = F.relu(outputs)
    predicted = (relu_outputs > 0).float()

print("\nClassification Report:")
print(classification_report(y_test, predicted, target_names=["Non-Face", "Face"]))

n_samples = 12
indices = np.random.choice(len(y_test), n_samples, replace=False)
test_images = ((X_test_original[indices] + 1) / 2)
true_labels = ['Face' if label == 1 else 'Non-Face' for label in y_test[indices]]
pred_labels = ['Face' if label == 1 else 'Non-Face' for label in predicted[indices]]
titles = [f"True: {t}\nPred: {p}" for t, p in zip(true_labels, pred_labels)]

plot_gallery(test_images, titles, 32, 32)
plt.show()
