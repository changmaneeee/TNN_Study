import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

from tnn_resnet_model import TNNResNet18
from tnn_modules import clip_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
CLIP_VALUES = 1.0

print("Preparing CIFAR-10 dataset...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


print("Initializing TNN ResNet-18 model...")
model = TNNResNet18(num_classes=10).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS*0.5), int(NUM_EPOCHS*0.75)], gamma=0.1)

def train(epoch, model, trainloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 학습 중 가중치 클리핑 (STE를 사용할 때 중요)
        clip_weights(model, -CLIP_VALUES, CLIP_VALUES)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0 or batch_idx == len(trainloader) -1 :
            print(f'Epoch: {epoch} [{batch_idx*len(inputs)}/{len(trainloader.dataset)} ({100.*batch_idx/len(trainloader):.0f}%)]\t'
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% ({correct}/{total})')
            
    avg_loss = train_loss / (batch_idx + 1)
    accuracy = 100. * correct / total
    epoch_time = time.time() - start_time
    print(f'--- Train Epoch {epoch} Summary --- \nTime: {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return avg_loss, accuracy

def test(epoch, model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / (batch_idx + 1)
    accuracy = 100. * correct / total
    epoch_time = time.time() - start_time
    print(f'--- Test Epoch {epoch} Summary --- \nTime: {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})\n')
    return avg_loss, accuracy

# --- 메인 학습 루프 및 결과 로깅 ---
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS} - LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(epoch, model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    if scheduler:
        scheduler.step()
    
    # (선택) 모델 저장
    # if test_acc > best_acc:
    #     torch.save(model.state_dict(), './tnn_resnet18_cifar10_best.pth')
    #     best_acc = test_acc

# 최종 모델 저장
torch.save(model.state_dict(), './tnn_resnet18_cifar10_final.pth')
print("Training finished. Model saved.")

# --- 결과 시각화 ---
epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tnn_resnet18_cifar10_training_curves.png')
print("Training curves saved to tnn_resnet18_cifar10_training_curves.png")
plt.show()






