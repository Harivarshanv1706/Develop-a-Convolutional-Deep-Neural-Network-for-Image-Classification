# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Handwritten digit recognition is a fundamental problem in computer vision with applications in postal mail sorting, bank check processing, and digital form recognition.
We use the MNIST dataset, which contains:

60,000 training images

10,000 testing images
Each image is grayscale, size 28 × 28 pixels, representing digits 0–9.

## Neural Network Model

<img width="1280" height="551" alt="image" src="https://github.com/user-attachments/assets/8105ff81-a799-4d56-8634-a254323d8301" />


## DESIGN STEPS

### Step 1: Data Preparation

Import MNIST dataset

Apply transformations (Tensor conversion and normalization)

Create DataLoader for training and testing

### Step 2: Model Definition

Define CNN with convolutional, pooling, and fully connected layers

Use ReLU activation for non-linearity

### Step 3: Training and Testing

Train using CrossEntropyLoss and Adam optimizer

Evaluate using accuracy, confusion matrix, and classification report

## PROGRAM
```python
class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjusted input features for fc1
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: HARI VARSHAN V ')
        print('Register Number:212224230092')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```
### Name: HARI VARSHAN V

### Register Number: 212224230092


### OUTPUT

## Training Loss per Epoch

<img width="340" height="230" alt="Screenshot 2026-11-26 091902" src="https://github.com/user-attachments/assets/03d5d566-5bf4-46be-8d36-3245b948afa2" />


## Confusion Matrix

<img width="727" height="726" alt="Screenshot 2026-11-26 091826" src="https://github.com/user-attachments/assets/503465af-c679-4668-a06b-635b8bcffe6d" />


## Classification Report
<img width="552" height="436" alt="Screenshot 2026-11-26 091835" src="https://github.com/user-attachments/assets/e29ae861-7ab9-460d-9026-a9a1e59aa267" />


### New Sample Data Prediction
<img width="562" height="613" alt="Screenshot 2026-11-26 091846" src="https://github.com/user-attachments/assets/8e824c46-54e5-4d08-8a30-42d3e95cca28" />


## RESULT
A CNN was successfully implemented for handwritten digit classification using MNIST. The model achieved high accuracy (~98%) and correctly classified new unseen digit samples.
