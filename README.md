# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


torch.manual_seed(71)
x=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*x+1+e


plt.scatter(x,y,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("generated data for linear regression")
plt.show()

class Model(nn.Module):
    def __init__(self,input_features,output_features):
        super().__init__()
        self.linear=nn.Linear(input_features,output_features)
    def forward(self,x):
        return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weights=model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("Name: SANJAYKUMAR N B")
print("Register No: 212223230189")
print(f"Initial Weights: {initial_weights:.4f},Initial Bias: {initial_bias:.4f}\n")

loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

epochs=100
losses=[]

for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(x)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
        f'weight: {model.linear.weight.item():10.8f}  '
        f'bias: {model.linear.bias.item():10.8f}')
plt.plot(range(epochs),losses,color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.show()

final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print(f"Final Weight: {final_weight:.4f}, Final Bias: {final_bias:.4f}")





x1=torch.tensor([x.min().item(),x.max().item()])
y1=x1*final_weight+final_bias

plt.scatter(x,y,label='Original Data')
plt.plot(x1,y1,'r',label="Best-fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Trained Model: Best-Fit Line")
plt.legend()
plt.show()


x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new)
print(f"Prediction for x={x_new.item()} : {y_new_pred.item():.4f}")
```

### Dataset Information
<img width="854" height="555" alt="image" src="https://github.com/user-attachments/assets/1945b313-f475-4993-8665-e137ecd89038" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="891" height="534" alt="image" src="https://github.com/user-attachments/assets/0a6758f6-d7aa-4d5a-b895-bba1d5b5f8a4" />

Best Fit line plot
<img width="910" height="584" alt="image" src="https://github.com/user-attachments/assets/4d8b61b7-3645-446d-aac8-42a14244b9a6" />

Include your plot here

### New Sample Data Prediction
<img width="484" height="86" alt="image" src="https://github.com/user-attachments/assets/5d4f62c0-a10b-4e97-aef0-677899d1b6f9" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
