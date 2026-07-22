import numpy as np
from Autograd import WhyyTorch, MLP, mse_loss

# Tiny batch example
x_batch = WhyyTorch(np.random.randn(8, 3), requires_grad=False)
y_batch = WhyyTorch(np.random.randn(8, 1), requires_grad=False)

model = MLP(3, [8,8,1])
epochs = 100
for epoch in range(epochs):
    #Forward pass
    pred_batch = model(x_batch)
    #Calculate loss
    loss = mse_loss(pred_batch, y_batch,epoch)
    #Backward pass and update
    loss.backward()
    #Optimizer step, Update parameters
    model.step(0.01)
    #Zero gradients for next step
    model.zero_grad()
print("Real Values: ", y_batch.data.flatten())
print("Predicted Values: ", pred_batch.data.flatten())