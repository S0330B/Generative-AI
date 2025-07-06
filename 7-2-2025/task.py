import torch

inputs = torch.tensor([
    [1,2],
    [3,4],
    [5,6],
    [7,8]
    ], dtype=torch.float32)


y_true = torch.tensor([
    [10],
    [20],
    [30],
    [40]
    ], dtype=torch.float32)

weights_hidden = torch.rand((2,3),requires_grad=True)
weights_output = torch.rand((3,1),requires_grad=True)
bias_hidden = torch.rand((3),requires_grad=True)
bias_output = torch.rand((1),requires_grad=True)

epochs = 1000
learning_rate = 0.01

def relu(x):
    return torch.maximum(torch.tensor(0.0),x)

for epoch in range(epochs):
    linear_output = inputs @ weights_hidden + bias_hidden
    y_pred = relu(linear_output)

    final_output = y_pred @ weights_output + bias_output
    output = relu(final_output)

    loss = torch.mean((output-y_true)**2)
    loss.backward()

    with torch.no_grad():
        weights_hidden -= learning_rate*weights_hidden.grad
        weights_output -= learning_rate * weights_output.grad
        bias_hidden -= learning_rate * bias_hidden.grad
        bias_output -= learning_rate * bias_output.grad

        weights_hidden.grad.zero_()
        weights_output.grad.zero_()
        bias_hidden.grad.zero_()
        bias_output.grad.zero_()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss {loss.item()}")