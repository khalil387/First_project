import torch
import sys
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from torch import nn

weight= 0.7
bias=0.5
start=0
end=1
step=0.02
X=torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X +bias

train_f = int(0.8 * len(X))
X_train, Y_train= X[:train_f] ,Y[:train_f]
X_test, Y_test= X[train_f:] ,Y[train_f:]

# Ploting code
import matplotlib.pyplot as plt

def plot_predictions(predictions= None,
                train_data= X_train,
                train_label= Y_train,
                test_label= Y_test,
                test_data= X_test,):
  """

  Args:
    predictions:
    train_data:
    train_label:
    test_label:
    test_data:
  """

  plt.figure(figsize=(10, 7))
  plt.scatter(train_data, train_label, c='b', s=4, label = 'training_set')
  plt.scatter(test_data, test_label, c='g', s=4, label = 'tetsing_set')
  if predictions is not None:
    plt.scatter(test_data, predictions, c= 'r', s=4, label='prediction')
  plt.legend(prop={'size' : 14});


class LinearRegressionModel(nn.Module):
  """

  Attributes:
    weights:
    bias:
  """
  def __init__(self):
    super().__init__()
    self.weights= nn.Parameter(torch.randn(1,
                                           requires_grad=True,
                                           dtype= torch.float))
    self.bias= nn.Parameter(torch.randn(1,
                                           requires_grad=True,
                                           dtype= torch.float))
  def forward(self, x:torch.tensor) -> torch.tensor :
    """

    Args:
      x:

    Returns:

    """
    return self.weights * x + bias

model_0 = LinearRegressionModel()
with torch.inference_mode():
  Y_preds = model_0(X_test)

loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.001)

#epoch is one loop through the data
torch.manual_seed(12)
epochs = 2750
epochs_count = []
train_loss_values =[]
test_loss_values = []
for epoch in range(epochs):
  #Training loop
    model_0.train()
    ypred = model_0(X_train)
    loss= loss_func(ypred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  #Testing loop
    #Evakuation mode
    model_0.eval()

    with torch.inference_mode():
        y_preds_new = model_0(X_test)


    test_loss = loss_func(y_preds_new, Y_test)
  # Every 100 epochs
    if epoch % 10 == 0 :
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        epochs_count.append(epoch)
plot_predictions(y_preds_new);

"""#Hyperparameters
Hyperparameters are parameters that we set ourselves
"""

import numpy as np
plt.plot(epochs_count, train_loss_values, label = 'Training loss')
plt.plot(epochs_count, test_loss_values, label = 'Testing loss')
plt.title('Training and testing loss curves')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend();

from pathlib import Path
m_Path = Path('models')
m_Path.mkdir(parents= True, exist_ok= True)
m_Name = 'pytorch_workflow_model_0.pth'
m_save_path = m_Path / m_Name
torch.save(obj = model_0.state_dict,
           f= m_save_path )

torch.save(model_0.state_dict(), 'pytorch_work_model_0.pth')

print(y_preds_new - Y_test)