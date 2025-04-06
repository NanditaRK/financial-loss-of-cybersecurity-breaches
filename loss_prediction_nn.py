import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# read data that is combined from both datasets three and four as well as dataset two for financial cost data because we want to have a dataset that is both generalized and focused (on healthcare sector)
df = pd.read_csv('data/breach_report.csv')

# data preprocessing
if df.isna().sum().any():
    df = df.dropna()
    
cost_of_individual_record = 169

df['Financial Loss'] = df['Individuals Affected'] * cost_of_individual_record

df = df.drop(columns=['Individuals Affected'])

# log transform to reduce variance
df['log_Financial_Loss'] = np.log1p(df['Financial Loss'])

# label the categories and set the target variables
categorical_columns = ['State', 'Covered Entity Type', 'Type of Breach', 
                       'Location of Breached Information', 'Business Associate Present']
target_column = 'log_Financial_Loss'

# encode the categorical columns into a numerical value because our neural network only likes eating numbers !!!
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

X = df[categorical_columns].values
y = df[target_column].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# visualize distribution of target
plt.figure(figsize=(10, 6))
plt.hist(df['log_Financial_Loss'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Log-transformed Financial Loss', fontsize=16)
plt.xlabel('Log(Financial Loss + 1)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y')
plt.show()

# split our data into 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# turn into tensors for neural network because we are using pytorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# deine architecture of oruy model 
class BreachImpactModel(nn.Module):
    def __init__(self):
        super(BreachImpactModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = BreachImpactModel()

# using kaiming weight initialization whcih is basicaly just a fancy shmancy type of way to initialize the wieghts to avoid exploding gradients or vanishing gradients
# this is a good weight initialization that pairs up with the relu activation function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# adam optimizer a built in optimizer (gradient descent) from pytorch
optimizer = optim.Adam(model.parameters(), lr=0.001)

# considered the huber loss, a type of loss functioon
loss_fn = nn.SmoothL1Loss()  

# the training loopp!!
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


    model.eval()
    train_loss = epoch_loss / len(train_loader)
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# evaluation of model 
train_loss = evaluate_model(model, train_loader, loss_fn)
test_loss = evaluate_model(model, test_loader, loss_fn)

print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

predicted_log_values = model(X_test_tensor).squeeze().detach().numpy()

# revert from log transformation
predicted_values = np.exp(predicted_log_values)

# graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_values, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Predicted vs Financial Loss', fontsize=16)
plt.xlabel('Actual Financial Loss', fontsize=14)
plt.ylabel('Predicted Financial Loss', fontsize=14)
plt.show()


# permutation analysis gives us the feature importance here
def permutation_importance(model, X, y, data_loader, loss_fn):
    baseline_loss = evaluate_model(model, data_loader, loss_fn)
    feature_importance = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32)
        permuted_dataset = TensorDataset(permuted_tensor, torch.tensor(y, dtype=torch.float32).view(-1, 1))
        permuted_loader = DataLoader(permuted_dataset, batch_size=32, shuffle=False)
        
        permuted_loss = evaluate_model(model, permuted_loader, loss_fn)
        importance = permuted_loss - baseline_loss
        feature_importance.append(importance)
        print(f'Feature {i}: Importance = {importance:.4f}')
    
    return feature_importance

# Evaluate permutation importance
feature_importance = permutation_importance(model, X_train, y_train, train_loader, loss_fn)
print("Feature Importance:")
for idx, imp in enumerate(feature_importance):
    print(f"Feature {categorical_columns[idx]}: Importance = {imp:.4f}")
