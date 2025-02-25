import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# read data
df = pd.read_csv('breach_report.csv')

# data preprocessing
if df.isna().sum().any():
    df = df.dropna() 

categorical_columns = ['State', 'Covered Entity Type', 'Type of Breach', 
                       'Location of Breached Information', 'Business Associate Present']
target_column = 'Individuals Affected'

encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

X = df[categorical_columns].values
y = df[target_column].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# load data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# feed forward neural network model definition
class BreachImpactModel(nn.Module):
    def __init__(self):
        super(BreachImpactModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, 16)  
        self.fc4 = nn.Linear(16, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))  
        x = torch.relu(self.fc3(x))  
        x = self.fc4(x)  
        return x

model = BreachImpactModel()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# weight initialization (xavier uniform)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

#training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for inputs, labels in train_loader:
        # gradient to zero
        optimizer.zero_grad()
        
        # forward propagation
        outputs = model(inputs)
        
        # loss calculation
        loss = loss_fn(outputs, labels)
        
        # backward propagation
        loss.backward()
        
       
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # update the weights
        optimizer.step()
        
        # add the loss for this batch
        epoch_loss += loss.item()
    

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)



import numpy as np

def permutation_importance(model, X, y, data_loader, loss_fn):
    # evaluation of the model on the original data
    baseline_loss = evaluate_model(model, data_loader, loss_fn)
    
    feature_importance = []
    
    # loops through each feature
    for i in range(X.shape[1]):
        # permutes the feature column
        X_permuted = X.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # dataloader for the permuted data
        permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32)
        permuted_dataset = TensorDataset(permuted_tensor, torch.tensor(y, dtype=torch.float32).view(-1, 1))
        permuted_loader = DataLoader(permuted_dataset, batch_size=32, shuffle=False)
        
        # evaluate model with permuted feature
        permuted_loss = evaluate_model(model, permuted_loader, loss_fn)
        
        # calculate the change in performance (increase in loss)
        importance = permuted_loss - baseline_loss
        feature_importance.append(importance)
        print(f'Feature {i}: Importance = {importance:.4f}')
    
    return feature_importance



# evalulates permutation importance
feature_importance = permutation_importance(model, X_train, y_train, train_loader, loss_fn)

print("Feature Importance:")
for idx, imp in enumerate(feature_importance):
    print(f"Feature {categorical_columns[idx]}: Importance = {imp:.4f}")


# shap values
import shap

# create a SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# plot the SHAP values
shap.summary_plot(shap_values, X_train, feature_names=categorical_columns)
