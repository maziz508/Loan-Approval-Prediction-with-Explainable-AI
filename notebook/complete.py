import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import comb
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#########################################
# Data Loading and Preprocessing
#########################################
loan_data = pd.read_csv('./loan_data.csv')

# One-hot encode categorical columns and convert boolean to int
categorical_columns = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
loan_data_encoded = pd.get_dummies(loan_data, columns=categorical_columns).astype(int)

# Move 'loan_status' column to the last position
loan_status = loan_data_encoded.pop('loan_status')
loan_data_encoded['loan_status'] = loan_status

# Split features and labels
features = loan_data_encoded.iloc[:, :-1].values
labels = loan_data_encoded['loan_status'].values

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data
num_samples = len(features)
indices = np.random.permutation(num_samples)
train_end = int(0.6 * num_samples)
val_end = int(0.8 * num_samples)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

X_train, y_train = features[train_indices], labels[train_indices]
X_val, y_val = features[val_indices], labels[val_indices]
X_test, y_test = features[test_indices], labels[test_indices]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

final_feature_names = loan_data_encoded.columns[:-1]

#########################################
# Model Definitions with Hyperparams
#########################################
class MLP(nn.Module):
    def __init__(self, input_size, hidden_units=64, num_layers=2):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_size, num_conv_layers=1, initial_filters=16):
        super(CNN, self).__init__()
        conv_layers = []
        filters = initial_filters
        in_channels = 1
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = filters
            filters *= 2
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Linear((filters//2) * input_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv(x)    
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads=4, num_layers=2, d_model=64, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_size, bottleneck_dim=32):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.classifier(encoded)
        return output, decoded

input_size = X_train.shape[1]

#########################################
# Training/Validation Function
#########################################
def train_validate_model(model, train_loader, val_loader, lr=0.001, max_epochs=100, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=False)
    criterion_classifier = nn.BCEWithLogitsLoss()
    criterion_reconstruction = nn.MSELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                out, decoded = outputs
                loss_class = criterion_classifier(out, targets)
                loss_recon = criterion_reconstruction(decoded, inputs)
                loss = loss_class + 0.5 * loss_recon
            else:
                loss = criterion_classifier(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss_classification = 0.0
        val_loss_reconstruction = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)
                val_out = model(inputs)
                if isinstance(val_out, tuple):
                    val_o, val_dec = val_out
                    val_loss_class = criterion_classifier(val_o, targets)
                    val_loss_recon = criterion_reconstruction(val_dec, inputs)
                    val_loss = val_loss_class + 0.5 * val_loss_recon
                    val_loss_classification += val_loss_class.item()
                    val_loss_reconstruction += val_loss_recon.item()
                else:
                    val_loss = criterion_classifier(val_out, targets)
                    val_loss_classification += val_loss.item()
                val_count += 1

        if isinstance(model(inputs), tuple):
            avg_val_loss = (val_loss_classification + 0.5 * val_loss_reconstruction) / val_count
        else:
            avg_val_loss = val_loss_classification / val_count

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

    return best_val_loss

def finalize_and_test_model(model, model_name, train_loader, val_loader, test_loader, lr, max_epochs=100, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=False)
    criterion_classifier = nn.BCEWithLogitsLoss()
    criterion_reconstruction = nn.MSELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Measure training time
    train_start = time.time()
    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                out, decoded = outputs
                loss_class = criterion_classifier(out, targets)
                loss_recon = criterion_reconstruction(decoded, inputs)
                loss = loss_class + 0.5 * loss_recon
            else:
                loss = criterion_classifier(outputs, targets)

            loss.backward()
            optimizer.step()

        # Validation check
        model.eval()
        val_loss_classification = 0.0
        val_loss_reconstruction = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.view(-1, 1).to(device)
                val_out = model(inputs)
                if isinstance(val_out, tuple):
                    val_o, val_dec = val_out
                    val_loss_class = criterion_classifier(val_o, targets)
                    val_loss_recon = criterion_reconstruction(val_dec, inputs)
                    val_loss = val_loss_class + 0.5 * val_loss_recon
                    val_loss_classification += val_loss_class.item()
                    val_loss_reconstruction += val_loss_recon.item()
                else:
                    val_loss = criterion_classifier(val_out, targets)
                    val_loss_classification += val_loss.item()
                val_count += 1

        if isinstance(model(inputs), tuple):
            avg_val_loss = (val_loss_classification + 0.5 * val_loss_reconstruction) / val_count
        else:
            avg_val_loss = val_loss_classification / val_count

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f'./models/{model_name}.pth')
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break
    train_end = time.time()
    training_time = train_end - train_start
    print(f"Training Time for {model_name}: {training_time:.2f} seconds")

    # Test performance
    model.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location=device))
    model.eval()

    test_start = time.time()
    model_actuals = []
    model_predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.view(-1,1).to(device)
            model_actuals.extend(targets.cpu().numpy().flatten())
            out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]
            preds = (torch.sigmoid(out).cpu().numpy().flatten() > 0.5).astype(int)
            model_predictions.extend(preds)
    test_end = time.time()
    testing_time = test_end - test_start
    print(f"Testing Time for {model_name}: {testing_time:.2f} seconds")

    accuracy = accuracy_score(model_actuals, model_predictions)
    precision = precision_score(model_actuals, model_predictions, zero_division=1)
    recall = recall_score(model_actuals, model_predictions, zero_division=1)
    f1 = f1_score(model_actuals, model_predictions, zero_division=1)

    print(f"\nTesting Metrics for {model_name} with best hyperparameters:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

#########################################
# Hyperparameter Search
#########################################
max_epochs = 100
patience = 10

# MLP search
mlp_lrs = [0.001, 0.0005]
mlp_hidden_units = [64, 128]
mlp_layers = [2, 3]

# CNN search
cnn_lrs = [0.001, 0.0005]
cnn_conv_layers = [1, 2]
cnn_filters = [16, 32]

# Transformer search
transformer_lrs = [0.001, 0.0005]
transformer_heads = [2, 4]
transformer_layers = [2, 3]
transformer_d_models = [64, 128]

# Autoencoder search
autoencoder_lrs = [0.001, 0.0005]
autoencoder_bottleneck = [32, 16]

print("Hyperparameter Ranges:")
print("MLP: LR:", mlp_lrs, "Hidden_Units:", mlp_hidden_units, "Layers:", mlp_layers)
print("CNN: LR:", cnn_lrs, "Conv_Layers:", cnn_conv_layers, "Initial_Filters:", cnn_filters)
print("Transformer: LR:", transformer_lrs, "Heads:", transformer_heads, "Layers:", transformer_layers, "d_model:", transformer_d_models)
print("Autoencoder: LR:", autoencoder_lrs, "Bottleneck:", autoencoder_bottleneck)

best_params = {}

# MLP search
print("\nSearching best hyperparameters for MLP...")
mlp_start = time.time()
best_score = float('inf')
best_config = None
for lr in mlp_lrs:
    for hu in mlp_hidden_units:
        for nl in mlp_layers:
            model = MLP(input_size, hidden_units=hu, num_layers=nl).to(device)
            val_loss = train_validate_model(model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience)
            if val_loss < best_score:
                best_score = val_loss
                best_config = (lr, hu, nl)
mlp_end = time.time()
print(f"Hyperparameter Search Time for MLP: {mlp_end - mlp_start:.2f} seconds")
best_params['MLP'] = best_config
print(f"Best hyperparameters for MLP: LR={best_config[0]}, Hidden_Units={best_config[1]}, Layers={best_config[2]}, Val Loss={best_score:.4f}")

# CNN search
print("\nSearching best hyperparameters for CNN...")
cnn_start = time.time()
best_score = float('inf')
best_config = None
for lr in cnn_lrs:
    for cl in cnn_conv_layers:
        for f in cnn_filters:
            model = CNN(input_size, num_conv_layers=cl, initial_filters=f).to(device)
            val_loss = train_validate_model(model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience)
            if val_loss < best_score:
                best_score = val_loss
                best_config = (lr, cl, f)
cnn_end = time.time()
print(f"Hyperparameter Search Time for CNN: {cnn_end - cnn_start:.2f} seconds")
best_params['CNN'] = best_config
print(f"Best hyperparameters for CNN: LR={best_config[0]}, Conv_Layers={best_config[1]}, Initial_Filters={best_config[2]}, Val Loss={best_score:.4f}")

# Transformer search
print("\nSearching best hyperparameters for Transformer...")
transformer_start = time.time()
best_score = float('inf')
best_config = None
for lr in transformer_lrs:
    for h in transformer_heads:
        for nl in transformer_layers:
            for dm in transformer_d_models:
                model = TransformerModel(input_size, num_heads=h, num_layers=nl, d_model=dm).to(device)
                val_loss = train_validate_model(model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience)
                if val_loss < best_score:
                    best_score = val_loss
                    best_config = (lr, h, nl, dm)
transformer_end = time.time()
print(f"Hyperparameter Search Time for Transformer: {transformer_end - transformer_start:.2f} seconds")
best_params['Transformer'] = best_config
print(f"Best hyperparameters for Transformer: LR={best_config[0]}, Heads={best_config[1]}, Layers={best_config[2]}, d_model={best_config[3]}, Val Loss={best_score:.4f}")

# Autoencoder search
print("\nSearching best hyperparameters for Autoencoder...")
auto_start = time.time()
best_score = float('inf')
best_config = None
for lr in autoencoder_lrs:
    for bdim in autoencoder_bottleneck:
        model = AutoencoderClassifier(input_size, bottleneck_dim=bdim).to(device)
        val_loss = train_validate_model(model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience)
        if val_loss < best_score:
            best_score = val_loss
            best_config = (lr, bdim)
auto_end = time.time()
print(f"Hyperparameter Search Time for Autoencoder: {auto_end - auto_start:.2f} seconds")
best_params['Autoencoder'] = best_config
print(f"Best hyperparameters for Autoencoder: LR={best_config[0]}, Bottleneck={best_config[1]}, Val Loss={best_score:.4f}")

#########################################
# Final Training and Testing with Best Hyperparameters
#########################################
os.makedirs('./models', exist_ok=True)

final_models = {}

# MLP final
mlp_lr, mlp_hu, mlp_nl = best_params['MLP']
mlp_final = MLP(input_size, hidden_units=mlp_hu, num_layers=mlp_nl).to(device)
finalize_and_test_model(mlp_final, 'MLP', train_loader, val_loader, test_loader, mlp_lr, max_epochs, patience)
final_models['MLP'] = mlp_final

# CNN final
cnn_lr, cnn_cl, cnn_f = best_params['CNN']
cnn_final = CNN(input_size, num_conv_layers=cnn_cl, initial_filters=cnn_f).to(device)
finalize_and_test_model(cnn_final, 'CNN', train_loader, val_loader, test_loader, cnn_lr, max_epochs, patience)
final_models['CNN'] = cnn_final

# Transformer final
trans_lr, trans_h, trans_nl, trans_dm = best_params['Transformer']
transformer_final = TransformerModel(input_size, num_heads=trans_h, num_layers=trans_nl, d_model=trans_dm).to(device)
finalize_and_test_model(transformer_final, 'Transformer', train_loader, val_loader, test_loader, trans_lr, max_epochs, patience)
final_models['Transformer'] = transformer_final

# Autoencoder final
auto_lr, auto_bdim = best_params['Autoencoder']
autoencoder_final = AutoencoderClassifier(input_size, bottleneck_dim=auto_bdim).to(device)
finalize_and_test_model(autoencoder_final, 'Autoencoder', train_loader, val_loader, test_loader, auto_lr, max_epochs, patience)
final_models['Autoencoder'] = autoencoder_final

#########################################
# Print Dataset Shapes and Feature Names
#########################################
print("\nDataset Shapes:")
print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

print("\nFinal feature names:")
print(list(final_feature_names))

#########################################
# Implementing OLGA, SHAP, LIME from Scratch
#########################################
def plot_feature_attributions(attributions, feature_names, title="Feature Attributions", output_path=None, sort_by_magnitude=False):
    assert len(attributions) == len(feature_names), "Length mismatch between attributions and feature names."
    if sort_by_magnitude:
        indices = np.argsort(np.abs(attributions))[::-1]
    else:
        indices = np.arange(len(attributions))

    sorted_attributions = attributions[indices]
    sorted_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(sorted_attributions)), sorted_attributions, color='steelblue')
    plt.xticks(range(len(sorted_attributions)), sorted_features, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.title(title)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        if height != 0:
            plt.text(bar.get_x() + bar.get_width()/2,
                     height + 0.01 * np.sign(height),
                     f"{height:.2f}",
                     ha='center',
                     va='bottom' if height > 0 else 'top',
                     fontsize=8,
                     color='black',
                     rotation=90)
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def olga_feature_attributions(model, x0, epsilon=1e-3, device='cpu'):
    start_time = time.time()
    model.eval()
    x0 = x0.to(device)
    D = x0.shape[0]

    with torch.no_grad():
        outputs = model(x0.unsqueeze(0))
        if isinstance(outputs, tuple):
            outputs = outputs[0]

    f_plus = np.zeros(D)
    f_minus = np.zeros(D)
    for i in range(D):
        x_plus = x0.clone()
        x_minus = x0.clone()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        with torch.no_grad():
            y_plus = model(x_plus.unsqueeze(0))
            y_minus = model(x_minus.unsqueeze(0))

            if isinstance(y_plus, tuple):
                y_plus = y_plus[0]
                y_minus = y_minus[0]

            f_plus[i] = y_plus.item()
            f_minus[i] = y_minus.item()

    gradients = (f_plus - f_minus) / (2.0 * epsilon)
    end_time = time.time()
    print(f"OLGA Computation Time: {end_time - start_time:.4f} seconds")
    return gradients

def manual_shap(model, x0, X_train, num_samples=100, device='cpu'):
    start_time = time.time()
    model.eval()
    x0_np = x0.cpu().numpy()
    D = x0_np.shape[0]
    X_mean = X_train.mean(axis=0)

    def shap_kernel(subset):
        s = len(subset)
        if s == 0 or s == D:
            return 0
        denom = comb(D, s)
        return (D - 1) / (denom * s * (D - s))

    subsets = []
    for _ in range(num_samples):
        s = np.random.randint(1, D)
        subset = np.random.choice(range(D), size=s, replace=False)
        subsets.append(subset)

    shap_values = np.zeros(D)
    with torch.no_grad():
        for subset in subsets:
            subset = list(subset)
            subset_comp = [i for i in range(D) if i not in subset]

            x_sub = X_mean.copy()
            x_sub[subset] = x0_np[subset]

            x_comp = X_mean.copy()
            x_comp[subset_comp] = x0_np[subset_comp]

            x_sub_tensor = torch.tensor(x_sub, dtype=torch.float32, device=device).unsqueeze(0)
            x_comp_tensor = torch.tensor(x_comp, dtype=torch.float32, device=device).unsqueeze(0)

            y_sub = model(x_sub_tensor)
            y_comp = model(x_comp_tensor)
            if isinstance(y_sub, tuple):
                y_sub = y_sub[0]
                y_comp = y_comp[0]

            contrib = (y_sub - y_comp).item()
            w = shap_kernel(subset)

            for idx in subset:
                shap_values[idx] += w * contrib
    end_time = time.time()
    print(f"SHAP Computation Time: {end_time - start_time:.4f} seconds")
    return shap_values

def manual_lime(model, x0, X_train, num_samples=100, epsilon=0.01, device='cpu'):
    start_time = time.time()
    model.eval()
    x0_np = x0.cpu().numpy()
    D = x0_np.shape[0]

    stds = X_train.std(axis=0)
    perturbed_samples = []
    for _ in range(num_samples):
        perturbation = np.random.normal(0, epsilon, size=D) * stds
        x_pert = np.clip(x0_np + perturbation, X_train.min(axis=0), X_train.max(axis=0))
        perturbed_samples.append(x_pert)
    perturbed_samples = np.array(perturbed_samples)

    distances = np.sqrt(np.sum((perturbed_samples - x0_np)**2, axis=1))
    sigma = np.mean(distances) + 1e-8
    weights = np.exp(- (distances**2) / (2 * sigma**2))

    with torch.no_grad():
        inputs = torch.tensor(perturbed_samples, dtype=torch.float32, device=device)
        outs = model(inputs)
        if isinstance(outs, tuple):
            outs = outs[0]
        preds = outs.cpu().numpy().flatten()

    X_centered = perturbed_samples - perturbed_samples.mean(axis=0)
    W = np.diag(weights)
    XTWX = X_centered.T.dot(W).dot(X_centered)
    XTWy = X_centered.T.dot(W).dot(preds)
    ridge = 1e-6
    coeffs = np.linalg.solve(XTWX + ridge * np.eye(D), XTWy)
    end_time = time.time()
    print(f"LIME Computation Time: {end_time - start_time:.4f} seconds")
    return coeffs

x0 = X_test_tensor[0].to(device)
X_train_np = X_train_tensor.numpy()
feature_names = final_feature_names

for model_name, model in final_models.items():
    model.eval()
    print(f"\n=== Explanations for {model_name} ===")
    olga_attr = olga_feature_attributions(model, x0, epsilon=1e-3, device=device)
    plot_feature_attributions(olga_attr, feature_names, 
                              title=f"OLGA - {model_name}", 
                              output_path=f"./xai_plots/OLGA_{model_name}.png",
                              sort_by_magnitude=True)
    print("OLGA Attributions:", olga_attr)

    shap_attr = manual_shap(model, x0, X_train_np, num_samples=100, device=device)
    plot_feature_attributions(shap_attr, feature_names,
                              title=f"SHAP - {model_name}",
                              output_path=f"./xai_plots/SHAP_{model_name}.png",
                              sort_by_magnitude=True)
    print("SHAP Attributions (Approx.):", shap_attr)

    lime_attr = manual_lime(model, x0, X_train_np, num_samples=100, epsilon=0.01, device=device)
    plot_feature_attributions(lime_attr, feature_names,
                              title=f"LIME - {model_name}",
                              output_path=f"./xai_plots/LIME_{model_name}.png",
                              sort_by_magnitude=True)
    print("LIME Attributions (Approx.):", lime_attr)

print("\nExplanations (OLGA, SHAP, LIME) computed and plots saved successfully.")

