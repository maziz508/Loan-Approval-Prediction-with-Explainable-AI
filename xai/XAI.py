import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
