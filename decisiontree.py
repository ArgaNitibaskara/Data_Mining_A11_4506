"""
Tugas Data Mining - Minggu ke-7

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
print("="*60)
print("DECISION TREE CLASSIFICATION - MUSIC GENRE")
print("="*60)

print("\n[1] Loading Dataset...")
# Load preprocessed data
df = pd.read_csv('musicgenre_preprocessed.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features and target
X = df.drop('genre_encoded', axis=1)
y = df['genre_encoded']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nGenre distribution:")
print(y.value_counts().sort_index())

# Genre names mapping (for better visualization)
genre_names = {
    0: 'Classic Pop & Rock',
    1: 'Classical',
    2: 'Dance & Electronica',
    3: 'Folk',
    4: 'Hip-Hop',
    5: 'Jazz & Blues',
    6: 'Metal',
    7: 'Pop',
    8: 'Punk',
    9: 'Soul & Reggae'
}

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================
print("\n[2] Splitting Data...")
# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# ==========================================
# 3. DECISION TREE CLASSIFIER - BASELINE
# ==========================================
print("\n[3] Training Baseline Decision Tree...")

# Create baseline model
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)

# Predictions
y_train_pred_baseline = dt_baseline.predict(X_train)
y_test_pred_baseline = dt_baseline.predict(X_test)

# Evaluate baseline
train_acc_baseline = accuracy_score(y_train, y_train_pred_baseline)
test_acc_baseline = accuracy_score(y_test, y_test_pred_baseline)

print(f"✓ Baseline Model Trained")
print(f"  Training Accuracy: {train_acc_baseline:.4f} ({train_acc_baseline*100:.2f}%)")
print(f"  Testing Accuracy: {test_acc_baseline:.4f} ({test_acc_baseline*100:.2f}%)")

# ==========================================
# 4. HYPERPARAMETER TUNING
# ==========================================
print("\n[4] Hyperparameter Tuning...")

# Define parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Best Parameters: {grid_search.best_params_}")
print(f"✓ Best CV Score: {grid_search.best_score_:.4f}")

# Get best model
dt_optimized = grid_search.best_estimator_

# Predictions with optimized model
y_train_pred = dt_optimized.predict(X_train)
y_test_pred = dt_optimized.predict(X_test)

# Evaluate optimized model
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nOptimized Model Performance:")
print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Testing Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ==========================================
# 5. CROSS-VALIDATION
# ==========================================
print("\n[5] Cross-Validation...")
cv_scores = cross_val_score(dt_optimized, X_train, y_train, cv=5, scoring='accuracy')
print(f"✓ 5-Fold CV Scores: {cv_scores}")
print(f"✓ Mean CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ==========================================
# 6. MODEL EVALUATION
# ==========================================
print("\n[6] Model Evaluation...")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=[genre_names[i] for i in range(10)]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================
print("\n[7] Feature Importance Analysis...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_optimized.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ==========================================
# 8. SAVE MODEL
# ==========================================
print("\n[8] Saving Model...")
joblib.dump(dt_optimized, 'decision_tree_model.pkl')
print("✓ Model saved to: decision_tree_model.pkl")

# Save evaluation results
with open('model_evaluation.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("DECISION TREE - MODEL EVALUATION\n")
    f.write("="*60 + "\n\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    f.write(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"Testing Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    f.write(f"Cross-Validation Scores: {cv_scores}\n")
    f.write(f"Mean CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred, target_names=[genre_names[i] for i in range(10)]))

print("✓ Evaluation results saved to: model_evaluation.txt")

# ==========================================
# 9. VISUALIZATIONS
# ==========================================
print("\n[9] Creating Visualizations...")

# Create visualization folder
import os
if not os.path.exists('dt_visualizations'):
    os.makedirs('dt_visualizations')

# Visualization 1: Decision Tree Diagram
print("  → Creating decision tree diagram...")
plt.figure(figsize=(25, 15))
plot_tree(
    dt_optimized,
    feature_names=X.columns,
    class_names=[genre_names[i] for i in range(10)],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3  # Only show top 3 levels for clarity
)
plt.title('Decision Tree Visualization (Max Depth 3)', fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('dt_visualizations/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ decision_tree.png saved")

# Visualization 2: Confusion Matrix Heatmap
print("  → Creating confusion matrix heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[genre_names[i] for i in range(10)],
    yticklabels=[genre_names[i] for i in range(10)],
    cbar_kws={'label': 'Count'},
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Predicted Genre', fontsize=12, fontweight='bold')
plt.ylabel('Actual Genre', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('dt_visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ confusion_matrix.png saved")

# Visualization 3: Feature Importance
print("  → Creating feature importance chart...")
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('dt_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ feature_importance.png saved")

# Visualization 4: Model Comparison (Baseline vs Optimized)
print("  → Creating model comparison chart...")
models = ['Baseline\n(Default Params)', 'Optimized\n(GridSearch)']
train_accs = [train_acc_baseline, train_acc]
test_accs = [test_acc_baseline, test_acc]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 7))
bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_accs, width, label='Testing Accuracy', color='salmon', edgecolor='black')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('dt_visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ model_comparison.png saved")

# Visualization 5: Class Distribution
print("  → Creating class distribution chart...")
plt.figure(figsize=(12, 7))
genre_counts = y.value_counts().sort_index()
genre_labels = [genre_names[i] for i in genre_counts.index]
colors = plt.cm.Set3(np.linspace(0, 1, len(genre_labels)))

plt.bar(genre_labels, genre_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Genre', fontsize=12, fontweight='bold')
plt.ylabel('Number of Songs', fontsize=12, fontweight='bold')
plt.title('Genre Distribution in Dataset', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (label, value) in enumerate(zip(genre_labels, genre_counts.values)):
    plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('dt_visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ class_distribution.png saved")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n✓ Decision Tree Model Successfully Trained!")
print(f"\nBest Model Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"\n✓ Testing Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✓ Cross-Validation Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"\nOutput Files:")
print(f"  1. decision_tree_model.pkl - Trained model")
print(f"  2. model_evaluation.txt - Detailed evaluation metrics")
print(f"  3. dt_visualizations/ - Visualizations folder")
print(f"     - decision_tree.png")
print(f"     - confusion_matrix.png")
print(f"     - feature_importance.png")
print(f"     - model_comparison.png")
print(f"     - class_distribution.png")
print("\n" + "="*60)
print("DECISION TREE CLASSIFICATION COMPLETED!")
print("="*60)
