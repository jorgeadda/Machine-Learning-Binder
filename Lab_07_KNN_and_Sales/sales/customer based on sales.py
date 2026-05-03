# Dataset url: https://www.kaggle.com/datasets/sidramazam/e-commerce-sales-performance-analysis
#By Aidan and Jorge 
# importing libraries to format data and visualize it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# importing libraries for machine learning test and train split
from sklearn.model_selection import (
    train_test_split,       # Split data into train/test sets
    cross_val_score         # Cross-validation for robust evaluation
)
# importing libraries to feature enginerring by label encoding as well as accuracy test, confusiion matrix and classification report.
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("Customer Spending Tier Classification")
print("=" * 60)
print()



df = pd.read_csv("ecommerce_sales_data.csv")

# Encode categorical columns as integers so KNN can use them 
le_cat    = LabelEncoder()
le_region = LabelEncoder()

df['Category_Enc'] = le_cat.fit_transform(df['Category'])
df['Region_Enc']   = le_region.fit_transform(df['Region'])


df['Spending_Tier'] = pd.qcut(
    df['Sales'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# Quick look at the data
print("STEP 2: Dataset Overview")
print("-" * 40)
print(f"Total samples:       {df.shape[0]}")
print(f"Number of features:  4 (Category_Enc, Region_Enc, Quantity, Profit)")
print(f"Number of classes:   {df['Spending_Tier'].nunique()}")
print()

# Show how many of each spending tier we have
print("Class distribution:")
print(df['Spending_Tier'].value_counts().sort_index().to_string())
# Each class should have ~1167 samples (balanced by pd.qcut)
print()

# Show a few sample rows
print("First 5 rows (selected columns):")
print(df[['Category', 'Region', 'Quantity', 'Profit', 'Sales', 'Spending_Tier']].head().to_string(index=False))
print()




print("STEP 3: Exploring the Data")
print("-" * 40)

# Average values for each spending tier
print("Average measurements per spending tier:")
plot_df = df[['Category_Enc', 'Region_Enc', 'Quantity', 'Profit', 'Spending_Tier']].copy()
print(plot_df.groupby('Spending_Tier').mean().round(3))
print()
# Notice how Profit and Quantity generally INCREASE from Low → Medium → High
# This makes intuitive sense — larger orders generate more profit!


# We pick 3 features + the label to keep the plot readable
plot_features = ['Quantity', 'Profit', 'Category_Enc', 'Spending_Tier']

sns.pairplot(
    df[plot_features],         # Only these columns
    hue='Spending_Tier',       # Color by spending tier
    palette='Set2',            # A nice color palette
    diag_kind='kde',           # Diagonal: show density curves instead of histograms
    plot_kws={                 # Settings for the scatter plots
        'alpha': 0.5,          # Semi-transparent dots
        'edgecolor': 'w'       # White edges around dots
    }
)
plt.suptitle('Sales Feature Relationships by Spending Tier', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

print("   -> Pair plot displayed")
print("   -> Look for features where the 3 colors separate well.")
print("      Those features will be most useful for KNN!")
print()

# Individual feature comparison 
# Let's also make a box plot for Profit by spending tier
# Box plots show the median, quartiles, and outliers
plt.figure(figsize=(7, 4))
sns.boxplot(
    data=df,
    x='Spending_Tier',               # Categories on x-axis
    y='Profit',                      # Values on y-axis
    palette='Set2',                  # Same colors as pair plot
    order=['Low', 'Medium', 'High']  # Force this order
)
plt.xlabel('Spending Tier', fontsize=12)
plt.ylabel('Profit (dollars)', fontsize=12)
plt.title('Profit by Spending Tier', fontsize=14)
plt.tight_layout()
plt.show()

print("   -> Box plot displayed")
print()




print("STEP 4: Preparing the Data")
print("-" * 40)

# Separate features from labels
feature_cols = ['Category_Enc', 'Region_Enc', 'Quantity', 'Profit']
X = df[feature_cols]             # Only the four feature columns
y = df['Spending_Tier']          # Just the label column

# Split: 80% training, 20% testing
# stratify=y ensures each spending tier is proportionally represented
# in both the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print()

# Verify the class balance in both sets
print("Class balance check:")
print(f"  Training: {dict(y_train.value_counts().sort_index())}")
print(f"  Testing:  {dict(y_test.value_counts().sort_index())}")
print("  (Should be roughly proportional)")
print()

# Scale features
# Remember: fit_transform on training, transform on testing!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn + scale
X_test_scaled  = scaler.transform(X_test)         # Scale only (no learning)

print("Features scaled with StandardScaler.")
print()




print("STEP 5: Training and Evaluating")
print("-" * 40)

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2%}")
print()

# Classification Report
# For multiclass problems, this shows precision/recall/f1 for EACH class
# plus macro avg (simple average) and weighted avg (accounts for class size)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix 
# Now it's 3x3! Diagonal = correct, off-diagonal = mistakes
# You can see exactly which spending tiers get confused with each other
cm = confusion_matrix(
    y_test, y_pred,
    labels=['Low', 'Medium', 'High']  # Force this order in the matrix
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,                                   # Show numbers
    fmt='d',                                      # Integer format
    cmap='Greens',                                # Green color scheme
    xticklabels=['Low', 'Medium', 'High'],        # Predicted labels
    yticklabels=['Low', 'Medium', 'High']         # Actual labels
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.title('Confusion Matrix — Customer Spending Tiers', fontsize=14)
plt.tight_layout()
plt.show()

print("   -> Confusion matrix displayed")
print("   -> TIP: Medium-tier orders are often the hardest to classify")
print("      because they overlap with both Low and High groups.")
print()




print("STEP 6: Finding the Best K with Cross-Validation")
print("-" * 40)

# Test K from 1 to 30
k_range = range(1, 31)

# For each K, we'll store:
# The mean CV accuracy (average across 5 folds)
# The std of CV accuracy (how much it varies between folds)
cv_means = []    # Average accuracy
cv_stds  = []    # Standard deviation (spread)

for k in k_range:
    # Create a model with this K
    model = KNeighborsClassifier(n_neighbors=k)

    # Run 5-fold cross-validation on the TRAINING data only
    # cross_val_score returns an array of 5 accuracy scores
    scores = cross_val_score(
        model,                   # The model to evaluate
        X_train_scaled,          # The training features (scaled)
        y_train,                 # The training labels
        cv=5,                    # 5 folds
        scoring='accuracy'       # Metric to use
    )

    # Store the mean and standard deviation
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

    # Print progress every 5 steps
    if k % 5 == 0 or k == 1:
        print(f"   K={k:2d} → CV Accuracy = {scores.mean():.2%} (±{scores.std():.2%})")

print()

# Plot the results
# We plot the mean accuracy as a line and add a shaded region
# showing ± 1 standard deviation (the uncertainty band)
plt.figure(figsize=(9, 5))

# Convert to numpy arrays for easy math
cv_means = np.array(cv_means)
cv_stds  = np.array(cv_stds)
k_list   = list(k_range)

# Main line (mean accuracy)
plt.plot(k_list, cv_means, marker='o', color='#27ae60',
         linewidth=2, markersize=4, label='Mean CV Accuracy')

# Shaded region (±1 standard deviation)
# This shows the uncertainty in our accuracy estimate
plt.fill_between(
    k_list,
    cv_means - cv_stds,     # Lower bound
    cv_means + cv_stds,     # Upper bound
    alpha=0.2,              # Transparency
    color='#27ae60',        # Same green
    label='±1 Std Dev'
)

plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('Finding the Best K with 5-Fold Cross-Validation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find the best K
best_k           = k_list[np.argmax(cv_means)]
best_cv_accuracy = cv_means.max()

print(f"   -> Best K = {best_k}")
print(f"   -> Best CV Accuracy = {best_cv_accuracy:.2%}")
print(f"   -> Chart displayed")
print()




print("STEP 7: Final Evaluation with Best K")
print("-" * 40)

# Retrain with the best K
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)

# Final predictions on the test set
y_pred_final = final_model.predict(X_test_scaled)

# Final accuracy
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final Test Accuracy (K={best_k}): {final_accuracy:.2%}")
print()

# Final classification report
print("Final Classification Report:")
print(classification_report(y_test, y_pred_final))




print("STEP 8: Predicting New Orders")
print("-" * 40)

# Three new orders with different characteristics
new_orders = pd.DataFrame({
    'Category_Enc': [0,   2,   1],             # Accessories, Office, Electronics
    'Region_Enc':   [0,   1,   3],             # East, North, West
    'Quantity':     [2,   5,   9],             # Small, Medium, Large order
    'Profit':       [75.0, 360.0, 1400.0],     # Low, Medium, High profit
})

print("New order measurements:")
print(new_orders.to_string(index=False))
print()


new_scaled = scaler.transform(new_orders)

# Get predictions and probabilities
predictions   = final_model.predict(new_scaled)
probabilities = final_model.predict_proba(new_scaled)

# Display results for each order
class_labels = final_model.classes_  # ['High', 'Low', 'Medium'] 

for i in range(len(new_orders)):
    print(f"Order #{i+1}:")
    print(f"   Predicted spending tier: {predictions[i]}")
    print(f"   Confidence breakdown:")

    for cls, prob in zip(class_labels, probabilities[i]):
        # Visual bar to make probabilities easy to read
        bar_length = int(prob * 25)
        bar = '█' * bar_length + '░' * (25 - bar_length)
        print(f"      {cls:8s} {bar} {prob:6.1%}")
    print()

print()

# The algorithm K-nearest neighbors (KNN) was used to predict the spending tier of new orders based on the features: Category_Enc, Region_Enc, Quantity, and Profit. 
# The purpose of this is to predict the spending tier of new orders, which can help the sales team prioritize high-value customers and tailor marketing strategies accordingly.
# another purpose is to identify potential high-value customers early in the sales process, allowing the company to allocate resources more effectively and improve customer satisfaction.

