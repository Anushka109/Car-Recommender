import numpy as np

# Print original and contextual vectors for one car
print("Original 42D vector for Honda N-Van:", car_data["Honda N-Van"])
print("Contextual 10D vector for Honda N-Van:", car_contextual_data["Honda N-Van"])

# Check variance explained by 10 components
print("Variance explained by 10 components:", pca.explained_variance_ratio_)
print("Total variance captured:", sum(pca.explained_variance_ratio_))

# Inspect top features in first principal component
feature_names = ["male", "age 20-40", "age 40-60", "age 60+", "Regular Office", "Hybrid", "Freelance", 
                 "Purpose: Commuting", "Purpose: Child Pickup", "Purpose: Travel", "Purpose: Shopping", 
                 "Purpose: Caregiving", "Purpose: Outdoor", "Purpose: Work Use", "Seating: 1", "Seating: 2", 
                 "Seating: 3", "Seating: 4", "Seating: 5", "Seating: 6+", "Budget: <1M", "Budget: 1-2M", 
                 "Budget: 2-3M", "Budget: 3-4M", "Budget: 4-5M", "Budget: 5M+", "Priority: Price", 
                 "Priority: Fuel Efficiency", "Priority: Design", "Priority: Safety", "Priority: Storage", 
                 "Priority: Interior Space", "Priority: Ease of Driving", "Priority: Eco-Friendly", 
                 "Priority: Ride Comfort", "Priority: Engine Power", "Priority: Advanced Tech", 
                 "Priority: Ease of Entry", "Hobby: Outdoor", "Hobby: Travel", "Hobby: Shopping", "Hobby: Pets"]
top_features = np.argsort(np.abs(pca.components_[0]))[-5:][::-1]
print("\nTop 5 features in first principal component:")
for idx in top_features:
    print(f"  {feature_names[idx]}: {pca.components_[0][idx]:.3f}")