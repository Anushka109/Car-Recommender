# Car Recommender

Car Recommender is an AI-driven personalized vehicle recommendation system that helps users identify the most suitable car based on their lifestyle, priorities, budget, and usage requirements. The system combines structured feature engineering, dimensionality reduction, and weighted similarity scoring to generate explainable recommendations in a user-friendly web application.

## Project Overview

Choosing a car is a multidimensional decision problem involving several factors, including:
- budget
- family size and seating needs
- daily commuting and travel patterns
- vehicle purpose
- fuel efficiency and safety requirements
- personal preferences and lifestyle

This project models both user preferences and vehicle characteristics as structured feature vectors and uses a recommendation pipeline to rank the most relevant car options. The recommendation engine is designed to deliver personalized suggestions while maintaining interpretability for the end user.

## Problem Statement

Consumers often struggle to compare multiple vehicle options because their decision criteria are heterogeneous and often conflicting. A buyer may prioritize:
- affordability
- safety
- comfort
- fuel efficiency
- interior space
- styling
- practicality for family or business usage

This project addresses that challenge by building a recommendation system that aligns user requirements with vehicle features in a mathematically consistent representation.

## Core AI / ML Concepts Used

### 1. Feature Engineering
User inputs such as age group, work type, seating preference, purpose of use, hobbies, and budget are encoded into fixed-length binary feature vectors. Similarly, each vehicle is represented using structured binary attributes aligned to the same feature space.

### 2. Feature Normalization
Categorical values are transformed into consistent numerical representations so that different user inputs can be compared in the same vector space.

### 3. Dimensionality Reduction with PCA
Principal Component Analysis (PCA) is applied to reduce the high-dimensional feature space while preserving the most informative variance. This creates a compact representation that enables more efficient similarity matching.

### 4. Weighted Similarity Matching
The recommendation engine computes similarity scores using weighted cosine similarity. Feature weights are derived from PCA explained variance, allowing more important dimensions to influence the recommendation more strongly.

### 5. Explainable Recommendation Layer
The system produces recommendation reasons such as:
- best match for the user's budget
- strong fit for family usage
- aligned with primary purchase priority
- suitable for commuting or travel needs

This makes the output more interpretable and user-trustworthy.

## System Architecture

The system follows a lightweight content-based recommendation architecture:

1. User answers a questionnaire
2. Input is transformed into binary feature vectors
3. Features are normalized into a consistent representation
4. PCA reduces dimensionality
5. Similarity is computed against vehicle profiles
6. Top-ranked recommendations are returned
7. The system explains the rationale behind each suggestion

## Key Features

- Personalized recommendation engine
- Binary vector representation for user and vehicle attributes
- PCA-based feature reduction
- Weighted cosine similarity scoring
- Preference-aware ranking
- Explainable AI for recommendation output
- Interactive Flask web application
- User-centered recommendation workflow

## Technologies Used

- Python
- Flask
- NumPy
- Pandas
- scikit-learn
- SQLAlchemy
- HTML / CSS / JavaScript

## Project Goal

The goal of this project is to build a practical AI-powered recommendation system that helps users make better purchase decisions using personalized, data-driven ranking instead of manual comparison alone.

## Use Cases

This project is applicable to:
- personalized product recommendation systems
- consumer decision support
- recommendation-based product selection
- explainable AI in user-facing applications

## Future Enhancements

- Add collaborative filtering to improve diversification and personalization
- Integrate real-world vehicle pricing and availability data
- Add model evaluation metrics for recommendation quality
- Improve interpretability with richer explanation generation
- Extend the system to support hybrid recommendation approaches

## Repository Notes

This project is developed as a portfolio and learning-focused AI application to demonstrate:
- feature engineering
- recommendation system design
- dimensionality reduction
- similarity-based ranking
- explainable recommendation logic


## Contact

Open to feedback, collaboration, and discussions around recommendation systems, AI product development, and applied machine learning.
