
# Prepare Data for Challenge Similarity Analysis

This README explains the process and reasoning behind the `prepare_data.ipynb` notebook, which is designed to analyze the similarity between different challenges using trained CNN models.

## Objective

The main objective of this notebook is to train a Convolutional Neural Network (CNN) model for each challenge, extract the weights from the penultimate layer of these models, and investigate whether similar challenges have similar weights in their penultimate layers. This approach is based on the hypothesis that challenges with similar characteristics or difficulty levels might produce similar internal representations in the trained models.

## Process

1. **Data Preparation**: 
   - Load and preprocess the data for each challenge.
   - Ensure that the data is in a suitable format for training CNN models.

2. **Model Training**:
   - For each challenge, train a CNN model using the prepared data.
   - The model architecture likely includes convolutional layers, pooling layers, and fully connected layers.
   - The penultimate layer (typically the layer before the final classification layer) is of particular interest.

3. **Weight Extraction**:
   - After training, extract the weights from the penultimate layer of each model.
   - These weights represent the high-level features learned by the model for each challenge.

4. **Data Collection**:
   - Compile a dataset containing:
     - Challenge identifiers
     - Extracted weights from the penultimate layer
     - Model accuracy for each challenge

5. **Analysis**:
   - Examine the distribution of model accuracies across challenges.
   - Compare the weights of the penultimate layers between different challenges.
   - Look for patterns or clusters in the weight space that might indicate similarity between challenges.

## Key Findings

Based on the code snippets provided:

- The models achieve high accuracy overall, with a mean accuracy of approximately 0.9916 (99.16%).
- There is some variation in model performance, with the minimum accuracy being 0 and the maximum being 1.
- The weights from the penultimate layer (fc2) have a shape of (1024,), indicating 1024 features are being used for the final classification.

## Implications

This analysis can provide valuable insights into:
- The relative difficulty of different challenges
- Grouping of challenges based on their inherent characteristics
- Potential transfer learning opportunities between similar challenges
- Identification of outlier challenges that might need special attention

## Future Work

- Perform clustering analysis on the extracted weights to identify groups of similar challenges.
- Visualize the weight space using dimensionality reduction techniques like PCA or t-SNE.
- Investigate the relationship between challenge similarity and other metadata (e.g., topic, difficulty rating).
- Explore how this information can be used to improve challenge recommendation systems or curriculum design.

