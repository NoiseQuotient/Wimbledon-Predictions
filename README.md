# Wimbledon 2025 Tennis Match Outcome Prediction

## Overview

This project aims to predict the outcomes of tennis matches at Wimbledon 2025 using historical match data and machine learning techniques. The core idea is to leverage player performance metrics, including a specialized Elo rating system that accounts for surface type (grass, clay, hard court), to estimate the probability of each player winning a match.

## What We Do

1. **Data Loading and Preparation**  
   We load historical ATP match data from the Sackmann dataset and recent 2025 match data that we have scraped from the ATP official website to get the most recent data. To ensure fairness and completeness, we "mirror" each match so that each player appears as Player 1 and Player 2 in separate rows.

2. **Round Normalization**  
   Tennis tournaments use various naming conventions for rounds (e.g., "QF", "Quarterfinals", "Round of 16"). We standardize these to a consistent format to make analysis easier.

3. **Elo Rating Calculation**  
   We calculate two Elo ratings for each player:  
   - A general Elo rating based on all matches.  
   - A surface-specific Elo rating that adjusts for performance on grass, clay, or hard courts.  
   These ratings are updated match-by-match, considering recency and tournament importance.

4. **Feature Engineering**  
   We create features such as rank differences, rolling averages of serve statistics, and Elo momentum (recent changes in Elo rating) to feed into our machine learning model.

5. **Model Training and Evaluation**  
   Using XGBoost, a powerful gradient boosting algorithm, we train models to predict match winners. We split the data so that all matches before Wimbledon 2025 are used for training, and Wimbledon 2025 matches are used for testing. This simulates a real-world scenario where we predict future matches based on past data.

6. **Analysis and Visualization**  
   We analyze feature importance, correlations, and model accuracy overall and by tournament round. This helps us understand which factors are most predictive and how well the model performs at different stages of the tournament.

## Why This Matters

Predicting tennis match outcomes is challenging due to the many factors involved: player form, surface, fatigue, and more. By combining domain knowledge (Elo ratings) with machine learning, we aim to build a model that can provide meaningful predictions, potentially useful for fans, analysts, or even betting strategies.

## How to Use This Code

- Place your historical ATP match CSV files in the specified folder.  
- Provide the 2025 match data Excel file in the project directory.  
- Run the main script to train the model and see detailed prediction results for Wimbledon 2025.  
- Review the printed accuracy reports and feature importance plots to understand model performance.

## Understanding the Results

- **Accuracy**: The percentage of matches where the model correctly predicted the winner.  
- **Classification Report**: Includes precision, recall, and F1-score, giving a deeper look at prediction quality.  
- **Feature Importance**: Shows which features (e.g., Elo difference, serve stats) the model relied on most.  
- **Accuracy by Round**: Helps identify if the model performs better in early rounds or later stages like quarterfinals and finals.

Results Summary
We trained and tested our tennis match outcome prediction models using historical ATP data combined with 2025 match data, focusing especially on Wimbledon 2025.

## Data Overview

Loaded 40 years of historical ATP matches (259,118 rows including mirrored matches).
Added 6,420 rows of 2025 matches from Excel data.

Combined dataset contains 265,538 rows.

## Model Training and Performance

General XGBoost Model trained on all data before Wimbledon 2025, tested on Wimbledon 2025 matches.
Training size: 263,356 matches; Test size: 478 matches. We find the best hyperparameters via randomized search.

Top features: Rank difference (log scale), Elo momentum, Elo difference.
Wimbledon-only subset model (trained and tested only on Wimbledon data):

Test accuracy on Wimbledon 2025: 72.0%
This model outperformed the general model, showing the value of surface-specific ratings.
Balanced precision and recall around 72%.

## Detailed Wimbledon 2025 Analysis

Accuracy by round shows the model performs better in later rounds (e.g., quarterfinals and semifinals near or at 100% accuracy).
Early rounds have lower but still respectable accuracy (~60-80%).
The surface-specific Elo model consistently outperforms the general model across most rounds.

![Model Accuracy by Round](results/Model Accuracy by Round.png)

Player-Level Insights
Some players had perfect prediction accuracy in Wimbledon 2025 matches (e.g., Facundo Diaz Acosta, Thiago Seyboth Wild).
The model’s confidence (predicted win probabilities) aligns well with actual outcomes in many cases.

Confusion Matrix Highlights
The surface Elo model correctly predicted 168 wins and 176 losses, with some misclassifications (71 false positives, 63 false negatives).
Predictions are roughly balanced between predicting Player 1 and Player 2 wins.

What This Means ist that the model’s ~73% accuracy on Wimbledon 2025 is a strong result given the inherent unpredictability of tennis.
Surface-specific Elo ratings add meaningful predictive power.

The model can be a useful tool for analysts or enthusiasts looking to understand match outcomes better.
There is room for improvement by adding more player-specific features, refining hyperparameters, or incorporating bookmaker odds.
