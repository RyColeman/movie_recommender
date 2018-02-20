# movie_recommender
A movie recommender app


This project was inpired by a case study that was conducted in my time at Galvanize's Data Science Immersive program. The motivation here was to build a movie recommender model that would yeild the highest profit. 

## Background:
- A fictional company called Movies-Legit, uses a movie recommender to their users and although this recommender is not optimized, it has yeilded the company a significant revenue stream so their management is hesitent to change it but they want to explore any possible optimized solutions.

- Currently Movies-Legit recommender uses what's called a 'mean-of-means'. How 'mean-of-means' works is that future un-seen movie ratings for a user are predicted by getting the mean of:

a) the rating for a particular movie accross users
b) the mean of all the particular user's ratings
c) the average overall rating of all ratings for everymovie by every user

Hence the "mean" of Means name.

- Ultimate Goal: Tie back revenue to the performance of the movie recommender and find a recommender that yeilds the highest profit. Hopefully this optimized recommender yeilds a higher profit than Movies-Legit's current mean-of-means recommender.

## Step 1: Gather prediction accuracy scores of other models vs. the current Mean-of-Means model.
### Error Metrics explained:
- These recommenders are predicting the ratings a user would give a movie that he/she has not yet seen. This is a regression problem, therefore we'll use score metrics of RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). MAE is literally the average absolute error between the model's predicted movie rating and the true rating that a user gave a movie. RMSE is similar but it gives extra error penalties when the difference between the predicted movie rating and the true rating value is large since it squares this difference. RMSE is more conservative so we'll pay special attention to this metric.

### Scaling ratings data:
- Before we test our models, we'll want to scale all the rating data. What this means is you're re-scaling the ratings a user gives from the 0.5-5 range to a 0-1 range. The reason for this is that many users rate movies differently. Some users will rate movies always on the lower end, perhaps from 0.5 - 3, while others will always rate movies from 3.5-5. In otherwords, one user's .5 rating is equivalent to another user's 3.5 rating (as an example). To make sure all user's ratings are on the same scale, we do what's called data scaling where we'll spread a given users' previous ratings on a 0-1 scale so we can evenly compare users ratings.

### Results:
![CV score of all models, RMSE & MAE](images/cross_val_box_plot_scaled_data_v2.png)
