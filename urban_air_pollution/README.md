# competitive-programming

#### Urban-air-pollution

Predicting air quality in cities around the world using satellite data.
The Challenge can be found on [Zindi](https://zindi.africa/hackathons/urban-air-pollution-challenge/) for more info.

Solution summary:

* Data cleaning:
> * Drop the target related features as they can't be used for testing.
> * Slice out far target-variable outliers in the train set to enhance model generalizability.
* A bit of feature engineering on the date feature.
* feature selection (dropped two features that hurt performance alot)

* Cross-validation strategy:
> * 8-fold KFold train data split and cross-validation without shuffling.

* Model: LightGBM
* evaluation metric: RMSE.

**Score:** 29.7419 **Rank:** 46/126
