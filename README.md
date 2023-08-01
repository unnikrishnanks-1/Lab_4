# Fish Weight Prediction Using Random Forest Regression and Polynomial Regression

## Problem Statement:
The goal of this project is to build machine learning models that predict the weight of fish based on various physical measurements. The Fish Market dataset is used for this purpose, which contains information on common fish species found in fish markets. Two regression models are implemented and compared: Random Forest Regression and Polynomial Regression.

## Data Exploration and Preprocessing:
- The Fish Market dataset is loaded, and the column names are modified for clarity.
- The dataset contains information on the weight and physical measurements (lengths, height, width) of fish, as well as the species they belong to.
- Data visualization techniques, such as countplots, barplots, pairplots, and distplots, are used to explore and understand the relationships between features and the target variable.
- Outliers are detected and removed to ensure the quality of the dataset for modeling.
- Dummy variables are created for the 'Species' column to convert categorical data into numerical form, which is required for model training.

## Model Building and Evaluation - Random Forest Regression:
- The dataset is split into training and test sets for model evaluation.
- A Random Forest Regressor is chosen as the machine learning model due to its ensemble nature and capability to handle regression tasks.
- Hyperparameter tuning is performed using RandomizedSearchCV to find the optimal hyperparameters for the Random Forest Regressor.
- The model is trained on the training set and evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) as the evaluation metrics.
- The final model is used to predict the weight of fish on the test set, and the RMSE score is reported to assess the model's performance.

## Model Building and Evaluation - Polynomial Regression:
- Polynomial features of different degrees (1 to 4) are created for the independent variables.
- The dataset is split into training and test sets for model evaluation.
- For each polynomial degree, a Linear Regression model is trained on the training set and evaluated on the test set using Root Mean Squared Error (RMSE).
- The RMSE values for different polynomial degrees are plotted to visualize the trade-off between model complexity and prediction accuracy.
- The final model with the optimal polynomial degree is chosen, and its RMSE score on the test set is reported.

## Usage:
- To use the Random Forest Regression model, users can access the provided Flask API, which accepts inputs for fish measurements (Vertical Length, Diagonal Length, Cross Length, Height, Diagonal Width) and the fish species.
- The API returns the predicted weight of the fish based on the given inputs.

## File Descriptions:
1. `Fish.csv`: The dataset containing fish weight and measurements.
2. `fish_weight_prediction.py`: The Python code for data exploration, preprocessing, model building, and evaluation using both Random Forest Regression and Polynomial Regression.
3. `new_model.pkl`: The saved model file obtained after training the Random Forest Regressor with the optimal hyperparameters.
4. `app.py`: The Flask API code for hosting the Random Forest Regression model and handling user input for predictions.
5. `templates/index.html`: The HTML file for the frontend webpage allowing users to input fish measurements and species.

## Getting Started:
1. Clone this repository to your local machine.
2. Install the required Python libraries specified in `requirements.txt`.
3. Run the Flask API using the command `python app.py` in the terminal.
4. Access the frontend webpage by visiting `http://localhost:5000/` in your web browser.
5. Input the fish measurements and species in the form and click "Predict the Weight" to get the predicted fish weight.

## Important Note:
This project is for educational purposes and demonstrates how to build and deploy machine learning models. The accuracy of the predictions may vary depending on the data and model tuning. Users are encouraged to explore both regression techniques and choose the best model for their specific application.
