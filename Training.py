import pandas as pd
import joblib

from DataCleaning import data_cleaning

from ModelBuilding import(
    train_test_split_and_features,
    evaluate)

# Read the Heart Disease Training Data from output_data.csv file in data folder
# If output_data.csv file in data folder doesn't exist, then first run the DB_to_CSV.py script
df = pd.read_csv('data/output_data.csv')
df=data_cleaning(df)

x_train, x_test, y_train, y_test,features=train_test_split_and_features(df)

random_model, r2score, mean_abs,mean_sq=evaluate(x_train,y_train,x_test,y_test,max_depth=10,min_samples_split=2,max_features=0.8,max_samples=0.8)

joblib.dump(random_model, "models\Model_Classifier_Electric.pkl")
joblib.dump(features, "models\Features_Columns.pkl")