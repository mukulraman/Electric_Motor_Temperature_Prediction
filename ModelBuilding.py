from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def train_test_split_and_features(df):
    x = df.drop(['pm'], axis=1)
    y = df['pm']
    print("Before split:")
    print("Shape of x:", x.shape)  # Should be (n_samples, 7)
    print("Shape of y:", y.shape)  # Should be (n_samples,)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("\nAfter split:")
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Type of x_train:", type(x_train))
    print("Type of x_test:", type(x_test))
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test, features

def evaluate(x_train, y_train, x_test, y_test, max_depth=10, min_samples_split=2, max_features=0.8, max_samples=0.8):
    random_model = RandomForestRegressor(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split,
                                         max_features=max_features, max_samples=max_samples)
    random_model = random_model.fit(x_train, y_train)
    y_test_pred = random_model.predict(x_test)
    r2score = r2_score(y_test, y_test_pred)
    mean_abs = mean_absolute_error(y_test, y_test_pred)
    mean_sq = mean_squared_error(y_test, y_test_pred)

    print("R² Score:", r2score)
    print("Mean Absolute Error:", mean_abs)
    print("Mean Squared Error:", mean_sq)
    return random_model, r2score, mean_abs, mean_sq