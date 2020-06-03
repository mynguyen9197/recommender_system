from surprise import accuracy
from sklearn.metrics import mean_squared_error

def evaluate_surprise_alg(predictions):

    # return cross_validate(alg, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    accuracy.rmse(predictions)
    

def evaluate_content_based(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    rmse = math.sqrt(mse)
    return rmse