import joblib
def exportRegressor(rf, path):
    joblib.dump(rf, path)
    
def importRegressor(rf, path):
    return joblib.dump(rf, path)