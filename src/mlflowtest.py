import mlflow
import pathlib
import pandas as pd

RDIR = pathlib.Path('C:/Users/Ayush/cid')

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

registered_model = {
    'name': 'tracking-quickstart',
    'version': 'latest'
}

model_uri = 'models:/{}/{}'.format(registered_model['name'], registered_model['version'])
test = pd.read_csv(RDIR.joinpath('models/data/test.csv'))


# Reference code for testing

# Load the model back for predictions as a generic Python Function model

# loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
# loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model = mlflow.sklearn.load_model(model_uri)

predictions = loaded_model.predict(test.drop(columns=['species']))

# iris_feature_names = sns.datasets.load_iris().feature_names

# result = pd.DataFrame(test.drop(columns=['species']), columns=iris_feature_names)
result = test.drop(columns = ['species'])
result["actual_class"] = test['species']
result["predicted_class"] = predictions

print(result[:4])