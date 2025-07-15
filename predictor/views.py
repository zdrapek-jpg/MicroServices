import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


from Data.One_hot_Encoder import OneHotEncoder
import numpy as np
import pandas as pd

from Data.Transformers import Transformations
from Data.load_user_data import modify_user_input_for_network

normalization_instance =None
one_hot_instance  = None
model_instance  = None

def load():
    global normalization_instance, one_hot_instance, model_instance
    normalization_instance = Transformations.load_data()  # instancja klasy normalizującej - ładujemy ją
    one_hot_instance = OneHotEncoder.load_data()  # plik json jest przypisane do klasy - tworzymy instancje klasy
    model_instance = NNetwork.create_instance()  # Create the model instance
    print(normalization_instance.std_type)
    print("Data loaded successfully!")
from NeuralNetwork.Network_single_class1 import NNetwork  # your model import here

# Make sure you have your load(), model_instance, one_hot_instance, normalization_instance, modify_user_input_for_network defined somewhere globally

@csrf_exempt
def predict_view(request):
    try:
        load()

        if request.method != "POST":
            return JsonResponse({"error": "Only POST requests allowed"}, status=405)
        if not model_instance or not normalization_instance or not one_hot_instance:
            return JsonResponse({"error": "Model or preprocessors not loaded"}, status=500)

        print(f"DEBUG: Raw request.body received: {request.body}")
        print(f"DEBUG: Type of request.body: {type(request.body)}")
        print("DEBUG: request.META content:")
        for key, value in request.META.items():
            print(f"  {key}: {value}")
        print("--- END DEBUG PRINTS (before json.loads) ---")

        json_data = json.loads(request.body)
        print(f"DEBUG: json_data after json.loads: {json_data}")
        print(f"DEBUG: Type of json_data: {type(json_data)}")

        if not json_data or "user_data" not in json_data:
            return JsonResponse({"error": "Missing 'user_data' field in JSON"}, status=400)

        user_data = json_data["user_data"]
        print(f"DEBUG: user_data extracted: {user_data}")
        print(f"DEBUG: Type of user_data: {type(user_data)}")
        for i, item in enumerate(user_data):
            print(f"DEBUG: user_data[{i}]: {item} (Type: {type(item)})")

        if len(user_data) != 14:
            return JsonResponse({"error": f"Expected 14 features, got {len(user_data)}"}, status=400)

        # --- CRITICAL DEBUGGING POINT ---
        print("DEBUG: Calling modify_user_input_for_network...")
        ready_for_model = modify_user_input_for_network(user_data, one_hot_instance, normalization_instance)
        print(f"DEBUG: modify_user_input_for_network returned: {ready_for_model}")
        print(f"DEBUG: Type of ready_for_model: {type(ready_for_model)}")

        print("DEBUG: Calling model_instance.pred...")
        prediction = round(model_instance.pred(ready_for_model[0])[0], 3)
        print(f"DEBUG: prediction calculated: {prediction}")
        # --- END CRITICAL DEBUGGING POINT ---


        data = pd.read_csv(r"TrainData/training_data.csv", delimiter=";")
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        acc, error = model_instance.perceptron(x, y)

<<<<<<< HEAD
        preds = [model_instance.pred(row)[0] for row in x]

        predictions = []
        for x_point in x:
            predictions.append(model_instance.pred(x_point)[0])

        y_origin= y.tolist()
        print(y.shape )
        #matrix = model_instance.confusion_matrix(preds, y.reshape(-1).tolist()).values.tolist()
        matrix = NNetwork.confusion_matrix_ground_truth(y_pred=predictions,y_origin =y_origin).values.tolist()
        print(matrix)
        stay_acc = matrix[0][0]
        leave_acc= matrix[1][1]

=======
        preds = [round(model_instance.pred(row)[0]) for row in x]
        matrix = model_instance.confusion_matrix(preds, y.reshape(-1).tolist()).values.tolist()
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        leave_acc = tp / (tp + fn) if (tp + fn) > 0 else 1
        stay_acc = tn / (tn + fp) if (tn + fp) > 0 else 1
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8

        chance = (
            "low" if prediction <= 0.20 else
            "average" if prediction <= 0.60 else
            "high" if prediction <= 0.95 else
            "very high"
        )

        return JsonResponse({
            "prediction": prediction,
            "accuracy": round(acc,4)*100,
            "error_on_data":  round(error,3),
            "leave_bank_acc": leave_acc,
            "stay_in_bank_acc": stay_acc,
            "chance": chance
        })

    except Exception as e:
        print(f"ERROR in predict_view: {e}")
        return JsonResponse({"error": str(e)}, status=500)
