import pandas as pd
from faker import Faker
from numpy.random import normal


def gen_dataset(length: int):
    faker_instance = Faker()
    fake_dict = {
        "ca": faker_instance.random_elements(elements=list(range(4)), length=length),
        "cp": faker_instance.random_elements(elements=list(range(4)), length=length),
        "exang": faker_instance.random_elements(elements=list(range(2)), length=length),
        "fbs": faker_instance.random_elements(elements=list(range(2)), length=length),
        "sex": faker_instance.random_elements(elements=list(range(2)), length=length),
        "restecg": faker_instance.random_elements(
            elements=list(range(3)), length=length
        ),
        "slope": faker_instance.random_elements(elements=list(range(3)), length=length),
        "thal": faker_instance.random_elements(elements=list(range(3)), length=length),
        "age": [normal(54.5, 9.0) for _ in range(length)],
        "chol": [normal(247, 52) for _ in range(length)],
        "oldpeak": [normal(1.06, 1.17) for _ in range(length)],
        "thalach": [normal(149.6, 22.94) for _ in range(length)],
        "trestbps": [normal(131.69, 17.76) for _ in range(length)],
        "condition": faker_instance.random_elements(
            elements=list(range(2)), length=length
        ),
    }

    fake_df = pd.DataFrame(fake_dict)
    return fake_df
