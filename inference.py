from model.model import Model


def input_gathering():
    return [
        [
            int(input("Int value for User ID. Empty for default: ") or 1001),
            int(input("Int value for Rating. Empty for default: ") or 5),
            int(input("Boolean value for relaxed driving. Empty for default: ") or "0"),
            int(input("Boolean value for sport driving. Empty for default: ") or "0"),
            int(input("Boolean value for coast line. Empty for default: ") or 1),
            int(input("Boolean value for country side. Empty for default: ") or "0"),
            int(input("Boolean value for mountains. Empty for default: ") or "0"),
            int(input("Boolean value for urban. Empty for default: ") or "0"),
            int(input("Boolean value for active mood. Empty for default: ") or "0"),
            int(input("Boolean value for happy mood. Empty for default: ") or "0"),
            int(input("Boolean value for lazy mood. Empty for default: ") or "0"),
            int(input("Boolean value for sad mood. Empty for default: ") or "0"),
            int(input("Boolean value for afternoon. Empty for default: ") or "0"),
            int(input("Boolean value for day. Empty for default: ") or "0"),
            int(input("Boolean value for morning. Empty for default: ") or "0"),
            int(input("Boolean value for night. Empty for default: ") or "0"),
            int(input("Boolean value for city. Empty for default: ") or "0"),
            int(input("Boolean value for highway. Empty for default: ") or "0"),
            int(input("Boolean value for serpentine. Empty for default: ") or "0"),
            int(input("Boolean value for awake. Empty for default: ") or "0"),
            int(input("Boolean value for sleepy. Empty for default: ") or "0"),
            int(input("Boolean value for free road. Empty for default: ") or "0"),
            int(input("Boolean value for lots of cars. Empty for default: ") or "0"),
            int(input("Boolean value for traffic jam. Empty for default: ") or "0"),
            int(input("Boolean value for cloudy. Empty for default: ") or "0"),
            int(input("Boolean value for rainy. Empty for default: ") or "0"),
            int(input("Boolean value for snowing. Empty for default: ") or "0"),
            int(input("Boolean value for sunny. Empty for default: ") or "0"),
        ]
    ]


def inference(input_data):
    model = Model("dataset.csv", model_path="model.pkl")
    result = model.predict(input_data)

    print(result)


if __name__ == "__main__":
    while True:
        input_data = input_gathering()
        inference(input_data)
        if input("type \"exit\" to exit: ") == "exit":
            break