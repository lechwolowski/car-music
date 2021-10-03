from model.model import Model


def main():
    model = Model("dataset.csv", model_path="model.pkl")
    result = model.predict(
        [
            [
                1001,
                4,
                0,  # relaxed driving
                0,  # sport driving
                1,  # coast line
                0,  # country side
                0,  # mountains
                0,  # urban
                0,  # active
                0,  # happy
                0,  # lazy
                0,  # sad
                0,  # afternoon
                0,  # day
                0,  # morning
                0,  # night
                0,  # city
                0,  # highway
                0,  # serpentine
                0,  # awake
                0,  # sleepy
                0,  # free road
                0,  # lots of cars
                0,  # traffic jam
                0,  # cloudy
                0,  # rainy
                0,  # snowing
                0,  # sunny
            ]
        ]
    )

    print(result)

if __name__ == "__main__":
    main()
