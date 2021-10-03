from model.model import Model


def main():
    model = Model("dataset.csv")
    model.train()
    model.save_model("model.pkl")


if __name__ == "__main__":
    main()