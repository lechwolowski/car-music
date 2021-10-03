import train
import inference


def main():
    train.main()
    while True:
        input_data = inference.input_gathering()
        inference.inference(input_data)
        if input("type \"exit\" to exit: ") == "exit":
            break
    

if __name__ == "__main__":
    main()