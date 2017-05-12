import pandas as pd

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def write_file(data, file_name):
    new_data = pd.DataFrame(data=data).to_csv(file_name)

if __name__ == "__main__":
    data = open_file("../resources/train.csv")
    #write_file(data, "../resources/output.csv")
