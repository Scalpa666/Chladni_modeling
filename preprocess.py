import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df):

    # Randomly split the data set with fixed random seed
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the split data set
    train_df.to_csv("./Data/train_data.csv", index=False)
    test_df.to_csv("./Data/test_data.csv", index=False)

    print("The data set has been successfully split and saved！")

if __name__ == "__main__":
    # Read the dataset
    data_path = "./Data/full_data.csv"
    df = pd.read_csv(data_path)

    split_dataset(df)