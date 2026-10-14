from pathlib import Path
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "UCI_Credit_Card.csv"

def load_data(path: Path | str = DEFAULT_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=['ID'])
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(f"Default rate: {df['default.payment.next.month'].mean():.3f}")
    print(df.head())