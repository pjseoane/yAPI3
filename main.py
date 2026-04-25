from yfinance_api3 import QuantAnalytics, StockClient


def main() -> None:
    client = StockClient()
    qa = QuantAnalytics(client)
    print("yfinance_api3 is installed and importable.")
    print(f"Client type: {type(client).__name__}")
    print(f"Analytics type: {type(qa).__name__}")


if __name__ == "__main__":
    main()
