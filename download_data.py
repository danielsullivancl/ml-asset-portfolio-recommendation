import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
SCREEN_SIZE = 250


def main():
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        EquityQuery("gt", ["avgdailyvol3m", 1_000_000]),
        EquityQuery("gt", ["intradayprice", 5]),
    ])

    response = yf.screen(
        query,
        size=SCREEN_SIZE,
        sortField="avgdailyvol3m",
        sortAsc=False
    )

    tickers = [item["symbol"] for item in response["quotes"]]

    print(f"Ativos encontrados: {len(tickers)}")
    print(tickers[:20])

    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        threads=True
    )

    prices = data["Close"].copy()
    volume = data["Volume"].copy()

    missing_ratio = prices.isna().mean()
    valid_tickers = missing_ratio[missing_ratio <= 0.05].index.tolist()

    prices_final = prices[valid_tickers].dropna().copy()
    volume_final = volume[valid_tickers].loc[prices_final.index].copy()

    returns = prices_final.pct_change().copy()

    prices_final.index.name = "date"
    volume_final.index.name = "date"
    returns.index.name = "date"

    prices_long = prices_final.reset_index().melt(
        id_vars="date",
        var_name="ticker",
        value_name="price"
    )

    volume_long = volume_final.reset_index().melt(
        id_vars="date",
        var_name="ticker",
        value_name="volume"
    )

    returns_long = returns.reset_index().melt(
        id_vars="date",
        var_name="ticker",
        value_name="return"
    )

    dataset = prices_long.merge(
        volume_long,
        on=["date", "ticker"]
    ).merge(
        returns_long,
        on=["date", "ticker"]
    )

    dataset = dataset.dropna()
    dataset = dataset[["date", "ticker", "volume", "price", "return"]]

    dataset.to_csv(OUTPUT_DIR / "us_liquid_assets_dataset.csv", index=False)

    pd.Series(valid_tickers, name="ticker").to_csv(
        OUTPUT_DIR / "us_liquid_assets_tickers.csv",
        index=False
    )

    print(f"Ativos finais: {len(valid_tickers)}")
    print(f"Dataset final: {dataset.shape}")
    print("Arquivos salvos na pasta data/")


if __name__ == "__main__":
    main()