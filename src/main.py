if __name__ == "__main__":
    import yaml
    from src.data.fetch_options import fetch_option_chain
    from src.signals.generate_signals import find_profitable_spreads

    with open("configs/strategy.yaml", 'r') as f:
        config = yaml.safe_load(f)

    chain = fetch_option_chain(config["ticker"])
    signals = find_profitable_spreads(chain, min_spread=config["min_spread"])
    print("Signals:", signals)
