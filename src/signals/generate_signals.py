def find_profitable_spreads(chain: list, min_spread: float = 0.2):
    signals = []
    for opt in chain:
        spread = opt['ask'] - opt['bid']
        if spread >= min_spread:
            signals.append({"strike": opt['strike'], "spread": spread})
    return signals