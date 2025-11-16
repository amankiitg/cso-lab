import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from cso_lab.pricing import kirk_cso_price


# Base parameters from your example
BASE_PARAMS = {
    'F1': 50.0,  # Forward price for asset 1 (e.g., HH)
    'F2': 45.0,  # Forward price for asset 2 (e.g., WTI)
    'K': 3.0,  # Strike price
    'sigma1': 0.30,  # Volatility for asset 1
    'sigma2': 0.25,  # Volatility for asset 2
    'rho': 0.2,  # Correlation
    'T': 0.5,  # Time to expiration in years
    'steps': 1  # Number of steps for path generation
}


def generate_synthetic_cso_chain(
        n_tenors: int = 5,
        n_strikes: int = 7,
        output_file: str = "data/synthetic/cso_chain.csv"
) -> pd.DataFrame:
    """
    Generate a synthetic CSO chain with realistic variations.

    Args:
        n_tenors: Number of different tenors to generate
        n_strikes: Number of different strikes per tenor
        output_file: Path to save the output CSV

    Returns:
        DataFrame containing the synthetic CSO chain
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate tenors (0.25 to 2 years)
    tenors = np.linspace(0.25, 2.0, n_tenors)

    # Generate strikes as percentage of F1-F2 spread
    spread = BASE_PARAMS['F1'] - BASE_PARAMS['F2']
    strikes = np.linspace(-0.5, 0.5, n_strikes) * spread + spread

    rows = []
    for tenor in tenors:
        # Add some randomness to forward prices and vols
        F1 = BASE_PARAMS['F1'] * (1 + np.random.normal(0, 0.05))
        F2 = BASE_PARAMS['F2'] * (1 + np.random.normal(0, 0.05))
        sigma1 = max(0.1, min(0.5, BASE_PARAMS['sigma1'] * (1 + np.random.normal(0, 0.1))))
        sigma2 = max(0.1, min(0.5, BASE_PARAMS['sigma2'] * (1 + np.random.normal(0, 0.1))))
        rho = max(-0.9, min(0.9, BASE_PARAMS['rho'] + np.random.normal(0, 0.1)))

        for K in strikes:
            try:
                price = kirk_cso_price(F1, F2, K, sigma1, sigma2, rho, tenor)
                if not (np.isfinite(price) and price >= 0):
                    price = max(0, F1 - F2 - K)  # Fallback to intrinsic

                rows.append({
                    'tenor': tenor,
                    'strike': K,
                    'price': price,
                    'F1': F1,
                    'F2': F2,
                    'vol1': sigma1,
                    'vol2': sigma2,
                    'rho': rho,
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d')
                })
            except Exception as e:
                print(f"Error pricing K={K}, T={tenor}: {str(e)}")
                continue

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} option contracts, saved to {output_path}")
    return df


# Example usage
if __name__ == "__main__":
    df = generate_synthetic_cso_chain(
        n_tenors=5,
        n_strikes=7,
        output_file="data/synthetic/cso_chain.csv"
    )

    # Show summary
    print("\nGenerated CSO Chain Summary:")
    print(f"Tenors: {sorted(df['tenor'].unique())}")
    print(f"Strikes: {sorted(df['strike'].unique())[:5]}...")
    print(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")