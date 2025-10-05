import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Visualization:
    @staticmethod
    def plot_training_history(csv_path="donki_data_cleaned.csv"):
        try:
            print("📦 Loading dataset...")

            # Read large CSV in chunks for better performance
            chunks = pd.read_csv(csv_path, chunksize=1000)
            df = pd.concat([chunk for chunk in tqdm(chunks, desc="Reading data")])

            print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")
            print("Columns:", list(df.columns))

            # Check if 'speed' column exists
            if 'speed' not in df.columns:
                raise KeyError("❌ 'speed' column not found in the CSV file.")

            # Convert to numeric (in case of string/NaN values)
            speed = pd.to_numeric(df['speed'], errors='coerce').dropna()

            if speed.empty:
                raise ValueError("❌ No valid numeric data found in 'speed' column.")

            # Create and show boxplot
            plt.figure(figsize=(8, 6))
            plt.boxplot(speed, patch_artist=True, boxprops=dict(facecolor="skyblue"))
            plt.title('CME Speed Distribution')
            plt.ylabel('Speed (km/s)')
            plt.grid(True, linestyle='--', alpha=0.6)

            # Save plot to file
            plt.savefig('cme_speed_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Plot saved as 'cme_speed_distribution.png'")

            plt.show()

        except FileNotFoundError:
            print(f"❌ File not found: {csv_path}")
        except Exception as e:
            print("⚠️ Error:", e)


if __name__ == "__main__":
    Visualization.plot_training_history()
