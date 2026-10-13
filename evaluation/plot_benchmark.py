import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_file = "results/benchmarks/benchmark_results.csv"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run benchmark_scenarios.py first.")
        return
        
    df = pd.read_csv(results_file)
    output_dir = "results/benchmarks/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ("WaitTime", "Average Wait Time (s)"),
        ("QueueLength", "Average Queue Length"),
        ("Throughput", "Total Throughput (Vehicles)"),
        ("TravelTime", "Average Travel Time (s)"),
        ("Speed", "Average Speed (m/s)"),
        ("Teleports", "Total Teleports"),
        ("Reward", "Episode Reward")
    ]
    
    for metric, ylabel in metrics:
        if metric not in df.columns:
            print(f"Skipping {metric}, not found in data.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        sns.barplot(
            data=df,
            x="Scenario",
            y=metric,
            hue="Model",
            palette="viridis"
        )
        
        plt.title(f"{ylabel} by Scenario", fontsize=14, pad=15)
        plt.xlabel("Traffic Scenario", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Controller Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{metric.lower()}_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    main()
