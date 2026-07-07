import json
import os
import matplotlib.pyplot as plt

def main():
    summary_path = "./results/benchmarks/phase4_summary.json"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run benchmark first.")
        return
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    policies = [d['policy'] for d in data]
    avg_queue = [d['mean_avg_queue_per_step'] for d in data]
    avg_wait = [d['mean_avg_waiting_time'] for d in data]
    throughput = [d['mean_throughput'] for d in data]
    
    # Create plots directory
    os.makedirs("./results/plots", exist_ok=True)
    
    # 1. Queue Length Plot
    plt.figure(figsize=(10, 6))
    plt.bar(policies, avg_queue, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Average Queue Length per Step')
    plt.ylabel('Vehicles')
    plt.grid(axis='y', alpha=0.7)
    plt.savefig('./results/plots/avg_queue.png')
    plt.close()
    
    # 2. Waiting Time Plot
    plt.figure(figsize=(10, 6))
    plt.bar(policies, avg_wait, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Average Waiting Time per Step')
    plt.ylabel('Seconds')
    plt.grid(axis='y', alpha=0.7)
    plt.savefig('./results/plots/avg_wait_time.png')
    plt.close()
    
    # 3. Throughput Plot
    plt.figure(figsize=(10, 6))
    plt.bar(policies, throughput, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Total Throughput (Vehicles completed route)')
    plt.ylabel('Vehicles')
    plt.grid(axis='y', alpha=0.7)
    plt.savefig('./results/plots/throughput.png')
    plt.close()
    
    print("Plots generated in ./results/plots/")

if __name__ == "__main__":
    main()
