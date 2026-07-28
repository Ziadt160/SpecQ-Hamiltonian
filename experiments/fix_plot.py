import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = 'results/case_study_fraud/q_vs_q_metrics.csv'
if os.path.exists(csv_path):
    df_res = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    plt.bar(df_res['Model'], df_res['AUC'], alpha=0.7, label='AUC', color='teal')
    plt.bar(df_res['Model'], df_res['F1'], alpha=0.7, label='F1', color='coral', width=0.4)
    plt.ylabel('Score')
    plt.title('Quantum vs Quantum: Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/case_study_fraud/q_vs_q_performance.png')
    print("Plot generated successfully.")
else:
    print("CSV not found.")
