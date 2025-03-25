import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
claiborne = pd.read_csv("claiborne.csv")
warren = pd.read_csv("warren.csv")
copiah = pd.read_csv("copiah.csv")

# Add county labels
claiborne['County'] = 'Claiborne'
warren['County'] = 'Warren'
copiah['County'] = 'Copiah'

# Combine all
df = pd.concat([claiborne, warren, copiah], ignore_index=True)

### 1. Demographic Summary ###
def plot_demographics(col):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='County')
    plt.title(f'{col} Distribution by County')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='County')
    plt.tight_layout()
    plt.show()

plot_demographics('Race')
plot_demographics('Gender')
plot_demographics('Education Level')

### 2. Risk Scores by Demographic Group ###
risk_summary = df.groupby(['County', 'Race', 'Gender'])['Risk Score'].mean().reset_index()
print("\nAverage Risk Scores:\n", risk_summary)

plt.figure(figsize=(12,6))
sns.barplot(data=risk_summary, x='Race', y='Risk Score', hue='County')
plt.title("Average Risk Score by Race and County")
plt.ylabel("Avg Risk Score")
plt.tight_layout()
plt.show()


### 3. Judge Decisions vs AI Risk Scores (Stacked Bar by Race & Gender) ###

# Ensure Risk Level binning
df['Risk Level'] = pd.cut(df['Risk Score'], bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'])

# Create a grouped dataframe
grouped = df.groupby(['Race', 'Gender', 'Risk Level', 'Judge Decision']).size().unstack(fill_value=0)

# Loop through race/gender combinations
for (race, gender), data in grouped.groupby(level=[0,1]):
    subset = data.droplevel([0,1])
    
    # Plot stacked bar chart
    subset[[0,1]].plot(kind='bar', stacked=True, figsize=(8,5), color=['#1f77b4', '#ff7f0e'])
    plt.title(f'Judge Decisions vs AI Risk Level ({race}, {gender})')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.legend(['Denied (0)', 'Granted (1)'])
    plt.tight_layout()
    plt.show()


### 4. Re-offense & Fairness Metrics ###

def calc_fairness_by_race(df):
    metrics = []
    for race in df['Race'].unique():
        group = df[df['Race'] == race]
        
        # Define positive (1) as re-offended
        # AI predicts risk > 5 as "high risk" (should deny bail)
        TP = ((group['Risk Score'] > 5) & (group['Re-offense'] == 1)).sum()
        FP = ((group['Risk Score'] > 5) & (group['Re-offense'] == 0)).sum()
        FN = ((group['Risk Score'] <= 5) & (group['Re-offense'] == 1)).sum()
        TN = ((group['Risk Score'] <= 5) & (group['Re-offense'] == 0)).sum()

        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

        metrics.append({
            'Race': race,
            'False Positive Rate (FPR)': round(fpr, 3),
            'False Negative Rate (FNR)': round(fnr, 3)
        })
    return pd.DataFrame(metrics)

fairness_df = calc_fairness_by_race(df)
print("\nFairness Metrics:\n", fairness_df)

# Plotting
fairness_df_melted = fairness_df.melt(id_vars='Race', var_name='Metric', value_name='Rate')

plt.figure(figsize=(10,6))
sns.barplot(data=fairness_df_melted, x='Race', y='Rate', hue='Metric')
plt.title("False Positive and False Negative Rates by Race")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
