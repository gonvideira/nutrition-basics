import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
import numpy as np
from matplotlib import pyplot as plt

FILE = 'NIA_2023/Excel IMC criancas.csv'
COLS = list(range(10))

df = pd.read_csv(FILE,sep=';',usecols=COLS,decimal=',',encoding='utf-8')
df.dropna(inplace=True)
df = df.astype({'Z_score_IMC':'int32'})

print(df)
print(df.dtypes)
print(df.describe())

# Calculate Pearson correlation coefficient
variable_X = df['Sexo']
z_scores = df['Z_score_IMC']
correlation_coefficient, p_value = pearsonr(variable_X, z_scores)
print(f"\nPearson correlation coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}.\n")

# Create a contingency table (cross-tab) from your data
contingency_table = pd.crosstab(df['Sexo'], df['Z_score_IMC'])

print(contingency_table)

# Perform the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display the results
print(f'Chi-Squared Statistic: {chi2:.4f}')
print(f'P-value: {p:.4f}')
print(f'Degrees of Freedom: {dof}')
print('Expected Frequencies:')
print(expected)
print('\n')

# define series of z-scores for each gender
gender_0_Z = df[df['Sexo']==0]['Z_score_IMC']
gender_1_Z = df[df['Sexo']==1]['Z_score_IMC']

# Perform independent samples t-test
t_statistic, p_value = ttest_ind(gender_0_Z, gender_1_Z)

print(f'T-statistic: {t_statistic:.4f}')
print(f'P-value: {p_value:.4f}')

# Check for statistical significance
alpha = 0.05
if p_value < alpha:
    print("The difference in Z-scores is statistically significant.")
else:
    print("There is no statistically significant difference in Z-scores.")
print('\n')


print_user = input('Print plots? (Y/N)')

if print_user == 'N':
    exit()

# Plotting the histogram for gender 0
plt.hist(gender_0_Z, bins=range(min(gender_0_Z), max(gender_0_Z) + 1), edgecolor='black')

# Calculate and display the average
average_value = np.mean(gender_0_Z)
plt.text(average_value, plt.ylim()[1]*0.95, f'Mean: {average_value:.2f}', color='red', ha='center', va='bottom')

# Calculate and display the standard deviation
std_deviation = np.std(gender_0_Z)
plt.text(average_value + std_deviation, plt.ylim()[1]*0.95, f'Std Dev: {std_deviation:.2f}', color='green', ha='center', va='bottom')

plt.title('Histogram of Z-scores for Gender 0')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# plot histograms for gender 1

# Plotting the histogram
plt.hist(gender_1_Z, bins=range(min(gender_1_Z), max(gender_1_Z) + 1), edgecolor='black')

# Calculate and display the average
average_value = np.mean(gender_1_Z)
plt.text(average_value, plt.ylim()[1]*0.95, f'Mean: {average_value:.2f}', color='red', ha='center', va='bottom')

# Calculate and display the standard deviation
std_deviation = np.std(gender_1_Z)
plt.text(average_value + std_deviation, plt.ylim()[1]*0.95, f'Std Dev: {std_deviation:.2f}', color='green', ha='center', va='bottom')

plt.title('Histogram of Z-scores for Gender 1')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
