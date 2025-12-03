import pandas as pd

df = pd.read_csv('enaho_2024_ingresos_individuales.csv')

total = len(df)
mas_30k = (df['ingreso_laboral_anual'] > 30000).sum()
pct = mas_30k / total * 100

print(f'Total trabajadores: {total:,}')
print(f'Con ingreso > S/ 30,000: {mas_30k:,} ({pct:.2f}%)')

print(f'\nDistribución por rangos de ingreso anual:')
print(f'  S/ 0 - 10,000: {(df["ingreso_laboral_anual"] <= 10000).sum():,} ({(df["ingreso_laboral_anual"] <= 10000).sum() / total * 100:.1f}%)')
print(f'  S/ 10,001 - 20,000: {((df["ingreso_laboral_anual"] > 10000) & (df["ingreso_laboral_anual"] <= 20000)).sum():,} ({((df["ingreso_laboral_anual"] > 10000) & (df["ingreso_laboral_anual"] <= 20000)).sum() / total * 100:.1f}%)')
print(f'  S/ 20,001 - 30,000: {((df["ingreso_laboral_anual"] > 20000) & (df["ingreso_laboral_anual"] <= 30000)).sum():,} ({((df["ingreso_laboral_anual"] > 20000) & (df["ingreso_laboral_anual"] <= 30000)).sum() / total * 100:.1f}%)')
print(f'  S/ 30,001 - 40,000: {((df["ingreso_laboral_anual"] > 30000) & (df["ingreso_laboral_anual"] <= 40000)).sum():,} ({((df["ingreso_laboral_anual"] > 30000) & (df["ingreso_laboral_anual"] <= 40000)).sum() / total * 100:.1f}%)')
print(f'  S/ 40,001 - 50,000: {((df["ingreso_laboral_anual"] > 40000) & (df["ingreso_laboral_anual"] <= 50000)).sum():,} ({((df["ingreso_laboral_anual"] > 40000) & (df["ingreso_laboral_anual"] <= 50000)).sum() / total * 100:.1f}%)')
print(f'  S/ 50,001+: {(df["ingreso_laboral_anual"] > 50000).sum():,} ({(df["ingreso_laboral_anual"] > 50000).sum() / total * 100:.1f}%)')
