import pandas as pd

df = pd.read_csv('enaho_2024_ingresos_individuales.csv')
df_clean = df[(df['nivel_educativo'] != 99) & (df['ingreso_laboral_anual'] > 0)].dropna()

print('Percentiles actuales (P1-P97):')
print(f'P1: S/ {df_clean["ingreso_laboral_anual"].quantile(0.01):,.0f}')
print(f'P97: S/ {df_clean["ingreso_laboral_anual"].quantile(0.97):,.0f}')

print('\nNuevos percentiles propuestos (P5-P95):')
p5 = df_clean["ingreso_laboral_anual"].quantile(0.05)
p95 = df_clean["ingreso_laboral_anual"].quantile(0.95)
print(f'P5: S/ {p5:,.0f}')
print(f'P95: S/ {p95:,.0f}')

registros_eliminar = ((df_clean["ingreso_laboral_anual"] < p5) | (df_clean["ingreso_laboral_anual"] > p95)).sum()
pct_eliminar = registros_eliminar / len(df_clean) * 100

print(f'\nRegistros a eliminar: {registros_eliminar:,} ({pct_eliminar:.1f}%)')
print(f'Registros a mantener: {len(df_clean) - registros_eliminar:,} ({100 - pct_eliminar:.1f}%)')
