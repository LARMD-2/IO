import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('enaho_2024_ingresos_individuales.csv', encoding='utf-8')

# Aplicar los mismos filtros del análisis
df = df[df['nivel_educativo'] != 99]
df = df[df['edad'] >= 18]
df = df[df['ingreso_laboral_anual'] > 0]
df = df[(df['ingreso_laboral_anual'] >= 8000) & (df['ingreso_laboral_anual'] <= 35000)]

print("=" * 80)
print("ANÁLISIS DE HORAS TRABAJADAS vs INGRESO")
print("=" * 80)

print(f"\nTotal registros: {len(df):,}")

print("\n--- ESTADÍSTICAS DE HORAS TRABAJADAS ---")
print(df['horas_trabajadas_semanal'].describe())

print("\n--- DISTRIBUCIÓN POR RANGOS DE HORAS ---")
df['rango_horas'] = pd.cut(df['horas_trabajadas_semanal'], 
                            bins=[0, 20, 40, 48, 60, 200],
                            labels=['≤20 hrs', '21-40 hrs', '41-48 hrs', '49-60 hrs', '>60 hrs'])
horas_dist = df.groupby('rango_horas', observed=True).agg({
    'ingreso_laboral_anual': ['mean', 'median', 'count']
}).round(0)
print(horas_dist)

print("\n--- INGRESO PROMEDIO POR CATEGORÍA OCUPACIONAL Y HORAS ---")
# Ver si el tipo de trabajo afecta la relación horas-ingreso
categorias = df.groupby('categoria_ocupacional').agg({
    'horas_trabajadas_semanal': 'mean',
    'ingreso_laboral_anual': 'mean',
    'cod_persona': 'count'
}).round(0)
categorias.columns = ['Horas Promedio', 'Ingreso Promedio', 'Cantidad']
print(categorias)

print("\n--- CORRELACIÓN HORAS-INGRESO POR CATEGORÍA ---")
for cat in df['categoria_ocupacional'].unique():
    if pd.notna(cat):
        subset = df[df['categoria_ocupacional'] == cat]
        if len(subset) > 10:
            corr = subset['horas_trabajadas_semanal'].corr(subset['ingreso_laboral_anual'])
            print(f"Categoría {int(cat)}: {corr:.3f} (n={len(subset):,})")

print("\n--- INGRESO POR HORA ---")
df['ingreso_por_hora'] = df['ingreso_laboral_anual'] / (df['horas_trabajadas_semanal'] * 52)
print(f"\nIngreso promedio por hora: S/ {df['ingreso_por_hora'].mean():.2f}")
print(f"Mediana ingreso por hora: S/ {df['ingreso_por_hora'].median():.2f}")
print(f"Desviación estándar: S/ {df['ingreso_por_hora'].std():.2f}")

print("\n--- CASOS EXTREMOS ---")
print("\nTrabajadores con >60 horas y bajo ingreso:")
casos_extremos = df[(df['horas_trabajadas_semanal'] > 60) & (df['ingreso_laboral_anual'] < 12000)]
print(f"Cantidad: {len(casos_extremos):,} ({100*len(casos_extremos)/len(df):.1f}%)")

print("\nTrabajadores con <30 horas y alto ingreso:")
casos_extremos2 = df[(df['horas_trabajadas_semanal'] < 30) & (df['ingreso_laboral_anual'] > 25000)]
print(f"Cantidad: {len(casos_extremos2):,} ({100*len(casos_extremos2)/len(df):.1f}%)")
