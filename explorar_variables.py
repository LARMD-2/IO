import pandas as pd

# Cargar módulo 500
df = pd.read_csv('Enaho01a-2024-500.csv', encoding='latin1', low_memory=False)

print("Variables relacionadas con empleo (P5xx) disponibles:\n")

# Variables clave de empleo que podrían ser relevantes
variables_interes = [
    'P505',  # Independiente/dependiente
    'P506',  # Grupo ocupacional (1 dígito)
    'P506R4',  # Ocupación (4 dígitos)
    'P507',  # Categoría ocupacional
    'P508A',  # Actividad económica (2 dígitos)
    'P508A1',  # Actividad económica (1 dígito)
    'P510',  # Tipo de empleador
    'P511A',  # Tipo de contrato
    'P512A',  # ¿Tiene seguro de salud?
    'P512B',  # Tipo de seguro
    'P513T',  # Horas trabajadas
    'P514',  # ¿Desea trabajar más horas?
    'P517D1',  # Tamaño de empresa
    'P519',  # ¿Recibe pago en especie?
    'P520',  # ¿Tiene otro trabajo?
    'P521',  # ¿Cuántas horas en segundo trabajo?
]

for var in variables_interes:
    if var in df.columns:
        # Mostrar valores únicos y frecuencias
        valores = df[var].value_counts().head(10)
        n_unicos = df[var].nunique()
        n_missing = df[var].isna().sum()
        total = len(df)
        
        print(f"\n{var}:")
        print(f"  Valores únicos: {n_unicos}")
        print(f"  Missing: {n_missing:,} ({n_missing/total*100:.1f}%)")
        if n_unicos <= 15:
            print(f"  Distribución:")
            for val, count in valores.items():
                print(f"    {val}: {count:,} ({count/total*100:.1f}%)")
        else:
            print(f"  Top 5 valores:")
            for val, count in list(valores.items())[:5]:
                print(f"    {val}: {count:,}")
