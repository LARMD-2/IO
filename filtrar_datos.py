"""
Script para filtrar y consolidar datos de ENAHO 2024
Proyecto: Análisis de Ingreso Laboral Individual
"""

import pandas as pd
import numpy as np

print("="*70)
print("FILTRADO Y CONSOLIDACIÓN DE DATOS ENAHO 2024")
print("="*70)

# ============================================================================
# 1. CARGAR MÓDULOS CON VARIABLES SELECCIONADAS
# ============================================================================

print("\n[1/4] Cargando Módulo 100 (Características del Hogar)...")
mod100_vars = ['CONGLOME', 'VIVIENDA', 'HOGAR', 'DOMINIO', 'ESTRATO', 'UBIGEO']

df_100 = pd.read_csv('Enaho01-2024-100.csv', 
                     usecols=mod100_vars,
                     encoding='latin1')
print(f"   ✓ Módulo 100: {len(df_100):,} registros")

# ----------------------------------------------------------------------------

print("\n[2/4] Cargando Módulo 200 (Características Demográficas)...")
mod200_vars = ['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO', 
               'P207', 'P208A', 'P209']

df_200 = pd.read_csv('Enaho01-2024-200.csv', 
                     usecols=mod200_vars,
                     encoding='latin1')
print(f"   ✓ Módulo 200: {len(df_200):,} registros (personas)")

# ----------------------------------------------------------------------------

print("\n[3/4] Cargando Módulo 300 (Educación)...")
mod300_vars = ['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO',
               'P301A', 'P301B']

df_300 = pd.read_csv('Enaho01A-2024-300.csv', 
                     usecols=mod300_vars,
                     encoding='latin1')
print(f"   ✓ Módulo 300: {len(df_300):,} registros")

# ----------------------------------------------------------------------------

print("\n[4/4] Cargando Módulo 500 (Empleo e Ingresos)...")
mod500_vars = ['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO',
               'P501', 'P506R4', 'P507', 'P510', 'P511A', 'P513A1', 'P513T', 'P517D1', 'I524E1']

df_500 = pd.read_csv('Enaho01a-2024-500.csv', 
                     usecols=mod500_vars,
                     encoding='latin1')
print(f"   ✓ Módulo 500: {len(df_500):,} registros")

# ============================================================================
# 2. INTEGRAR TODOS LOS MÓDULOS
# ============================================================================

print("\n" + "="*70)
print("INTEGRANDO MÓDULOS")
print("="*70)

# Paso 1: Unir 200 con 100 (agregar características del hogar)
df_merge = df_200.merge(df_100, 
                        on=['CONGLOME', 'VIVIENDA', 'HOGAR'], 
                        how='left')
print(f"\n[1/3] Módulo 200 + 100: {len(df_merge):,} registros")

# Paso 2: Unir con 300 (agregar educación)
df_merge = df_merge.merge(df_300, 
                          on=['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO'], 
                          how='left')
print(f"[2/3] + Módulo 300: {len(df_merge):,} registros")

# Paso 3: Unir con 500 (agregar empleo e ingresos)
df_merge = df_merge.merge(df_500, 
                          on=['CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO'], 
                          how='left')
print(f"[3/3] + Módulo 500: {len(df_merge):,} registros")

# ============================================================================
# 3. FILTRAR SOLO INDIVIDUOS CON INGRESO LABORAL
# ============================================================================

print("\n" + "="*70)
print("FILTRADO DE DATOS")
print("="*70)

print(f"\nRegistros totales antes de filtrar: {len(df_merge):,}")

# Convertir I524E1 a numérico (puede venir como texto)
df_merge['I524E1'] = pd.to_numeric(df_merge['I524E1'], errors='coerce')

# Filtrar solo personas que trabajaron y tienen ingreso
# Convertir I524E1 a numérico antes de filtrar
df_merge['I524E1'] = pd.to_numeric(df_merge['I524E1'], errors='coerce')

df_filtrado = df_merge[
    (df_merge['P501'] == 1) &  # Trabajó la semana pasada
    (df_merge['I524E1'].notna()) &  # Tiene ingreso registrado
    (df_merge['I524E1'] > 0)  # Ingreso mayor a 0
].copy()

print(f"Registros después de filtrar (solo trabajadores con ingreso): {len(df_filtrado):,}")
print(f"Reducción: {len(df_merge) - len(df_filtrado):,} registros eliminados")

# ============================================================================
# 4. RENOMBRAR VARIABLES PARA MAYOR CLARIDAD
# ============================================================================

print("\n" + "="*70)
print("RENOMBRANDO VARIABLES")
print("="*70)

df_filtrado = df_filtrado.rename(columns={
    # Identificación
    'CODPERSO': 'cod_persona',
    'CONGLOME': 'conglomerado',
    'VIVIENDA': 'vivienda',
    'HOGAR': 'hogar',
    
    # Demográficas
    'P207': 'sexo',  # 1=Hombre, 2=Mujer
    'P208A': 'edad',
    'P209': 'estado_civil',
    
    # Educación
    'P301A': 'nivel_educativo',  # Nivel educativo aprobado
    'P301B': 'anios_educacion',  # Años de educación aprobados
    
    # Empleo
    'P501': 'trabajo_semana_pasada',
    'P506R4': 'ocupacion',  # Código de ocupación (2 dígitos)
    'P507': 'categoria_ocupacional',
    'P510': 'tipo_empleador',
    'P511A': 'tipo_contrato',  # 1-8: indefinido, plazo fijo, sin contrato, etc.
    'P513A1': 'anios_en_ocupacion',  # Años trabajando en la ocupación actual
    'P513T': 'horas_trabajadas_semanal',
    'P517D1': 'tamano_empresa',  # 1-5: hasta 20, 21-50, 51-100, 101-499, 500+
    
    # Ingreso (variable dependiente)
    'I524E1': 'ingreso_laboral_anual',
    
    # Geográficas
    'DOMINIO': 'dominio_geografico',
    'ESTRATO': 'estrato',
    'UBIGEO': 'ubigeo'
})

print("✓ Variables renombradas correctamente")

# ============================================================================
# 5. INFORMACIÓN DEL DATASET FILTRADO
# ============================================================================

print("\n" + "="*70)
print("RESUMEN DEL DATASET FILTRADO")
print("="*70)

print(f"\nDimensiones: {df_filtrado.shape[0]:,} filas x {df_filtrado.shape[1]} columnas")
print(f"\nColumnas incluidas:")
for i, col in enumerate(df_filtrado.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nEstadísticas de Ingreso Laboral Anual:")
print(f"  - Mínimo:    S/ {df_filtrado['ingreso_laboral_anual'].min():,.2f}")
print(f"  - Máximo:    S/ {df_filtrado['ingreso_laboral_anual'].max():,.2f}")
print(f"  - Media:     S/ {df_filtrado['ingreso_laboral_anual'].mean():,.2f}")
print(f"  - Mediana:   S/ {df_filtrado['ingreso_laboral_anual'].median():,.2f}")

print(f"\nDistribución por Sexo:")
sexo_dist = df_filtrado['sexo'].value_counts()
print(f"  Hombres (1): {sexo_dist.get(1, 0):,}")
print(f"  Mujeres (2): {sexo_dist.get(2, 0):,}")

print(f"\nDistribución por Dominio Geográfico:")
print(df_filtrado['dominio_geografico'].value_counts().sort_index().head(10).to_string())

# ============================================================================
# 6. GUARDAR DATASET FILTRADO
# ============================================================================

print("\n" + "="*70)
print("GUARDANDO DATASET")
print("="*70)

output_file = 'enaho_2024_ingresos_individuales.csv'
df_filtrado.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ Dataset guardado exitosamente: {output_file}")
print(f"  Tamaño: {len(df_filtrado):,} registros")
print(f"  Variables: {len(df_filtrado.columns)}")

# ============================================================================
# 7. VERIFICAR VALORES NULOS
# ============================================================================

print("\n" + "="*70)
print("VERIFICACIÓN DE VALORES NULOS")
print("="*70)

nulos = df_filtrado.isnull().sum()
if nulos.sum() > 0:
    print("\nVariables con valores nulos:")
    print(nulos[nulos > 0].to_string())
else:
    print("\n✓ No hay valores nulos en el dataset")

print("\n" + "="*70)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"\nArchivo generado: {output_file}")
print("Puedes proceder con el análisis de regresión.")
