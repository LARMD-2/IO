# -*- coding: utf-8 -*-
"""
ANÁLISIS DE REGRESIÓN LINEAL - INGRESOS LABORALES ENAHO 2024
Proyecto de Investigación de Operaciones

Este script realiza:
1. Análisis exploratorio de datos
2. Regresión Lineal Simple
3. Regresión Lineal Múltiple
4. Cálculo de métricas (MAE, MSE, RMSE, MAPE)
5. Comparación de modelos
6. Visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Modo no-interactivo, evita ventanas y errores de tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("ANÁLISIS DE REGRESIÓN LINEAL - INGRESOS LABORALES ENAHO 2024")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

print("\n[1/6] Cargando datos...")
df = pd.read_csv('enaho_2024_ingresos_individuales.csv')
print(f"✓ Dataset cargado: {len(df):,} registros, {len(df.columns)} variables")

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

print("\n[2/6] Análisis Exploratorio de Datos...")

# Limpiar y convertir variables numéricas
variables_numericas_cols = ['edad', 'nivel_educativo', 'anios_educacion', 
                            'horas_trabajadas_semanal', 'ingreso_laboral_anual']
for col in variables_numericas_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar códigos de "no especificado" (99) en nivel educativo
df = df[df['nivel_educativo'] != 99]

# Filtrar solo mayores de 18 años y máximo 70 años
df = df[(df['edad'] >= 18) & (df['edad'] <= 70)]

# Eliminar ingresos igual a 0 o negativos
df = df[df['ingreso_laboral_anual'] > 0]

# Eliminar valores extremos (menores a 9,000 y mayores a 35,000)
print("\nEliminando valores extremos...")
limite_inferior = 9000
limite_superior = 35000
df_antes = len(df)
df = df[(df['ingreso_laboral_anual'] >= limite_inferior) & 
        (df['ingreso_laboral_anual'] <= limite_superior)]
print(f"✓ Valores extremos eliminados: {df_antes - len(df):,} registros (rango: S/ {limite_inferior:,.0f} - S/ {limite_superior:,.0f})")

# Filtrar horas trabajadas menores o iguales a 75 por semana
print("\nFiltrando horas trabajadas por semana...")
df_antes = len(df)
df = df[df['horas_trabajadas_semanal'] <= 75]
print(f"✓ Registros con horas > 75 eliminados: {df_antes - len(df):,} registros")

# Eliminar filas con valores nulos en variables importantes
df = df.dropna(subset=variables_numericas_cols)
print(f"✓ Datos limpiados: {len(df):,} registros válidos (sin missing, sin outliers extremos)")
print(f"✓ Rango de ingresos: S/ {df['ingreso_laboral_anual'].min():.0f} - S/ {df['ingreso_laboral_anual'].max():.0f}")
print(f"✓ Dataset final: {len(df):,} registros")

# Matriz de correlación con datos filtrados
print("\nGenerando matriz de correlación (datos finales filtrados)...")
variables_para_correlacion = ['edad', 'sexo', 'estado_civil', 'nivel_educativo', 
                              'anios_educacion', 'anios_en_ocupacion', 'ocupacion',
                              'categoria_ocupacional', 'tipo_empleador', 'tipo_contrato',
                              'tamano_empresa', 'horas_trabajadas_semanal', 
                              'ingreso_laboral_anual']

# Crear copia para correlación asegurando que todas las variables sean numéricas
df_corr = df[variables_para_correlacion].copy()
for col in df_corr.columns:
    df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
df_corr = df_corr.dropna()

correlacion = df_corr.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlacion, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8})
plt.title('Matriz de Correlación - Todas las Variables del Modelo', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('02_matriz_correlacion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_matriz_correlacion.png")
plt.close()

# Variables a analizar
print("\nVariables en el dataset:")
print(df.columns.tolist())

# Estadísticas descriptivas de la variable dependiente
print("\n" + "="*80)
print("ESTADÍSTICAS DEL INGRESO LABORAL ANUAL")
print("="*80)
print(df['ingreso_laboral_anual'].describe())

# Estadísticas por variables categóricas
print("\n" + "="*80)
print("INGRESO PROMEDIO POR SEXO")
print("="*80)
print(df.groupby('sexo')['ingreso_laboral_anual'].agg(['mean', 'median', 'count']))

print("\n" + "="*80)
print("INGRESO PROMEDIO POR NIVEL EDUCATIVO")
print("="*80)
print(df.groupby('nivel_educativo')['ingreso_laboral_anual'].agg(['mean', 'median', 'count']))

print("\n" + "="*80)
print("INGRESO PROMEDIO POR CATEGORÍA OCUPACIONAL")
print("="*80)
print(df.groupby('categoria_ocupacional')['ingreso_laboral_anual'].agg(['mean', 'median', 'count']))

# Crear visualizaciones EDA
print("\nGenerando gráficos exploratorios...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribución del ingreso
axes[0, 0].hist(df['ingreso_laboral_anual'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Ingreso Laboral Anual (S/)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución del Ingreso Laboral Anual')
axes[0, 0].axvline(df['ingreso_laboral_anual'].mean(), color='red', 
                    linestyle='--', label=f'Media: S/ {df["ingreso_laboral_anual"].mean():,.0f}')
axes[0, 0].legend()

# 2. Boxplot por sexo (mejorado)
df_sexo = df.copy()
df_sexo['sexo'] = df_sexo['sexo'].map({1: 'Hombre', 2: 'Mujer'})

# Crear boxplot con mejor formato
data_hombre = df_sexo[df_sexo['sexo'] == 'Hombre']['ingreso_laboral_anual']
data_mujer = df_sexo[df_sexo['sexo'] == 'Mujer']['ingreso_laboral_anual']

bp = axes[0, 1].boxplot([data_hombre, data_mujer], labels=['Hombre', 'Mujer'], 
                         patch_artist=True, widths=0.6)

# Colorear las cajas
for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0, 1].set_xlabel('Sexo', fontweight='bold')
axes[0, 1].set_ylabel('Ingreso Laboral Anual (S/)', fontweight='bold')
axes[0, 1].set_title('Ingreso por Sexo', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x/1000:.0f}k'))


# 3. Scatter: Edad vs Ingreso
axes[1, 0].scatter(df['edad'], df['ingreso_laboral_anual'], alpha=0.3)
axes[1, 0].set_xlabel('Edad (años)')
axes[1, 0].set_ylabel('Ingreso Laboral Anual (S/)')
axes[1, 0].set_title('Relación Edad vs Ingreso')

# 4. Scatter: Horas trabajadas vs Ingreso
axes[1, 1].scatter(df['horas_trabajadas_semanal'], df['ingreso_laboral_anual'], alpha=0.3)
axes[1, 1].set_xlabel('Horas Trabajadas por Semana')
axes[1, 1].set_ylabel('Ingreso Laboral Anual (S/)')
axes[1, 1].set_title('Relación Horas Trabajadas vs Ingreso')

plt.tight_layout()
plt.savefig('01_analisis_exploratorio.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_analisis_exploratorio.png")
plt.close()

# ============================================================================
# 3. PREPARACIÓN DE DATOS PARA REGRESIÓN
# ============================================================================

print("\n[3/6] Preparando datos para regresión...")

# Variable dependiente (Y)
y = df['ingreso_laboral_anual']

# Aplicar transformación logarítmica a la variable dependiente
y_log = np.log(y)
print(f"✓ Transformación logarítmica aplicada al ingreso")

# Limpiar variables categóricas adicionales
variables_categoricas = ['sexo', 'estado_civil', 'ocupacion', 'categoria_ocupacional', 
                         'tipo_empleador', 'dominio_geografico', 'estrato', 'tipo_contrato', 'tamano_empresa']
for col in variables_categoricas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Usar años en ocupación declarados (experiencia real)
df['anios_en_ocupacion'] = pd.to_numeric(df['anios_en_ocupacion'], errors='coerce')
df['experiencia'] = df['anios_en_ocupacion'].fillna(0).clip(lower=0)  # Llenar NaN con 0 y no negativa
df['experiencia_cuadrado'] = df['experiencia'] ** 2

# Crear variables de interacción
df['educacion_experiencia'] = df['nivel_educativo'] * df['experiencia']
df['edad_cuadrado'] = df['edad'] ** 2

# Nuevas interacciones adicionales
df['edad_horas'] = df['edad'] * df['horas_trabajadas_semanal']
df['sexo_educacion'] = df['sexo'] * df['nivel_educativo']
df['estado_civil_edad'] = df['estado_civil'] * df['edad']
df['ocupacion_horas'] = df['ocupacion'] * df['horas_trabajadas_semanal']
df['experiencia_cuadrado'] = df['experiencia'] ** 2

print(f"✓ Variables de interacción creadas: 9 variables")

# Mapear ESTRATO a nombres descriptivos
estrato_nombres = {
    1: 'Lima_Metropolitana',
    2: 'Resto_Costa_Urbana', 
    3: 'Sierra_Urbana',
    4: 'Selva_Urbana',
    5: 'Costa_Rural',
    6: 'Sierra_Rural',
    7: 'Selva_Rural',
    8: 'Lima_Rural'
}
df['estrato_nombre'] = df['estrato'].map(estrato_nombres)

# Convertir ESTRATO a variables dummy (área urbana/rural)
estrato_dummies = pd.get_dummies(df['estrato_nombre'], drop_first=True)
df = pd.concat([df, estrato_dummies], axis=1)
print(f"✓ Dummies de estrato creadas: {len(estrato_dummies.columns)} variables")

# Eliminar filas con valores nulos en variables para regresión múltiple
variables_modelo_multiple = ['edad', 'edad_cuadrado', 'sexo', 'estado_civil', 'nivel_educativo', 
                              'anios_educacion', 'experiencia', 'educacion_experiencia', 
                              'ocupacion', 'categoria_ocupacional', 'tipo_empleador', 'tipo_contrato', 'tamano_empresa',
                              'horas_trabajadas_semanal',
                              'edad_horas', 'sexo_educacion', 'estado_civil_edad', 
                              'ocupacion_horas', 'experiencia_cuadrado'] + list(estrato_dummies.columns)

df = df.dropna(subset=variables_modelo_multiple)
print(f"✓ Registros válidos para regresión múltiple: {len(df):,}")

# Variables independientes para regresión múltiple (X)
# Seleccionar solo variables numéricas relevantes
X_multiple = df[variables_modelo_multiple]

# Variable para regresión simple (elegimos la más correlacionada)
# Según análisis, usaremos 'nivel_educativo'
X_simple = df[['nivel_educativo']]

# Actualizar y después del filtrado (tanto normal como log)
y = df['ingreso_laboral_anual']
y_log = np.log(y)

print(f"✓ Variable dependiente (Y): ingreso_laboral_anual (original y log)")
print(f"✓ Variable para regresión simple (X): nivel_educativo")
print(f"✓ Variables para regresión múltiple (X): {X_multiple.shape[1]} variables")
print(f"  - Variables base: 14 (+ tipo_contrato + tamano_empresa)")
print(f"  - Variables de interacción: 5")
print(f"  - Dummies de estrato: {len(estrato_dummies.columns)}")

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_simple_train, X_simple_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
    X_simple, y, y_log, test_size=0.2, random_state=42
)

X_multiple_train, X_multiple_test, _, _, _, _ = train_test_split(
    X_multiple, y, y_log, test_size=0.2, random_state=42
)

print(f"✓ Datos de entrenamiento: {len(X_simple_train):,} registros (80%)")
print(f"✓ Datos de prueba: {len(X_simple_test):,} registros (20%)")

# ============================================================================
# 4. REGRESIÓN LINEAL SIMPLE
# ============================================================================

print("\n[4/6] Regresión Lineal Simple...")
print("="*80)

# Entrenar modelo
modelo_simple = LinearRegression()
modelo_simple.fit(X_simple_train, y_train)

# Predicciones
y_pred_simple_train = modelo_simple.predict(X_simple_train)
y_pred_simple_test = modelo_simple.predict(X_simple_test)

# Coeficientes
print("\nEcuación del modelo:")
print(f"Ingreso = {modelo_simple.intercept_:.2f} + {modelo_simple.coef_[0]:.2f} × Nivel_Educativo")

# Métricas en datos de entrenamiento
print("\n--- Métricas en Datos de Entrenamiento ---")
mae_simple_train = mean_absolute_error(y_train, y_pred_simple_train)
mse_simple_train = mean_squared_error(y_train, y_pred_simple_train)
rmse_simple_train = np.sqrt(mse_simple_train)
mape_simple_train = np.mean(np.abs((y_train - y_pred_simple_train) / y_train)) * 100
r2_simple_train = r2_score(y_train, y_pred_simple_train)

print(f"MAE:  S/ {mae_simple_train:,.2f}")
print(f"MSE:  {mse_simple_train:,.2f}")
print(f"RMSE: S/ {rmse_simple_train:,.2f}")
print(f"MAPE: {mape_simple_train:.2f}%")
print(f"R²:   {r2_simple_train:.4f}")

# Métricas en datos de prueba
print("\n--- Métricas en Datos de Prueba ---")
mae_simple_test = mean_absolute_error(y_test, y_pred_simple_test)
mse_simple_test = mean_squared_error(y_test, y_pred_simple_test)
rmse_simple_test = np.sqrt(mse_simple_test)
mape_simple_test = np.mean(np.abs((y_test - y_pred_simple_test) / y_test)) * 100
r2_simple_test = r2_score(y_test, y_pred_simple_test)

print(f"MAE:  S/ {mae_simple_test:,.2f}")
print(f"MSE:  {mse_simple_test:,.2f}")
print(f"RMSE: S/ {rmse_simple_test:,.2f}")
print(f"MAPE: {mape_simple_test:.2f}%")
print(f"R²:   {r2_simple_test:.4f}")

# Visualización regresión simple
print("\nGenerando gráfico de regresión simple...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Datos de entrenamiento
axes[0].scatter(X_simple_train, y_train, alpha=0.3, label='Datos reales')
axes[0].plot(X_simple_train, y_pred_simple_train, color='red', linewidth=2, 
             label='Línea de regresión')
axes[0].set_xlabel('Nivel Educativo')
axes[0].set_ylabel('Ingreso Laboral Anual (S/)')
axes[0].set_title(f'Regresión Lineal Simple - Entrenamiento\nR² = {r2_simple_train:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Datos de prueba
axes[1].scatter(X_simple_test, y_test, alpha=0.3, label='Datos reales', color='green')
axes[1].plot(X_simple_test, y_pred_simple_test, color='red', linewidth=2, 
             label='Línea de regresión')
axes[1].set_xlabel('Nivel Educativo')
axes[1].set_ylabel('Ingreso Laboral Anual (S/)')
axes[1].set_title(f'Regresión Lineal Simple - Prueba\nR² = {r2_simple_test:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_regresion_simple.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_regresion_simple.png")
plt.close()

# ============================================================================
# 5. REGRESIÓN LINEAL MÚLTIPLE (CON TRANSFORMACIÓN LOG)
# ============================================================================

print("\n[5/6] Regresión Lineal Múltiple (con transformación logarítmica)...")
print("="*80)

# Entrenar modelo con variable dependiente transformada
modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple_train, y_log_train)

# Predicciones en escala log
y_log_pred_multiple_train = modelo_multiple.predict(X_multiple_train)
y_log_pred_multiple_test = modelo_multiple.predict(X_multiple_test)

# Convertir predicciones de log a escala original
y_pred_multiple_train = np.exp(y_log_pred_multiple_train)
y_pred_multiple_test = np.exp(y_log_pred_multiple_test)

# Coeficientes
print(f"\nCoeficientes del modelo (escala logarítmica):")
print(f"Intercepto: {modelo_multiple.intercept_:.4f}")
print("\nCoeficientes de las variables:")
for i, col in enumerate(X_multiple.columns):
    print(f"  {col:30s}: {modelo_multiple.coef_[i]:>10,.4f}")

# Métricas en datos de entrenamiento
print("\n--- Métricas en Datos de Entrenamiento ---")
mae_multiple_train = mean_absolute_error(y_train, y_pred_multiple_train)
mse_multiple_train = mean_squared_error(y_train, y_pred_multiple_train)
rmse_multiple_train = np.sqrt(mse_multiple_train)
mape_multiple_train = np.mean(np.abs((y_train - y_pred_multiple_train) / y_train)) * 100
r2_multiple_train = r2_score(y_train, y_pred_multiple_train)

print(f"MAE:  S/ {mae_multiple_train:,.2f}")
print(f"MSE:  {mse_multiple_train:,.2f}")
print(f"RMSE: S/ {rmse_multiple_train:,.2f}")
print(f"MAPE: {mape_multiple_train:.2f}%")
print(f"R²:   {r2_multiple_train:.4f}")

# Métricas en datos de prueba
print("\n--- Métricas en Datos de Prueba ---")
mae_multiple_test = mean_absolute_error(y_test, y_pred_multiple_test)
mse_multiple_test = mean_squared_error(y_test, y_pred_multiple_test)
rmse_multiple_test = np.sqrt(mse_multiple_test)
mape_multiple_test = np.mean(np.abs((y_test - y_pred_multiple_test) / y_test)) * 100
r2_multiple_test = r2_score(y_test, y_pred_multiple_test)

print(f"MAE:  S/ {mae_multiple_test:,.2f}")
print(f"MSE:  {mse_multiple_test:,.2f}")
print(f"RMSE: S/ {rmse_multiple_test:,.2f}")
print(f"MAPE: {mape_multiple_test:.2f}%")
print(f"R²:   {r2_multiple_test:.4f}")

# Visualización regresión múltiple
print("\nGenerando gráficos de regresión múltiple...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Valores reales vs predichos (entrenamiento)
axes[0].scatter(y_train, y_pred_multiple_train, alpha=0.3)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', linewidth=2, label='Predicción perfecta')
axes[0].set_xlabel('Ingreso Real (S/)')
axes[0].set_ylabel('Ingreso Predicho (S/)')
axes[0].set_title(f'Regresión Múltiple - Entrenamiento\nR² = {r2_multiple_train:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Valores reales vs predichos (prueba)
axes[1].scatter(y_test, y_pred_multiple_test, alpha=0.3, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Predicción perfecta')
axes[1].set_xlabel('Ingreso Real (S/)')
axes[1].set_ylabel('Ingreso Predicho (S/)')
axes[1].set_title(f'Regresión Múltiple - Prueba\nR² = {r2_multiple_test:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_regresion_multiple.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 04_regresion_multiple.png")
plt.close()

# Importancia de variables (usando coeficientes estandarizados)
print("\nGenerando gráfico de importancia de variables...")

# Entrenar modelo con datos estandarizados y y_log (igual a como se entrena el modelo múltiple)
from sklearn.preprocessing import StandardScaler as StandardScaler_import

scaler_importance = StandardScaler_import()
X_multiple_scaled = scaler_importance.fit_transform(X_multiple)

# Usar y_log porque el modelo se entrena con log
modelo_importance = LinearRegression()
modelo_importance.fit(X_multiple_scaled, y_log)

# Calcular importancia
importancia = pd.DataFrame({
    'Variable': X_multiple.columns,
    'Coeficiente': np.abs(modelo_importance.coef_)
}).sort_values('Coeficiente', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importancia['Variable'], importancia['Coeficiente'])
plt.xlabel('Valor Absoluto del Coeficiente Estandarizado')
plt.title('Importancia de Variables en Regresión Múltiple', fontweight='bold')
plt.tight_layout()
plt.savefig('05_importancia_variables.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 05_importancia_variables.png")
plt.close()

# ============================================================================
# 6. MODELOS ADICIONALES (Ridge, Lasso, Random Forest)
# ============================================================================

print("\n[6/9] Regresión Ridge...")
print("="*80)

# Entrenar Ridge con log
modelo_ridge = Ridge(alpha=1.0, random_state=42)
modelo_ridge.fit(X_multiple_train, y_log_train)

# Predicciones
y_log_pred_ridge_test = modelo_ridge.predict(X_multiple_test)
y_pred_ridge_test = np.exp(y_log_pred_ridge_test)

# Métricas Ridge - Test
mae_ridge = mean_absolute_error(y_test, y_pred_ridge_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge_test)
rmse_ridge = np.sqrt(mse_ridge)
mape_ridge = np.mean(np.abs((y_test - y_pred_ridge_test) / y_test)) * 100
r2_ridge = r2_score(y_test, y_pred_ridge_test)

print(f"MAE:  S/ {mae_ridge:,.2f}")
print(f"RMSE: S/ {rmse_ridge:,.2f}")
print(f"MAPE: {mape_ridge:.2f}%")
print(f"R²:   {r2_ridge:.4f}")

# Visualización Ridge
print("\nGenerando gráfico de Ridge...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge_test, alpha=0.4, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Predicción perfecta')
plt.xlabel('Ingreso Real (S/)')
plt.ylabel('Ingreso Predicho (S/)')
plt.title(f'Regresión Ridge - Prueba\nR² = {r2_ridge:.4f}, MAPE = {mape_ridge:.2f}%', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('09_ridge_predicciones.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 09_ridge_predicciones.png")
plt.close()

print("\n[7/7] Random Forest...")
print("="*80)

# Entrenar Random Forest con hiperparámetros optimizados
modelo_rf = RandomForestRegressor(n_estimators=400,        # Más árboles
                                  max_depth=18,            # Mayor profundidad
                                  min_samples_split=5,     # Más flexibilidad
                                  min_samples_leaf=3,      # Nodos más pequeños
                                  max_features='sqrt',     # Mantener sqrt
                                  random_state=42, 
                                  n_jobs=-1)
modelo_rf.fit(X_multiple_train, y_log_train)

# Predicciones
y_log_pred_rf_test = modelo_rf.predict(X_multiple_test)
y_pred_rf_test = np.exp(y_log_pred_rf_test)

# Métricas Random Forest - Test
mae_rf = mean_absolute_error(y_test, y_pred_rf_test)
mse_rf = mean_squared_error(y_test, y_pred_rf_test)
rmse_rf = np.sqrt(mse_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf_test) / y_test)) * 100
r2_rf = r2_score(y_test, y_pred_rf_test)

print(f"MAE:  S/ {mae_rf:,.2f}")
print(f"RMSE: S/ {rmse_rf:,.2f}")
print(f"MAPE: {mape_rf:.2f}%")
print(f"R²:   {r2_rf:.4f}")

# Visualización Random Forest
print("\nGenerando gráfico de Random Forest...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf_test, alpha=0.4, color='darkgreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Predicción perfecta')
plt.xlabel('Ingreso Real (S/)')
plt.ylabel('Ingreso Predicho (S/)')
plt.title(f'Random Forest - Prueba\nR² = {r2_rf:.4f}, MAPE = {mape_rf:.2f}%', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('11_random_forest_predicciones.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 11_random_forest_predicciones.png")
plt.close()

# Importancia de variables en Random Forest
print("\nGenerando gráfico de importancia (Random Forest)...")
importancia_rf = pd.DataFrame({
    'Variable': X_multiple.columns,
    'Importancia': modelo_rf.feature_importances_
}).sort_values('Importancia', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importancia_rf['Variable'], importancia_rf['Importancia'])
plt.xlabel('Importancia')
plt.title('Importancia de Variables - Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig('07_importancia_random_forest.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 07_importancia_random_forest.png")
plt.close()

# ============================================================================
# 7. COMPARACIÓN DE TODOS LOS MODELOS
# ============================================================================

print("\n[8/8] Comparación de Modelos...")
print("="*80)

# Tabla comparativa
# Tabla comparativa de todos los modelos
comparacion = pd.DataFrame({
    'Métrica': ['MAE (S/)', 'MSE', 'RMSE (S/)', 'MAPE (%)', 'R²'],
    'Regresión Simple': [
        f'{mae_simple_test:,.2f}',
        f'{mse_simple_test:,.2f}',
        f'{rmse_simple_test:,.2f}',
        f'{mape_simple_test:.2f}',
        f'{r2_simple_test:.4f}'
    ],
    'Regresión Múltiple': [
        f'{mae_multiple_test:,.2f}',
        f'{mse_multiple_test:,.2f}',
        f'{rmse_multiple_test:,.2f}',
        f'{mape_multiple_test:.2f}',
        f'{r2_multiple_test:.4f}'
    ],
    'Ridge': [
        f'{mae_ridge:,.2f}',
        f'{mse_ridge:,.2f}',
        f'{rmse_ridge:,.2f}',
        f'{mape_ridge:.2f}',
        f'{r2_ridge:.4f}'
    ],
    'Random Forest': [
        f'{mae_rf:,.2f}',
        f'{mse_rf:,.2f}',
        f'{rmse_rf:,.2f}',
        f'{mape_rf:,.2f}',
        f'{r2_rf:.4f}'
    ]
})

print("\nCOMPARACIÓN DE TODOS LOS MODELOS (Datos de Prueba):")
print(comparacion.to_string(index=False))

# Visualización comparativa - Métricas de error
print("\nGenerando gráfico comparativo de métricas...")
modelos = ['Simple', 'Múltiple', 'Ridge', 'Random\nForest']
mae_vals = [mae_simple_test, mae_multiple_test, mae_ridge, mae_rf]
rmse_vals = [rmse_simple_test, rmse_multiple_test, rmse_ridge, rmse_rf]
mape_vals = [mape_simple_test, mape_multiple_test, mape_ridge, mape_rf]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE
axes[0].bar(modelos, mae_vals, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[0].set_ylabel('MAE (S/)')
axes[0].set_title('Error Absoluto Medio (MAE)', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(mae_vals):
    axes[0].text(i, v + 100, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

# RMSE
axes[1].bar(modelos, rmse_vals, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_ylabel('RMSE (S/)')
axes[1].set_title('Raíz del Error Cuadrático Medio (RMSE)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v + 100, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

# MAPE
axes[2].bar(modelos, mape_vals, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[2].set_ylabel('MAPE (%)')
axes[2].set_title('Error Porcentual Absoluto Medio (MAPE)', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(mape_vals):
    axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('06_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 06_comparacion_modelos.png")
plt.close()

# Visualización R²
print("\nGenerando gráfico comparativo de R²...")
r2_vals = [r2_simple_test, r2_multiple_test, r2_ridge, r2_rf]

plt.figure(figsize=(10, 6))
bars = plt.bar(modelos, r2_vals, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])
plt.ylabel('R² (Coeficiente de Determinación)')
plt.title('Comparación de R² entre Modelos', fontweight='bold', fontsize=14)
plt.ylim(0, max(r2_vals) * 1.2)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(r2_vals):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('08_comparacion_r2.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 08_comparacion_r2.png")
plt.close()

# Conclusiones
print("\n" + "="*80)
print("CONCLUSIONES")
print("="*80)

# Encontrar mejor modelo por R²
modelos_dict = {
    'Regresión Simple': r2_simple_test,
    'Regresión Múltiple': r2_multiple_test,
    'Ridge': r2_ridge,
    'Random Forest': r2_rf
}
mejor_modelo = max(modelos_dict, key=modelos_dict.get)
print(f"\n✓ Mejor modelo según R²: {mejor_modelo} (R² = {modelos_dict[mejor_modelo]:.4f})")

# Ranking de modelos
print("\n✓ Ranking de modelos por R²:")
for i, (modelo, r2) in enumerate(sorted(modelos_dict.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"  {i}. {modelo}: {r2:.4f}")

# Mejor modelo por MAE
mae_dict = {
    'Regresión Simple': mae_simple_test,
    'Regresión Múltiple': mae_multiple_test,
    'Ridge': mae_ridge,
    'Random Forest': mae_rf
}
mejor_mae = min(mae_dict, key=mae_dict.get)
print(f"\n✓ Mejor modelo según MAE: {mejor_mae} (MAE = S/ {mae_dict[mejor_mae]:,.2f})")

mejoria_mae = ((mae_simple_test - mae_multiple_test) / mae_simple_test) * 100
mejoria_rmse = ((rmse_simple_test - rmse_multiple_test) / rmse_simple_test) * 100

print(f"\n✓ Mejora de la Regresión Múltiple vs Simple:")
print(f"  - MAE: {mejoria_mae:.2f}% de reducción")
print(f"  - RMSE: {mejoria_rmse:.2f}% de reducción")

# ============================================================================
# 9. ANÁLISIS POR CATEGORÍAS
# ============================================================================

print("\n" + "="*80)
print("GENERANDO ANÁLISIS POR CATEGORÍAS")
print("="*80)

# Gráfico 1: Boxplot de Ingreso por Nivel Educativo
print("\n[1/4] Generando boxplot por nivel educativo...")
plt.figure(figsize=(14, 6))
df['nivel_educativo'] = pd.to_numeric(df['nivel_educativo'], errors='coerce')
df_plot = df[df['nivel_educativo'].notna()].copy()
df_plot = df_plot.sort_values('nivel_educativo')

# Crear etiquetas descriptivas
nivel_labels = {
    1: 'Sin nivel',
    2: 'Inicial',
    3: 'Primaria',
    4: 'Secundaria Inc.',
    5: 'Secundaria',
    6: 'Sup. no univ. Inc.',
    7: 'Sup. no univ.',
    8: 'Sup. univ. Inc.',
    9: 'Sup. univ.',
    10: 'Maestría',
    11: 'Doctorado',
    12: 'Posgrado'
}

df_plot['nivel_educativo_label'] = df_plot['nivel_educativo'].map(nivel_labels)
boxplot_data = [df_plot[df_plot['nivel_educativo'] == nivel]['ingreso_laboral_anual'].values 
                for nivel in sorted(df_plot['nivel_educativo'].unique())]
labels = [nivel_labels.get(int(nivel), f'Nivel {int(nivel)}') 
          for nivel in sorted(df_plot['nivel_educativo'].unique())]

bp = plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#4ECDC4')
    patch.set_alpha(0.7)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Ingreso Laboral Anual (S/)', fontsize=11)
plt.title('Distribución de Ingresos por Nivel Educativo', fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('12_ingreso_por_educacion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 12_ingreso_por_educacion.png")
plt.close()

# Gráfico 2: Barras de Ingreso por Tipo de Contrato
print("\n[2/4] Generando gráfico por tipo de contrato...")
df['tipo_contrato'] = pd.to_numeric(df['tipo_contrato'], errors='coerce')
contrato_stats = df.groupby('tipo_contrato')['ingreso_laboral_anual'].agg(['mean', 'count']).reset_index()
contrato_stats = contrato_stats[contrato_stats['count'] >= 20]  # Solo categorías con suficientes datos
contrato_stats = contrato_stats.sort_values('mean', ascending=False)

contrato_labels = {
    1: 'Indefinido',
    2: 'Plazo fijo',
    3: 'Sin contrato',
    4: 'Convenio',
    5: 'Locación servicios',
    6: 'Part-time',
    7: 'Modalidad formativa',
    8: 'Otros'
}

fig, ax = plt.subplots(figsize=(12, 6))
contrato_stats['label'] = contrato_stats['tipo_contrato'].map(lambda x: contrato_labels.get(int(x) if pd.notna(x) else 0, f'Tipo {int(x)}'))
bars = ax.bar(range(len(contrato_stats)), contrato_stats['mean'], color='#FF6B6B', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(contrato_stats)))
ax.set_xticklabels(contrato_stats['label'], rotation=45, ha='right')
ax.set_ylabel('Ingreso Promedio Anual (S/)', fontsize=11)
ax.set_title('Ingreso Promedio por Tipo de Contrato', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)

# Añadir valores encima de barras
for i, (v, n) in enumerate(zip(contrato_stats['mean'], contrato_stats['count'])):
    ax.text(i, v + 500, f'S/ {v:,.0f}\n(n={n:,})', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('13_ingreso_por_contrato.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 13_ingreso_por_contrato.png")
plt.close()

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print("\nArchivos generados:")
print("  1. 01_analisis_exploratorio.png")
print("  2. 02_matriz_correlacion.png")
print("  3. 03_regresion_simple.png")
print("  4. 04_regresion_multiple.png")
print("  5. 05_importancia_variables.png")
print("  6. 06_comparacion_modelos.png")
print("  7. 07_importancia_random_forest.png")
print("  8. 08_comparacion_r2.png")
print("  9. 09_ridge_predicciones.png")
print(" 10. 10_random_forest_predicciones.png")
print(" 11. 12_ingreso_por_educacion.png")
print(" 12. 13_ingreso_por_contrato.png")
print(" 13. 14_ingreso_por_empresa.png")
print(" 14. 15_ingreso_por_region.png")
print("\nPuedes usar estos gráficos y resultados en tu informe PDF.")
