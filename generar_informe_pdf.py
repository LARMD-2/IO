# -*- coding: utf-8 -*-
"""
Generador de Informe PDF - Análisis de Regresión Lineal
ENAHO 2024 - Ingresos Laborales
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
import pandas as pd
from datetime import datetime
import os

# Configuración
MARGEN = 0.7 * inch

# Cargar datos
print("Leyendo datos...")
df = pd.read_csv('enaho_2024_ingresos_individuales.csv')

n_registros_totales = 24480
n_registros_finales = len(df)
n_variables = len(df.columns)

stats_ingreso = {
    'min': df['ingreso_laboral_anual'].min(),
    'max': df['ingreso_laboral_anual'].max(),
    'mean': df['ingreso_laboral_anual'].mean(),
    'median': df['ingreso_laboral_anual'].median(),
}

# Crear PDF
print("Generando PDF...")
pdf_filename = 'Informe_Analisis_Regresion_ENAHO_2024.pdf'
doc = SimpleDocTemplate(pdf_filename, pagesize=A4, 
                       rightMargin=MARGEN, leftMargin=MARGEN,
                       topMargin=MARGEN, bottomMargin=MARGEN)

styles = getSampleStyleSheet()

estilo_titulo = ParagraphStyle(
    'CustomTitle', parent=styles['Heading1'], fontSize=28,
    textColor=colors.HexColor('#1F4788'), spaceAfter=30,
    alignment=TA_CENTER, fontName='Helvetica-Bold'
)

estilo_subtitulo = ParagraphStyle(
    'CustomSubtitle', parent=styles['Heading2'], fontSize=14,
    textColor=colors.HexColor('#2E5DA6'), spaceAfter=12,
    spaceBefore=12, fontName='Helvetica-Bold'
)

estilo_seccion = ParagraphStyle(
    'CustomSection', parent=styles['Heading3'], fontSize=12,
    textColor=colors.HexColor('#1F4788'), spaceAfter=10,
    spaceBefore=10, fontName='Helvetica-Bold'
)

estilo_normal = ParagraphStyle(
    'CustomNormal', parent=styles['Normal'], fontSize=10,
    alignment=TA_JUSTIFY, spaceAfter=10
)

story = []

# ============================================================================
# PORTADA
# ============================================================================

story.append(Spacer(1, 2*inch))
story.append(Paragraph("INFORME DE ANÁLISIS", estilo_titulo))
story.append(Paragraph("Regresión Lineal: Análisis de Ingresos Laborales", estilo_titulo))
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("Dataset: ENAHO 2024", 
            ParagraphStyle('center', parent=styles['Normal'], 
                          fontSize=14, alignment=TA_CENTER)))
story.append(Paragraph("Encuesta Nacional de Hogares del Perú",
            ParagraphStyle('center', parent=styles['Normal'], 
                          fontSize=11, alignment=TA_CENTER, textColor=colors.grey)))
story.append(Spacer(1, 1*inch))

info_portada = [
    ['Fuente:', 'INEI (Instituto Nacional de Estadística e Informática)'],
    ['Período:', 'Año 2024'],
    ['Análisis:', 'Regresión Lineal Simple y Múltiple'],
    ['Variable Dependiente:', 'Ingreso Laboral Anual (Soles Peruanos)'],
    ['Fecha de Generación:', datetime.now().strftime('%d de %B de %Y')],
]

tabla_info = Table(info_portada, colWidths=[2*inch, 3*inch])
tabla_info.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_info)
story.append(PageBreak())

# ============================================================================
# ÍNDICE
# ============================================================================

story.append(Paragraph("ÍNDICE", estilo_titulo))
story.append(Spacer(1, 0.3*inch))

indice_items = [
    "1. Introducción y Descripción del Dataset",
    "2. Variables Utilizadas",
    "3. Análisis Exploratorio de Datos",
    "4. Regresión Lineal Simple",
    "5. Regresión Lineal Múltiple",
    "6. Comparación de Modelos y Métricas",
    "7. Interpretación de Resultados",
    "8. Conclusiones",
    "9. Referencias de Código Fuente",
]

for item in indice_items:
    story.append(Paragraph(item, estilo_normal))
    story.append(Spacer(1, 0.15*inch))

story.append(PageBreak())

# ============================================================================
# 1. INTRODUCCIÓN Y DATASET
# ============================================================================

story.append(Paragraph("1. Introducción y Descripción del Dataset", estilo_subtitulo))

story.append(Paragraph(
    "<b>Objetivo del Análisis:</b> Este informe presenta un análisis de regresión lineal "
    "para identificar y cuantificar los factores que influyen en el ingreso laboral anual "
    "de los trabajadores peruanos, utilizando datos de la Encuesta Nacional de Hogares (ENAHO) 2024.",
    estilo_normal
))

story.append(Paragraph(
    "<b>Fuente de Datos:</b> La Encuesta Nacional de Hogares (ENAHO) es la encuesta oficial del Perú "
    "realizada por el INEI. Proporciona información socioeconómica representativa a nivel nacional, "
    "regional y por dominios geográficos. Los datos utilizados corresponden al período 2024.",
    estilo_normal
))

story.append(Paragraph("<b>Características del Dataset:</b>", estilo_seccion))

stats_dataset = [
    ['Registros Originales:', f'{n_registros_totales:,}'],
    ['Registros Después del Filtrado:', f'{n_registros_finales:,}'],
    ['Variables Utilizadas:', f'{n_variables}'],
    ['Rango de Ingresos:', f"S/ {stats_ingreso['min']:,.0f} - S/ {stats_ingreso['max']:,.0f}"],
    ['Ingreso Promedio:', f"S/ {stats_ingreso['mean']:,.2f}"],
    ['Ingreso Mediano:', f"S/ {stats_ingreso['median']:,.2f}"],
]

tabla_stats = Table(stats_dataset, colWidths=[3*inch, 2.5*inch])
tabla_stats.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_stats)
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph(
    "<b>Filtros Aplicados:</b> Se eliminaron registros con valores extremos "
    "(ingresos menores a S/ 9,000 y mayores a S/ 40,000), se filtraron personas menores de 18 años "
    "o mayores de 70 años, y se removieron registros con datos faltantes en variables clave.",
    estilo_normal
))

story.append(PageBreak())

# ============================================================================
# 2. VARIABLES UTILIZADAS
# ============================================================================

story.append(Paragraph("2. Variables Utilizadas", estilo_subtitulo))

story.append(Paragraph(
    "El análisis incluye tanto variables demográficas como laborales, permitiendo "
    "comprender cómo diferentes factores impactan en el ingreso laboral anual:",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Variable Dependiente (Y):</b>", estilo_seccion))

var_dependiente = [
    ['Nombre', 'Descripción', 'Unidad'],
    ['ingreso_laboral_anual', 'Ingreso laboral percibido en el año', 'Soles (S/)'],
]

tabla_dep = Table(var_dependiente, colWidths=[2*inch, 2.5*inch, 1.5*inch])
tabla_dep.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_dep)

story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("<b>Variables Independientes (X):</b>", estilo_seccion))

variables_indep = [
    ['Nombre', 'Descripción', 'Tipo'],
    ['edad', 'Edad en años cumplidos', 'Cuantitativa'],
    ['sexo', 'Sexo del trabajador (1=Hombre, 2=Mujer)', 'Cualitativa'],
    ['nivel_educativo', 'Nivel educativo aprobado', 'Ordinal'],
    ['anios_educacion', 'Años de educación aprobados', 'Cuantitativa'],
    ['horas_trabajadas_semanal', 'Horas trabajadas por semana', 'Cuantitativa'],
    ['categoria_ocupacional', 'Categoría de ocupación', 'Cualitativa'],
    ['tipo_empleador', 'Tipo de empleador', 'Cualitativa'],
    ['tipo_contrato', 'Tipo de contrato laboral', 'Cualitativa'],
    ['tamano_empresa', 'Tamaño de la empresa', 'Ordinal'],
    ['ocupacion', 'Código de ocupación', 'Cualitativa'],
    ['estado_civil', 'Estado civil', 'Cualitativa'],
    ['anios_en_ocupacion', 'Años en la ocupación actual', 'Cuantitativa'],
]

tabla_indep = Table(variables_indep, colWidths=[2*inch, 2.5*inch, 1.5*inch])
tabla_indep.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 1), (0, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_indep)

story.append(PageBreak())

# ============================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS
# ============================================================================

story.append(Paragraph("3. Análisis Exploratorio de Datos", estilo_subtitulo))

story.append(Paragraph(
    "El análisis exploratorio inicial permite entender la estructura y características "
    "principales del dataset, identificar patrones y preparar los datos para el modelado.",
    estilo_normal
))

story.append(Spacer(1, 0.2*inch))

if os.path.exists('01_analisis_exploratorio.png'):
    story.append(Paragraph("Distribuciones de Variables", estilo_seccion))
    try:
        img = Image('01_analisis_exploratorio.png', width=5*inch, height=2.8*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

if os.path.exists('02_matriz_correlacion.png'):
    story.append(Paragraph("Matriz de Correlación", estilo_seccion))
    try:
        img = Image('02_matriz_correlacion.png', width=4.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

story.append(Paragraph(
    "<b>Hallazgos Principales:</b> El análisis exploratorio revela que existen correlaciones "
    "moderadas entre el nivel educativo, experiencia laboral y el ingreso. Las variables categóricas "
    "como tipo de contrato y tamaño de empresa también muestran influencia en los ingresos.",
    estilo_normal
))

story.append(PageBreak())

# ============================================================================
# 4. REGRESIÓN LINEAL SIMPLE
# ============================================================================

story.append(Paragraph("4. Regresión Lineal Simple", estilo_subtitulo))

story.append(Paragraph(
    "Se realizó una regresión lineal simple considerando el nivel educativo como única "
    "variable independiente. Este modelo establece una relación lineal fundamental entre "
    "educación e ingreso laboral.",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Modelo:</b>", estilo_seccion))
story.append(Paragraph(
    "Ingreso = β₀ + β₁ × Nivel_Educativo + ε",
    ParagraphStyle('formula', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER,
                  fontName='Courier')
))

story.append(Spacer(1, 0.2*inch))

if os.path.exists('03_regresion_simple.png'):
    story.append(Paragraph("Visualización del Modelo", estilo_seccion))
    try:
        img = Image('03_regresion_simple.png', width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

story.append(Paragraph(
    "<b>Resultados:</b> La regresión simple muestra que cada año adicional de educación "
    "se asocia con un aumento en el ingreso anual. Este modelo proporciona una baseline "
    "para comparación con modelos más complejos.",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Métricas de Desempeño - Datos de Entrenamiento:</b>", estilo_seccion))

metricas_simple_train = [
    ['Métrica', 'Valor'],
    ['MAE (Error Absoluto Promedio)', 'S/ 5,372.96'],
    ['MSE (Error Cuadrático Medio)', '45,779,170.54'],
    ['RMSE (Raíz del Error Cuadrático Medio)', 'S/ 6,766.03'],
    ['MAPE (Error Porcentual Absoluto Promedio)', '31.91%'],
    ['R² (Coeficiente de Determinación)', '0.1300'],
]

tabla_metricas = Table(metricas_simple_train, colWidths=[3.2*inch, 1.8*inch])
tabla_metricas.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_metricas)

story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("<b>Métricas de Desempeño - Datos de Prueba:</b>", estilo_seccion))

metricas_simple_test = [
    ['Métrica', 'Valor'],
    ['MAE (Error Absoluto Promedio)', 'S/ 5,107.64'],
    ['MSE (Error Cuadrático Medio)', '42,265,862.77'],
    ['RMSE (Raíz del Error Cuadrático Medio)', 'S/ 6,501.22'],
    ['MAPE (Error Porcentual Absoluto Promedio)', '30.56%'],
    ['R² (Coeficiente de Determinación)', '0.1247'],
]

tabla_metricas2 = Table(metricas_simple_test, colWidths=[3.2*inch, 1.8*inch])
tabla_metricas2.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_metricas2)

story.append(PageBreak())

# ============================================================================
# 5. REGRESIÓN LINEAL MÚLTIPLE
# ============================================================================

story.append(Paragraph("5. Regresión Lineal Múltiple", estilo_subtitulo))

story.append(Paragraph(
    "Se desarrolló un modelo de regresión lineal múltiple que incluye 14 variables independientes, "
    "junto con términos de interacción y variables dummy para capturar efectos no lineales y "
    "variaciones geográficas.",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Variables Incluidas:</b>", estilo_seccion))

story.append(Paragraph(
    "Base: edad, sexo, nivel_educativo, años_educación, experiencia, ocupación, "
    "categoría_ocupacional, tipo_empleador, tipo_contrato, tamaño_empresa, "
    "horas_trabajadas_semanal, estado_civil. Interacciones: edad×horas, sexo×educación. "
    "Dummies: 7 dominios geográficos.",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

if os.path.exists('04_regresion_multiple.png'):
    try:
        img = Image('04_regresion_multiple.png', width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

if os.path.exists('05_importancia_variables.png'):
    story.append(Paragraph("Ranking de Importancia de Variables", estilo_seccion))
    try:
        img = Image('05_importancia_variables.png', width=5*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

story.append(Paragraph(
    "<b>Importancia de Variables:</b> El análisis de coeficientes estandarizados identifica "
    "que el nivel educativo es la variable más importante, seguida por la edad y las horas "
    "trabajadas por semana.",
    estilo_normal
))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Métricas de Desempeño - Datos de Entrenamiento:</b>", estilo_seccion))

metricas_multi_train = [
    ['Métrica', 'Valor'],
    ['MAE (Error Absoluto Promedio)', 'S/ 4,847.32'],
    ['MSE (Error Cuadrático Medio)', '41,185,186.09'],
    ['RMSE (Raíz del Error Cuadrático Medio)', 'S/ 6,417.57'],
    ['MAPE (Error Porcentual Absoluto Promedio)', '26.88%'],
    ['R² (Coeficiente de Determinación)', '0.2173'],
]

tabla_m3 = Table(metricas_multi_train, colWidths=[3.2*inch, 1.8*inch])
tabla_m3.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_m3)

story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("<b>Métricas de Desempeño - Datos de Prueba:</b>", estilo_seccion))

metricas_multi_test = [
    ['Métrica', 'Valor'],
    ['MAE (Error Absoluto Promedio)', 'S/ 4,563.97'],
    ['MSE (Error Cuadrático Medio)', '37,655,693.25'],
    ['RMSE (Raíz del Error Cuadrático Medio)', 'S/ 6,136.42'],
    ['MAPE (Error Porcentual Absoluto Promedio)', '25.50%'],
    ['R² (Coeficiente de Determinación)', '0.2201'],
]

tabla_m4 = Table(metricas_multi_test, colWidths=[3.2*inch, 1.8*inch])
tabla_m4.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_m4)

story.append(PageBreak())

# ============================================================================
# 6. COMPARACIÓN DE MODELOS Y MÉTRICAS
# ============================================================================

story.append(Paragraph("6. Comparación de Modelos y Métricas", estilo_subtitulo))

story.append(Paragraph(
    "A continuación se presenta una comparación detallada entre el modelo simple y el modelo múltiple.",
    estilo_normal
))

story.append(Spacer(1, 0.2*inch))

if os.path.exists('06_comparacion_modelos.png'):
    try:
        img = Image('06_comparacion_modelos.png', width=5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

if os.path.exists('08_comparacion_r2.png'):
    try:
        img = Image('08_comparacion_r2.png', width=5*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    except:
        pass

story.append(Paragraph("<b>Tabla Comparativa - Datos de Prueba:</b>", estilo_seccion))

comparacion = [
    ['Métrica', 'Modelo Simple', 'Modelo Múltiple', 'Mejora'],
    ['MAE', 'S/ 5,107.64', 'S/ 4,563.97', '↓ 10.6%'],
    ['RMSE', 'S/ 6,501.22', 'S/ 6,136.42', '↓ 5.6%'],
    ['MAPE', '30.56%', '25.50%', '↓ 16.6%'],
    ['R²', '0.1247', '0.2201', '↑ 76.5%'],
]

tabla_comp = Table(comparacion, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 1.1*inch])
tabla_comp.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ('FONTNAME', (1, 1), (-1, -1), 'Courier'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 0), (-1, 0), [colors.HexColor('#1F4788')]),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabla_comp)

story.append(Spacer(1, 0.2*inch))

story.append(Paragraph(
    "<b>Conclusión Comparativa:</b> El modelo múltiple presenta un desempeño significativamente "
    "superior al modelo simple. El R² de 0.2201 indica que el modelo explica aproximadamente "
    "22% de la varianza en ingresos, lo que representa una mejora del 76.5% respecto al modelo simple. "
    "El MAPE del 25.50% sugiere un error promedio razonable.",
    estilo_normal
))

story.append(PageBreak())

# ============================================================================
# 7. INTERPRETACIÓN DE RESULTADOS
# ============================================================================

story.append(Paragraph("7. Interpretación de Resultados", estilo_subtitulo))

story.append(Paragraph(
    "Los resultados del análisis de regresión revelan insights importantes sobre los determinantes "
    "del ingreso laboral en el contexto peruano:",
    estilo_normal
))

story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("<b>a) Factor Educativo</b>", estilo_seccion))
story.append(Paragraph(
    "La educación emerge como el factor más importante en la determinación de ingresos. "
    "Cada nivel educativo adicional está asociado con aumentos significativos en el ingreso laboral.",
    estilo_normal
))

story.append(Paragraph("<b>b) Edad y Experiencia</b>", estilo_seccion))
story.append(Paragraph(
    "La edad (como proxy de experiencia) es el segundo factor más importante. La relación es "
    "principalmente positiva, pero con rendimientos decrecientes.",
    estilo_normal
))

story.append(Paragraph("<b>c) Horas de Trabajo</b>", estilo_seccion))
story.append(Paragraph(
    "Las horas trabajadas semanalmente muestran una relación positiva con los ingresos, "
    "aunque la magnitud del efecto es menor que la educación y edad.",
    estilo_normal
))

story.append(Paragraph("<b>d) Factores Ocupacionales</b>", estilo_seccion))
story.append(Paragraph(
    "El tipo de contrato, tamaño de empresa y tipo de empleador tienen impactos moderados. "
    "Los trabajadores con contratos permanentes en empresas grandes tienden a tener ingresos más altos.",
    estilo_normal
))

story.append(Paragraph("<b>e) Limitaciones del Modelo</b>", estilo_seccion))
story.append(Paragraph(
    "El R² del 0.2201 sugiere que aproximadamente el 78% de la varianza en ingresos se debe a factores "
    "no capturados por el modelo. Estos pueden incluir: capital social, conexiones empresariales, "
    "factores macroeconómicos, y características no observables.",
    estilo_normal
))

story.append(PageBreak())

# ============================================================================
# 8. CONCLUSIONES
# ============================================================================

story.append(Paragraph("8. Conclusiones", estilo_subtitulo))

story.append(Paragraph(
    "<b>1. Superioridad del Modelo Múltiple:</b> El modelo de regresión múltiple proporciona "
    "predicciones significativamente mejores que el modelo simple.",
    estilo_normal
))

story.append(Spacer(1, 0.08*inch))

story.append(Paragraph(
    "<b>2. Determinantes Principales:</b> La educación y la edad son los determinantes más importantes "
    "del ingreso laboral. Políticas enfocadas en mejorar acceso educativo podrían tener impactos significativos.",
    estilo_normal
))

story.append(Spacer(1, 0.08*inch))

story.append(Paragraph(
    "<b>3. Desempeño Predictivo:</b> Aunque el modelo explica el 22% de la varianza, otros factores "
    "no incluidos juegan roles importantes en la determinación de ingresos.",
    estilo_normal
))

story.append(Spacer(1, 0.08*inch))

story.append(Paragraph(
    "<b>4. Aplicabilidad Práctica:</b> Las métricas de error (MAE: S/ 4,563.97, RMSE: S/ 6,136.42) "
    "son razonables para predicción de ingresos en contextos de investigación socioeconómica.",
    estilo_normal
))

story.append(Spacer(1, 0.08*inch))

story.append(Paragraph(
    "<b>5. Futuras Investigaciones:</b> Se sugiere explorar modelos no lineales, incluir variables "
    "de capital social, considerar análisis por sector económico.",
    estilo_normal
))

story.append(PageBreak())

# ============================================================================
# 9. CÓDIGO FUENTE
# ============================================================================

story.append(Paragraph("9. Referencias de Código Fuente", estilo_subtitulo))

story.append(Paragraph(
    "El código fuente utilizado para este análisis se encuentra disponible en el siguiente enlace:",
    estilo_normal
))

story.append(Spacer(1, 0.3*inch))

story.append(Paragraph(
    "[AGREGUE AQUÍ EL ENLACE A SU REPOSITORIO DE CÓDIGO]",
    ParagraphStyle('link', parent=styles['Normal'], fontSize=11, 
                  alignment=TA_CENTER, textColor=colors.blue, fontName='Courier')
))

story.append(Spacer(1, 0.3*inch))

story.append(Paragraph("<b>Estructura de Archivos:</b>", estilo_seccion))

story.append(Paragraph(
    "• <b>filtrar_datos.py</b> - Script para consolidar datos de módulos ENAHO<br/>"
    "• <b>analisis_regresion.py</b> - Script principal con análisis de regresión<br/>"
    "• <b>enaho_2024_ingresos_individuales.csv</b> - Dataset procesado (17,281 registros)<br/>"
    "• <b>Gráficos:</b> 13 visualizaciones en formato PNG",
    estilo_normal
))

story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("<b>Librerías Utilizadas:</b>", estilo_seccion))

story.append(Paragraph(
    "pandas, numpy, scikit-learn, matplotlib, seaborn",
    estilo_normal
))

story.append(Spacer(1, 0.8*inch))

story.append(Paragraph(
    f"<b>Fecha de generación:</b> {datetime.now().strftime('%d de %B de %Y a las %H:%M')}",
    ParagraphStyle('footer', parent=styles['Normal'], fontSize=9, 
                  alignment=TA_CENTER, textColor=colors.grey)
))

# ============================================================================
# CONSTRUIR PDF
# ============================================================================

print(f"Guardando PDF: {pdf_filename}")
doc.build(story)

print(f"Informe PDF generado exitosamente: {pdf_filename}")
print(f"El informe está listo para usar.")
print(f"Instrucciones: Abra el PDF, busque '[AGREGUE AQUÍ EL ENLACE...]' y reemplace con su enlace.")
