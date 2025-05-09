# %%
import pandas as pd

df_skincare=pd.read_csv("product_info_skincare.csv")

# %%
df_skincare.head()

# %%
df_skincare["ingredients"]

# %%
df_skincare.columns

# %%
print("Shape:", df_skincare.shape)

# %%
# 5. Información general de columnas (tipos de datos, nulos)
print("\nInfo general:")
print(df_skincare.info())

# %%
# 6. Conteo de valores nulos por columna
print(df_skincare.isnull().sum().sort_values(ascending=False))


# %%
# 7. Conteo de filas duplicadas
print("\nFilas duplicadas:", df_skincare.duplicated().sum())

# %%
# 8. Convertir columnas de texto a minúsculas 
cols_texto = ['product_id', 'product_name', 'brand_id', 'brand_name', 'loves_count',
       'rating', 'reviews', 'size', 'variation_type', 'variation_value',
       'variation_desc', 'ingredients', 'price_usd', 'limited_edition', 'new',
       'online_only', 'out_of_stock', 'sephora_exclusive', 'highlights',
       'primary_category', 'secondary_category', 'tertiary_category',
       'child_count']

# Convertir a minúsculas y quitar espacios extra
for col in cols_texto:
    if col in df_skincare.columns:
        df_skincare[col] = df_skincare[col].astype(str).str.lower().str.strip()

# %%
df_skincare_copy=df_skincare.copy

# %%
#eliminar columnas innocesarias en análisis y muchos nulos 
df_skincare.drop(columns=["Unnamed: 0","value_price_usd", "sale_price_usd", "child_max_price", "child_min_price" ], inplace=True)

# %%
df_skincare.head(2)

# %%
df_skincare.columns

# %%
df_skincare.shape

# %%
#limpiar la columna de ingredientes 
# Eliminar la palabra 'water' (en cualquier posición y formato)
df_skincare["ingredients_clean"] = df_skincare["ingredients"].str.lower().str.replace(r'\bwater\b', '', regex=True)

# %%
df_skincare.columns

# %%
df_skincare.drop(columns=["ingredients"], inplace=True)

# %%
df_skincare.columns

# %%
#categorías de productos, crearcolumna . Ya hemos hecho la de ingredientes limpia y ahora esta. 

def categorize_product(name):
    name = name.lower()
    if "moisturizer" in name or "cream" in name or "lotion" in name:
        return "moisturizer"
    elif "cleanser" in name or "wash" in name or "soap" in name:
        return "cleanser"
    elif "toner" in name or "mist" in name:
        return "toner"
    elif "serum" in name or "ampoule" in name:
        return "serum"
    elif "mask" in name or "sheet" in name:
        return "mask"
    elif "sunscreen" in name or "spf" in name:
        return "sunscreen"
    elif "scrub" in name or "exfoli" in name or "peel" in name:
        return "exfoliant"
    elif "eye" in name:
        return "eye cream"
    else:
        return "other"

df_skincare["product_category"] = df_skincare["product_name"].apply(categorize_product)


# %%
df_skincare["product_category"]

# %%
df_skincare.info()

# %%
df_skincare.dtypes

# %%
# Lista de columnas que queremos convertir
cols_to_convert = ['price_usd', 'loves_count', 'rating', 'reviews']

# Convertimos a float usando pd.to_numeric con errores ignorados (valores inválidos se convierten en NaN)
df_skincare[cols_to_convert] = df_skincare[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Verificamos los nuevos tipos
print(df_skincare[cols_to_convert].dtypes)

# %%
df_skincare.dtypes

# %%
# Lista de columnas actual
cols = list(df_skincare.columns)

# Remover la columna que quieres mover
cols.remove('product_category')

# Insertarla en la posición deseada (por ejemplo, después de 'columna_1')
pos = cols.index('product_name') + 1
cols.insert(pos, 'product_category')

# %%
df_skincare = df_skincare[cols]

# %%
cols_2 = list(df_skincare.columns)

# Remover la columna que quieres mover
cols_2.remove('ingredients_clean')

# Insertarla en la posición deseada (por ejemplo, después de 'columna_1')
pos_2 = cols_2.index('brand_name') + 1
cols_2.insert(pos_2, 'ingredients_clean')

# %%
df_skincare = df_skincare[cols_2]

# %%
df_skincare.columns

# %%
df_skincare['price_category'] = pd.cut(df_skincare['price_usd'],
                                       bins=[0, 25, 60, 120, float('inf')],
                                       labels=['gama baja', 'gama media', 'gama alta', 'lujo'])


# %%
df_skincare.columns

# %%
cols_3 = list(df_skincare.columns)

# Remover la columna que quieres mover
cols_3.remove('price_category')

# Insertarla en la posición deseada (por ejemplo, después de 'columna_1')
pos_3 = cols_3.index('ingredients_clean') + 1
cols_3.insert(pos_3, 'price_category')

# %%
df_skincare = df_skincare[cols_3]

# %%
df_skincare.columns

# %%
df_skincare["price_category"].value_counts()

# %%
df_skincare.shape

# %% [markdown]
# # ANÁLISIS

# %%
#análisis más en profundidad de los ingredientes 
from collections import Counter

# 1. Convertir todos los ingredientes en una sola lista
all_ingredients = df_skincare["ingredients_clean"].str.split(', ').explode()



# %%
# 2. Contar ocurrencias
#Usa collections.Counter, una clase de Python que cuenta cuántas veces aparece cada valor.
#Aplica esto sobre la serie all_ingredients.
#Resultado: un diccionario donde:
#Clave = ingrediente
#Valor = número de veces que aparece en el dataset

ingredient_counts = Counter(all_ingredients)



# %%
# 3. Pasar a DataFrame ordenado
df_frecuencias = (
    pd.DataFrame(ingredient_counts.items(), columns=["ingredient", "count"])
    .sort_values(by="count", ascending=False)
    .reset_index(drop=True)
)

# Mostrar los 10 ingredientes más comunes
df_frecuencias.head(10)

# %%
pip install seaborn

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Tomamos los 10 ingredientes más comunes
top_10 = df_frecuencias.head(10)

# Estilo de seaborn
sns.set(style="whitegrid")

# Crear el gráfico
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x="count", y="ingredient", palette="viridis")

# Añadir etiquetas y título
plt.title("Top 10 ingredientes más frecuentes", fontsize=14)
plt.xlabel("Frecuencia")
plt.ylabel("Ingrediente")

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# %%
# Contar la cantidad de productos de cada marca
#las marcas con más productos son: bumble and bumble	67, kérastase	60,oribe	42, briogeo	38, virtue	32
most_count_brand=df_skincare.groupby(['product_category', 'brand_name']).size().reset_index(name='product_count').sort_values(by='product_count',ascending=False)
most_count_brand

#sacar el porcentaje de estas marcas respecto al total

# %%
most_count_product_category= most_count_brand.groupby('product_category')['product_count'].idxmax().sort_values(ascending=False)
most_count_product_category

# %%
#Top 10 brand por número de 'loves'. 
top_10_brands= df_skincare.groupby(['brand_name', 'product_category'])['loves_count'].sum().sort_values(ascending=False).head(10).reset_index(name="loves count")
top_10_brands

# %%
#Top 10 product category por número de 'loves'. 
top_10_product_category= df_skincare.groupby(['product_category'])['loves_count'].sum().sort_values(ascending=False).head(10).reset_index(name="loves count")
top_10_product_category

# %% [markdown]
# # Por cada categoría de producto qué marca es la más popular?  Primero en absolutos top 10 y luego su % y gráfica

# %%
#hacemos una función para que en cada categoría de producto nos saque la marca.
#la marca más popular en base al número de love_counts count . 

def top_brands_by_category(df_skincare, top_n=10):
    # Obtener categorías únicas
    categorias = df_skincare['product_category'].dropna().unique()

    # Diccionario para guardar resultados
    resultados = {}

    for categoria in categorias:
        # se queda con las filas donde product_category es igual a la categoría del bucle.
        df_filtrado = df_skincare[df_skincare['product_category'] == categoria]

        # Agrupar por marca y sumar 'loves_count'
        top_marcas = (
            df_filtrado.groupby('brand_name')['loves_count']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        # Guardar en el diccionario
        resultados[categoria] = top_marcas

    return resultados

top_marcas_por_categoria = top_brands_by_category(df_skincare)

# Por ejemplo, mostrar el top 10 para 'cleanser'
print("Top marcas para 'cleanser':")
print(top_marcas_por_categoria['cleanser'])


# %%
print("Top marcas para 'eye cream':")
print(top_marcas_por_categoria['eye cream'])


# %%
print("Top marcas para 'exfoliant':")
print(top_marcas_por_categoria['exfoliant'])

# %%
print("Top marcas para 'serum':")
print(top_marcas_por_categoria['serum'])

# %%
print("Top marcas para 'sunscreen':")
print(top_marcas_por_categoria['sunscreen'])

# %%
print("Top marcas para 'toner':")
print(top_marcas_por_categoria['toner'])

# %%
print("Top marcas para 'moisturizer':")
print(top_marcas_por_categoria['moisturizer'])


# %%
print("Top marcas para 'mask':")
print(top_marcas_por_categoria['mask'])


# %%
print("Top marcas para 'other':")
print(top_marcas_por_categoria['other'])

# %%
#rehacemos la función para que nos saque tmabién el gráfico pastel con porcentaje y números enteros. 
import matplotlib.pyplot as plt

def pastel_top10_marcas_por_categoria(df_skincare, categoria, top_n=10):
    """
    Muestra un gráfico pastel con el top N marcas más populares por loves_count dentro de una categoría,
    incluyendo el porcentaje y el valor absoluto.
    """
    # Filtrar el DataFrame por la categoría deseada
    df_filtrado_2 = df_skincare[df_skincare['product_category'] == categoria]
    
    # Agrupar por marca y sumar loves_count
    top_marcas_2 = (
        df_filtrado_2.groupby('brand_name')['loves_count']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Calcular el total del top N
    total_top = top_marcas_2.sum()

    # Función para etiquetas con porcentaje + número
    def etiqueta(pct, allvals):
        valor_abs = int(round(pct / 100. * total_top))
        return f"{pct:.1f}%\n({valor_abs:,})"

    # Crear gráfico pastel
    plt.figure(figsize=(9, 9))
    plt.pie(
        top_marcas_2,
        labels=top_marcas_2.index,
        autopct=lambda pct: etiqueta(pct, top_marcas_2),
        startangle=140,
        colors=plt.cm.tab20.colors
    )

    plt.title(f"Top {top_n} marcas por loves_count en '{categoria}'", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# %%
pastel_top10_marcas_por_categoria(df_skincare, 'cleanser')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'eye cream')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'exfoliant')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'serum')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'sunscreen')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'toner')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'moisturizer')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'mask')

# %%
pastel_top10_marcas_por_categoria(df_skincare, 'other')

# %%
#Top 10 productos por número de 'loves'. 
top_loved = df_skincare[['product_name', 'brand_name', 'product_category',"price_category", 'loves_count']].sort_values(by='loves_count', ascending=False).head(10)
top_loved

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo
sns.set(style="whitegrid")

# Crear figura
plt.figure(figsize=(12, 6))

# Crear gráfico de barras
ax = sns.barplot(data=top_loved, 
                 x='loves_count', 
                 y='product_name', 
                 hue='price_category', 
                 dodge=False, 
                 palette='Set2')

# Título y etiquetas
plt.title("Top 10 productos más populares (por loves_count)", fontsize=14)
plt.xlabel("Número de 'loves'")
plt.ylabel("Nombre del producto")

# Añadir los valores de cada barra
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', 
                (p.get_width() + 5000, p.get_y() + p.get_height() / 2), 
                ha='center', va='center', 
                color='black', fontsize=12)

# Ajustar diseño
plt.tight_layout()
plt.legend(title="Categoría de precio", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%
# Top 10 productos por reviews
top_reviewed = df_skincare[['product_name', 'brand_name', 'product_category',"price_category", 'reviews']].sort_values(by='reviews', ascending=False).head(10)
top_reviewed 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo
sns.set(style="whitegrid")

# Crear figura
plt.figure(figsize=(12, 6))

# Crear gráfico de barras
ax = sns.barplot(data=top_reviewed, 
                 x='reviews', 
                 y='product_name', 
                 hue='price_category', 
                 dodge=False, 
                 palette='Set2')

# Título y etiquetas
plt.title("Top 10 productos más populares por número de 'reviews'", fontsize=14)
plt.xlabel("Número de 'reviews'")
plt.ylabel("Nombre del producto")

# Añadir los valores de cada barra
for p in ax.patches:
    width = p.get_width()
    if width > 0:  # Asegurarnos de que solo se anoten barras con valores mayores que 0
        ax.annotate(f'{int(width)}', 
                    (width + 5000, p.get_y() + p.get_height() / 2), 
                    ha='center', va='center', 
                    color='black', fontsize=12)

# Ajustar diseño
plt.tight_layout()
plt.legend(title="Categoría de precio", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%
price_analysis=df_skincare.groupby('price_category')[['loves_count', 'reviews']].mean().round(2).sort_values(by='loves_count',ascending=False)
price_analysis

# %%
#El porcentaje de cada categoría de precio vs el total de loves_count . Qué supone cada una.
# Paso 1: calcular el total de 'loves_count'
total_loves = df_skincare['loves_count'].sum()

# Paso 2: agrupar por categoría y sumar los 'loves_count'
price_loves = df_skincare.groupby('price_category')['loves_count'].sum()

# Paso 3: calcular el porcentaje
price_percent = (price_loves / total_loves) * 100

# Paso 4: redondear y ordenar
price_percent = price_percent.round(2).sort_values(ascending=False)
price_percent

# %%
plt.figure(figsize=(8, 6))
plt.pie(price_percent, labels=price_percent.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de loves_count por Rango de Precio')
plt.axis('equal')
plt.show()

# %%
category_analysis=df_skincare.groupby('product_category')[['loves_count', "rating",'reviews']].mean().round(2).sort_values(by='loves_count',ascending=False)
category_analysis

# %%
import matplotlib.pyplot as plt

# Datos: promedio de loves_count por categoría
category_analysis = df_skincare.groupby('product_category')[['loves_count', "rating", 'reviews']].mean().round(2).sort_values(by='loves_count', ascending=False)

# Filtrar solo la columna de loves_count
loves_mean = category_analysis['loves_count']

# Crear gráfico de pastel
plt.figure(figsize=(8, 8))
plt.pie(loves_mean, labels=loves_mean.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución media de loves_count por categoría de producto')
plt.axis('equal')  # Para que el gráfico sea un círculo
plt.tight_layout()
plt.show()


# %%
#El porcentaje de cada categoría de producto vs el total de loves_count . Qué supone cada una.
# Paso 1: calcular el total de 'loves_count'
total_loves = df_skincare['loves_count'].sum()

# Paso 2: agrupar por categoría y sumar los 'loves_count'
category_loves = df_skincare.groupby('product_category')['loves_count'].sum()

# Paso 3: calcular el porcentaje
category_percent = (category_loves / total_loves) * 100

# Paso 4: redondear y ordenar
category_percent = category_percent.round(2).sort_values(ascending=False)


# %%
category_percent 

# %%
plt.figure(figsize=(8, 6))
plt.pie(category_percent, labels=category_percent.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de loves_count por Categoría de Producto')
plt.axis('equal')  # para que sea un círculo perfecto
plt.show()

# %%
# Filtrar el dataset por la categoría "other" en la columna "product_category"
df_other = df_skincare[df_skincare['product_category'] == 'other']

# Mostrar las primeras filas del nuevo dataset filtrado
df_other.head()

# %%
other_analysis=df_other.groupby('primary_category')[['loves_count','reviews']].mean().round(2).sort_values(by='loves_count',ascending=False)
other_analysis

# %%
# Filtrar el dataset por la categoría "skincare" en la columna "product_category"
df_skincare_primary_category = df_skincare[df_skincare['primary_category'] == 'skincare']

# Mostrar las primeras filas del nuevo dataset filtrado
df_skincare_primary_category.head()

# %%
df_skincare['primary_category'].value_counts()

# %% [markdown]
# # CSVs reviews 

# %%
#primer data set de reviews 
df_reviews_250=pd.read_csv("reviews_0_250_masked.csv")

# %%
df_skincare.to_csv("skincare_limpio_ok.csv")

# %%
df_reviews_250.columns

# %%
df_reviews_250.dtypes

# %%
df_reviews_250.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'rating', 'is_recommended', 'helpfulness','submission_time', 'review_text',
       'review_title', 'skin_tone', 'eye_color', 'skin_type', 'hair_color' ], inplace=True)

# %%
df_reviews_250.columns

# %%
# 8. Convertir columnas de texto a minúsculas 
cols_texto_2 = ['total_feedback_count', 'total_neg_feedback_count',
       'total_pos_feedback_count', 'product_id', 'product_name', 'brand_name',
       'price_usd']

# Convertir a minúsculas y quitar espacios extra
for col in cols_texto_2:
    if col in df_skincare.columns:
        df_skincare[col] = df_skincare[col].astype(str).str.lower().str.strip()

# %%
df_reviews_250.head()

# %%
df_reviews_250.nunique()

# %%
# Eliminar filas donde total_feedback_count es 0
df_reviews_250 = df_reviews_250[df_reviews_250['total_feedback_count'] != 0]

# Opcional: guardar el nuevo DataFrame en un archivo CSV
df_reviews_250.to_csv('reviews_250_filtrado.csv', index=False)

# %%
df_reviews_250.head(3)

# %%
df_reviews_250.shape

# %%
#todas las filas en minúsuclas 
for col in df_reviews_250.select_dtypes(include='object').columns:
    df_reviews_250[col] = df_reviews_250[col].str.lower()

# %%
df_reviews_250.head(3)

# %%
df_reviews_250_reducido=df_reviews_250[['product_name', 'total_feedback_count', 'total_neg_feedback_count', 'total_pos_feedback_count']]
df_reviews_250_reducido.head(3).reset_index()

# %%
#segundo data set de reviews 
df_reviews_250=pd.read_csv("reviews_0_250_masked.csv")

# %%
import pandas as pd

# Función de limpieza y estandarización
def limpiar_dataset(path_csv):
    df = pd.read_csv(path_csv)

    # Convertir a minúsculas todas las columnas de texto
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()

    # Filtrar solo las columnas necesarias
    columnas_necesarias = ['total_feedback_count', 'total_neg_feedback_count', 'total_pos_feedback_count','product_id']
    df = df[columnas_necesarias]

    # Eliminar filas con total_feedback_count igual a 0
    df = df[df['total_feedback_count'] != 0]

    # Resetear el índice
    df = df.reset_index(drop=True)

    return df

# Cargar y limpiar los 5 datasets
df1 = limpiar_dataset('reviews_0_250_masked.csv')
df2 = limpiar_dataset('reviews_250-500_masked.csv')
df3 = limpiar_dataset('reviews_500-750_masked.csv')
df4 = limpiar_dataset('reviews_750-1250_masked.csv')
df5 = limpiar_dataset('reviews_1250-end_masked.csv')

# Unir todos los DataFrames en uno solo
df_combinado_reviews = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# (Opcional) Guardar el dataset combinado
df_combinado_reviews.to_csv('reviews_combinado.csv', index=False)


# %%
df_combinado_reviews=df_combinado_reviews.groupby("product_id").agg({"total_neg_feedback_count":"sum","total_pos_feedback_count":"sum"}).reset_index()
df_combinado_reviews

# %%
df_combinado_reviews['total_feedback_count']=df_combinado_reviews['total_neg_feedback_count']+ df_combinado_reviews['total_pos_feedback_count']
df_combinado_reviews

# %%
# Unir los DataFrames con merge (left join)
df_merged = pd.merge(df_skincare, df_combinado_reviews, on='product_id', how='left')
df_merged.head(3)

# %%
df_merged.isnull().sum()

# %%
#poner 0 a todos los nulos! de mi data set final que es el merge 
# Convertir columnas categóricas a string (texto)
for col in df_merged.select_dtypes(include='category').columns:
    df_merged[col] = df_merged[col].astype(str)

# Reemplazar nulos por 0
df_merged = df_merged.fillna(0)


# %%
df_merged.to_csv("df_merged.csv")

# %%
df_merged.isnull().sum()

# %%
# Top 10 productos por total_pos_feedback_count
top_positive_feedback = df_merged.sort_values(by='total_pos_feedback_count', ascending=False).head(10)
top_positive_feedback

# %%
top_loved_positive_feedback = df_merged[['product_name', 'brand_name', 'product_category',"price_category", 'loves_count','total_pos_feedback_count']].sort_values(by='total_pos_feedback_count', ascending=False).head(10).reset_index()
top_loved_positive_feedback

# %%
# Crear una nueva columna que combine ambos criterios
df_merged['popularity_score'] = df_merged['loves_count'] + df_merged['total_pos_feedback_count']

# Top 10 productos más populares por score combinado
top_popular_combined = df_merged[['product_name', 'brand_name', 'product_category',"price_category",'popularity_score']].sort_values(by='popularity_score', ascending=False).head(10)
top_popular_combined 

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Preparar el DataFrame top 10 productos populares
top_popular_combined = df_merged[['product_name', 'brand_name', 'product_category', 'price_category', 'popularity_score']] \
    .sort_values(by='popularity_score', ascending=False).head(10)

# Crear etiquetas combinadas de producto + marca
top_popular_combined['producto_marca'] = top_popular_combined['product_name'] + ' (' + top_popular_combined['brand_name'] + ')'

# Calcular el total de popularity_score en el top 10
total_score_top10 = top_popular_combined['popularity_score'].sum()

# Calcular el porcentaje que representa cada producto
top_popular_combined['porcentaje'] = (top_popular_combined['popularity_score'] / total_score_top10) * 100

# Crear gráfico de barras horizontal
plt.figure(figsize=(10, 6))
bars = plt.barh(top_popular_combined['producto_marca'], top_popular_combined['popularity_score'], color='skyblue')
plt.xlabel('Popularity Score')
plt.title('Top 10 Productos Más Populares (Loves + Feedback Positivo)')
plt.gca().invert_yaxis()  # El más popular arriba

# Desactivar notación científica
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.ticklabel_format(style='plain', axis='x')

# Añadir texto con el porcentaje al final de cada barra
for bar, porcentaje in zip(bars, top_popular_combined['porcentaje']):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{porcentaje:.1f}%', va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.show()


# %%
df_merged.columns

# %%
#rehacemos la función para que nos saque tmabién el gráfico pastel con porcentaje y números enteros. 
#más populares según la nueva variable popularity_score
import matplotlib.pyplot as plt

def pastel_top_marcas_por_categoria_popularity(df_merged, categoria, top_n=10):
    """
    Muestra un gráfico pastel con el top N marcas más populares por loves_count dentro de una categoría,
    incluyendo el porcentaje y el valor absoluto.
    """
    # Filtrar el DataFrame por la categoría deseada
    df_filtrado_3 = df_merged[df_merged['product_category'] == categoria]
    
    # Agrupar por marca y sumar loves_count
    top_marcas_3 = (
        df_filtrado_3.groupby('brand_name')['popularity_score']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Calcular el total del top N
    total_top = top_marcas_3.sum()

    # Función para etiquetas con porcentaje + número
    def etiqueta(pct, allvals):
        valor_abs = int(round(pct / 100. * total_top))
        return f"{pct:.1f}%\n({valor_abs:,})"

    # Crear gráfico pastel
    plt.figure(figsize=(9, 9))
    plt.pie(
        top_marcas_3,
        labels=top_marcas_3.index,
        autopct=lambda pct: etiqueta(pct, top_marcas_3),
        startangle=140,
        colors=plt.cm.tab20.colors
    )

    plt.title(f"Top {top_n} marcas por popularity_score en '{categoria}'", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# %%
df_merged['product_category'].value_counts()

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'other')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'mask')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'moisturizer')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'eye cream')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'exfoliant')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'sunscreen')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'toner')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'cleanser')

# %%
pastel_top_marcas_por_categoria_popularity(df_merged, 'serum')

# %%
# Top 10 productos con más feedback negativo
top_hate = df_merged[['product_name', 'brand_name', 'product_category',"price_category",'popularity_score','total_neg_feedback_count']].sort_values(by='total_neg_feedback_count', ascending=False).head(10)
top_hate

# %%
price_analysis_popularity=df_merged.groupby('price_category')[['popularity_score']].count().round(2).sort_values(by='popularity_score',ascending=False)
price_analysis_popularity

# %%
#del total de reviews hacer un varemo del total de reviews ver qué porcentaje es positivo y negativo
#elegir el producto de menos riesgo porque la diferencia entre el porcentaje de los positivos y negativos se aleje 

# %% [markdown]
# Objetivo: Elegir3 productos con el menor riesgo y mayor popularidad
# 
# Utilizaremos:
# 
# popularity_score para medir demanda/potencial.
# La diferencia entre comentarios positivos y negativos, en porcentajes, como medida de riesgo.

# %% [markdown]
# Paso 1: Popularidad por categoría (product_category)
# 
# Calculamos la suma total del popularity_score por categoría y lo convertimos a porcentaje del total

# %%
#popularidad total por categoría
pop_cat = df_merged.groupby('product_category')['popularity_score'].sum()

# Convertir a porcentaje
pop_cat_pct = (pop_cat / pop_cat.sum()) * 100

# Redondear y ajustar el total a 100%
pop_cat_pct = pop_cat_pct.round(1)
ajuste = 100 - pop_cat_pct.sum()
pop_cat_pct.iloc[-1] += ajuste  # Ajustar el último para que sume 100%

# %%
pop_cat_pct

# %% [markdown]
# Paso 2: Análisis de riesgo (positivos vs negativos)

# %%
# Agrupar por categoría y sumar feedback
feedback = df_merged.groupby('product_category')[['total_pos_feedback_count', 'total_neg_feedback_count']].sum()

# Calcular diferencia positiva neta
feedback['neto'] = feedback['total_pos_feedback_count'] - feedback['total_neg_feedback_count']

# Calcular el porcentaje que representa cada diferencia neta dentro del total
feedback['porcentaje_neto'] = (feedback['neto'] / feedback['neto'].sum()) * 100
feedback['porcentaje_neto'] = feedback['porcentaje_neto'].round(1)
ajuste = 100 - feedback['porcentaje_neto'].sum()
feedback['porcentaje_neto'].iloc[-1] += ajuste


# %%
feedback

# %% [markdown]
# Paso 3: Unimos los dos análisis

# %%
# Unimos ambos análisis en un solo DataFrame
resultado = pd.concat([pop_cat_pct.rename('popularidad_%'), 
                       feedback['porcentaje_neto'].rename('riesgo_% (positivo neto)')], axis=1)

# Ordenamos por menor riesgo (más diferencia positiva)
resultado_ordenado = resultado.sort_values(by='riesgo_% (positivo neto)', ascending=False)

# Mostrar top 3 recomendaciones
top_3_productos = resultado_ordenado.head(3)
print("🔝 Recomendación de productos con menor riesgo y más potencial:")
print(top_3_productos)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Datos para el gráfico
categorias = resultado_ordenado.index
popularidad = resultado_ordenado['popularidad_%']
riesgo = resultado_ordenado['riesgo_% (positivo neto)']

x = np.arange(len(categorias))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, popularidad, width, label='Popularidad (%)', color='skyblue')
plt.bar(x + width/2, riesgo, width, label='Diferencia Positiva (%)', color='lightgreen')

plt.xlabel('Categoría de producto')
plt.ylabel('Porcentaje')
plt.title('Popularidad vs Riesgo por Categoría de Producto')
plt.xticks(x, categorias, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# %%
df_merged['price_usd'] = pd.to_numeric(df_merged['price_usd'], errors='coerce')

# %%
# Calcular precio medio por categoría
precio_medio = df_merged.groupby('product_category')['price_usd'].mean().round(2)

# Añadir al DataFrame anterior
resultado['precio_medio'] = precio_medio

# Reordenar columnas (opcional)
resultado = resultado[['popularidad_%', 'riesgo_% (positivo neto)', 'precio_medio']]

# Reordenar por menor riesgo (más diferencia positiva)
resultado_ordenado = resultado.sort_values(by='riesgo_% (positivo neto)', ascending=False)

# Mostrar el top 3 de categorías recomendadas
top_3_productos = resultado_ordenado.head(3)
print("🔝 Recomendación de categorías con menor riesgo y mayor potencial:")
print(top_3_productos)

# %%
top_3_productos.to_csv("top3_categoria_productos.csv", index=False)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos
data = {
    'product_category': ['mask', 'cleanser', 'moisturizer'],
    'popularidad_%': [26.4, 12.5, 9.3],
    'riesgo_%': [43.1, 24.9, 16.2],
    'precio_medio_usd': [28.7, 22.5, 34.1]
}

df = pd.DataFrame(data)

# Gráfico combinado
fig, ax1 = plt.subplots(figsize=(9, 6))

# Barras para popularidad y riesgo
df_melt = df.melt(id_vars='product_category', 
                  value_vars=['popularidad_%', 'riesgo_%'], 
                  var_name='Indicador', value_name='Porcentaje')
sns.barplot(data=df_melt, x='product_category', y='Porcentaje', hue='Indicador', palette='Set2', ax=ax1)

# Eje secundario para el precio medio
ax2 = ax1.twinx()
sns.lineplot(data=df, x='product_category', y='precio_medio_usd', color='black', marker='o', linewidth=2, label='Precio Medio (USD)', ax=ax2)
ax2.set_ylabel("Precio Medio (USD)", fontsize=10)
ax2.legend(loc='upper right')

# Estilo y títulos
ax1.set_title("🔝 Recomendación: Popularidad, Bajo Riesgo y Precio Medio por Categoría", fontsize=13)
ax1.set_ylabel("Porcentaje (%)")
ax1.set_xlabel("Categoría de Producto")
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, 50)

plt.tight_layout()
plt.show()


# %% [markdown]
# Ahora sabemos cuales son las tres mejores categorías donde invertir. 
# Vamos a centrarnos en la categoría mask y que top 5 productos debemos "imitar".
# Recordemos que las gamas según rpecio: 0, 25, 60, 120, infinito. 'gama baja', 'gama media', 'gama alta', 'lujo'

# %%
#Nos centraremos en la categoría mask 
#Filtrar solo productos de la categoría "mask"
df_mask = df_merged[df_merged['product_category'] == 'mask']
#Agrupar por product_name para calcular métricas
top5_mask = (
    df_mask.groupby('product_name')
    .agg({
        'popularity_score': 'sum',
        'total_pos_feedback_count': 'sum',
        'total_neg_feedback_count': 'sum',
        'price_usd': 'mean',
        'brand_name': 'first'
    })
    .reset_index()
)

# %%
#Seleccionar el top 5 por popularidad
top5_mask = top5_mask.sort_values(by='popularity_score', ascending=False).head(5)


# %%
top5_mask

# %%
#Calcular indicadores porcentuales
# Diferencia positiva
top5_mask['neto'] = top5_mask['total_pos_feedback_count'] - top5_mask['total_neg_feedback_count']

# Porcentajes
top5_mask['popularidad_%'] = (top5_mask['popularity_score'] / top5_mask['popularity_score'].sum()) * 100
top5_mask['riesgo_% (positivo neto)'] = (top5_mask['neto'] / top5_mask['neto'].sum()) * 100

# Redondear y ajustar a 100%
top5_mask['popularidad_%'] = top5_mask['popularidad_%'].round(1)
top5_mask['riesgo_% (positivo neto)'] = top5_mask['riesgo_% (positivo neto)'].round(1)

# Ajustes para que sumen 100%
ajuste_pop = 100 - top5_mask['popularidad_%'].sum()
ajuste_riesgo = 100 - top5_mask['riesgo_% (positivo neto)'].sum()
top5_mask.loc[top5_mask.index[-1], 'popularidad_%'] += ajuste_pop
top5_mask.loc[top5_mask.index[-1], 'riesgo_% (positivo neto)'] += ajuste_riesgo

# Redondear precio
top5_mask['precio_medio'] = top5_mask['price_usd'].round(2)


# %%
import matplotlib.pyplot as plt
import numpy as np

top5_mask['producto_marca'] = top5_mask['product_name'] + ' (' + top5_mask['brand_name'] + ')'
x = np.arange(len(top5_mask))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, top5_mask['popularidad_%'], width, label='Popularidad (%)', color='skyblue')
plt.bar(x + width/2, top5_mask['riesgo_% (positivo neto)'], width, label='Diferencia Positiva (%)', color='lightgreen')

plt.xlabel('Producto (marca)')
plt.ylabel('Porcentaje')
plt.title('Top 5 Productos de la Categoría "Mask"')
plt.xticks(x, top5_mask['producto_marca'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# %%
top5_mask.to_csv("top5_productos_mask.csv", index=False)

# %%
df_merged.head()

# %% [markdown]
# Ver dentro de la categoría mask cuáles se venden solo online y cuales edicion limitada. Ver su popularidad, a ver si van de la mano o no . 

# %%
# Filtrar solo productos de la categoría 'mask'
mask_df = df_merged[df_merged['product_category'] == 'mask'].copy()

# Agrupar por 'online_only' y calcular la popularidad media
mask_online_pop = mask_df.groupby('online_only')['popularity_score'].mean().reset_index()
mask_online_pop

# %%
# Agrupar por 'limited_edition' y calcular la popularidad media
mask_limited_pop = mask_df.groupby('limited_edition')['popularity_score'].mean().reset_index()
mask_limited_pop


