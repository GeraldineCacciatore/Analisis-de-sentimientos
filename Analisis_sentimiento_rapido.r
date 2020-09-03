### ---- Análisis ---- ###
# En este trabajo vamos a hacer un análisis de texto de un dataset que contiene titulares de noticias financieras, con el objetivo de encontrar las palabras con más repetición según el tipo de sentimiento, y crear un modelo bayeiano que nos prediga el tipo de sentimiento que le genera el titular al lector. 

### ---- Lectura de los datos ---- ###
#He descargado los datos de  [Kaggle](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news/kernels) y los guardé en mi directorio de la materia. Para abrirlos usamos la función read_csv (no se utilizó la funcion read.csv porque generaba errores).
#El dataset contiene dos columnas a las que renombraremos como 'sentimiento' y 'titular'. El sentimiento puede ser negativo, neutral o positivo. 

# -- Cargo librerias
library(readr)

# -- Cargo datos
all_data <- read_csv('all-data.csv',local = locale(encoding = "latin1"))

# -- Cambiamos el nombre de las variables
names(all_data) <- c('sentimiento', 'titular')

# -- Vemos la dimensión
#Es un dataset pequeño
dim(all_data) 

# -- Un vistazo
summary(all_data)

# -- Pasamos la variable **sentimiento** de clase character a factor
all_data$sentimiento = factor(all_data$sentimiento)

# -- Vemos más en detalle los valores de la columna 'sentimiento'
table(all_data$sentimiento, useNA = 'ifany')
#Vemos que este dataset no esta balanceado

# -- Vemos los resultados gráficamente 
library(ggplot2)
ggplot(all_data, aes(x = all_data$sentimiento, fill = factor(all_data$sentimiento))) + geom_bar()

#Podemos observar claramente que el dataset esta desbalanceado. Lo que estamos diciendo es que hay bastantes más valores **neutral** (59%) que **positive** (28%) y **negative** (13%). 
#Hay que tener en cuenta esto, porque es importante al momento de dividir un dataset en train y test, o al aplicar algunos algoritmos. 

# -- ETL y Corpus
#Ahora vamos a usar la librería **tm**, entre otras cosas para crear un corpus. Para usar la función **corpus**, le pasamos el valor de all_data$titular con **VectorSource()** para decirle que cada vector ha de ser tratado como un documento
library(tm)
datos_corpus <- Corpus(VectorSource(all_data$titular))

#Vemos la clase y lo que contiene
class(datos_corpus)

#Los cinco primeros elementos
inspect(datos_corpus[1:5])

#El tamaño
print(datos_corpus)

#Ahora vamos a usar la función **tm_map** para mejorar el corpus y guardarlo en un nuevo objeto: **corpus_clean**. 
#Esta función tiene distintas funcionalidades. Empezamos por pasar el texto a minúsculas y ver si es necesario para éste análisis quitar los números (en principio las dejaremos). 

# -- Pasar a minúsculas
corpus_clean <- tm_map(datos_corpus, tolower)

#Vemos cómo quedo
inspect(corpus_clean[1:5])

#El paquete tm, nos proporciona la posibilidad de eliminar un grupo de palabras del corpus, bien sea pasándole la lista o con funciones específicas. Entre esas funciones, la más importante es **stopwords()** que va a ser la que utilizaremos en este trabajo (se utilizará la versión en inglés).

#La siguiente línea le dice a la función **tm_map** que elimine las palabras (**removeWords**) de corpus_clean que están en **stopwords()**
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())

#Y ahora vamos a eliminar los simbolos de puntuación con con la opción **removePuntuation**
corpus_clean <- tm_map(corpus_clean, removePunctuation)

#Después de todos estos pasos, ahora tenemos esto:
inspect(corpus_clean[1:5])

#Como hemos eliminado varias cosas, en esos lugares aparecen espacios. Para limpiarlos continuamos utilizando la mimsa funcion, pero esta vez con la opción **stripWhitespace**
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

#Vemos las diferencias de los datos iniciales y los finales 

#### Inicial
inspect(datos_corpus[1:5]) 

#### Limpio
inspect(corpus_clean[1:5])

# -- Document-Term Matrix
#Ahora vamos a crear una matriz dtm (document-term matrix). Son matrices en las que se representan los términos que aparecen en el documento y con qué frecuencia
datos_dtm <- DocumentTermMatrix(corpus_clean) 
datos_dtm

#Este objeto es de clase DTM
class(datos_dtm)

#Vemos la estructura del documento
names(datos_dtm) #Las filas son los documentos, en nuestro caso los titulares

datos_dtm$nrow #Las columnas son las palabras que han ido apareciendo

datos_dtm$ncol

# Tenemos 10809 palabras distintas
# **datos_dtm** almacena cuantas veces aparece cada una de esas 10809 palabras en cada uno de los 4845 documentos. 
# Una celda concreta corresponde a un documento (en este caso un titular) concreto que se cruza con uno de los posibles términos distinto y el valor que aparece en esa celda es el número de repeticiones

# Vemos que tenemos:
str(datos_dtm)

#Podemos identifcar esa estructura:
#  - Los documentos (4845), aparecen cómo **Docs** y cómo filas **nrow**
#  - Los términos (10809), aparecen cómo **Terms** y cómo columnas **ncol**
  
# -- Dividir el dataset (Train / Test)
# Ahora vamos a intentar hacer un modelo que prediga el sentimiento según el titular. Para realizarlo, dividimos el dataset en dos, uno de train y otro de test y lo vamos a hacer de forma que la proporción del tipo de sentimiento se mantenga. Esto es muy importante cuando el dataset está desbalanceado (como vimos que se cumple en este caso) y hay más valores de un tipo que de otro.

# Damos un 80% al train y 20% al test
library(caret)

# Antes creamos una semilla para que siempre se den los mismos resultados
set.seed(1234)

# Creamos una copia de 'all_data' en formato de data frame para poder asignar el train y tesat
all_data2 = data.frame(all_data) 

# Creamos un índice que mantenga esa proporción
index         <- createDataPartition(all_data2$sentimiento, p=0.8, list=FALSE)

all_data_train <- all_data2[ index,]
all_data_test  <- all_data2[-index,]

# ****** Tener en cuenta que estamos haciendo el modelo con los datos en bruto

#Para el dtm

# La datos de train
datos_dtm_train <- datos_dtm[ index, ]

# Los datos de test
datos_dtm_test  <- datos_dtm[-index, ]


#Hacemos lo mismo para el corpus
datos_corpus_train <- corpus_clean[index]
datos_corpus_test  <- corpus_clean[-index]

#Chequeamos las proporciones entre los tipos de sentiminetos que tenemos

# En train
prop.table(table(all_data_train$sentimiento))

# En test
prop.table(table(all_data_test$sentimiento))

#Vemos que las proporciones son similares

#Antes de continuar vamos a hacer una visualización con word cloud de las palabras que más aparecen

# --  Word cloud (nube de palabras)

#Nos quedamos sólo con las que aparezcan al menos 40 veces
library(wordcloud)
wordcloud(datos_corpus_train, min.freq = 40, random.order = FALSE,colors = brewer.pal(8, "Set2"))

# Ahora vamos a realizar otro word club pero por tipo de sentimiento. 
negative <- subset(all_data_train, sentimiento == "negative")
neutral  <- subset(all_data_train, sentimiento == "neutral")
positive  <- subset(all_data_train, sentimiento == "positive")

#### Negative
wordcloud(negative$titular, max.words = 30, scale = c(6, 0.5),colors = brewer.pal(8, "Set2"))

#### Neutral
wordcloud(neutral$titular, max.words = 30, scale = c(6, 0.5), colors = brewer.pal(8, "Set2")) 

#### Positive
wordcloud(positive$titular, max.words = 30, scale = c(5, 0.5),colors = brewer.pal(8, "Set2"))

# Se puede ver a simple vista que cambian las pablabras que dominan para cada grupo. 

# -- Palabras frecuentes
#La lista de términos con una frecuencia mínima de 40
findFreqTerms(datos_dtm_train, 40)

### Diccionario
#Ahora vamos a crear un diccionario con las palabras con un número mínimo de apariciones de 5. 

```{r}
Dictionary <- function(x) {
  if( is.character(x) ) {
    return (x)
  }
  stop('x is not a character vector')
}

#La aplicamos el diccionario en el train
datos_dict  <- Dictionary(findFreqTerms(datos_dtm_train, 5))
head(datos_dict)

#Para train y test
datos_train <- DocumentTermMatrix(datos_corpus_train, list(dictionary = datos_dict))
datos_test  <- DocumentTermMatrix(datos_corpus_test, list(dictionary = datos_dict))

#Creamos una función que convierte el número de apariciones en un contador de tipo factor Si/No
convert.counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Si"))
}

#Y lo aplicamos a datos_train y datos_test 
datos_train <- apply(datos_train, MARGIN = 2, convert.counts)
datos_test  <- apply(datos_test, MARGIN = 2, convert.counts)

### Modelo Bayesiano Naive
#Vamos a crear un modelo bayesiano
library(e1071)
set.seed(1234)
datos_classifier <- naiveBayes(datos_train, all_data_train$sentimiento)

#Predecimos en el dataset de test
datos_test_pred <- predict(datos_classifier, datos_test)

#Vemos que tal hemos hecho la predicción
library(gmodels)
CrossTable(datos_test_pred, all_data_test$sentimiento,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# La tabla anterior es una [matriz de confusión](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n) donde cruzamos datos reales (actual) con predicciones (predicted). Lo ideal es que todos los valores estén en la diagonal, es decir que las predicciones **negative** sean realmente **negative**,  las **neutral** sean realmente **neutral** y las predicciones **positive** sean realmente **positive**
# Este modelo ha predicho que hay 169 valores **negative** y ha acertado en 71 y se ha equivocado en 98.
# Por otro lado, ha dicho que hay 608 valores de **neutral** y ha acertado en 475 y se ha equivocado en 130.
# Finalmente, ha predecido que hay 190 valores **positive** y ha hacertado en 121, equivocandose en 69.
#En total ha acertado 667 (71 + 475 + 121) de 967. Un 69%


### Mejoramos el modelo
#Modificamos un poco los parámetros con el objetivo de intentar mejorar el modelo

# Entrenamos
datos_classifier2 <- naiveBayes(datos_train, all_data_train$sentimiento, laplace = 1)

# Predecimos
datos_test_pred2  <- predict(datos_classifier2, datos_test)

# Y mostramos el rersultado
CrossTable(datos_test_pred2, all_data_test$sentimiento,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'),colors = brewer.pal(8, "Set2"))

E#ste modelo ha predicho que hay 118 valores **negative** y ha acertado en 62 y se ha equivocado en 56.
#Por otro lado, ha dicho que hay 661 valores de **neutral** y ha acertado en 507 y se ha equivocado en 154.
#Finalmente, ha predecido que hay 188 valores **positive** y ha hacertado en 126, equivocandose en 62.
#En total ha acertado 695 (62 + 507 + 126) de 967. Un 72%

#Este modelo es un poco mejor. 

# -- Conclusiones
#En base a este pequeño estudio podemos concluir que:
  
# - Casi el 60% de los titulares no generan un sentimiento positivo ni negativo Se puede ver que las palabras que mayor se repiten por sentimiento son visualmente diferentes
# - El modelo creado no ha sido de lo mejor, pero finalmente con la mejora realizada, ha podido acertar en el 72% de los datos. 
