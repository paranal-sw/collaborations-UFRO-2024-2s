from Structure import Core
import nltk
from nltk.corpus import stopwords
import itertools
import numpy as np
import pandas as pd

if __name__ == '__main__':
    nltk.download('stopwords')
    # Lista de instrumentos que se van a analizar
    nombres_inst=['MATISSE']
    n= len(nombres_inst) # Número de instrumentos
    BASES={}  # Diccionario para almacenar instancias de la clase Core
    
    # Iterar sobre cada instrumento
    for nombre_instrumento in nombres_inst:

        # Se crea una instancia de la clase Core, se puede modificar el rango de tiempo
        Base = Core(INSTRUMENT=nombre_instrumento ,RANGE='6m',time_it=True)
        BASES[nombre_instrumento] = Base  # Almacenar Core en el diccionario
        
        # Número de errores en el conjunto de metadatos del instrumento actual
        n_errores = BASES[nombre_instrumento].n_errors
        BASES[nombre_instrumento].meta_tokenization()  # Tokenizar los metadatos
        stop_words = stopwords.words('english')  # Obtener la lista de stopwords en inglés
        BASES[nombre_instrumento].meta_vectorization(stop_words = stop_words)  # Vectorizar los metadatos utilizando las stopwords


        # Crear un rango de valores para el parámetro alpha y para fit_prior
        alpha_range = np.arange(0,1,0.2).tolist()
        fit_prior = [True, False]
        #sampled_success = range(n_errores, n_errores*7 + 1, n_errores)
        combinaciones = list(itertools.product(alpha_range,fit_prior))  # Crear combinaciones de parámetros

        regexp_list = [r'\D+', r'[a-zA-Z]+', r'\s+', r'\b[a-zA-Z]+\b'] # Se testearán 4 expresiones regulares
        for regexp in regexp_list:
            if regexp != r'\s+': BASES[nombre_instrumento].meta_tokenization(regexp=regexp)
            else: BASES[nombre_instrumento].meta_tokenization(regexp=regexp, gaps=True)
            for combinacion in combinaciones:  # Iterar sobre cada combinación de parámetros
                alpha, fit_prior  = combinacion

                # Crear un diccionario con los parámetros para el clasificador Naive Bayes
                NB_args = {'alpha': alpha,
                        'fit_prior': fit_prior}
                
                # Llamar al pipeline de clasificación usando Naive Bayes
                BASES[nombre_instrumento].call_pipeline(method = 'Multinomial_NB', NB_kwargs = NB_args)

        # Definir un rango de umbrales para el análisis de sentimientos
        tresholds = np.arange(0.6,1.4,0.1).tolist()

        # Iterar sobre cada umbral y ejecutar el análisis de sentimientos
        for regexp in regexp_list:
            if regexp != r'\s+': BASES[nombre_instrumento].meta_tokenization(regexp=regexp)
            else: BASES[nombre_instrumento].meta_tokenization(regexp=regexp, gaps=True)
            for treshold in tresholds:
                BASES[nombre_instrumento].call_pipeline(method = 'TextBlob', treshold = treshold)

        # Crear un DataFrame resumen con los resultados del pipeline
        summary = pd.DataFrame(BASES[nombre_instrumento].pipeline.data)

        # Guardar el resumen en un archivo CSV
        #summary.to_csv(f'{nombre_instrumento}.csv')
    



