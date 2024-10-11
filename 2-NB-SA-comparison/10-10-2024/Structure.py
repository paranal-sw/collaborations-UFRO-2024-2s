# PY
import os 
#import re
import numpy as np
#from collections import Counter
import pandas as pd
import time
#import itertools
#import math
from urllib.request import urlretrieve

# NLP
import nltk
from gensim.corpora import Dictionary
#from gensim.models.word2vec import Word2Vec  # Para entrenar un nuevo modelo
from textblob import TextBlob

# ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer


"""
Este avance propone una plantilla que debe desarrollarse por partes para la resolución del problema principal.

Se monta una estructura central llamada "Core" que actuará como "coordinador" para realizar descarga de datos, tokenizaciones, 
vectorizaciones, entrenamientos, etc... donde cada uno será resuelto en una clase separada para poder realizar modificaciones
más fácilmente.
"""


""" Funciones para descarga de datos
"""
def load_dataset(INSTRUMENT: str, RANGE: str):
    """
    Carga datos de un conjunto de trazas relacionadas con un instrumento y rango específico desde un repositorio 
    en Hugging Face.

    La función descarga varios archivos Parquet (trazas de instrumentos, subsistemas y telescopios) 
    según los parámetros proporcionados (`INSTRUMENT` y `RANGE`). 
    Verifica si los archivos ya existen localmente; de lo contrario, 
    los descarga y los guarda en un directorio adecuado (compatible con Colab o un sistema local).
    Luego, concatena todos los DataFrames descargados, los ordena por la columna '@timestamp',
    y devuelve el conjunto de datos consolidado.

    Args:
        INSTRUMENT (str): El nombre del instrumento del cual se desea cargar las trazas.
        RANGE (str): El rango temporal o específico que define el subconjunto de datos a cargar.

    Returns:
        pd.DataFrame: Un DataFrame consolidado que contiene todas las trazas ordenadas y listas para ser usadas.
    """
    if 'COLAB_RELEASE_TAG' in os.environ.keys():
        #!mkdir -p data
        PATH='data/' # Convenient name to be Colab compatible
    else:
        PATH='../data/' # Local directory to your system    
    REPO_URL='https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'
    fname = f'{INSTRUMENT}-{RANGE}-traces.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_inst=pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-SUBSYSTEMS.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_subs=pd.read_parquet(f'{PATH}/{fname}')

    fname = f'{INSTRUMENT}-{RANGE}-traces-TELESCOPES.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_tele=pd.read_parquet(f'{PATH}/{fname}')

    all_traces = [df_inst, df_subs, df_tele]

    df_all = pd.concat(all_traces)
    df_all.sort_values('@timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all


def load_trace(INSTRUMENT: str, RANGE: str, trace_id: int) -> pd.DataFrame:
    """
    Carga los detalles de una traza específica asociada a un evento en los metadatos.

    Esta función recupera un DataFrame que contiene información detallada sobre una entrada de log específica 
    identificada por 'trace_id'. Cada entrada de log proporciona un registro de eventos o acciones dentro del 
    sistema, con múltiples columnas que describen los detalles del evento.

    El DataFrame contiene las siguientes columnas:
    - timestamp: Marca de tiempo de la entrada de log en milisegundos.
    - system   : Nombre del sistema (e.g., PIONIER) de donde proviene la entrada de log.
    - hostname : Nombre del host de la máquina donde se generó la entrada de log.
    - loghost  : El host del sistema de registro que generó la entrada.
    - logtype  : Tipo de entrada de log (e.g., LOG, FEVT, ERR), indicando si es un log general, un evento o un error.
    - envname  : Nombre del entorno donde se generó la entrada de log, proporcionando contexto adicional.
    - procname : Nombre del proceso que generó la entrada de log.
    - procid   : ID del proceso asociado con la entrada de log.
    - module   : Módulo del sistema desde el cual se originó la entrada de log.
    - keywname : Nombre de cualquier palabra clave asociada con la entrada de log (si aplica).
    - keywvalue: Valor de la palabra clave mencionada en `keywname` (si aplica).
    - keywmask : Máscara o contexto adicional para la palabra clave (si aplica).
    - logtext  : Texto del log, que proporciona información detallada sobre el evento o acción.
    - trace_id : Identificador único asociado a cada entrada de log, que corresponde a un id en la tabla de metadatos.

    Args:
        INSTRUMENT (str): El nombre del instrumento a cargar, puede ser "PIONIER", "MATISSE" o "GRAVITY".
        RANGE (str): El rango de tiempo para la observación, que puede ser "1d" (diario), "1w" (semanal),
                     "1m" (mensual) o "6m" (seis meses).
        trace_id (int): El identificador del evento registrado que se desea seguir.

    Returns:
        pd.DataFrame: Un DataFrame de pandas que contiene los detalles del evento seleccionado por `trace_id`.
    """
    print("Loading Trace for ID: {}".format(trace_id))
    df_all = load_dataset(INSTRUMENT, RANGE)
    df_all = df_all[ df_all['trace_id']==trace_id ]
    print("Done.")
    return df_all


def load_meta(INSTRUMENT: str, RANGE: str) -> pd.DataFrame:
    """
    Carga los metadatos de un instrumento específico en un rango de tiempo determinado.

    Esta función recupera y carga un archivo de metadatos en formato parquet, que contiene 
    información sobre la ejecución de plantillas para un instrumento de observación astronómica 
    (como PIONIER, MATISSE o GRAVITY) en un intervalo de tiempo específico.

    El archivo de metadatos incluye las siguientes columnas:
    - START   : Marca de tiempo de inicio de la ejecución de la plantilla (en milisegundos).
    - END     : Marca de tiempo de finalización de la ejecución de la plantilla (en milisegundos).
    - TIMEOUT : Indica si la ejecución excedió un límite de tiempo predefinido.
    - system  : Nombre del sistema utilizado (e.g., PIONIER).
    - procname: Nombre del proceso asociado con la ejecución de la plantilla.
    - TPL_ID  : Nombre del archivo de la plantilla correspondiente.
    - ERROR   : Indica si hubo un error durante la ejecución.
    - Aborted : Indica si la ejecución de la plantilla fue abortada (manualmente o debido a un error).
    - SECONDS : Duración de la ejecución de la plantilla en segundos.
    - TEL     : Clase de telescopio utilizado en la observación (solo se utiliza AT en este conjunto de datos).

    Args:
        INSTRUMENT (str): El nombre del instrumento a cargar, puede ser "PIONIER", "MATISSE" o "GRAVITY".
        RANGE (str): El rango de tiempo para la observación, que puede ser "1d" (diario), "1w" (semanal), 
                     "1m" (mensual) o "6m" (seis meses).

    Returns:
        pd.DataFrame: Un DataFrame de pandas que contiene los metadatos del instrumento en el rango de tiempo seleccionado.
    """
    print("Loading Metadata for: \nINSTRUMENT: {} \nRANGE: {} \n".format(INSTRUMENT,RANGE))
    if 'COLAB_RELEASE_TAG' in os.environ.keys():
        #!mkdir -p data
        PATH='data/' # Convenient name to be Colab compatible
    else:
        PATH='../data/' # Local directory to your system
    REPO_URL='https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'
    fname = f'{INSTRUMENT}-{RANGE}-meta.parket'
    if not os.path.exists(f'{PATH}/{fname}'):
        urlretrieve(f'{REPO_URL}/{fname}', f'{PATH}/{fname}')
    df_meta=pd.read_parquet(f'{PATH}/{fname}')
    print("Done.")
    return df_meta


class Core():
    """ Esta clase es la base (o esqueleto) para la ejecución de procesos y almacenamiento.
    Los atributos del constructor los utilizaré para el almacenamiento de registros.
    
    Methods:
        - load_meta: llamará a la función load_meta() definida en "task_description.ipynb".
        - load_trace: cargará la traza asociada al índice.
        - trace_tokenization: 

    """
    
    def __init__(self, INSTRUMENT: str = 'MATISSE', RANGE: str = '1d', time_it: bool = False) -> None:
        """
    Constructor de la clase Core que inicializa las propiedades necesarias para almacenar, leer y procesar datos.

    Esta clase tiene como objetivo crear una estructura flexible para manejar y analizar metadatos y trazas relacionadas.

    Args:
        INSTRUMENT (str): Nombre del instrumento a analizar (por defecto: 'MATISSE').
        RANGE (str)     : Rango de tiempo a analizar (por defecto: '1d'), opciones pueden incluir '1w', '1m', '6m'.
        time_it (bool)  : Si es True, se utilizará un cronómetro para medir los tiempos de ejecución de ciertos métodos.
    
    Attributes:
        INSTRUMENT (str): El nombre del instrumento astronómico a analizar, e.g., "MATISSE", "PIONIER", "GRAVITY".
        RANGE (str): El rango de tiempo que se analizará en los datos, e.g., "1d", "1w", "1m", "6m".
        meta_data (pd.DataFrame): DataFrame que contiene los metadatos cargados, se obtiene a través de 'load_meta()'.
        trace_data (pd.DataFrame): DataFrame que contiene los detalles de la traza seleccionada, inicialmente 'None'.
        trace_index (int): El índice de la traza actualmente seleccionada, inicializado en 0.
        dataset (pd.DataFrame): El conjunto completo de datos cargado a partir del instrumento y el rango de tiempo,
                                se obtiene mediante 'load_dataset()'. Cargar el dataset completo optimiza la velocidad
                                de acceso a las trazas pero puede sacrificar RAM.
        n_errors (int): El número de errores en los metadatos, obtenido filtrando los registros con 'ERROR == True'.
        regExp (str): Una expresión regular utilizada para análisis y filtrado, inicializada como cadena vacía.
        time_it (bool): Indica si el cronómetro debe activarse para medir tiempos de ejecución.
        clock (Clock): Instancia de la clase `Clock` que puede usarse para medir tiempos de ejecución.
        pipeline (Custom_Pipeline): Instancia de la clase 'Custom_Pipeline' que proporciona métodos para entrenar
                                    y evaluar modelos de clasificación.
    """
        self.INSTRUMENT : str = INSTRUMENT
        self.RANGE :str = RANGE
        self.meta_data :pd.DataFrame = load_meta(INSTRUMENT=INSTRUMENT, RANGE=RANGE)
        self.n_errors = len(self.meta_data[self.meta_data['ERROR']==True])
        self.regExp: str = ''  # Inicializado en cadena vacía hasta definirla en un método

        # Cargar el dataset completo es sacrificar RAM por velocidad en la carga de trazas
        self.dataset: pd.DataFrame = load_dataset(INSTRUMENT=INSTRUMENT, RANGE=RANGE)
        self.trace_data : pd.DataFrame = None  # Inicialmente None, será asignado a un DataFrame más adelante
        self.trace_index: int = 0  # Índice inicial de la traza objetivo
        self.time_it: bool = time_it

        # Instancias de clases adicionales para medición de tiempo y manejo de pipelines de modelos ("arms")
        self.clock: Clock = Clock()
        self.pipeline = Custom_Pipeline()
        pass

    
    def load_meta(self, INSTRUMENT: str, RANGE: str) -> None:
        """
    Carga el DataFrame de Metadatos para el instrumento e intervalo de tiempo especificados.

    Este método actualiza los atributos `INSTRUMENT` y `RANGE` de la instancia y carga los 
    metadatos correspondientes en el atributo `meta_data`. Si la carga de los metadatos falla, 
    se lanza un ValueError.

    Args:
        INSTRUMENT (str): Nombre del instrumento a analizar (e.g., "PIONIER", "MATISSE", "GRAVITY").
        RANGE (str): Rango de tiempo a analizar (e.g., "1d", "1w", "1m", "6m").

    Raises:
        ValueError: Si no se puede cargar el DataFrame de metadatos, se lanza este error.
    """
        self.INSTRUMENT = INSTRUMENT
        self.RANGE = RANGE
        try:
            self.meta_data = load_meta(INSTRUMENT=self.INSTRUMENT, RANGE=self.RANGE)
            pass # Método vacío, actúa solo sobre los atributos
        except:
            raise ValueError('No se ha podido cargar los datos en el lector (Error en método load_meta)')
    
    
    def load_trace(self, index: int, **kwargs) -> None:
        """
    Carga la traza asociada a un índice dado que pertenece a 'meta_data'.

    Este método busca la entrada correspondiente en el conjunto de datos completo utilizando el 
    'trace_id' proporcionado. Si se encuentra, se carga la información de la traza en el 
    atributo 'trace_data'. Además, mide el tiempo de ejecución si se activa.

    Args:
        index (int): El índice de la traza que se desea cargar, debe corresponder a un 
                      'trace_id' en 'meta_data'.
        **kwargs: Argumento adicional 'timer', para controlar la temporización.

    Raises:
        ValueError: Si el índice solicitado excede la cantidad de trazas disponibles o 
                     si hay un error al cargar la traza.
    """
        # begin Clock {
        timer = kwargs.get('timer', True)
        if self.time_it and timer: self.clock.begin()
        # }
        
        if index > len(self.meta_data):
            if self.time_it and timer: self.clock.stop(func=str(self.load_trace.__name__)) # end clock
            raise ValueError('El número de traza excede a las disponibles:\n Máximo: ',len(self.meta),
                             '\n Solicitado: ',index)
        self.trace_index = index
        try: # Cargar la traza correspondiente al index
            self.trace_data = self.dataset[self.dataset['trace_id']==index]
            if self.time_it and timer: self.clock.stop(func=str(self.load_trace.__name__)) # end clock
            pass
        except:
            if self.time_it and timer: self.clock.stop(func=str(self.load_trace.__name__)) # end clock
            raise ValueError('No se ha podido cargar la traza para el evento registrado')
    

    def trace_tokenization(self, column: str='logtext', setting: str ='RegExp', **kwargs) -> pd.Series:
        """
    Tokeniza cada log de la traza en una columna específica.

    Este método aplica un tokenizador a la columna especificada del DataFrame 'trace_data' 
    y devuelve una serie de pandas con la tokenización de cada entrada de log.

    Args:
        column (str): El nombre de la columna que contiene los textos a tokenizar. 
                      Por defecto es 'logtext'.
        setting (str): Define el método de tokenización a utilizar. 
                       'RegExp' usa expresiones regulares (default), 
                       'GenDict' es un método en desarrollo.
        **kwargs: Argumentos adicionales para personalizar la tokenización, como la expresión 
                  regular y la opción de "gaps".
                -   regexp (raw str): Expresión regular a utilizar en el tokenizador RegExp
                -   gaps (bool): Booleano que manipula los espacios entre palabras.

    Returns:
        pd.Series: Una serie de pandas que contiene los logs tokenizados.
    """
        # begin Clock {
        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()
        # }

        custom_tokenizer = Custom_Tokenizer()  # Llamar al tokenizador y luego eliminarlo

        # Configuración del tokenizador basado en el método elegido
        if setting == 'RegExp':
            ### kwargs: {
            regexp=kwargs.get('regexp',r'\b[a-zA-Z]+\b')
            gaps=kwargs.get('gaps',False)
            ### }
            custom_tokenizer.set_regexp_tokenizer(regexp=regexp,gaps=gaps)
            self.regExp = regexp  # Almacenar la expresión regular utilizada

        #elif setting == 'GenDict':  # Esto aún no está desarrollado
            #custom_tokenizer.set_gensim_tokenizer()
        
        # Aplicar el tokenizador a la columna especificada
        result = self.trace_data[column].apply(lambda x: custom_tokenizer.c_tokenize(x))

        if timer and self.time_it: self.clock.stop(func=self.trace_tokenization.__name__)  # end clock
        pass
        return result
    

    def meta_tokenization(self,**kwargs):
        """  
    Tokeniza todas las trazas en `meta_data` y agrega los resultados en una lista.

    Este método recorre el DataFrame de metadatos, carga las trazas correspondientes 
    a cada entrada y aplica la tokenización. Los resultados se almacenan en una nueva 
    columna llamada 'Trace_tokens'.

    Keyword Args:
        setting (str): Define el tokenizador a usar (default: 'RegExp').
        regexp (str) : Expresión regular a usar si se elige 'RegExp' (default: r"\b[a-zA-Z]+\b").
        gaps (bool)  : Si es True, se omiten espacios en la tokenización.

    PARALELIZABLE*, AGREGAR EXCEPCIONES*
    """
        # begin Clock{
        timer = kwargs.get('timer', True)
        if self.time_it and timer: 
            self.clock.begin()
        # }
        
        trace_tokens_list = []
        for k in range(0, len(self.meta_data)):  # Recorrer todas las entradas de meta_data
            self.load_trace(index=k, timer=False)  # Cargar la traza correspondiente

            ### kwargs:{
            setting = kwargs.get('setting','RegExp')
            regexp=kwargs.get('regexp',r'\b[a-zA-Z]+\b')
            gaps = kwargs.get('gaps',False)
            ### } 
            # Tokenizar la traza y agregar los resultados a la lista
            trace_tokens_list.append(self.trace_tokenization(setting=setting, regexp=regexp, gaps=gaps, timer=False).tolist())
        self.meta_data['Trace_tokens'] = trace_tokens_list
        if self.time_it and timer: self.clock.stop(func=self.meta_tokenization.__name__)  # end Clock
        pass

    
    def meta_vectorization(self,method: str = 'CountVectorizer',**kwargs) -> None:
        """
    Agrega una columna con representación vectorial de cada sucesión de logs por registro.

    Actualmente, el vectorizador solo asocia cada registro a un vector utilizando 
    CountVectorizer. Se deben implementar modificaciones para asociar cada registro 
    a una sucesión de vectores (matriz), lo que permitirá analizar los logs junto 
    a una variable de tiempo, en lugar de solo el resultado final.

    También se debe considerar la reducción de dimensiones mediante PCA.

    Params:
        method (str): define el método de vectorización (actualmente solo se ofrece CountVectorizer).

    Keyword Args:
        stop_words (list): Lista de stopwords en formato string.
    """
        # begin Clock{
        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()
        # }
        
        # Crear una columna de clasificación basada en errores(conversión booleana/númerica)
        self.meta_data['Classification'] = self.meta_data['ERROR'].to_numpy().astype(int)

        #self.meta_tokenization(timer=False)  # Tokenizar automáticamente antes de vectorizar ?
        if method == 'CountVectorizer':
            ### kwargs: {
            stop_words : list = kwargs.get('stop_words', [])
            ###}

            # Inicializar el vectorizador
            vectorizer = CountVectorizer(stop_words=stop_words)  # Si en algún punto resulta necesario convertir vector a texto habrá que extraer el corpus.
            # Convertir cada sucesión de logs en una cadena de texto.
            self.meta_data['Word_vector'] = self.meta_data['Trace_tokens'].apply(lambda token_list:' '.join([' '.join(tokens) for tokens in token_list]))
            
            # Asignar un vector a cada sucesión.
            vector_series = vectorizer.fit_transform(self.meta_data['Word_vector'])
            vector_series = vector_series.toarray().tolist() # transformar a lista
            self.meta_data['Word_vector'] = vector_series  # Almacenar la lista de vectores en meta_data
            if timer and self.time_it: self.clock.stop(self.meta_vectorization.__name__)  # end Clock
        else:
            pass
    
    
    def transfer_matrix(self,sample_succes: int = 42, **kwargs) -> tuple:
        """
        Convierte columnas vectoriales a matrices de NumPy y realiza un muestreo de la clase mayoritaria.

        Este método filtra y transfiere una columna que contiene vectores de palabras, apilándolos en una matriz de NumPy 
        (aún debe construirse el caso de un tensor de mayor orden).
        También realiza un muestreo de la clase mayoritaria (éxitos) de acuerdo al parámetro 'sample_succes', 
        y combina este subconjunto con la clase minoritaria (errores) para balancear el conjunto.

        Args:
            sample_succes (int): Número de muestras a tomar de la clase mayoritaria (éxitos). Valor por defecto: 42.
            **kwargs:
                - timer (bool): True para imprimir el tiempo demorado, False para omitir.

        Returns:
            tuple: Una tupla que contiene:
                - X (np.ndarray): Matriz de vectores de palabras.
                - y (np.ndarray): Arreglo con las etiquetas de clasificación correspondientes.
    """
        # begin Clock{
        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()
        # }
        meta_errors = self.meta_data[self.meta_data['Classification']==1]  # filtro
        meta_success = self.meta_data[self.meta_data['Classification']==0]
        meta_success = meta_success.sample(n=sample_succes)  # sample
        meta_success = pd.concat([meta_success,meta_errors])

        X = meta_success['Word_vector']  # Columna con palabras vectorizadas
        X = np.stack(X.values)
        y = meta_success['Classification'].to_numpy()
        
        if self.time_it and timer: self.clock.stop(func=str(self.transfer_matrix.__name__)) # End Clock
        return X, y
    

    def call_pipeline(self, method: str = 'Multinomial_NB', sample_success: int = 42, test_size: float = 0.2,
                       NB_kwargs: dict = {'alpha': 1.0, 'fit_prior': True}, treshold: float = 1.0 ,**kwargs):
        """
    Realiza los análisis de clasificación o análisis de sentimientos basados en el método especificado.

    Parámetros:
        method (str): El método de análisis a utilizar ('Multinomial_NB' o 'TextBlob').
        sample_success (int): Número de muestras a utilizar (solo se aplica a 'Multinomial_NB').
        test_size (float): Proporción del conjunto de datos para el conjunto de prueba (solo se aplica a 'Multinomial_NB').
        NB_kwargs (dict): Parámetros adicionales para el clasificador Naive Bayes (solo se aplica a 'Multinomial_NB').
        threshold (float): Umbral para el análisis de sentimientos (solo se aplica a 'TextBlob').

    Keyword Args:
        timer (bool): Si se debe medir el tiempo de ejecución.

    REQUIERE MANEJO DE ERRORES*
    """
        # begin Clock{
        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()
        #}
        # begin Multinomial_NB {
        if method == 'Multinomial_NB':
            X,y = self.transfer_matrix(sample_succes=sample_success,timer=False)
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=100)  # ¿Fijar semilla?
            self.pipeline.Multinomial_Naive_Bayes(X_train=X_train,
                                                    X_test=X_test,
                                                    y_test=y_test,
                                                    y_train=y_train,
                                                    NB_args= NB_kwargs,
                                                    success_sampling=sample_success,
                                                    RegExp=self.regExp) # }
        # begin TextBlob {
        elif method == 'TextBlob':
            sample_meta = self.meta_data.sample(n=sample_success)
            sample_meta['Trace_tokens'] = sample_meta['Trace_tokens'].apply(lambda token_list: [' '.join(sublist) for sublist in token_list])
            self.pipeline.TextBlob_Sentiment_Analysis(treshold=treshold,meta_data=sample_meta, regExp=self.regExp)  
            # }

        if self.time_it and timer: self.clock.stop(func=str(self.call_pipeline.__name__)) # End Clock
        pass                


    def meta_predict(self,model) -> dict:
        """Aplicar modelo a la columna de logs y comparar con y_true = Classification,
        En desarrollo..."""
        if isinstance(model,MultinomialNB):
            y_pred = model.predict(np.stack(self.meta_data['Word_vector'].values))
            matrix = confusion_matrix(y_true=self.meta_data['Classification'].values,
                                      y_pred=y_pred)
            accuracy = accuracy_score(y_true=self.meta_data['Classification'].values,
                                      y_pred=y_pred)
            precision = precision_score(y_true=self.meta_data['Classification'].values,
                                        y_pred=y_pred)
            recall = recall_score(y_true=self.meta_data['Classification'].values,
                                  y_pred=y_pred)
            summary = {
                'True_negative' : matrix[0][0],
                'False_negative' : matrix[1][0],
                'True_positive' : matrix[1][1],
                'False_positive' : matrix[0][1],
                'Accuracy': accuracy,
                'Precision': precision,  # Precisión de positivos
                'Recall': recall,
                'Predictions': y_pred}   #  Positivos reales
            return summary

        
class Custom_Tokenizer():
    """ Se deben utilizar distintos tokenizadores para realizar la limpieza de los logs,
    y buscar una estructura adecuada. Por ahora dejaré un simple tokenizador de expresiones
    regulares.
    En el caso de un tokenizador nltk se puede aplicar expresiones regulares inmediatamente.
    Si se trata de un tokenizador de Gensim puede que sea mejor aplicar un método de limpieza
    anterior a la tokenización.
    """

    def __init__(self, ) -> None:
        self.tokenizer = None
        self.setting: str = None
        pass
    
    
    def set_regexp_tokenizer(self, regexp: str = r'\b[a-zA-Z]+\b' , gaps: bool = False) -> None:
        """ Se deben modificar los argumentos de acuerdo a las necesidades de un RegexpTokenizer.
        Argumentos a probar (por ahora):
        regexp = r'\D+'      : Eliminar todo lo que  sea un número.
        regexp = r'[a-zA-Z]+': Eliminar cualquier número y símbolo
        r'\s+', gaps=True    : Conservar solo letras y reemplazar el resto por espacios.
        ----------------------
        Params: 
            regexp (str) : La expresión regular que define el patrón de los tokens.
                            (default: busca solo palabras)
            gaps (bool)  : Indica si los patrones encontrados deben ser considerados como separadores 
                           entre tokens (True) o como los propios tokens (False).
            *flags (int) : (agregar más tarde si es necesario)
        returns:
            nltk.tokenize.RegexpTokenizer con la configuración establecida.
        """
        self.setting = 'RegExp'
        self.tokenizer = nltk.tokenize.RegexpTokenizer(pattern=regexp,gaps=gaps)
        pass

    
    def set_gensim_tokenizer(self):
        """REQUIERE DE PREPROCESAMIENTO DE TEXTO CON RE
        """
        self.setting = 'GenDict'
        self.tokenizer = Dictionary()
        pass


    def c_tokenize(self, log: str) -> list:
        """ Custom_tokenize: Agregar aquí cualquier modificación necesaria para el tokenizador.
        Lo importante es que reciba un log y devuelva una lista (o tal vez un conjunto) para que
        pueda ser leído por el Core. 
        """
        #### Aquí es posible definir modificaciones al tokenizador que se ajusten a las necesidades
        if self.setting == 'RegExp':
            result = self.tokenizer.tokenize(log)
        #elif self.setting == 'GenDict':
            #log = preprocesador_texto(log)
            #self.tokenizer.add_documents([log])
            #result = list(self.tokenizer.token2id.values())
        return result
    

    ###################################


    #class Custom_Vectorizer():
    """Probablemente se requiera un vectorizador adaptable, si en algún punto se utiliza
    un modelo Word2Vec será necesario aplicar tensores de mayor grado, mientras que una 
    vectorización como tf-idf solo necesitará matrices, por lo que las salidas deben ajustarse
    para ser compatibles."""


class Custom_Pipeline():
    """
    Clase para estructurar y ejecutar el entrenamiento y evaluación de múltiples modelos de clasificación.

    Esta clase permite entrenar varios modelos de clasificación con distintos parámetros, incluyendo Naive Bayes
    y análisis de sentimiento con TextBlob. Los resultados se almacenan para su análisis posterior.

    Attributes:
        model (str): El modelo que se va a utilizar. El valor predeterminado es 'Multinomial_NB'.
        data (list): Lista que almacena los resultados de los modelos entrenados y sus métricas.
    """
    def __init__(self, model: str='Multinomial_NB') -> None:
        """
        Inicializa la clase Custom_Pipeline con el modelo especificado.

        Args:
            model (str): El nombre del modelo a entrenar (default: 'Multinomial_NB').
        """
        self.model = model
        self.data: list = []
        pass


    def Multinomial_Naive_Bayes(self,
                                X_train: np.array,
                                y_train: np.array,
                                X_test: np.array,
                                y_test: np.array,
                                NB_args: dict,
                                success_sampling: int,
                                RegExp: str) -> dict:
        """
    Entrena y evalúa un modelo de clasificación Multinomial Naive Bayes.

    Este método entrena un clasificador Naive Bayes utilizando los datos de entrenamiento y el 
    conjunto de parámetros proporcionados en `NB_args`. 
    Luego realiza predicciones en los datos de prueba y calcula varias métricas de rendimiento, 
    incluyendo la precisión, el recall, y la matriz de confusión. Finalmente, devuelve un diccionario
    con el clasificador entrenado y los resultados de las métricas.

    Args:
        X_train (np.array): Matriz de características para el entrenamiento.
        y_train (np.array): Etiquetas de entrenamiento correspondientes a `X_train`.
        X_test (np.array): Matriz de características para el conjunto de prueba.
        y_test (np.array): Etiquetas reales correspondientes al conjunto de prueba.
        NB_args (dict): Diccionario con los argumentos del modelo `MultinomialNB` de `sklearn`.
        success_sampling (int): Número de muestras exitosas utilizadas durante el muestreo.
        RegExp (str): Expresión regular utilizada en el análisis para seguimiento o documentación.

    Returns:
        dict: Un diccionario con los siguientes elementos:
            - 'Classifier' (MultinomialNB): El modelo Naive Bayes entrenado.
            - 'Success_Sampling' (int): Cantidad de muestras exitosas usadas en el muestreo.
            - 'Params' (dict): Los parámetros utilizados para entrenar el clasificador.
            - 'RegExp' (str): Expresión regular para referencia.
            - 'True_negative' (int): Cantidad de verdaderos negativos de la matriz de confusión.
            - 'False_negative' (int): Cantidad de falsos negativos de la matriz de confusión.
            - 'True_positive' (int): Cantidad de verdaderos positivos de la matriz de confusión.
            - 'False_positive' (int): Cantidad de falsos positivos de la matriz de confusión.
            - 'Accuracy' (float): Precisión del modelo (proporción de predicciones correctas).
            - 'Precision' (float): Precisión del modelo en la predicción de positivos.
            - 'Recall' (float): Tasa de verdaderos positivos con respecto a los positivos reales.
    """
        classifier = MultinomialNB(**NB_args)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # Metrics
        accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
        precision = precision_score(y_true=y_test,y_pred=y_pred)
        recall = recall_score(y_true=y_test,y_pred=y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        # Summary
        result = {
        'Classifier' : classifier,  # Claisificador entrenado
        'Success_Sampling' : success_sampling,
        'Params' : NB_args,  # parámetros del NB entrenado
        'RegExp' : RegExp,
        'True_negative' : matrix[0][0],
        'False_negative' : matrix[1][0],
        'True_positive' : matrix[1][1],
        'False_positive' : matrix[0][1],
        'Accuracy': accuracy,
        'Precision': precision,  # Precisión de positivos
        'Recall': recall}   #  Positivos reales
        self.data.append(result)
        return result


    def TextBlob_Sentiment_Analysis(self,treshold: float,  meta_data: pd.DataFrame, regExp: str, success_sampling: int):
        """
    Realiza un análisis de sentimiento usando el clasificador TextBlob y calcula métricas de evaluación.

    Este método aplica el clasificador de sentimiento 'TextBlob_Classifier' a una columna de tokens de trazas en 'meta_data', 
    evaluando los resultados con respecto a la clasificación real. 
    
    Calcula diversas métricas de rendimiento como precisión, exactitud, y recall, además de la matriz de confusión. 
    Al final, guarda los resultados en un diccionario y los agrega a la lista de datos de la instancia.

    Args:
        treshold (float)        : El umbral usado por el clasificador para determinar la clasificación de una traza.
        meta_data (pd.DataFrame): El DataFrame que contiene las trazas tokenizadas bajo la columna 'Trace_tokens' 
                                  y las etiquetas de clasificación reales en la columna 'Classification'.
        regExp (str)            : La expresión regular utilizada en el análisis o para la selección de datos 
                                  (parámetro para propósitos de seguimiento).

    Returns:
        dict: Un diccionario con los siguientes elementos:
            - 'Classifier' (TextBlob_Classifier): El clasificador utilizado.
            - 'Success_Sampling' (None): Tamaño de la muestra exitosa utilizada para la evaluación.
            - 'Params' (dict): Los parámetros utilizados, incluyendo el umbral ('treshold').
            - 'RegExp' (str): La expresión regular proporcionada para referencia.
            - 'True_negative' (int): Cantidad de verdaderos negativos de la matriz de confusión.
            - 'False_negative' (int): Cantidad de falsos negativos de la matriz de confusión.
            - 'True_positive' (int): Cantidad de verdaderos positivos de la matriz de confusión.
            - 'False_positive' (int): Cantidad de falsos positivos de la matriz de confusión.
            - 'Accuracy' (float): La exactitud del modelo (proporción de predicciones correctas).
            - 'Precision' (float): La precisión de la predicción de la clase positiva.
            - 'Recall' (float): La proporción de verdaderos positivos con respecto al total de positivos reales.

        """
        classifier = TextBlob_Classifier(treshold=treshold)
        y_pred = meta_data['Trace_tokens'].apply(lambda trace: classifier.predict(trace)).values
        y_test = meta_data['Classification'].values
        # Metrics
        accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
        precision = precision_score(y_true=y_test,y_pred=y_pred)
        recall = recall_score(y_true=y_test,y_pred=y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        # Summary
        result = {
        'Classifier' : classifier,  # Claisificador
        'Success_Sampling' : success_sampling,
        'Params' : {'treshold' : treshold},  # Umbral de TextBlob
        'RegExp' : regExp,
        'True_negative' : matrix[0][0],
        'False_negative' : matrix[1][0],
        'True_positive' : matrix[1][1],
        'False_positive' : matrix[0][1],
        'Accuracy': accuracy,
        'Precision': precision,  # Precisión de positivos
        'Recall': recall}   #  Positivos reales
        self.data.append(result)
        return result


class TextBlob_Classifier:
    """
    Un clasificador de sentimientos basado en la librería TextBlob con un sistema de umbral configurable.

    Esta clase ofrece dos métodos principales para la clasificación de sentimientos en textos:
    
    1. 'classify_sentiment': Clasifica el sentimiento de un log individual en positivo, 
        negativo o neutro utilizando la polaridad del análisis de TextBlob.
    2. 'predict': Clasifica el sentimiento general de una secuencia de logs (traza) utilizando la función 'classify_sentiment'
        en cada log y evaluando la proporción de logs positivos y negativos. 
        El resultado final está influenciado por un umbral definido en la instancia de la clase.

    Attributes:
        treshold (float): El umbral que controla cómo se clasifica una traza de logs. 
        Si la proporción de logs negativos frente a logs positivos es menor que este valor, 
        la traza se clasifica como exitosa.

    Methods:
        classify_sentiment(log: str) -> int:
            Clasifica el sentimiento de un log basado en su polaridad:
            - Retorna 1 si es positivo.
            - Retorna -1 si es negativo.
            - Retorna 0 si es neutro.
        
        predict(trace: list) -> int:
            Clasifica una traza de logs completa evaluando cada uno de sus elementos.
            - Retorna 0 (éxito) si la proporción de logs negativos frente a positivos es menor que el umbral.
            - Retorna 1 (error) si todos los logs son negativos o si la proporción de logs negativos supera el umbral.
    """

    def __init__(self,treshold: float = 1.0):
        self.treshold = treshold
        pass
    
    def classify_sentiment(self, log: str) -> int:
        """
        Clasifica el sentimiento de un log utilizando TextBlob.

        Este método analiza el sentimiento de una cadena de texto ('log') a través de la librería TextBlob. 
        Devuelve un valor que indica si el sentimiento es positivo, negativo o neutro, basado en la polaridad del texto:
    
        - Si la polaridad es mayor a 0, clasifica el log como positivo (retorna 1).
        - Si la polaridad es menor a 0, lo clasifica como negativo (retorna -1).
        - Si la polaridad es 0, lo clasifica como neutro (retorna 0).

        Args:
            log (str): El texto del log cuyo sentimiento se desea clasificar.
            Returns:
                int: Un valor que representa el sentimiento del log:
                -    1 para positivo,
                -   -1 para negativo,
                -    0 para neutro.
        """
        analisis = TextBlob(log)
        if analisis.sentiment.polarity > 0:
            return 1
        elif analisis.sentiment.polarity < 0:
            return -1
        else:
            return 0

    def predict(self, trace: list) -> int:
        """
    Clasifica el sentimiento general de una traza de logs usando 'classify_sentiment' y un evaluador de umbral.
    Este método analiza cada log en la lista 'trace' utilizando la función 'classify_sentiment',
    que clasifica el sentimiento de cada log individualmente. Luego, cuenta los resultados positivos y 
    negativos y evalúa la relación entre ellos.
    Si el ratio de logs negativos frente a logs positivos es menor que el umbral definido por 
    'self.treshold' y existe al menos un log positivo, clasifica la traza como éxito (retorna 0).
    Si no hay logs positivos, la traza se clasifica como negativa (retorna 1).
    En cualquier otro caso, se clasifica como error (retorna 1).

    Args:
        trace (list): Una lista de logs que representan la traza a evaluar.

    Returns:
        int: El resultado de la clasificación de la traza:
            - 0 para éxito,
            - 1 para negativo.
    """
        trace_sentiment = [self.classify_sentiment(log) for log in trace]
        #trace_sentiment = sum(trace_sentiment)
        conteo_positivo = trace_sentiment.count(1)
        conteo_negativo = trace_sentiment.count(-1)
        if conteo_negativo/conteo_positivo < self.treshold and conteo_positivo != 0:
            return 0
        elif conteo_positivo == 0:
            return 1
        else:
            return 1


class Clock():
    """
    Un cronómetro simple para medir tiempos de ejecución.

    La clase Clock permite medir y mostrar el tiempo transcurrido entre dos puntos en el código,
    facilitando el seguimiento del rendimiento de funciones o bloques de código.

    Attributes:
        tic (float): Almacena el tiempo inicial cuando se inicia el cronómetro.
        toc (float): Almacena el tiempo final cuando se detiene el cronómetro.
    """
    def __init__(self) -> None:
        """
        Inicializa el cronómetro con valores de tiempo nulos.
        """
        self.toc = None
        # self.history = {method, time}
        pass

    def begin(self):
        """
        Inicia el cronómetro capturando el tiempo actual.
        """
        self.tic = time.perf_counter()
        pass

    def stop(self, func: str):
        """
        Detiene el cronómetro, calcula el tiempo transcurrido desde 'begin', y lo imprime.

        Args:
            func (str): Nombre de la función o tarea cuyo tiempo de ejecución se mide.
        """
        self.toc = time.perf_counter()
        if self.tic is None:
            return None
        print(func," Execution time: ",self.toc - self.tic," Seconds")
        self.tic = None
        self.toc = None
        pass

