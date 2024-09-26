# PY
import os 
import re
import numpy as np
#from collections import Counter
import pandas as pd
import time
#import math

# NLP
import nltk
from urllib.request import urlretrieve
from gensim.corpora import Dictionary
#from gensim.models.word2vec import Word2Vec  # Para entrenar un nuevo modelo

# ML
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer


"""
Este avance propone una plantilla que debe desarrollarse por partes para la resolución del problema principal.

Se monta una estructura central llamada "Core" que actuará como "coordinador" para realizar descarga de datos, tokenizaciones, 
vectorizaciones, entrenamientos, etc... donde cada uno será resuelto en una clase separada para poder realizar modificaciones
más fácilmente.

Espero no enrredar mucho el código, puede parecer muy desordenado pero el usuario final no tendrá que indagar en este código.
Sino que solo utilizar los métodos que se programarán en la estructura final.

"""


""" Funciones para descarga de datos
"""
def load_dataset(INSTRUMENT, RANGE):
    """ *DESCRIPTION REQUIRED*
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
    """ Traces are Dataframes containing detailed information about an especific entry of the metadata.
    @timestamp: The timestamp of the log entry in milliseconds.
    system 	  : The name of the system (e.g., PIONIER) from which the log entry originates.
    hostname  :	The hostname of the machine where the log entry was generated.
    loghost   :	The host of the logging system that generated the entry.
    logtype   :	Type of the log entry (e.g., LOG, FEVT, ERR), indicating its nature such as general log, event, or error.
    envname   :	The environment name where the log was generated, providing context for the log entry.
    procname  :	The name of the process that generated the log entry.
    procid 	  : The process ID associated with the log entry.
    module    : The module from which the log entry originated, indicating the specific part of the system.
    keywname  :	Name of any keyword associated with the log entry, if applicable. It is always paired with keywvalue
    keywvalue : Value of the keyword mentioned in keywname, if applicable.
    keywmask  : Mask or additional context for the keyword, if applicable.
    logtext   :	The actual text of the log entry, providing detailed information about the event or action.
    trace_id  :	A unique identifier associated with each log entry, corresponds to id in metadata table.
    Params:
        - INSTRUMENT (str): may be "PIONIER", "MATISSE", "GRAVITY".
        - RANGE (str): represents time range which are "1d"(daily), "1w"(weekly), "1m"(monthly)
                      and "6m" (six months).
        - trace_id (int): index of the registered event to follow.

    Returns: pandas.DataFrame object containing deatils of the selected event.

    """
    print("Loading Trace for ID: {}".format(trace_id))
    df_all = load_dataset(INSTRUMENT, RANGE)
    df_all = df_all[ df_all['trace_id']==trace_id ]
    print("Done.")
    return df_all


def load_meta(INSTRUMENT: str, RANGE: str) -> pd.DataFrame:
    """ Metadata format: 
    START   : The start timestamp of the template execution in milliseconds.
    END     : The end timestamp of the template execution in milliseconds.
    TIMEOUT : Indicates if the execution exceeded a predefined time limit.
    system 	: The name of the system used (e.g., PIONIER).
    procname: The process name associated with the template execution.
    TPL_ID 	: The filename of the corresponding template file.
    ERROR 	: Indicates if there was an error during execution.
    Aborted : Indicates if the template execution was aborted (manually or because an error).
    SECONDS : The duration of the template execution in seconds.
    TEL 	: The class of telescope used in the observation, in this dataset it is only AT.
    
    Params:
        -INSTRUMENT (str): may be "PIONIER", "MATISSE", "GRAVITY".
        -RANGE (str): represents time range which are "1d"(daily), "1w"(weekly), "1m"(monthly)
                      and "6m" (six months).

    Returns: pandas.DataFrame object with metadata of the instrument in the time range selected. 
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


def preprocesador_texto(log) -> str:
        """Debo definir preprocesamiento de texto aún

        """
        log = re.sub(r'\d+', '', log)  # Eliminar cualquier número.
        #log = re.sub(r'[^a-zA-Z\s]', '', log)  # Eliminar números y símbolos
        #log = re.sub(r'[^a-zA-Z]', ' ', log)  # Conservar solo letras
        return log


class Core():
    """ Esta clase es la base (o esqueleto) para la ejecución de procesos y almacenamiento.
    Los atributos del constructor los utilizaré para el almacenamiento de registros.
    
    Methods:
        - load_meta: llamará a la función load_meta() definida en "task_description.ipynb".
        - load_trace: cargará la traza asociada al índice.

    """
    
    def __init__(self, INSTRUMENT: str = 'MATISSE', RANGE: str = '1d', time_it: bool = False) -> None:
        """ La idea de este "esqueleto" es crear una base flexible para almacenar, leer y procesar
        los datos. Para un determinado DataFrame de metadatos se utiliza el atributo "index" como 
        puntero para realizar la lectura de una traza.
        Debido a la cantidad de datos es importante ir desechando variables que no sean necesarias
        para así mantener lo más despejada posible la memoria RAM.

        Atrs:
        INSTRUMENT (str) : Nombre del instrumento a analizar (Default: MATISSE)
        RANGE (str) : Rango de tiempo a analizar  (Default: '1d')
        meta_data (pandas.Dataframe) : con el formato de metadatos de la función load_meta()
        trace_data (pandas.Dataframe): con el formato de traza de la función load_trace()
        trace_index (int) : traza objetivo.
        """
        self.INSTRUMENT : str = INSTRUMENT
        self.RANGE :str = RANGE
        self.meta_data :pd.DataFrame = load_meta(INSTRUMENT=INSTRUMENT, RANGE=RANGE)

        # Cargar el dataset completo es sacrificar RAM por velocidad en la carga de trazas
        self.dataset: pd.DataFrame = load_dataset(INSTRUMENT=INSTRUMENT, RANGE=RANGE)

        self.trace_data : pd.DataFrame = None  # Se inicializará en None pero se considera que es un pd.DataFrame
        self.trace_index: int = 0
        self.time_it: bool = time_it
        self.clock: Clock = Clock()
        pass

    
    def load_meta(self, INSTRUMENT: str, RANGE: str) -> None:
        """ Cargar el DataFrame de Metadatos para el instrumento e intervalo de tiempo.
        Se guardan en el atributo correspondiente.
        """
        self.INSTRUMENT = INSTRUMENT
        self.RANGE = RANGE
        try:
            self.meta_data = load_meta(INSTRUMENT=self.INSTRUMENT, RANGE=self.RANGE)
            pass # Método vacío, actúa solo sobre los atributos
        except:
            raise ValueError('No se ha podido cargar los datos en el lector (Error en método load_meta)')
    
    
    def load_trace(self, index: int, **kwargs):
        """Para un índice dado que pertenezca a "meta_data" se carga la traza asociada al evento.
        """

        timer = kwargs.get('timer', True)
        if self.time_it and timer: self.clock.begin()
        if index > len(self.meta_data):
            raise ValueError('El número de traza excede a las disponibles:\n Máximo: ',len(self.meta),
                             '\n Solicitado: ',index)
        self.trace_index = index
        try:
            self.trace_data = self.dataset[self.dataset['trace_id']==index]
            if self.time_it and timer: self.clock.stop(func=str(self.load_trace.__name__))
            pass
        except:
            raise ValueError('No se ha podido cargar la traza para el evento registrado')
    

    def trace_tokenization(self, column: str='logtext', setting: str ='RegExp', **kwargs) -> pd.Series:
        """Tokenizar por columnas cada traza.
        Por ahora el objetivo será devolver un pandas.series con la tokenización de 
        cada log de la traza.
        """

        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()

        custom_tokenizer = Custom_Tokenizer()  # Llamar al tokenizador y luego eliminarlo
        if setting == 'RegExp':
            # kwargs: {
            regexp=kwargs.get('regexp',r'\b[a-zA-Z]+\b')
            gaps=kwargs.get('gaps',False)
            # }
            custom_tokenizer.set_regexp_tokenizer(regexp=regexp,gaps=gaps)
        elif setting == 'GenDict':  # Esto aún no está desarrollado
            custom_tokenizer.set_gensim_tokenizer()
        result = self.trace_data[column].apply(lambda x: custom_tokenizer.c_tokenize(x))

        if timer and self.time_it: self.clock.stop(func=self.trace_tokenization.__name__)
        return result
    

    def meta_tokenization(self,**kwargs):
        """Tokenizar todas las trazas en meta_data y agregar en una lista.
        Keyword args:
            'setting' : define tokenizador (default: 'RegExp')
            'regexp' : Expresión regular en caso de elegir setting= 'RegExp' (default: r"\b[a-zA-Z]+\b")
            'gaps' : booleano que decide la omisión de espacios.
        """
        timer = kwargs.get('timer', True)
        if self.time_it and timer: 
            self.clock.begin()

        trace_tokens_list = []
        for k in range(0, len(self.meta_data)):
            self.load_trace(index=k, timer=False)
            ##### kwargs:{
            setting = kwargs.get('setting','RegExp')
            regexp=kwargs.get('regexp',r'\b[a-zA-Z]+\b')
            gaps = kwargs.get('gaps',False)
            ##### } 
            trace_tokens_list.append(self.trace_tokenization(setting=setting, regexp=regexp, gaps=gaps, timer=False).tolist())
        self.meta_data['Trace_tokens'] = trace_tokens_list
        if self.time_it and timer: self.clock.stop(func=self.meta_tokenization.__name__)
        pass

    #def vectorize(self, column: str):
        """ Luego del paso de tokenización lo ideal es realizar una vectorización para
        el análisis de sentimiento, este paso va depender mucho del modelo a elegir.
        El método debe recibir una entrada compatible con la salida del tokenizador y
        una dar un output para aplicar algún modelo de procesamiento de lenguaje.
        """

    
    def meta_vectorization(self,method: str = 'CountVectorizer',**kwargs) -> None:
        """ Agregar columna con representación vectorial de cada sucesión de logs por registro.
        
        El vectorizador por ahora solo asocia cada registro a un vector (utilizando CountVectorizer), 
        debo agregar modificaciones para asociar cada registro a una sucesión de vectores (matriz) 
        lo que permitiría analizar los logs junto a una variable de tiempo, en lugar de solo el resultado 
        final.

        Debe agregarse una reducción de dimensión por PCA
        Nota: Tal vez el dataset deba iniciarse aquí y eliminarse luego de ser utilizado.
        
        Params:
            method (str): define el método de vectorización (por ahora solo hay un CountVectorizer).

        Keyword args:
            'stop_words' (list): Lista de stopwords en formato string. 

        """

        timer = kwargs.get('timer', True)
        if timer and self.time_it: self.clock.begin()
        
        # Vector de clasificaciones
        self.meta_data['Classification'] = self.meta_data['ERROR'].to_numpy().astype(int)

        #self.meta_tokenization(timer=False)  # Tokenizar automáticamente antes de vectorizar ?
        if method == 'CountVectorizer':
            ##### kwargs: {
            stop_words : list = kwargs.get('stop_words', [])
            #####}

            vectorizer = CountVectorizer(stop_words=stop_words)  # Si en algún punto resulta necesario convertir vector a texto habrá que extraer el corpus.
            # Convertir cada sucesión de logs en una cadena de texto.
            self.meta_data['Word_vector'] = self.meta_data['Trace_tokens'].apply(lambda token_list:' '.join([' '.join(tokens) for tokens in token_list]))
            # Asignar un vector a cada sucesión.
            vector_series = vectorizer.fit_transform(self.meta_data['Word_vector'])
            vector_series = vector_series.toarray().tolist() # transformar a lista
            self.meta_data['Word_vector'] = vector_series  # lista de vectores
            if timer and self.time_it: self.clock.stop(self.meta_vectorization.__name__)
        else:
            pass
    
    
    def transfer_matrix(self) -> tuple:
        """ El método debe convertir todas las columnas vectoriales a numpy, por ahora dejaré
        un "parche". Transfiere una columna de vectores a una matriz (debo probar con tensores de mayor orden).
        """
        X = self.meta_data['Word_vector']
        X = np.stack(X.values)
        y = self.meta_data['Classification'].to_numpy()
        return X, y


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
        elif self.setting == 'GenDict':
            log = preprocesador_texto(log)
            self.tokenizer.add_documents([log])
            result = list(self.tokenizer.token2id.values())
        return result
    

    ###################################


#class Custom_Vectorizer():
    """Probablemente se requiera un vectorizador adaptable, si en algún punto se utiliza
    un modelo Word2Vec será necesario aplicar tensores de mayor grado, mientras que una 
    vectorización como tf-idf solo necesitará matrices, por lo que las salidas deben ajustarse
    para ser compatibles."""


class Custom_Pipeline():
    """ Estructura para entrenar varios modelos con distintos parámetros
    """
    def __init__(self) -> None:
        pass
    

class Clock():
    """Esto es solo un cronómetro para medir tiempos de ejecución.
    """
    def __init__(self) -> None:
        self.tic = None
        self.toc = None
        # self.history = {method, time}
        pass

    def begin(self):
        self.tic = time.perf_counter()
        pass

    def stop(self, func: str):
        self.toc = time.perf_counter()
        if self.tic is None:
            return None
        print(func," Execution time: ",self.toc - self.tic," Seconds")
        self.tic = None
        self.toc = None
        pass

