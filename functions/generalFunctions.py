import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning, module="optimum")


class ModelRetrieval(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # def __init__(self, persist_directory="chroma_store", model_name=BI_ENCODER_NAME):
    #     if hasattr(self, "initialized") and self.initialized:
    #         return  # Evita re-inicializar
    #     self.persist_directory = os.path.join(os.getcwd(), persist_directory)
    #     self.embedding_fn = HuggingFaceEmbeddings(model_name="./models/" + model_name)
    #     # Cargar la base vectorial una vez
    #     self.db = Chroma(
    #         persist_directory=self.persist_directory,
    #         embedding_function=self.embedding_fn
    #     )
    #     self.llm_client = Client()
    #     self.initialized = True