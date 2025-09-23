from dataclasses import dataclass

@dataclass
class RAGConfig:
    """"
    Configuration d'un RAG léger.

    Attributes:
        chunk_size (int): Nombre maximal de mots par chunk.
        overlap (int): Nombre de mots qui se chevauchent entre chunks.
        model_name (str): Nom du modèle d'embedding Sentence-Transformers.
        top_k (int): Nombre de chunks à récupérer lors de la recherche.
    """
    chunk_size: int = 500
    overlap: int = 50
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3

@dataclass
class Chunk:
    """
    Représente chunk (segment) extrait d'un document
    pour être indexé et utilisé dans un pipeline RAG.

    Attributes:
        text (str): Contenu textuel du chunk, nettoyé et prêt à l'indexation.
        source (str): Identifiant de la source d'origine
        chunk_id (int): Numéro séquentiel du chunk dans la source.
    """
    text: str
    source: str
    chunk_id: int
