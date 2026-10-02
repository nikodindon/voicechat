"""voicechat — chat console avec LLM local + synthèse vocale Kokoro locale."""

import os
import warnings

__version__ = "1.2.1"

# Bruit de bibliothèques tierces : rien d'actionnable pour l'utilisateur, mais ça
# remplit la console de dizaines de lignes à chaque lancement. On le coupe ici,
# avant tout import de torch / huggingface_hub (ce module est chargé en premier).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=r"dropout option adds dropout")
warnings.filterwarnings("ignore", message=r".*weight_norm.*")
warnings.filterwarnings("ignore", message=r".*sending unauthenticated requests.*")
