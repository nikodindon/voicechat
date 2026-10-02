# voicechat ★

[![tests](https://github.com/nikodindon/voicechat/actions/workflows/tests.yml/badge.svg)](https://github.com/nikodindon/voicechat/actions/workflows/tests.yml)

Chat en console avec un **LLM local** (API OpenAI-compatible) qui **répond aussi à voix haute** grâce à **Kokoro TTS**, exécuté **en local sur la machine du client** (GPU utilisé si disponible).

```
   ┌──────────────────────────┐         HTTP / SSE          ┌───────────────────────┐
   │  voicechat (ce projet)   │  ─────────────────────────► │  serveur LLM          │
   │  console + TTS + audio   │  ◄───────────────────────── │  100.91.114.49:8080   │
   └──────────────────────────┘      tokens en streaming    └───────────────────────┘
              │
              │ Kokoro (torch) + sounddevice
              ▼
        🔊 haut-parleur local
```

L'idée directrice : **le LLM peut tourner ailleurs** (ici sur `niko-tv`), **la voix reste locale**
sur la machine avec laquelle on parle. On ne dépend donc jamais d'un service TTS distant.

---

## 1. État actuel

| Élément | Statut |
|---|---|
| Client console + streaming SSE | ✅ implémenté et vérifié (v0.1) |
| Kokoro TTS local + lecture audio | ✅ implémenté et vérifié (v0.1) |
| Saisie clavier : collage multi-lignes = un seul message | ✅ implémenté et vérifié (v0.1) |
| **Interruption à chaud** (`Ctrl+C` coupe tout, `Échap` coupe la voix) | ✅ implémenté et vérifié (v0.2) |
| **Tokens et débit réels** (usage + timings du serveur) | ✅ implémenté et vérifié (v0.2) |
| **Conversations** (`/save`, `/load`, `--continue`) | ✅ implémenté et vérifié (v0.2) |
| Nom de modèle court à l'affichage | ✅ implémenté et vérifié (v0.2) |
| **Recherche `Ctrl+R` + complétion `Tab`** | ✅ implémenté et vérifié (v0.3) |
| **Lecture audio continue** (flux persistant) | ✅ implémenté et vérifié (v0.3) — ~32 ms gagnées par phrase |
| **Profils de prompt système** (`--profil`, `/profil`) | ✅ implémenté et vérifié (v0.3) |
| **Export markdown** (`/export`) | ✅ implémenté et vérifié (v0.3) |
| **Entrée vocale** : micro + VAD Silero v6 + `faster-whisper` (`--micro`, `/ecoute`) | ⚠️ **expérimental** — transcription parfaite depuis un fichier (RTF 0,11 sur GPU), mais la capture micro est hachée par PortAudio sur cette machine : cause racine non identifiée, `parec` est propre (cf. §10) |
| Transcription d'un fichier, diagnostic micro (`--transcrire`, `--diag-micro`) | ✅ implémenté et vérifié (v0.4) |
| **Nettoyage markdown avant synthèse** (titres, listes, tableaux, liens, code) | ✅ implémenté et vérifié (v0.5) — mesuré de bout en bout sur ce qui part à la synthèse, cf. §11 |
| **Unités et abréviations FR** (`%`, `°C`, `€`, `km/h`, `M.`, `Mme`, `n°`) | ✅ implémenté et vérifié (v0.5) |
| **Mélange de voix pondéré** (`ff_siwis:3+ef_dora:1`) | ✅ implémenté et vérifié (v0.5) — intelligibilité mesurée par Whisper, cf. §11 |
| **Voix portée par le persona** (en-tête de profil) | ✅ implémenté et vérifié (v0.5) |
| **Réessais avec délai croissant** (connexion refusée, 5xx, 429) | ✅ implémenté et vérifié (v0.6) — 3 essais, 0,5 s → 1 s → 2 s |
| **Serveurs de secours** (`--secours`, `VOICECHAT_SECOURS`) | ✅ implémenté et vérifié (v0.6) — bascule annoncée à l'écran, jamais silencieuse |
| **Mode dégradé** : phrases non synthétisées gardées et rejouables (`/rejoue`) | ✅ implémenté et vérifié (v0.6) |
| **`--dire` / `--dire-fichier`** : lire un texte sans le LLM | ✅ implémenté et vérifié (v1.1) — par son WAV dans la réserve, cf. §14.1 |
| **`/cherche <mot>`** dans les conversations sauvées (accents compris) | ✅ implémenté et vérifié (v1.1) — 17 tests, cf. §14.2 |
| **`/resume`** : compacter les vieux échanges au lieu de les perdre | ✅ implémenté et vérifié (v1.1) — 1,1× à 1,6× selon la densité du texte, cf. §14.3 |
| **`-q`** : mode non-interactif, scriptable (cron, autre programme) | ✅ implémenté et vérifié (v1.1) — testé **sans terminal**, cf. §14.4 |
| **Intégration continue** (pytest + contrôle « sans paquets lourds ») | ✅ en place (v1.1) — la suite tourne sans torch, Kokoro ni GPU, cf. §14.5 |
| **Réserve d'audio** : une phrase déjà dite n'est pas resynthétisée | ✅ implémenté et mesuré (v1.0) — 282 ms → ~0 ms par phrase répétée, cf. §13.1 |
| **Paquet installable** (`pip install ./voicechat`, commande `voicechat`) | ✅ vérifié (v1.0) — installation et point d'entrée réellement exécutés, cf. §13.2 |
| **Fichier de configuration TOML** + réglages par modèle et par voix | ✅ implémenté et vérifié (v1.0) — précédence testée, cf. §13.3 |
| **Serveur TTS partagé** (`--serveur-tts` / `--tts-distant`) | ⚠️ fonctionnel et vérifié (v1.0), audio retranscrit à 100 % — mais testé en boucle locale seulement, pas entre deux machines, cf. §13.4 |
| Sélection GPU `auto/cuda/cpu` | ✅ **GPU opérationnel** — RTF 0,09 (cf. §8 pour l'obligation de build cu126 sur Pascal) |
| Serveur LLM `100.91.114.49:8080` (niko-1650-super) | ✅ **joignable** — Ornith-1.5-35B-A3B Q4_K_M |
| Suite de tests hors ligne | ✅ 275 tests — et 271 passent même **sans torch, Kokoro ni faster-whisper** |

> **Cible réelle du serveur LLM** — `100.91.114.49` = `niko-1650-super` dans le tailnet,
> llama.cpp exposant une API OpenAI-compatible :
>
> ```
> $ tailscale status | grep 100.91.114.49
> 100.91.114.49   niko-1650-super   nikodindon@   linux   active; direct 192.168.1.32:41641
>
> $ curl -s -m 10 http://100.91.114.49:8080/v1/models | python3 -m json.tool | head -18
> {
>     "models": [
>         {
>             "name": "/mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf",
>             "model": "/mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf",
>             "modified_at": "",
>             "size": "",
>             "digest": "",
>             "type": "model",
>             "description": "",
>             "tags": [
>                 ""
>             ],
>             "capabilities": [
>                 "completion"
>             ],
>             "parameters": "",
>             "details": {
> ```
>
> Le nom du modèle renvoyé par llama.cpp est un **chemin de fichier complet**. Le client le
> reprend tel quel depuis `/v1/models` — c'est exactement pourquoi il ne faut pas le coder en
> dur dans une constante.

> **Note historique** : la v0.1 a d'abord été développée en pointant sur `100.108.224.60`
> (`niko-tv`), qui était en fait **éteint depuis 2 jours** — et surtout n'était pas la bonne
> machine. Le §9 conserve la trace de ce diagnostic raté et de la validation intermédiaire
> contre `tests/fake_llm_server.py`, car le chemin de code exercé était le même.

---

## 2. Pré-requis

- Python ≥ 3.10 (testé sur 3.12.3)
- Un serveur LLM exposant `/v1/chat/completions` en streaming (llama.cpp, vLLM, Ollama…)
- Facultatif : un GPU NVIDIA (testé sur **GTX 1050, 4 Go, driver 580.173.02**)
- Facultatif : `espeak-ng` (meilleure prononciation des mots hors dictionnaire).
  `misaki` embarque une copie de la bibliothèque via le paquet pip `espeakng-loader`,
  donc **ce n'est pas bloquant** pour l'anglais et le français.

---

## 3. Installation

```bash
cd ~/projects/voicechat
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` :

```
kokoro         # TTS (tire torch automatiquement)
sounddevice    # lecture audio (PortAudio)
numpy
faster-whisper # reconnaissance vocale locale (v0.4) — tire ctranslate2 et onnxruntime
av==18.1.0     # épinglé : voir « Pièges d'installation » en §10
soundfile      # écriture de WAV (tests)
pytest
```

⚠️ La borne sur `av` n'est pas cosmétique : `faster-whisper` déclare `av` sans version,
donc `pip` installe **PyAV 19**, avec qui il plante à la première transcription
(`metadata_errors`). Détail et trace en **§10**.

### ⚠️ Étape obligatoire sur GTX 1050 / Pascal (sm_61)

`pip install kokoro` tire le `torch` par défaut de PyPI, construit en **CUDA 13.0**, qui ne
contient **plus** les noyaux Pascal. Il faut le remplacer par le build `cu126` :

```bash
.venv/bin/pip uninstall -y $(.venv/bin/pip freeze | \
    grep -iE "^(torch|triton|nvidia-|cuda-toolkit|cuda-bindings)" | cut -d= -f1)

.venv/bin/pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126

# contrôle : la liste doit contenir sm_50 et sm_60, et le calcul doit passer
.venv/bin/python -c "import torch; print(torch.cuda.get_arch_list()); \
    a=torch.zeros(8,device='cuda'); print('GPU OK ->', (a+1).sum().item())"
```

Attendu :

```
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
GPU OK -> 8.0
```

Sur Turing (20xx) ou plus récent, cette étape est **facultative** : le wheel par défaut suffit.

> Le premier lancement télécharge les poids Kokoro depuis Hugging Face (~350 Mo)
> dans `~/.cache/huggingface`. C'est fait **une seule fois**.

---

## 4. Configuration

Tout se règle par variable d'environnement (voir `.env.example`) **ou** par option de ligne
de commande (l'option gagne). Aucune clé n'est obligatoire : llama.cpp ignore `Authorization`.

| Variable | Défaut | Rôle |
|---|---|---|
| `VOICECHAT_BASE_URL` | `http://100.91.114.49:8080/v1` | racine de l'API OpenAI-compatible |
| `VOICECHAT_SECOURS` | *(vide)* | autres serveurs, séparés par des `,` — essayés dans l'ordre (cf. §12) |
| `VOICECHAT_MODEL` | *(auto)* | nom du modèle ; si vide → détecté via `/v1/models` |
| `VOICECHAT_API_KEY` | *(vide)* | jeton éventuel |
| `VOICECHAT_VOICE` | `ff_siwis` | voix Kokoro, ou mélange pondéré (`ff_siwis:3+ef_dora:1`) — cf. §11 |
| `VOICECHAT_LANG` | `f` | code langue Kokoro (`a`=en, `f`=fr, `e`=es, `i`=it, `p`=pt, `j`=ja, `z`=zh, `h`=hi) |
| `VOICECHAT_SPEED` | `1.0` | vitesse de lecture |
| `VOICECHAT_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOICECHAT_SYSTEM` | *(court prompt FR)* | prompt système |
| `VOICECHAT_TTS` | `1` | `0` pour désactiver la voix |
| `VOICECHAT_CACHE` | `1` | `0` = resynthétiser chaque phrase (cf. §13.1) |
| `VOICECHAT_CACHE_DIR` | `~/.cache/voicechat/tts` | où ranger l'audio déjà synthétisé |
| `VOICECHAT_TTS_URL` | *(vide)* | synthèse déportée sur un autre poste (cf. §13.4) |
| `VOICECHAT_CONFIG` | `~/.config/voicechat/config.toml` | fichier de configuration TOML (cf. §13.3) |
| `VOICECHAT_TEMPERATURE` | `0.7` | température d'échantillonnage |
| `VOICECHAT_MAX_TOKENS` | `0` | `0` = laisser le serveur décider |
| `VOICECHAT_TIMEOUT` | `300` | délai de lecture du flux LLM, en secondes |
| `VOICECHAT_DATA` | `~/.local/share/voicechat` | dossier des conversations sauvegardées |
| `VOICECHAT_PROFILS` | `~/.config/voicechat/profils` | dossier des profils de prompt système |
| `VOICECHAT_PROFIL` | *(vide)* | profil à charger au démarrage |
| `VOICECHAT_MICRO` | `0` | `1` = démarrer en mains libres |
| `VOICECHAT_STT_MODELE` | `small` | modèle faster-whisper (`base`, `small`, `medium`, `large-v3`) |
| `VOICECHAT_STT_DEVICE` | `auto` | appareil de transcription : `auto` / `cuda` / `cpu` |
| `VOICECHAT_STT_LANGUE` | `fr` | langue forcée ; vide = détection automatique |
| `VOICECHAT_MICRO_DEVICE` | *(défaut)* | périphérique d'entrée (nom ou index) |

---

## 5. Utilisation

```bash
# Lancement normal (voix + GPU auto)
.venv/bin/python -m voicechat

# Vérifier la joignabilité du serveur sans lancer le chat
.venv/bin/python -m voicechat --probe

# Chat muet (utile pour tester le LLM seul)
.venv/bin/python -m voicechat --no-tts

# Lister les voix Kokoro disponibles
.venv/bin/python -m voicechat --list-voices

# Reprendre la dernière conversation sauvegardée
.venv/bin/python -m voicechat --continue

# Charger un profil de prompt système
.venv/bin/python -m voicechat --profil brain-wash

# Changer de cible (utile pour tester contre un autre llama-server)
.venv/bin/python -m voicechat --base-url http://192.168.1.32:8080/v1

# Forcer le CPU (pas le GPU) pour ne pas chauffer pendant une longue session
.venv/bin/python -m voicechat --device cpu

# Voix anglaise, débit ralenti
.venv/bin/python -m voicechat --voice af_heart --lang a --speed 0.9
```

### Commandes en session

| Commande | Effet |
|---|---|
| `/help` | aide |
| `/quit` (`/q`) | quitter |
| `/reset` | vide l'historique de conversation |
| `/voice <nom>` | change la voix, ou mélange : `/voice ff_siwis:3+ef_dora:1` |
| `/lang <code>` | change la langue Kokoro |
| `/speed <x>` | change la vitesse (ex. `/speed 1.15`) |
| `/tts on\|off` | active/coupe la voix |
| `/model <nom>` | change de modèle pour les tours suivants |
| `/system <texte>` | remplace le prompt système |
| `/profil [nom]` | liste les profils, ou en charge un |
| `/profil save <nom>` | enregistre le prompt système courant comme profil |
| `/save [nom]` | enregistre la conversation (défaut : `derniere`) |
| `/load <nom>` | recharge une conversation sauvegardée |
| `/conversations` | liste les conversations sauvegardées |
| `/cherche <mot>` | cherche un passage dans les conversations sauvegardées (accents ignorés) |
| `/forget <nom>` | supprime une conversation sauvegardée |
| `/export [fichier]` | écrit la conversation en markdown |
| `/voices` | liste les voix Kokoro |
| `/device` | liste les sorties audio détectées |
| `/ecoute` | écoute le micro et envoie ce qui est dit (puis retour clavier) |
| `/micro on\|off` | bascule le mode mains libres |
| `/rejoue` | réentend les phrases que la synthèse n'avait pas pu produire |
| `/resume` | compacte les vieux échanges en un résumé (au lieu de les perdre) |
| `/cache` | état de la réserve d'audio déjà synthétisé (`/cache vider`) |
| `/stats` | latences (TTFT, débit en tokens, RTF TTS) |
| `/debug` | bascule l'affichage des stats à chaque tour |

### Entrée vocale (v0.4 — ⚠️ expérimental)

```bash
# mode mains libres dès le lancement
.venv/bin/python -m voicechat --micro

# en session : /ecoute dicte un seul message, puis rend la main au clavier
vous › /ecoute
```

Pendant l'écoute, chaque bloc de 32 ms jugé « parole » par le VAD affiche un `●` : on voit
que le micro est entendu **avant** de connaître la transcription.

La chaîne complète (écoute → découpage → transcription) est mesurée par
`tests/verif_ecoute.py`, lancé pour de vrai :

```bash
$ .venv/bin/python tests/verif_ecoute.py
VAD     : silero v6 chargé (0,02 sur silence, 0,74 sur parole — cf. README)
VRAM    : 363 MiB

[1/3] phrase à reconnaître : « Allume la lumière du salon et vérifie que la porte est fermée. »
      audio de 4.05 s préparé (/tmp/verif_ecoute.wav)

[2/3] écoute en parallèle de la lecture (boucle monitor)
      ● parole détectée
      énoncé capturé : 3.94 s (source 4.05 s)

[3/3] transcription
      3.94 s d'audio transcrites en 0.60 s (RTF 0.15) — langue fr (100%)
      attendu : Allume la lumière du salon et vérifie que la porte est fermée.
      obtenu  : Aligne l'alignement du salin et vérifie que la porte est fermée.
```

La fin est juste, le début est haché : c'est le défaut de capture décrit en **§10**. Le
script ne masque pas l'écart, il le montre — c'est tout son intérêt.

Diagnostic et transcription de fichier, sans conversation :

```bash
.venv/bin/python -m voicechat --diag-micro          # ton micro est-il exploitable ?
.venv/bin/python -m voicechat --transcrire a.wav    # transcrire un fichier
```

**À lire avant d'essayer** : sur cette machine la capture micro est hachée (PortAudio) et le
micro écrête au gain par défaut. Les deux problèmes, leurs mesures et les contournements
sont en **§10** — c'est la section à lire en premier si la reconnaissance vous déçoit.

### Interrompre une réponse

| Touche | Effet |
|---|---|
| `Échap` | coupe **la voix seulement** : la réponse continue d'arriver à l'écran |
| `Ctrl+C` | coupe **tout** : génération, synthèse et lecture |

`Ctrl+C` ferme la connexion HTTP, invalide ce qui est en cours de synthèse et vide la file
de lecture. La session survit : on retombe sur le prompt, et la réponse partielle reçue est
conservée dans l'historique (c'est ce que l'utilisateur a lu).

Les touches frappées pendant une réponse ne sont pas perdues : elles réapparaissent
pré-remplies au prompt suivant.

### Conversations

```bash
vous › /save projet                # → ~/.local/share/voicechat/conversations/projet.json
vous › /conversations
conversations dans /home/niko/.local/share/voicechat/conversations :
  projet                         7 messages  02/10 12:10  Ornith-1.5-35B-A3B-APEX-i-mini
vous › /load projet
```

Un fichier JSON par conversation, écrit de façon atomique (jamais de fichier à moitié
écrit). Le nom est assaini : `/save ../../etc/passwd` ne peut pas sortir du dossier de
données. Au lancement, `--continue` reprend la plus récente.

### Profils de prompt système

Un profil est un simple fichier texte dans `~/.config/voicechat/profils/`. Le but : ne plus
retaper un prompt système long à chaque session.

```bash
vous › /system Tu réponds en français, en phrases courtes, sans markdown.
vous › /profil save court        # → ~/.config/voicechat/profils/court.md

# la fois suivante :
$ voicechat --profil court
Profil : court
```

`/profil` seul liste les profils disponibles en marquant celui qui est actif. `/profil <nom>`
en charge un à chaud (le message système est remplacé immédiatement). `Tab` complète les noms.

### Export markdown

```bash
vous › /export projet            # → ./projet.md
exporté : /home/niko/projet.md  (7 messages)
```

Le fichier contient la date, le modèle, la voix, le prompt système puis les échanges
alternés en `**Vous**` / `**Assistant**` — directement lisible ou collable ailleurs.

### Stats de la dernière réponse

```
ia › Bonjour !
   · 1er token 0.74 s | 3 tok (+54 prompt) | 30.6 tok/s
```

Les compteurs viennent du serveur (`usage` + `timings.predicted_per_second` de llama.cpp) :
c'est la seule mesure qui permette de comparer deux modèles. Si le serveur ne les fournit
pas, l'affichage retombe sur les caractères — sans jamais mentir avec un « 0 tok ».

### Saisie clavier et collage

`input()` ne convenait pas : quand on colle un texte de plusieurs lignes, le terminal
livre chaque ligne comme si l'utilisateur les avait tapées une par une. `input()` en
consommait une, et la boucle traitait toutes les suivantes comme autant de nouveaux
messages. Mesuré, avec l'ancien code :

```
prompt:   -> tour 1 : 'Ligne un'
  -> tour 2 : 'Ligne deux'
  -> tour 3 : 'Ligne trois'
```

`voicechat/editor.py` lit donc le clavier en **mode brut** et distingue trois choses :

* une **frappe** — le texte s'affiche au fil de l'eau ;
* un **collage**, repéré par les marqueurs du *bracketed paste* (`\x1b[?2004h`, activé
  par le programme) : les sauts de ligne sont conservés **à l'intérieur** du message et
  **rien n'est envoyé** ;
* la touche **Entrée**, seul et unique déclencheur d'envoi.

| Touche | Effet |
|---|---|
| `Entrée` | envoyer le message |
| `↑` / `↓` | historique des messages |
| `Ctrl+R` | rechercher dans l'historique — `Ctrl+R` à nouveau remonte plus loin, `Ctrl+G` abandonne |
| `Tab` | compléter les commandes **et leurs arguments** |
| `←` / `→` | déplacer le curseur |
| `Ctrl+U` / `Ctrl+W` | effacer la ligne / le mot précédent |
| `Ctrl+C` | effacer le brouillon (sur ligne vide : quitter) |
| `Ctrl+D` | quitter |

La complétion connaît le contexte : `Tab` après `/voice ` propose les voix Kokoro, après
`/load ` les conversations sauvegardées, après `/profil ` les profils. La liste des
commandes utilisée par `Tab` est **générée depuis le même tableau que `/help`** : les deux
ne peuvent pas diverger. Un `Tab` qui n'a qu'un candidat complète et ajoute l'espace ; s'il
y en a plusieurs, on complète jusqu'au plus long préfixe commun, et un second `Tab` affiche
les choix.

Un texte collé est résumé à l'écran plutôt que redessiné frappe par frappe :

```
vous › [12 lignes, 842 car.] Voici un long texte que je te copie-colle…  ⏎ envoyer · Ctrl+C annuler
```

**Repli** : si le terminal n'implémente pas le bracketed paste, un paquet de plusieurs
octets (ou deux lectures à moins de 30 ms) est considéré comme un collage — un humain écrit
un octet à la fois, il n'en envoie pas quinze d'un coup. **Conséquence à connaître** : une
ligne entière livrée d'un seul bloc (ce que fait un script, pas un clavier) est vu comme un
collage, donc son saut de ligne est conservé et il faut refaire `Entrée`. C'est bénin — aucune
donnée n'est perdue — mais ça explique pourquoi un collage et une frappe ne se ressemblent pas.

**Limite assumée** : au-delà d'une ligne écran le message est résumé au lieu d'être
redessiné — l'édition (retour arrière, `Ctrl+U`) reste exacte, seul l'affichage est
condensé.

En entrée **non** interactive (tube, redirection), le comportement est inchangé : une
ligne = un message. Un script qui écrit plusieurs lignes attend plusieurs tours de
conversation, ce n'est pas un collage.

---

## 6. Architecture du code

```
voicechat/
├── __main__.py     # python -m voicechat
├── config.py       # Config : env + args → objet unique
├── llm.py          # client OpenAI-compatible, streaming SSE (urllib, zéro dépendance)
├── text.py         # découpage du flux en phrases + nettoyage markdown (pur, testable)
├── tts.py          # KokoroTTS (synthèse) + SpeechPipeline (file de phrases → voix → son)
├── audio.py        # Speaker : file de tampons audio + thread de lecture PortAudio
├── editor.py       # lecture clavier en mode brut : frappe / collage / Entrée
├── micro.py        # capture micro + VAD Silero v6 + découpage en énoncés (v0.4)
├── stt.py          # transcription faster-whisper + choix du type de calcul (v0.4)
├── store.py        # conversations sauvegardées (JSON, écriture atomique)
└── cli.py          # boucle console, routage des commandes, orchestration
tests/              # pytest hors ligne, faux serveur OpenAI, vérifications en pseudo-terminal
```

**Pourquoi cette découpe ?** Le pipeline voix doit se déclencher **pendant** la génération,
pas après : sinon on attend la fin de la réponse avant d'entendre le premier mot. Le flux est
donc découpé en phrases au vol (`text.split_sentences`), chaque phrase part dans une file
(`tts.SpeechPipeline`) consommée par un thread qui synthétise puis empile le son dans
`audio.Speaker`. Un seul thread de synthèse → les phrases restent dans l'ordre, et le GPU
n'est jamais appelé en concurrence.

**Pourquoi un flux audio persistant ?** `sounddevice.play()` ouvre puis referme un flux
PortAudio à chaque appel, soit ~35 ms de silence **par phrase** — c'est exactement le « trou »
qu'on entend entre deux phrases. `audio.Speaker` ouvre donc un flux unique pour toute la
session et le remplit par un rappel (*callback*). Deux bénéfices : plus de latence de mise en
route, et une comptabilité exacte des échantillons réellement sortis — « attendre la fin de la
voix » ne repose plus sur une approximation. Le coût de la contrepartie (un flux toujours
ouvert) est mesuré en §9 : 0,35 % d'un cœur au repos.

### Tests (sans réseau, sans audio)

```bash
.venv/bin/python -m pytest tests/ -q
```

Pour valider la v0.1 sans dépendre du serveur distant, un faux serveur OpenAI-compatible
est fourni :

```bash
python3 tests/fake_llm_server.py 8099          # terminal 1
.venv/bin/python -m voicechat --base-url http://127.0.0.1:8099/v1 --no-tts   # terminal 2
```

Trois scripts de vérification bout en bout pilotent le vrai CLI dans un pseudo-terminal
(ils choisissent eux-mêmes un port libre et démarrent leur propre faux serveur) :

```bash
.venv/bin/python tests/verif_collage.py       # un collage = un seul message
.venv/bin/python tests/verif_interruption.py  # Ctrl+C / Échap
.venv/bin/python tests/verif_completion.py    # Tab / Ctrl+R
.venv/bin/python tests/bench_audio.py         # surcoût de lecture par phrase
.venv/bin/python tests/verif_ecoute.py        # micro → VAD → transcription (boucle monitor)
.venv/bin/python tests/bench_capture.py       # notre capture vs parec, par corrélation
.venv/bin/python tests/verif_nettoyage.py     # ce qui part VRAIMENT à la synthèse (markdown)
.venv/bin/python tests/bench_voix.py          # mélanges de voix : intelligibilité + timbre
.venv/bin/python tests/verif_profil_voix.py   # un profil porte bien sa voix
.venv/bin/python tests/verif_reprise.py       # bascule réseau, dans le vrai CLI
.venv/bin/python tests/verif_sans_lourds.py   # toute la suite, sans torch/Kokoro/whisper
.venv/bin/python tests/verif_confort.py       # -q, --dire, /cherche, /resume
.venv/bin/python tests/verif_config.py        # précédence TOML/env/arguments
```

---

## 7. Roadmap

### v0.1 — socle ✅ (ce commit)
- [x] Client console avec streaming token par token
- [x] Historique de conversation + `/reset`
- [x] Kokoro TTS local, lecture en file d'attente pendant la génération
- [x] Sélection du périphérique `auto/cuda/cpu`, repli CPU propre
- [x] Détection GPU par **vrai calcul** (test de fumée CUDA) — `is_available()` seul ne suffit pas
- [x] Détection automatique du nom du modèle via `/v1/models`
- [x] `--probe` : diagnostic réseau clair quand le serveur est éteint
- [x] Messages d'erreur lisibles (timeout, 404 API, modèle inconnu)
- [x] Saisie clavier maison : un collage multi-lignes reste **un seul message**, Entrée seule déclenche l'envoi
- [x] Tests unitaires hors ligne + faux serveur OpenAI-compatible

### v0.2 — confort d'usage ✅ (ce commit)
- [x] Interruption à chaud : `Ctrl+C` coupe la génération **et** la voix, `Échap` coupe la voix seule
- [x] Afficher un **nom de modèle court** au lieu du chemin complet renvoyé par llama.cpp
- [x] Comptage **réel** des tokens (`usage` + `timings`) avec repli sur les caractères
- [x] `/save`, `/load`, `/conversations`, `/forget` et reprise au lancement (`--continue`)
- [x] Les frappes faites pendant une réponse sont conservées et rendues au prompt suivant

### v0.3 — confort restant ✅ (ce commit)
- [x] Historique de saisie : recherche `Ctrl+R` et complétion des `/commandes` (Tab)
- [x] Streaming audio par morceaux : flux de sortie persistant (plus de trou entre phrases)
- [x] Profils de prompt système (`--profil`, `/profil`, `/profil save`)
- [x] Export d'une conversation en markdown (`/export`)

### v0.4 — entrée vocale (mode mains libres) ⚠️ expérimental
- [x] Capture micro (`sounddevice`) + VAD (détection d'activité vocale)
- [x] `faster-whisper` en local pour la reconnaissance (GPU, `small`)
- [x] Boucle « on parle → transcription → LLM → voix » (`--micro`)
- [ ] **Qualité de capture** : PortAudio hache la parole sur cette machine (`parec` est
      propre sur la même source). Cause racine non identifiée — voir §10.

### v0.5 — voix de meilleure qualité ✅
- [x] **Normalisation des réponses markdown avant synthèse** (listes, code, liens, tableaux)
- [x] **Césures/abréviations et unités FR** (nombres, sigles, `%`, `°C`, `€`)
- [x] Mélange de voix Kokoro, pondéré (`ff_siwis:3+ef_dora:1`)
- [x] Sélection de voix par persona (en-tête `voix:` du profil)

### v0.6 — robustesse réseau ✅
- [x] Reconnexion automatique + réessais à délai croissant si le serveur redémarre
- [x] Bascule automatique vers un serveur de secours (`--secours`, `VOICECHAT_SECOURS`)
- [x] Mode dégradé : phrases non synthétisées **gardées** au lieu d'être jetées, rejouables
      avec `/rejoue` — le texte reste à l'écran dans tous les cas

### v1.0 — distribué
- [x] Cache disque des phrases déjà synthétisées (le LLM se répète souvent)
- [x] Paquet `pip install ./voicechat` + point d'entrée `voicechat`
- [x] Fichier de config TOML + réglages par modèle et par voix
- [x] Serveur TTS partagé (le poste léger envoie le texte, un poste GPU synthétise)

**Reste ouvert, d'avant la v1.0** : la capture micro hachée (§10). C'est le seul point
du projet qui ne soit pas résolu.

### v1.1 — confort d'usage ✅

Une soirée de travail, effet immédiat, aucun risque : ces quatre-là ne touchent ni au
réseau, ni au GPU, ni au découpage du texte.

- [x] **`--dire "texte"` / `--dire-fichier part.txt`** — parler sans passer par le LLM.
      C'est l'outil qui manque pour comparer deux voix côte à côte ou se faire lire un
      document. Réutilise le nettoyage markdown et la file de synthèse déjà là.
- [x] **`/cherche <mot>`** dans les conversations sauvegardées — les fichiers sont déjà
      sur disque, il ne manquait que la recherche. Insensible à la casse **et aux accents**.
- [x] **`/resume`** — compacter l'historique quand le contexte se remplit. Vrai besoin, pas
      du confort : la fenêtre est de 32 768 tokens et la seule issue était `/reset`,
      donc tout perdre. L'historique qui débordait le dit maintenant, au lieu de laisser
      tomber des messages en silence.
- [x] **`-q "question"`** — mode non-interactif : répond et sort. C'est ce qui rend le
      projet scriptable (cron qui annonce un rappel, appel depuis un autre projet).
- [x] **Intégration continue** — pytest sur chaque push + badge dans le README. La suite
      tourne **sans torch, sans Kokoro, sans faster-whisper et sans GPU** : un contrôle
      dédié le prouve et échoue si quelqu'un ajoute un import lourd en haut d'un module.

### v1.2 — deux voix

Le « waouh » pour le moins d'effort : la plomberie existe déjà (mélange de voix pondéré,
voix portée par le profil, réserve d'audio, serveur TTS). Il ne reste que le prompt et
l'ordonnancement.

- [ ] **Mode dialogue** : deux voix alternées — question/réponse, ou deux personas qui
      se relancent. Chaque voix garde son profil (voix, vitesse, prompt, modèle).
- [ ] **`/enregistre`** : la session entière en un seul fichier audio. La réserve contient
      déjà chaque phrase en WAV, il suffit de les recoller dans l'ordre.
- [ ] **Voix automatique selon la langue** : Kokoro couvre 8 langues, mais `lang` est figé
      à `f` ; quand le modèle répond en anglais, c'est la voix française qui le lit.

### v1.3 — l'assistant dans le navigateur, et dans la poche

Le serveur TTS de la v1.0 parle déjà HTTP : cette version rend le reste du service
accessible, et fait entrer le téléphone dans la boucle.

- [ ] **Interface web** (chat + lecture audio) servie par `voicechat --web` : on ouvre une
      page, on tape, on entend. Rayon d'action immédiat depuis n'importe quelle machine
      du tailnet.
- [ ] **Service STT distant** (`--serveur-stt`), symétrique du serveur TTS : le téléphone
      envoie l'**audio**, le poste GPU renvoie le **texte**. Sans ça, un client léger doit
      embarquer Whisper — ce qui est justement ce qu'on veut éviter.
- [ ] **Installable depuis le navigateur** (PWA) : utilisable sur le téléphone sans store,
      et sans rien compiler.

### v2.0 — l'assistant qui agit

Le saut de nature : passer de « il répond » à « il fait ». llama.cpp expose déjà l'API
d'outils OpenAI, et ARIA utilise déjà ce motif (`[DELEGATE]` → sous-processus).

- [ ] **Outils / function calling** : lire un fichier, chercher sur le web, lancer une
      commande.
- [ ] **Garde-fous** : ce qui peut s'exécuter sans demander, et ce qui exige une
      confirmation explicite. Le point délicat n'est pas technique.
- [ ] **RAG local** sur les documents : indexer notes et projets pour répondre à
      « qu'est-ce que je disais sur X ? ». Le morceau le plus profond — découpage, index,
      seuil de pertinence — à ne lancer que si le besoin est réel.

### v3.0 — application Android

L'aboutissement logique du mode distribué : le téléphone est déjà dans le tailnet, donc
il n'a besoin ni de GPU, ni de modèle local — juste d'être sur le réseau.

- [ ] **Application Android** qui parle aux postes du tailnet : LLM sur un poste, Kokoro sur
      l'autre (ou le même), Whisper ailleurs. Tous les services existent déjà après la
      v1.3 : API OpenAI-compatible, `--serveur-tts`, `--serveur-stt`.
- [ ] **Découverte des postes** : dire une fois où sont LLM / TTS / STT, garder ça en
      configuration. Le tailnet donne des adresses `100.x.y.z` stables mais il y a
      plusieurs machines (niko-1650-super, niko-tv, niko-nitro-an515-52) : ne pas confondre.
- [ ] **Mode voiture / casque** : dialogue mains libres de bout en bout, ce qui suppose
      que la capture micro soit enfin réglée (voir ci-dessous).

### Chantier ouvert depuis la v0.4 : la capture micro

Ce n'est pas une fonctionnalité, c'est le blocage qui décide de la suite. PortAudio hache
la parole sur cette machine alors que `parec` est propre sur la même source : cause racine
non identifiée (§10).

Tant qu'il tient, trois idées restent **bloquées**, même si elles sont séduisantes :

- la **dictée continue** (parler pour écrire, sans le LLM) ;
- le **barge-in** (interrompre l'assistant en parlant, plutôt qu'avec `Ctrl+C`) ;
- le **mot de réveil** (« hey … ») qui rendrait le mode mains libres vraiment utilisable.

`tests/bench_capture.py` est prêt pour s'y attaquer : il compare notre capture à `parec`
par corrélation. La première étape n'est pas de corriger, c'est de comprendre.

---

## 8. Dépannage

**`[ERREUR] Connexion impossible à http://100.91.114.49:8080/v1`**
La machine distante est éteinte, ou Tailscale est down, ou l'IP a changé (les adresses
`100.x.y.z` du tailnet sont **stables** mais il y a plusieurs machines : ne pas confondre
`niko-tv`, `niko-1650-super` et `niko-nitro-an515-52`).
```bash
tailscale status | grep 100.91.114.49    # « offline, last seen … » ?
tailscale ping 100.91.114.49
```
Un serveur LLM doit écouter en `0.0.0.0` sur le port 8080, sinon Tailscale ne le voit pas :

```bash
# exemple llama.cpp côté niko-1650-super
llama-server -m modele.gguf --host 0.0.0.0 --port 8080
```

Astuce diagnostic : `--probe` teste la cible **sans** charger Kokoro. Séparer les deux moitiés
du problème (réseau vs GPU/TTS) fait gagner beaucoup de temps.

**`Torch not compiled with CUDA enabled` / repli CPU silencieux**
Vérifier ce que voit PyTorch :
```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**⚠️ Piège spécifique GTX 1050 (Pascal, sm_61) — le point le plus important de ce README**

Le wheel `torch` installé par défaut sur PyPI est construit pour **CUDA 13.0**, qui a
**retiré le support de Pascal**. Conséquence vicieuse : `torch.cuda.is_available()` répond
**`True`**, donc le code croit que le GPU est disponible, et l'échec n'arrive qu'à la
première opération :

```
torch 2.14.1+cu130 | CUDA compilé: 13.0
architectures supportées par ce wheel: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
is_available(): True
GPU: NVIDIA GeForce GTX 1050 | capability (6, 1)
ECHEC CUDA: AcceleratorError CUDA error: no kernel image is available for execution on the device
```

`KokoroTTS.load()` fait donc un **test de fumée** (`zeros(8)` + addition sur GPU) et
retombe sur le CPU avec un message explicite plutôt que de planter au milieu d'une réponse.

Pour retrouver le GPU, il faut un build torch qui contient encore les noyaux sm_61
(cu126 ou antérieur). Le dépôt PyTorch le suggère lui-même dans l'avertissement — **mais la
commande qu'il propose ne fonctionne pas telle quelle** :

```bash
$ .venv/bin/pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
Requirement already satisfied: torch==2.14.1 ... (2.14.1+cu130)
# → pip ne fait RIEN : la version 2.14.1+cu130 satisfait déjà la contrainte "==2.14.1",
#   le tag local (+cu130) n'entre pas dans l'égalité. On croit avoir corrigé, on n'a rien changé.
```

Il faut viser la version exacte, avec son tag local, et désinstaller l'ancienne au préalable :

```bash
# 1. vérifier ce que l'index propose vraiment
$ .venv/bin/pip index versions torch --index-url https://download.pytorch.org/whl/cu126
torch (2.14.1+cu126)
Available versions: 2.14.1+cu126, 2.14.0+cu126, 2.13.0+cu126, ... 2.6.0+cu126

# 2. sortir torch ET les libs CUDA du venv (sinon cohabitation cu12/cu13)
$ .venv/bin/pip uninstall -y $(.venv/bin/pip freeze | \
      grep -iE "^(torch|triton|nvidia-|cuda-toolkit|cuda-bindings)" | cut -d= -f1)

# 3. installer la version exacte, tag local inclus
$ .venv/bin/pip install "torch==2.14.1+cu126" \
      --index-url https://download.pytorch.org/whl/cu126

# 4. prouver que ça marche par un vrai calcul, pas par is_available()
$ .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); \
      a=torch.zeros(8,device='cuda'); print((a+1).sum().item())"
```

Repli si cu126 ne contenait plus Pascal : `.venv/bin/pip install "torch==2.6.0"`
(le wheel PyPI par défaut de cette version était construit en cu124).

Repères de compatibilité : `cu118`…`cu124` → Pascal OK · `cu128` à partir de torch 2.8 →
Pascal retiré · `cu13x` → plus rien sous Turing.

**Pas de son**
```bash
aplay -l                      # cartes détectées
.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```
Forcer une sortie : `--output-device 0` (index donné par `query_devices()`).

**La voix prononce mal un mot** → installer `espeak-ng` (`sudo apt install espeak-ng`)
améliore les mots hors dictionnaire ; c'est optionnel.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`** (v0.4)
`pip` a installé PyAV 19, que `faster-whisper` 1.2.1 ne supporte pas (il ne borne pas `av`).
```bash
.venv/bin/pip install "av==18.1.0"
```
C'est déjà épinglé dans `requirements.txt` — le problème n'apparaît que sur une install
où `av` a été mis à jour. Détail en §10.

**Le micro n'entend rien, ou entend la sortie des haut-parleurs** (v0.4)
La source PulseAudio par défaut est souvent le « monitor » du puits de sortie, pas le micro.
```bash
pactl get-default-source                       # est-ce un .monitor ?
pactl list short sources                       # la vraie liste des micros
PULSE_SOURCE=alsa_input.pci-0000_00_1f.3.analog-stereo \
    .venv/bin/python -m voicechat --diag-micro
```
`--diag-micro` affiche le niveau, le taux d'écrêtage et la réaction du VAD. Si l'écrêtage est
élevé, baisser le gain de capture (`amixer -c 0 sset Capture 25`). **Sur cette machine la
capture est de toute façon hachée** : voir §10 pour l'état exact et les mesures.

**La reconnaissance vocale ne marche pas** — checklist dans l'ordre :
```bash
.venv/bin/python -m voicechat --transcrire un_fichier.wav   # 1. whisper seul ? (si OK :)
.venv/bin/python -m voicechat --diag-micro                  # 2. micro exploitable ?
.venv/bin/python tests/verif_ecoute.py                      # 3. la chaîne complète ?
```
Séparer les trois moitiés (modèle / micro / chaîne) évite de chercher au mauvais endroit,
exactement comme `--probe` sépare le réseau du GPU.

---

## 9. Journal de validation (v0.1)

Sorties **réelles**, copiées telles quelles. Le serveur `niko-tv` étant éteint, la partie
LLM a été validée contre `tests/fake_llm_server.py` — le chemin de code exercé est exactement
le même (`voicechat.llm`), seule la cible change.

### Tests hors ligne

```
$ .venv/bin/python -m pytest tests/ -q
....................                                                     [100%]
20 passed in 2.33s
```

### Streaming + découpage en phrases, en direct

```
$ python3 -c "... stream_with_usage + split_sentences ..."
--- streaming + decoupage en phrases ---
texte recu: 'Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. Elle sert à
vérifier que le découpage en phrases fonctionne. Et que chaque phrase part bien vers la
synthèse vocale au fur et à mesure. Fin du message. '
  phrase 1: Bonjour !
  phrase 2: Voici une réponse de test, découpée en plusieurs phrases.
  phrase 3: Elle sert à vérifier que le découpage en phrases fonctionne.
  phrase 4: Et que chaque phrase part bien vers la synthèse vocale au fur et à mesure.
  phrase 5: Fin du message.
stats: ttft=0.003s total=1.185s chars=220 debit=186 car/s
```

### Le point clé : la voix démarre avant la fin de la génération

```
[matériel] ── sorties audio détectées ──
  [0] HDA Intel PCH: ALC255 Analog (hw:0,0) (44100 Hz)
  [13] pipewire (44100 Hz)
  [19] default (44100 Hz)

[t+ 0.00s] audio dispo=True
[t+25.49s] Kokoro pret (device=cpu)
[t+25.61s] phrase 1 envoyee a la voix: 'Bonjour !'
[t+25.85s] phrase 2 envoyee a la voix: 'Ceci est une demonstration.'
[t+26.39s] phrase 3 envoyee a la voix: 'La voix doit demarrer avant la fin du texte.'
[t+26.58s] phrase 4 envoyee a la voix: 'Derniere phrase ici.'
[t+26.58s] --- fin du flux LLM ---
[t+40.27s] --- tout est joue ---

premiere phrase envoyee au TTS a t+25.61s, flux termine a t+26.58s
>>> la voix demarre 0.96 s AVANT la fin de la generation
audio reellement joue par PortAudio: 8.45 s | erreur: None
RTF du TTS: 1.44
```

**Ce que ça dit, sans enjoliver :**

- le pipeline fonctionne : la première phrase part en synthèse pendant que le LLM écrit la suite ;
- PortAudio a bien joué 8,45 s d'audio, sans erreur ;
- **mais le RTF de 1,44 sur CPU est mauvais** : Kokoro met 1,44 s à produire 1 s d'audio, donc
  la file d'attente se remplit plus vite qu'elle ne se vide (8,45 s d'audio ont mis ~13,7 s à
  sortir). Sur ce portable, le CPU est en outre bridé par le profil économie d'énergie.
  → **le GPU n'est pas un luxe ici, c'est la condition pour que la voix suive le texte.**
  C'est précisément ce que règle la question du build torch Pascal ci-dessus (§8) — et la
  section suivante montre le résultat une fois le build corrigé.

### Détection du GPU au démarrage — désormais honnête

Avant correction, le client annonçait `auto → GPU` et ne découvrait le problème qu'au premier
mot prononcé. La détection fait maintenant un vrai calcul sur GPU avant de promettre quoi que
ce soit :

```
$ .venv/bin/python -m voicechat
[matériel] auto → CPU. torch 2.14.1+cu130 — GPU visible (sm_61) mais aucun noyau compilé
pour lui — ce wheel ne contient que sm_75, sm_80, sm_86, sm_90, sm_100, sm_120.
Détail : CUDA error: no kernel image is available for execution on the device
```

(capture faite **avant** le remplacement du wheel ; seule la ligne utile est reproduite)

### Le build cu126 change tout — mesure CPU vs GPU

Après remplacement du wheel (procédure §3), le même calcul qui échouait passe :

```
version   : 2.14.1+cu126
cuda      : 12.6
arch list : ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
is_avail  : True
GPU       : NVIDIA GeForce GTX 1050 | capability (6, 1)
MATMUL GPU: OK en 518.8 ms -> 7210.73
```

Et l'effet sur la synthèse, mesuré sur la **même phrase** :

```
  [cpu]  4.58 s audio en 5.75 s -> RTF 1.256
  [cuda] 4.58 s audio en 0.42 s -> RTF 0.092
```

**≈ 13× plus rapide.** Avec un RTF de 0,09, la synthèse produit 1 s d'audio en 0,09 s : la
voix n'est plus jamais le goulot, elle suit confortablement le flux de tokens.

En session réelle (faux serveur, `--debug`) :

```
[matériel] auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo)
voicechat 0.1.0  ·  serveur http://127.0.0.1:8099/v1
Modèle : fake-local-model (détecté, 1 disponible(s))
Voix   : chargement de Kokoro « ff_siwis »…
GPU    : auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo) → device=cuda

vous › ia › Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. […]
   · 1er token 0.00 s | 220 car. en 1.18 s (186.9 car/s) | TTS RTF 0.13
```

### Le serveur réel : premier échange complet

`--probe`, sans aucun argument — donc sur la cible par défaut :

```
$ .venv/bin/python -m voicechat --probe
Serveur   : http://100.91.114.49:8080/v1
Test      : GET http://100.91.114.49:8080/v1/models
OK en 0.05 s — 1 modèle(s)
  • /mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf
```

Puis le vrai chat, avec la voix sur GPU (modèle détecté tout seul, aucun `--model`) :

```
$ .venv/bin/python -m voicechat --debug
[matériel] auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo)
voicechat 0.1.0  ·  serveur http://100.91.114.49:8080/v1
Modèle : /mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf (détecté, 1 disponible(s))
Voix   : chargement de Kokoro « ff_siwis »…
GPU    : auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo) → device=cuda

vous › Reponds en une seule phrase courte : quelle est la capitale de la France ?
ia › La capitale de la France est Paris.
   · 1er token 1.81 s | 35 car. en 2.20 s (15.9 car/s) | TTS RTF 0.28
```

**Ce qui est prouvé ici** : la chaîne complète tourne — question tapée → streaming depuis
`niko-1650-super` (35B MoE, 3B actifs, Q4_K_M) → découpage en phrases → Kokoro sur la GTX 1050
→ audio dans les haut-parleurs. Le débit texte (15,9 car/s) est bien plus lent que la synthèse
(RTF 0,28) : la voix n'attend jamais le texte, c'est le texte qui attend la voix.

Réponse plus longue, avec **aucun argument de ligne de commande** (donc cible par défaut) :

```
$ .venv/bin/python -m voicechat --debug
vous › Explique en trois phrases courtes pourquoi le ciel est bleu.
ia › Le ciel est bleu à cause de la diffusion de la lumière du soleil par l'atmosphère.

Les rayons lumineux bleus ont une longueur d'onde plus courte, donc ils se dispersent
davantage que les autres couleurs.

C'est ce phénomène, appelé diffusion de Rayleigh, qui donne au ciel sa couleur bleue.
   · 1er token 0.97 s | 291 car. en 3.89 s (74.8 car/s) | TTS RTF 0.12
```

Le modèle sépare ses phrases par des lignes vides : le découpage les traite comme trois
phrases distinctes, donc **la première est déjà en train d'être prononcée** pendant que la
troisième n'est pas encore écrite. C'est tout l'intérêt de découper le flux au vol.

### Réglage : le nom du modèle

llama.cpp renvoie un **chemin de fichier** comme identifiant :

```
Modèle : /mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf
```

C'est laid dans la console mais fonctionnel — et c'est exactement pourquoi la détection
automatique via `/v1/models` vaut mieux qu'un nom codé en dur. Pour un affichage propre :

```bash
.venv/bin/python -m voicechat --model Ornith-1.5-35B-A3B-APEX-i-mini.gguf
```

### Collage multi-lignes : avant / après

Le bug rapporté, reproduit puis vérifié corrigé en pilotant le vrai CLI dans un
pseudo-terminal :

```bash
$ .venv/bin/python tests/verif_collage.py
```

```
=== après le collage, AVANT Entrée ===
    réponses du modèle : 0   (attendu : 0)

=== après Entrée ===
    réponses du modèle : 1   (attendu : 1)
    extrait : ia › Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. […]
```

Et la confirmation du mécanisme d'origine — avec `input()`, le même collage devenait
trois conversations :

```
prompt:   -> tour 1 : 'Ligne un'
  -> tour 2 : 'Ligne deux'
  -> tour 3 : 'Ligne trois'
```

`tests/test_editor.py` verrouille tout ça (39 tests) via de vrais pseudo-terminaux :
collage balisé, collage non balisé détecté par rafale, frappe lente qui ne doit **pas**
ressembler à un collage, retours arrière, `Ctrl+U`, historique, recherche `Ctrl+R`,
complétion `Tab`, Échap pendant une génération, et entrée non interactive.

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 65%]
......................................                                   [100%]
110 passed in 11.85s
```

### Interruption à chaud : vérifiée sur le vrai CLI

`tests/verif_interruption.py` pilote le CLI dans un pseudo-terminal et envoie un vrai
`SIGINT` (ce que fait la touche `Ctrl+C`) en pleine génération, puis une touche `Échap` :

```
=== 1. Ctrl+C en pleine génération ===
ia › Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. Elle sert à vérifier que le
[Ctrl+C] génération, synthèse et lecture coupées
   · 1er token 0.00 s | 96 car. en 0.50 s (192.0 car/s) | réponse tronquée | voix coupée

vous ›
   -> coupé, et la session survit

=== 2. Une réponse qui finit normalement ===
ia › Bonjour ! ... Fin du message.
   · 1er token 0.00 s | 39 tok (+42 prompt) | 33.0 tok/s

=== 3. Échap en pleine génération (voix coupée, réponse gardée) ===
ia › Bonjour ! ... Fin du message.
[Échap] voix coupée — la réponse reste à l'écran
   · 1er token 0.00 s | 39 tok (+42 prompt) | 33.0 tok/s | voix coupée

>>> OK : Ctrl+C coupe tout, Échap ne coupe que la voix
```

Deux détails volontaires dans ces sorties :

- au cas 1, l'affichage retombe sur les **caractères** (`96 car.`) : la connexion a été
  fermée avant que le serveur n'envoie son bloc `usage`. Le repli fonctionne, il ne ment pas
  en affichant « 0 tok » ;
- au cas 3, `Fin du message.` est bien présent : `Échap` n'a **pas** entamé la génération.

### Lecture audio : d'où venait le trou entre les phrases

`tests/bench_audio.py` joue 10 tampons d'une seconde et compare les deux méthodes :

```
$ .venv/bin/python tests/bench_audio.py
10 tampons de 1 s = 10 s d'audio à jouer

flux persistant   : 10.028 s de temps réel  (surcoût +0.028 s)  |  joué 10.00 s
sd.play par tampon: 10.352 s de temps réel  (surcoût +0.352 s)

>>> environ 32.4 ms de silence économisées par phrase
```

Soit ~35 ms de latence d'ouverture de flux **par phrase** avec l'ancienne méthode : c'est
exactement ce qu'on entendait entre deux phrases. La contrepartie est un flux toujours ouvert,
dont le coût est mesuré plutôt que supposé :

```
au repos 10 s, flux ouvert : 35.4 ms de CPU  (0.354 % d'un cœur)
```

Et la coupure reste nette, ce qui valide `abort()` comme choix pour `Échap`/`Ctrl+C` :

```
apres flush: joue +0.00 s en 0.6 s (un flux non coupe aurait joue ~0.6 s)
```

### Ctrl+R et Tab dans le vrai CLI

`tests/verif_completion.py` tape sur un vrai pseudo-terminal, dans l'application complète :

```
=== 1. Tab complète une commande unique ===
   écran : vous › /vous › /svous › /savous › /save
   -> /save

=== 2. Tab s'arrête au plus long préfixe commun ===
   -> /voice (les deux candidats /voice et /voices partagent ce préfixe)

=== 3. Deux Tab montrent les choix ===
   -> liste affichée

=== 4. Ctrl+R rappelle un message précédent ===
   -> message retrouvé et renvoyé

>>> OK : Tab complète, Ctrl+R recherche, dans le vrai CLI
```

### Voix disponibles

```
$ .venv/bin/python -m voicechat --list-voices
54 voix Kokoro — préfixe : a/b = anglais, f = français, e = espagnol,
i = italien, p = portugais, j = japonais, z = chinois, h = hindi

  [a] af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa
  [b] bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
  [e] ef_dora, em_alex, em_santa
  [f] ff_siwis
  [h] hf_alpha, hf_beta, hm_omega, hm_psi
  [i] if_sara, im_nicola
  [j] jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
  [p] pf_dora, pm_alex, pm_santa
  [z] zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang
```

⚠️ Kokoro-82M ne propose **qu'une seule voix française : `ff_siwis`** (féminine). Pour varier
la voix en français sans changer de langue, la v0.5 ajoute le **mélange pondéré** de styles :
`/voice ff_siwis:3+ef_dora:1` — les voix d'autres langues servent de source de timbre, la
prononciation restant française. Mesures et syntaxe complète en **§11**. Les 53 autres voix
servent aux langues ci-dessus.

### Entrée vocale : ce qui a été mesuré (v0.4)

**Premier réflexe, et il a payé** : avant d'écrire une ligne de code vocal, vérifier le micro.
La source PulseAudio par défaut s'est révélée être le **monitor des haut-parleurs**, pas le
micro. Le micro réel donne, lui, un signal saturé :

```
gain 63/63 (100 %) : RMS  -4.9 dBFS  sature=6.8 %   <- réglage d'usine, inutilisable
gain 40/63 ( 63 %) : RMS -17.2 dBFS  sature=0.2 %
gain 25/63 ( 40 %) : RMS -30.3 dBFS  sature=0.0 %   <- bon compromis
gain 15/63 ( 24 %) : RMS -39.0 dBFS  sature=0.0 %
```

**Le VAD a été choisi par mesure, pas par goût.** Silero est livré dans `faster-whisper` :
aucune dépendance de plus. Encore fallait-il la bonne interface — v6, pas v5 :

```
$ .venv/bin/python -c "…/silero_vad_v6.onnx …"
entrées : input ['seq_len', 576], h [1, 1, 128], c [1, 1, 128]
sorties : speech_probs, hn, cn

silence          : max 0.024
bruit faible     : max 0.065
bruit fort       : max 0.057
sinusoïde 200 Hz : max 0.419
parole (Kokoro)  : moyenne 0.741, 196/258 blocs > 0.5
```

**Le piège d'installation, découvert en le heurtant** :

```
$ .venv/bin/python -c "from faster_whisper import WhisperModel; …"
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

`faster-whisper` 1.2.1 attend PyAV ≤ 18 ; `pip` installe av 19.0.0 (non borné en amont).
`av==18.1.0` règle le problème — c'est épinglé dans `requirements.txt`.

**Le type de calcul, mesuré sur la GTX 1050** :

```
cuda / float16      : Requested float16 compute type, but the target device
                      or backend do not support efficient float16 computation.
cuda / float32      : 5.53 s transcrites en 1.08 s (RTF 0.194)
cuda / int8_float32 : 5.53 s transcrites en 0.59 s (RTF 0.107)   <- retenu
cpu  / int8         : 5.53 s transcrites en 2.07 s (RTF 0.375)
```

**Puis le test qui a mal tourné.** En jouant une phrase et en la recapturant, la
transcription du début était fausse. Un témoin a tranché :

```
fichier source direct  : Allume la lumière du salon et vérifie que la porte est fermée.   OK
capté avec parec       : Allume la lumière du salon et vérifie que la porte est fermée.   OK
capté avec notre code  : Arrime l'alignement du soleil et vérifie la porte aussi.         KO
```

**Et une conclusion que j'ai dû retirer.** Une comparaison de nombres d'échantillons avait
laissé croire à « 128 ms perdues sur 8 s ». C'était faux : `parec` et notre capture ne
couvraient pas la même fenêtre temporelle, et PortAudio signalait bien 0 débordement. La
piste « perte d'échantillons » est donc écartée — ce qui *n'est pas* la même chose que
« problème résolu ». La cause reste inconnue ; c'est écrit comme tel en §10.

**Enfin, deux bugs trouvés par les tests, dans notre code** : le pré-roll perdait toujours
un bloc (le bloc déclencheur était empilé avant le test, donc il évinçait le plus ancien),
et `int()` au lieu de `round()` sur une division flottante. Les deux sont corrigés, avec les
tests de régression correspondants.

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 51%]
....................................................................     [100%]
140 passed in 12.16s
```

### Mélanges de voix : une hypothèse de roadmap fausse (v0.5)

La roadmap disait « mélange de styles, cf. `kokoro.StyleTTS` ». **Vérification faite avant
d'écrire quoi que ce soit** : cette classe n'existe pas dans la version installée.

```
$ .venv/bin/python -c "import kokoro; print([n for n in dir(kokoro) if not n.startswith('_')])"
['KModel', 'KPipeline', 'custom_stft', 'istftnet', 'logger', 'model', 'modules', 'pipeline', 'sys']
```

En revanche, en lisant `KPipeline.load_voice`, une bonne surprise :

```python
packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
if len(packs) == 1:
    return packs[0]
self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
```

Kokoro sait donc **déjà** mélanger des voix — mais seulement en moyenne simple, sans poids.
Et une ligne plus haut, le même code accepte un `torch.FloatTensor` en entrée. D'où la
solution : composer le mélange pondéré nous-mêmes et le lui passer. Mesures en §11.

### Un bug trouvé par la vérification, pas par les tests (v0.5)

`tests/verif_profil_voix.py` — écrit justement pour prouver qu'un profil porte sa voix — a
planté au premier essai :

```
AttributeError: 'ChatSession' object has no attribute 'tts'
```

Le formulaire `--profil` appliquait ses réglages dans le constructeur, avant la création de
`self.tts`. Tout profil portant une voix aurait donc empêché le lancement. Les tests
unitaires ne l'ont pas vu parce qu'aucun d'eux ne construisait de session à partir d'un
profil : c'est le fait de **lancer pour de vrai** qui l'a révélé. Corrigé, et cinq tests
ajoutés pour que ça ne revienne pas.

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 38%]
........................................................................ [ 76%]
.............................................                            [100%]
189 passed in 12.14s
```

### Robustesse réseau : ce qui a été mis en place (v0.6)

Trois pannes différentes, trois réponses. Le détail est en §12 ; ce qui suit est le chemin.

**Ce qui a été fait en premier : lire le code existant.** Le client ne réessayait rien du
tout : une connexion refusée levait une `LLMError` immédiatement. Et côté voix, une phrase
dont la synthèse échouait était comptée puis jetée (`continue`) — le son manquait, et rien
ne le disait. C'est ce second point qui était le plus gênant : un échec silencieux.

**Le choix qui mérite d'être noté** : ne jamais réessayer en cours de flux. Après le premier
mot reçu, un nouvel essai rejouerait le début de la réponse. On signale donc la coupure et on
garde la réponse partielle, ce qui est aussi ce qui a été fait pour `Ctrl+C` depuis la v0.2.

**Deux erreurs de ma part, trouvées par les tests :**

1. `stream_chat()` refusait `delai` — je l'avais ajouté à `_ouvrir_flux` sans le faire
   remonter. Six tests ont échoué d'un coup sur `TypeError`.
2. Mon faux serveur « fragile » interceptait `_send`, mais le chemin **streaming** écrit
   directement dans `wfile` : les 503 n'atteignaient jamais les vraies requêtes de chat. Le
   serveur n'échouait donc jamais, et les tests de reprise ne testaient rien — ils passaient
   même au vert pour certains. Corrigé en interceptant `do_GET`/`do_POST`.

Puis, en relançant les vérifications, un troisième problème — cette fois dans le faux serveur
partagé : son mot-clé « markdown » matchait aussi le **prompt système par défaut**, qui
contient le mot (« sans listes à puces ni markdown »). Toutes les questions recevaient donc la
réponse markdown. Le mot-clé ne se cherche plus que dans les messages utilisateur, et les
vérifications de la v0.5 ont été relancées pour confirmer qu'elles passaient encore.

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 33%]
........................................................................ [ 67%]
....................................................................     [100%]
212 passed in 22.40s
```

---

## 10. Entrée vocale (v0.4) — ⚠️ expérimental

### Ce qu'on peut faire

```bash
voicechat --micro                  # mains libres : après chaque réponse, il écoute
voicechat                          # puis /ecoute  : dicter un seul message
voicechat --transcrire phrase.wav  # transcrire un fichier (ni conversation, ni voix)
voicechat --diag-micro             # santé du micro : niveau, saturation, réaction du VAD
```

En session : `/ecoute` dicte un message puis rend la main au clavier, `/micro on|off`
bascule les mains libres. Pendant une écoute, `Ctrl+C` rend la main au clavier.

La boucle est **séquentielle, pas full-duplex** : on écoute, on transcrit, on génère, on
parle — le micro n'est **pas** ouvert pendant la lecture de la réponse. C'est délibéré :
ça supprime d'emblée toute boucle audio (le micro qui se réentend parler).

### Choix techniques

| Point | Choix | Pourquoi |
|---|---|---|
| Détection de parole | **Silero v6** via `onnxruntime` | Le modèle est **déjà livré** dans `faster_whisper/assets/silero_vad_v6.onnx` : aucune dépendance de plus. Attention, l'interface v6 n'est pas celle de v5 : entrée de 576 échantillons (64 de contexte + 512 neufs) et états LSTM `h`/`c` séparés. |
| Découpage | blocs de 512 échantillons (32 ms) | Contrainte du modèle. Pré-roll de 250 ms (pour ne pas couper la première syllabe), fin d'énoncé après 600 ms de silence, énoncés de moins de 350 ms ignorés. |
| Transcription | `faster-whisper` `small`, `int8_float32`, GPU | Mesures ci-dessous. |
| Chargement | **à la première utilisation** | Une session au clavier ne doit payer ni 1 s de chargement ni 330 Mo de VRAM. |

### Performances mesurées

Même phrase de 5,53 s, transcrite avec chaque type de calcul :

```
cuda / float16      : ECHEC (voir « Pièges » plus bas)
cuda / float32      : 5.53 s transcrites en 1.08 s (RTF 0.194)
cuda / int8_float32 : 5.53 s transcrites en 0.59 s (RTF 0.107)   <- retenu
cpu  / int8         : 5.53 s transcrites en 2.07 s (RTF 0.375)
```

VRAM occupée sur les 4 Go de la GTX 1050 :

```
au repos                        :  365 MiB
+ whisper small (int8_float32)  :  695 MiB
+ whisper small + Kokoro        : 1335 MiB
```

Kokoro et Whisper cohabitent donc largement dans les 4 Go.

VAD Silero, probabilité de parole par bloc de 32 ms :

```
silence          : max 0.024
bruit faible     : max 0.065
bruit fort       : max 0.057
sinusoïde 200 Hz : max 0.419
parole (Kokoro)  : moyenne 0.741 — 196/258 blocs au-dessus du seuil
```

### Transcription réelle

```bash
$ .venv/bin/python -m voicechat --transcrire /tmp/stt1.wav
Micro  : chargement de whisper « small »…
Micro  : whisper « small » sur cuda (int8_float32)
Fichier: /tmp/stt1.wav
Résumé : 5.53 s d'audio transcrites en 0.73 s (RTF 0.13) — langue fr (100%)

Bonjour, ceci est un test de reconnaissance vocale avec le modèle WISP SMALL.
```

Le mot attendu était « whisper » : Kokoro le prononce à l'anglaise et Whisper entend
« WISP ». **C'est une faute de synthèse, pas de reconnaissance.** Sur les autres phrases,
la transcription est exacte — y compris avec des nombres, où « trente-deux » ressort en
« 32 », ce qui est même préférable à donner au LLM.

### ⚠️ Limite connue : la capture micro est hachée

**État : cause racine non identifiée.** Ce paragraphe existe pour que personne ne croie
que le micro est fiable aujourd'hui.

Le protocole : on joue une phrase dans les haut-parleurs et on la recapture via la sortie
« monitor » de PulseAudio (boucle numérique). Deux clients qui captent le même signal
doivent produire le *même* audio.

```
phrase attendue        : Allume la lumière du salon et vérifie que la porte est fermée.
fichier source direct  : Allume la lumière du salon et vérifie que la porte est fermée.   OK
capté avec parec       : Allume la lumière du salon et vérifie que la porte est fermée.   OK
capté avec notre code  : Arrime l'alignement du soleil et vérifie la porte aussi.         KO
```

`parec` (paquet `pulseaudio-utils`) capture donc proprement la **même source, au même
moment, à la même fréquence** (16 kHz mono). Notre capture via PortAudio/sounddevice hache
la parole.

Essayé, sans effet :

- blocs de rappel ×8 et `latency="high"` (contre un éventuel débordement de tampon) ;
- `onnxruntime` limité à un seul fil (il disputait le GIL au rappel audio de PortAudio) ;
- capture à la fréquence native 48 kHz puis rééchantillonnage FIR maison (fenêtre de
  Blackman, décimation 3:1) ;
- vérification que PortAudio signale bien **0 débordement d'entrée**.

**Piste écartée** : la perte d'échantillons. Une comparaison de longueurs avait laissé
croire à 128 ms perdues sur 8 s ; c'était faux — les deux captures ne couvraient pas la
même fenêtre temporelle. Conclusion retirée.

**Mise en garde sur le protocole** : tout ceci est mesuré à travers la boucle « monitor »,
qui est un chemin particulier. Sur un vrai micro acoustique, le problème n'existe peut-être
pas du tout. Les trois commandes qui tranchent :

```bash
.venv/bin/python tests/bench_capture.py   # corrélation entre notre capture et parec
voicechat --diag-micro                    # niveau, saturation, réaction du VAD
voicechat --micro                         # et on parle
```

`tests/bench_capture.py` compare plusieurs méthodes de capture à `parec` par corrélation :
sur une boucle numérique, elle doit valoir ~1.

### Le micro écrête à 100 % de gain

Indépendamment de ce qui précède, le micro interne de cette machine sature au gain par
défaut. Mesures (`alsa_input.pci-0000_00_1f.3.analog-stereo`, gain de capture) :

```
gain 63/63 (100 % = 30 dB) : RMS  -4.9 dBFS   écrêtage 6.8 %   <- inutilisable
gain 40/63 ( 63 %)         : RMS -17.2 dBFS   écrêtage 0.2 %
gain 25/63 ( 40 %)         : RMS -30.3 dBFS   écrêtage 0.0 %   <- bon compromis
gain 15/63 ( 24 %)         : RMS -39.0 dBFS   écrêtage 0.0 %   <- trop bas
```

`--diag-micro` détecte la saturation et propose la commande :

```bash
$ .venv/bin/python -m voicechat --diag-micro
Micro  : mesure du bruit de fond pendant 3 s (ne parle pas)…
  échantillons : 47616 (3.0 s)
  niveau       : RMS 0.5387 (-5.4 dBFS)
  écrêtage     : 4.35 % des échantillons
  VAD (seuil 0.5) : max 0.244, moyen 0.017 — 0/93 blocs jugés parole

  ⚠ micro SATURÉ : la reconnaissance sera mauvaise. Baisse le gain :
      amixer -c 0 sget Capture      # voir la valeur actuelle
      amixer -c 0 sset Capture 25   # ~40 % : bon compromis mesuré
```

Le taux d'écrêtage varie d'une mesure à l'autre (4 à 26 % selon le bruit du moment) —
il dépend du ventilateur et de l'activité de la machine.

Bonne nouvelle au passage : même sur ce signal saturé, le VAD ne s'est **pas** laissé
berner (0 bloc sur 93 déclaré parole). Le détecteur tient ; c'est le signal qui ne vaut rien.

Autre piège système à connaître : la **source PulseAudio par défaut** peut être le
« monitor » des haut-parleurs et non le micro. Pour forcer le micro :
`PULSE_SOURCE=alsa_input.pci-0000_00_1f.3.analog-stereo`, ou `--micro-device`.

### Pièges d'installation

**`av` doit être épinglé.** `faster-whisper` 1.2.1 appelle `av.open(..., metadata_errors=…)`,
paramètre que **PyAV 19 a retiré** — et `faster-whisper` ne borne pas `av` :

```
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

Le log d'installation montre le défaut (c'est `pip` qui prend la dernière version) :

```
Downloading av-19.0.0-cp312-abi3-manylinux_2_28_x86_64.whl (35.0 MB)
Successfully installed av-19.0.0 ctranslate2-4.8.2 faster-whisper-1.2.1
```

`requirements.txt` épingle donc `av==18.1.0`.

**`float16` est à éviter.** C'est le choix « évident »… et le seul qui échoue ici :

```
Requested float16 compute type, but the target device or backend do not
support efficient float16 computation.
```

CTranslate2 le refuse sur Pascal (GTX 1050, sm_61). `voicechat/stt.py` n'essaie jamais
`float16` et teste dans l'ordre `int8_float32` → `float32` → repli CPU ; deux tests
verrouillent cette liste pour qu'on ne la « simplifie » pas par erreur.

---

## 11. Qualité de la voix (v0.5)

### Le problème

Un LLM local répond en markdown. Lu tel quel, l'auditeur entend « astérisque astérisque
Baisse la luminosité astérisque astérisque », les barres verticales d'un tableau, et les
crochets d'un lien. Le nettoyage existait depuis la v0.1, mais **il ne voyait qu'une phrase
à la fois** (il est appliqué juste avant la synthèse, après découpage) et il laissait passer
des choses. Exemple réel, réponse du modèle :

```
Brut du modèle                                        Ce que la voix prononçait (avant)
----------------------------------------------------  ------------------------------------------
[ce guide Ubuntu](https://doc.ubuntu-fr.org/...)      [ce guide Ubuntu](lien
- 2. Ferme les processus inutiles.                    2.  /  Ferme les processus inutiles.
| Réglage | Gain |                                    | Réglage | Gain |
|---|---|---|---|                                        |---|---|
- **Baisse la luminosité**                            Baisse la luminosité
Traitement de M. Dupont sur 1 000 tours               M. Dupont sur 1 000 tours
Il fait 32 °C, soit 30 % de plus                      32 °C … 30 %
Bravo 😀🎉                                            Bravo 😀🎉 (emoji conservés)
```

Le cas du lien est le plus parlant : l'ancien code retirait l'URL et laissait
`[ce guide Ubuntu](` — l'auditeur entendait littéralement « crochet ».

### Ce que fait le nettoyage maintenant

| Entrée | Ce qui est prononcé |
|---|---|
| `[ce guide](https://…)` | `ce guide` |
| `![capture](img.png)` | *(rien : une image n'est pas du texte)* |
| `### Titre` / `> citation` | `Titre` / `citation` |
| `- item`, `* item`, `— item` | `item` |
| `1. item`, `2) item` | `item` |
| `\| Réglage \| Gain \|` | `Réglage, Gain` |
| `\|---\|---\|`, `---` | *(rien)* |
| ` ```bash … ``` ` | *(rien : le bloc n'est pas prononcé)* |
| `30 %`, `32 °C`, `15 €`, `90 km/h` | `30 pour cent`, `32 degrés`, `15 euros`, `90 kilomètres par heure` |
| `M. Dupont`, `Mme Martin`, `Dr House`, `n° 12` | `Monsieur Dupont`, `Madame Martin`, `Docteur House`, `numéro 12` |
| `1 000` (espaces insécables ou fines) | `1000` |
| `👍🎉` | *(rien)* |
| `A → B` | `A puis B` |

### Deux pièges du **flux** (et pas du texte)

C'est la partie qui a demandé le plus de soin : le nettoyage reçoit des fragments, parce que
le texte arrive morceau par morceau du serveur.

**1. Un bloc de code était prononcé.** Le découpage en phrases se fait sur les sauts de
ligne, donc ```` ```bash\nls -l\n``` ```` était haché en trois « phrases » avant que le
nettoyage ne voie quoi que ce soit : l'auditeur entendait `bash`, `ls -l`, puis le backtick
de clôture. Le découpeur **retient** désormais tout ce qui suit un ```` ``` ```` non refermé,
et ignore ce qui se trouve à l'intérieur d'un bloc.

**2. Une abréviation ou un marqueur de liste coupé en deux.** Si le paquet réseau s'arrête
exactement après `M.` ou après `2.`, l'ancienne garde (qui exigeait de voir la suite du
texte) ne s'appliquait pas : « 2. » partait seul à la synthèse et l'auditeur entendait
« deux » ; `M. Dupont` arrivait en deux morceaux et n'était plus développé. Ces marqueurs ne
sont donc **jamais** considérés comme une fin de phrase, même en fin de tampon. Le texte
retenu n'est pas perdu : le CLI vide son tampon à la fin de la réponse (`cli.py`).

### Vérification de bout en bout

Lire la réponse à l'écran ne prouve rien sur ce que la voix prononce. `tests/verif_nettoyage.py`
lance le **vrai CLI avec le vrai Kokoro**, contre un faux serveur qui renvoie du markdown, et
capture la trace de chaque phrase juste avant sa synthèse (`VOICECHAT_TRACE_TTS=1`). Le stderr
est capturé par un tuyau séparé du pseudo-terminal, pour ne pas confondre la réponse streamée
avec les traces.

```bash
$ .venv/bin/python tests/verif_nettoyage.py
CLI prêt, Kokoro chargé.

=== 10 segments réellement envoyés à la synthèse ===
   'Trois conseils pour la batterie'
   'Voici les points qui comptent :'
   'Baisse la luminosité :'
   "réduis l'éclairage du panneau."
   'Ferme les processus inutiles.'
   'Le mode économie limite la puissance.'
   'Réglage, Gain'
   'Écran, 30 pour cent'
   'Voir la doc Ubuntu, chapitre de Monsieur Dupont.'
   'La température idéale est 32 degrés et le disque tourne à 1000 tours/min.'

=== contrôles ===
  OK  aucun bloc de code prononcé
  OK  aucun backtick
  OK  aucune barre de tableau
  OK  aucun crochet de lien
  OK  lien réduit à son texte
  OK  abréviation développée
  OK  pourcentage prononçable
  OK  degré prononçable
  OK  marqueur de liste retiré

>>> OK : ce qui est prononcé ne contient plus de balisage markdown
```

On voit aussi, sur cette sortie, que le premier segment (`Trois conseils pour la batterie`)
a perdu son `#` et que le tableau a produit deux segments lisibles au lieu de barres.

### Voir ce qui part à la synthèse

`VOICECHAT_TRACE_TTS=1` affiche, sur la sortie d'erreur, chaque phrase telle qu'elle est
envoyée au synthétiseur — c'est le seul moyen de savoir ce que la voix dira vraiment :

```bash
VOICECHAT_TRACE_TTS=1 .venv/bin/python -m voicechat
```

### Mélange de voix

Kokoro-82M ne propose **qu'une seule voix française** (`ff_siwis`). Sortir de cette voix
unique demande donc de mélanger des styles.

**Ce que Kokoro sait faire, et ce qu'il ne sait pas.** Son `load_voice` accepte plusieurs
noms séparés par des virgules et renvoie la **moyenne** des styles — donc `ff_siwis,ef_dora`
donne 50/50, mais aucun moyen de pondérer. En lisant son code on voit en revanche qu'il
accepte aussi un **tenseur de style** en entrée : on compose donc le mélange pondéré
nous-mêmes (`sum(poids × style)`) et on le lui passe. Une voix seule continue de passer par
le chemin normal, donc son rendu ne change pas d'un iota.

Le mélange inter-langues est **volontaire** : `ef_dora` (espagnol) ou `if_sara` (italien)
servent de source de timbre, tandis que la prononciation reste française (c'est le G2P du
pipeline qui décide). Kokoro avertit « Language mismatch » dans ce cas — on tait ce message
précis, et lui seul.

**Mesuré** (`tests/bench_voix.py` : la phrase est synthétisée, puis retranscrite par Whisper
pour vérifier qu'elle reste intelligible ; le « timbre » est la similarité spectrale avec
`ff_siwis`) :

```
  ff_siwis (référence)     : 6.95 s | justesse  90.5% | timbre vs réf. 1.000 | RTF 0.13
  + ef_dora (es) 50/50     : 6.22 s | justesse  90.5% | timbre vs réf. 0.916 | RTF 0.12
  + if_sara (it) 50/50     : 6.33 s | justesse 100.0% | timbre vs réf. 0.963 | RTF 0.11
  + af_heart (en) 50/50    : 6.88 s | justesse  90.5% | timbre vs réf. 0.964 | RTF 0.11
  + ef_dora 80/20          : 6.70 s | justesse 100.0% | timbre vs réf. 0.980 | RTF 0.09
  + if_sara 80/20          : 6.65 s | justesse 100.0% | timbre vs réf. 0.989 | RTF 0.09
  + ef_dora 60/40          : 6.38 s | justesse 100.0% | timbre vs réf. 0.936 | RTF 0.09
  3 voix 60/20/20          : 6.38 s | justesse 100.0% | timbre vs réf. 0.965 | RTF 0.09
```

Trois exécutions donnent les mêmes durées, la même justesse et le même timbre (au millième
près). **Seul le RTF bouge** d'une exécution à l'autre — c'est une mesure de temps, elle
dépend de la charge de la machine. Les valeurs ci-dessus sont celles du dernier passage.

Trois enseignements :

- **aucun mélange n'est moins intelligible que la référence** — plusieurs font même mieux
  (100 % contre 90,5 %). Sur cette phrase, la voix d'origine est celle qui se trompe le plus
  (« dort » entendu « d'or », « allumée » entendu « allumé ») ;
- **le timbre bouge vraiment** : de 0,916 (ef_dora à 50 %) à 0,989. À noter que 80/20 donne
  0,980–0,989 contre 0,916–0,963 à 50/50 : les poids sont donc bien respectés, un mélange
  léger s'écarte moins de la référence qu'un mélange fort ;
- **le mélange ne coûte rien** en vitesse (RTF 0,09–0,11, contre 0,13 pour la référence).

⚠️ **Ce que cette mesure ne dit pas** : si une voix *sonne bien*. Ça, aucune métrique ne le
dit. Le script écrit donc un WAV par variante dans `/tmp/voix/` pour que l'oreille tranche :

```bash
.venv/bin/python tests/bench_voix.py
# → Échantillons écrits dans /tmp/voix — écoute-les
#   /tmp/voix/ff-siwis-reference.wav
#   /tmp/voix/ef-dora-es-50-50.wav
#   ...
```

### Voix portée par le persona

Un profil de prompt peut désormais porter sa voix, sa vitesse et sa langue dans un **en-tête
facultatif**, ce qui garde le fichier lisible et modifiable à la main :

```
voix: ff_siwis:3+ef_dora:1
vitesse: 1.1

Tu es un narrateur posé, tu réponds en français.
```

Chargement et enregistrement :

```bash
voicechat --profil narrateur          # au lancement : prompt + voix + vitesse
/profil narrateur                     # ou à chaud, en session
/profil save narrateur                # enregistre le prompt courant ET sa voix
```

`/profil save` n'écrit un réglage **que s'il s'écarte du défaut** : un profil qui ne change
pas la voix reste un simple fichier de prompt, comme avant la v0.5. Seules `voix`, `vitesse`
et `langue` sont reconnues en en-tête ; toute autre clé est laissée dans le prompt, pour ne
jamais perdre de texte par inadvertance.

**Un bug trouvé par la vérification, pas par les tests unitaires.** `tests/verif_profil_voix.py`
a planté au premier essai :

```
AttributeError: 'ChatSession' object has no attribute 'tts'
```

Le profil était appliqué dans le constructeur de `ChatSession`, **avant** la création de
`self.tts`, que les réglages consultent. Conséquence : tout profil portant une voix aurait
fait échouer le lancement. Les champs `tts`/`speaker`/`speech` sont désormais initialisés en
premier, et cinq tests verrouillent le comportement (`tests/test_cli.py`).

```
$ .venv/bin/python tests/verif_profil_voix.py
[1/4] profil écrit : /tmp/vc-profils-…/narrateur.md
[2/4] profil appliqué par ChatSession :
      OK  prompt système
      OK  voix
      OK  vitesse
[3/4] chargement de Kokoro avec le mélange…
Voix   : chargement de Kokoro « ff_siwis 75% + ef_dora 25% »…
[4/4] synthèse réelle avec la voix du profil :
      71400 échantillons = 2.98 s d'audio

>>> OK : le profil a porté sa voix jusqu'à la synthèse
```

---

## 12. Robustesse réseau (v0.6)

Trois mécanismes, pour trois pannes différentes.

### 1. Réessais à délai croissant

**La panne visée** : le serveur LLM redémarre. Pendant quelques secondes, la connexion est
refusée — et rien ne justifie d'abandonner, il suffit d'attendre. Trois essais, à 0,5 s, 1 s
puis 2 s : le total reste sous 4 s, donc une machine vraiment éteinte n'impose pas une
attente interminable.

Ce qui est réessayé, et ce qui ne l'est pas :

| Erreur | Réessayé ? | Pourquoi |
|---|---|---|
| Connexion refusée, timeout, erreur réseau | **oui** | transitoire : le serveur redémarre |
| HTTP 5xx, HTTP 429 | **oui** | surcharge passagère |
| HTTP 404 | **non** | l'URL est fausse, insister ne la corrigera pas |
| HTTP 400 hors `stream_options` | **non** | la requête est refusée pour ce qu'elle contient |

**Une exception volontaire : on ne réessaie jamais en cours de flux.** Si la connexion casse
après le premier mot, réessayer rejouerait le début de la réponse — et l'utilisateur lirait
« Bonjour, je vais vous expliquer… Bonjour, je vais vous expliquer… ». On signale donc la
coupure et on garde la réponse partielle. C'est un choix, pas un oubli.

Chaque attente est **annoncée** — un retry silencieux ressemble à un blocage :

```
[réseau] essai 1 échoué — nouvelle tentative dans 0.5 s
         (Connexion impossible à 127.0.0.1:60583 (<urlopen error [Errno 111] Connection refused>).)
[réseau] essai 2 échoué — nouvelle tentative dans 1.0 s
         (Connexion impossible à 127.0.0.1:60583 (<urlopen error [Errno 111] Connection refused>).)
[réseau] http://127.0.0.1:60583/v1 n'a pas répondu → réponse de http://127.0.0.1:54061/v1
ia › Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. […]
```

### 2. Serveurs de secours

```bash
voicechat --secours http://100.64.0.9:8080/v1          # en ligne de commande
VOICECHAT_SECOURS=http://100.64.0.9:8080/v1,http://autre:8080/v1   # ou dans .env
```

Les cibles sont essayées dans l'ordre : la principale, puis les secours. La bascule a lieu
**avant** le premier mot, et elle est annoncée — parler à une autre machine que celle qu'on
croit est exactement le genre de chose qu'il ne faut pas taire.

`--probe` teste chaque cible séparément, et dit laquelle servira :

```bash
$ .venv/bin/python -m voicechat --probe \
      --base-url http://127.0.0.1:9/v1 --secours http://100.91.114.49:8080/v1
Cibles    : 2 serveur(s), essayés dans l'ordre
  ÉCHEC http://127.0.0.1:9/v1 (principal) en 0.0 s
        Connexion impossible à 127.0.0.1:9 (<urlopen error [Errno 111] Connection refused>).
          → la machine distante est probablement éteinte, ou le serveur LLM n'écoute pas sur 0.0.0.0.
          → vérifier : tailscale status ; ou lancer le serveur avec --host 0.0.0.0
  OK    http://100.91.114.49:8080/v1 (secours) en 0.01 s — 1 modèle(s)
        • /mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf

La cible principale ne répond pas : c'est « http://100.91.114.49:8080/v1 » qui servira.
```

### 3. Mode dégradé : la voix ne jette plus les phrases

**La panne visée** : le GPU lâche en pleine session (mémoire pleine, pilote qui tombe) — pas
le réseau. Jusqu'à la v0.5, la phrase concernée était comptée dans `synth_errors` puis
**jetée** : le son manquait et rien ne le disait. Elle est maintenant mise de côté, et
signalée :

```
[voix] 3 phrase(s) non synthétisée(s) — /rejoue quand le GPU est revenu
```

`/rejoue` vide cette réserve et remet les phrases dans la file de synthèse. Le rejeu n'est
**jamais automatique** : le déclencher au milieu d'une réponse mélangerait deux textes. C'est
l'utilisateur qui sait quand le GPU est revenu.

Deux détails décidés explicitement :

- la réserve est **bornée** (40 phrases, les plus récentes) — une longue session ne doit pas
  remplir la mémoire ;
- un `Ctrl+C` (vidage de la file) **ne perd pas** ces phrases : sinon il suffirait
  d'interrompre une réponse pour perdre définitivement ce que le GPU n'avait pas pu dire.

### Un bug de mes propres tests, trouvé en les relançant

Le faux serveur de test choisissait sa réponse markdown si le mot « markdown » apparaissait
dans la requête. Or le **prompt système par défaut** contient ce mot :

```
"Réponds en phrases courtes et parlées, sans listes à puces ni markdown."
```

Toutes les questions recevaient donc la réponse markdown — et `tests/verif_reprise.py`
passait « pour la bonne raison » tout en ne testant pas ce qu'il croyait. Le mot-clé se
cherche désormais dans les messages **utilisateur** uniquement. Les tests de la v0.5 ont été
relancés pour vérifier qu'ils passaient toujours après cette correction.

### Vérification

```bash
$ .venv/bin/python tests/verif_reprise.py
cible principale : http://127.0.0.1:60583/v1   (rien n'écoute)
secours          : http://127.0.0.1:54061/v1   (faux serveur)

=== contrôles ===
  OK  la bascule est annoncée
  OK  le secours est nommé
  OK  l'URL fautive est nommée
  OK  une réponse est arrivée
  OK  aucune erreur bloquante

>>> OK : le secours a servi, et l'utilisateur l'a su
```

Les réessais et les erreurs définitives sont couverts par `tests/test_reprise.py`, avec un
faux serveur qui échoue **à la demande** (503, 404) : 15 tests, dont « un 404 ne part qu'une
fois » et « le délai double à chaque essai ». Le mode dégradé est couvert par
`tests/test_degrade.py` (8 tests), avec un synthétiseur qui échoue à la demande lui aussi.

---

## 13. Version distribuée (v1.0)

Quatre chantiers, menés dans l'ordre du rapport bénéfice/risque : la réserve d'audio
d'abord (elle ne dépend de rien et se mesure), le paquet ensuite (il conditionne le
reste), le fichier de configuration, et pour finir le serveur TTS partagé.

### 13.1 La réserve d'audio : 282 ms → 0 ms

Un LLM se répète énormément. « Bien sûr ! », « Voici les points qui comptent : »,
« N'hésite pas si tu as d'autres questions. » — la synthèse étant l'étape la plus
lente d'un tour, resservir le son de ces phrases est le gain le moins coûteux du
projet. Mesure réelle, Kokoro sur la GTX 1050, 7 phrases dont 2 répétitions :

```
$ .venv/bin/python tests/bench_cache.py
Kokoro sur cuda (auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo))
réserve : /tmp/vc-bench-cache-vqefcleu
7 phrases, dont 2 répétitions volontaires

[1/2] réserve vide — tout est synthétisé
     1.62 s d'audio en    584 ms — Bien sûr !
     2.12 s d'audio en    257 ms — Voici les points qui comptent :
     2.52 s d'audio en    282 ms — N'hésite pas si tu as d'autres questions.
     2.83 s d'audio en    299 ms — En résumé, il y a deux choses à retenir.
     3.67 s d'audio en    375 ms — Le mode économie limite la puissance du proc
     2.83 s d'audio en      0 ms — En résumé, il y a deux choses à retenir.
     1.62 s d'audio en      0 ms — Bien sûr !

[2/2] réserve pleine — relecture depuis le disque
     1.62 s d'audio en      0 ms — Bien sûr !
     ...

total à froid :    1797 ms   (médiane 282 ms/phrase)
total à chaud :       1 ms   (médiane 0 ms/phrase)
facteur       : 1556× plus rapide
réserve       : 10 reprise(s) / 15 phrase(s) (67%), 23.3 s d'audio resservies
~3.09 s de synthèse évitées (estimation)

Écart max entre le premier passage et le rejeu : 9.72e-02
(0 veut dire que l'audio resservi est identique au bit près)
```

Trois choses à lire dans cette sortie.

**Les deux répétitions sont déjà à 0 ms dans le premier passage** : la réserve sert dès
la deuxième occurrence, pas seulement au tour suivant.

**Le facteur 1556× ne veut pas dire « 1556 fois mieux »** : il compare 282 ms de calcul
à 0,1 ms de lecture disque. Le chiffre honnête est « ~280 ms par phrase répétée ».

**L'écart max n'est pas nul, et ce n'est pas un bug.** Deux synthèses du même texte
avec Kokoro ne donnent pas le même tableau : l'écart mesuré a été 5.99e-02 puis
9.72e-02 sur deux exécutions (donc ni une constante, ni un décalage fixe). Le GPU est
simplement non déterministe. Conséquence à connaître : la réserve ressert *un* audio
valide, pas une copie bit-à-bit de ce qu'une nouvelle synthèse aurait produit.

Ce qui rend le cache sûr, c'est la **clé** : elle inclut la voix, la vitesse, la langue
et le texte.

```python
def cle(texte, voix, vitesse, langue) -> str:
    return hashlib.sha1(f"{langue}|{voix}|{vitesse:.4f}|{texte}".encode()).hexdigest()
```

Changer de voix ne peut donc pas resservir l'audio de l'ancienne — c'est le piège
classique d'un cache audio, et un test le vérifie explicitement.

Le reste des précautions, chacune pour un échec réel :

| Situation | Traitement | Raison |
|---|---|---|
| Processus tué en pleine écriture | écriture dans `.tmp` puis `os.replace` | un WAV tronqué serait relu comme du son valide |
| Fichier corrompu ou d'une vieille version | `lire()` le supprime et renvoie `None` | une réserve abîmée ne doit pas casser la voix |
| Réserve qui grossit sans fin | plafond (2000 fichiers), éviction du plus ancien **par date d'accès** | relire une phrase rafraîchit sa date : elle survit |
| Voix changée, réserve gardée | clé différente | aucune chance de servir la mauvaise voix |

### 13.2 Le paquet installable

`pyproject.toml` avec la version lue dynamiquement depuis `voicechat/__init__.py` — une
seule source de vérité, sinon les deux divergent à la première montée de version.

```
$ .venv/bin/pip install -e . --no-deps
$ .venv/bin/voicechat --version
voicechat 1.0.0
$ .venv/bin/pip show voicechat | grep -E "^(Name|Version|Requires)"
Name: voicechat
Version: 1.0.0
Requires: av, faster-whisper, kokoro, numpy, sounddevice, soundfile
```

`--no-deps` est **volontaire** : laisser pip réinstaller les dépendances ramènerait
`av` en 19 (casserait Whisper, cf. §10) et `torch` en cu130 (sans sm_61, donc sans GPU
sur cette machine). Sur une installation neuve, faire `pip install ./voicechat`, **puis**
épingler à la main les deux paquets fragiles :

```bash
pip install ./voicechat
pip install "av==18.1.0" "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Deux détails relevés en installant :

* le point d'entrée atterrit dans `.venv/bin/` — le `voicechat` du PATH reste celui de
  `~/bin/` (le PATH ne contient pas `.venv/bin`), donc **le launcher existant continue
  de fonctionner** ; les deux mènent au même code ;
* `main()` renvoie un entier, que `sys.exit()` du point d'entrée utilise comme code de
  sortie : `--probe` sur un serveur mort sort donc en 2, pas en 0.

### 13.3 Le fichier de configuration TOML

Un fichier `~/.config/voicechat/config.toml` (modèle fourni : `config.example.toml`).
Précédence, du plus fort au plus faible :

```
arguments de la ligne de commande
  > variables d'environnement
  > config.toml
  > valeurs codées en dur
```

Le TOML sert de *valeur par défaut*, ce qui est le seul ordre qui ne surprenne pas : une
variable d'environnement posée pour une session reste prioritaire.

Les réglages **par modèle** vivent au premier niveau du fichier, pas dans `[voicechat]` :

```toml
[voicechat]
voice = "ff_siwis"

[modeles.Ornith-1.5-35B-A3B-APEX-i-mini]
voice = "ff_siwis:3+ef_dora:1"
temperature = 0.2
```

Un nom qui contient un point doit être entre guillemets (`[modeles."..."]`), sinon TOML
y voit une sous-table. Les noms de GGUF en contiennent presque toujours — c'est noté dans
le fichier d'exemple.

Vérification, sur le vrai chemin de code (`config_from_args`) puis avec le vrai CLI :

```
$ .venv/bin/python tests/verif_config.py
--- A. précédence, sur le vrai chemin de code ---
  TOML seul          : voice=af_heart speed=1.25 temperature=0.9
  + env VOICE=if_sara: voice=if_sara (le TOML disait af_heart)
  + args --voice     : voice=bm_george speed=0.9 (l'env disait if_sara, le TOML 1.25)
  pour_modele        : {'voice': 'ff_siwis:3+ef_dora:1', 'temperature': 0.2}

--- B. le vrai CLI affiche-t-il les réglages par modèle ? ---
  voicechat 1.0.0  ·  serveur http://100.91.114.49:8080/v1
  Modèle : réglages du fichier de configuration — temperature=0.2, voice=ff_siwis:3+ef_dora:1

=== contrôles ===
  OK  le TOML fournit les valeurs par défaut
  OK  l'environnement surclasse le TOML
  OK  les arguments surclassent l'environnement
  OK  les réglages par modèle sont trouvés
  OK  le CLI annonce les réglages appliqués
```

**Un trou trouvé par ce contrôle.** La première exécution a donné `temperature=0.7` au
lieu de `0.9` : `temperature`, `max_tokens` et `timeout` figuraient bien dans `Config`,
mais `from_env()` ne les lisait pas du tout. Conséquence incohérente : `[modeles.x]
temperature = 0.2` fonctionnait (il passe par la surcharge par champ) alors que
`[voicechat] temperature = 0.9` était silencieusement ignoré. Les trois sont maintenant
lus par les trois sources, et `Config` sert de définition aux valeurs par défaut pour
que les littéraux ne puissent plus diverger.

Une clé inconnue est **signalée** plutôt qu'ignorée, avec la liste des clés valides
tirée de la dataclass :

```
[config] /home/niko/.config/voicechat/config.toml — clé(s) inconnue(s) dans [voicechat] : voix
```

### 13.4 Le serveur TTS partagé

Le poste léger envoie du **texte**, le poste à GPU renvoie de l'**audio**. Le premier
n'a alors besoin ni de torch, ni de Kokoro, ni d'un GPU — il ne lui reste que la lecture
du son. Le protocole tient en deux routes :

```
GET  /sante   →  {"service": "voicechat-tts/1", "pret": true, "voix": "ff_siwis",
                  "device": "cuda", "rtf": 0.0, "reprises": 0, "appels": 0}
POST /parle   →  corps JSON {"texte", "voix", "vitesse", "langue"}
                 réponse  audio/wav  (float32, 24 kHz mono)
```

Sur le poste GPU :

```
$ voicechat --serveur-tts --port 8090
Serveur TTS : chargement de Kokoro « ff_siwis »…
GPU         : auto → GPU. torch 2.14.1+cu126 — NVIDIA GeForce GTX 1050 (4031 Mo)
Réserve     : /home/niko/.cache/voicechat/tts
Écoute      : http://192.168.1.22:36733
                GET  http://192.168.1.22:36733/sante   (état)
                POST http://192.168.1.22:36733/parle   (texte → WAV)

Depuis un autre poste :
  voicechat --tts-distant http://192.168.1.22:36733
```

Sur le poste léger : `voicechat --tts-distant http://192.168.1.22:8090` (ou la variable
`VOICECHAT_TTS_URL`).

Le WAV a été choisi pour deux raisons : float32 est exact après relecture, et le fichier
reste écoutable dans n'importe quel lecteur si un doute apparaît.

La vérification ne se contente pas de regarder si l'audio fait la bonne taille — un WAV
valide mais muet la passerait. L'audio revenu **par le réseau** est retranscrit :

```
$ .venv/bin/python tests/verif_distant.py
  GET /sante → 200 {"service": "voicechat-tts/1", "pret": true, "voix": "ff_siwis", "device": "cuda", ...}
  client chargé : serveur TTS http://127.0.0.1:36733 (voix ff_siwis, device cuda)

  appel 1 (froid)  : 92400 échantillons = 3.85 s d'audio en 0.85 s (RTF 0.22)
  appel 2 (réserve): 92400 échantillons en 0.002 s

  retranscription de l'audio revenu par le réseau (Whisper small)…
  entendu : « Le serveur de synthèse répond depuis un autre processus. »
  mots significatifs retrouvés : 6/6 (100%)

=== contrôles ===
  OK  le serveur a annoncé son adresse
  OK  GET /sante répond 200
  OK  le client s'est chargé
  OK  l'audio a la bonne taille
  OK  le second appel est plus rapide
  OK  Whisper reconnaît la phrase
```

Trois choix à connaître :

* **le GPU ne sert qu'une synthèse à la fois** (verrou côté serveur) : un client qui
  arrive pendant une phrase attend son tour au lieu de faire échouer les deux ;
* **la réserve vit côté serveur** : elle profite donc à *tous* les clients, et la
  deuxième phrase identique est servie en 2 ms sans toucher au GPU ;
* **`--port 0`** fait choisir un port libre par l'OS : pratique pour lancer deux
  services sur la même machine, et c'est ce qu'utilisent les tests.

**Limite honnête de cette section** : la vérification ci-dessus tourne en boucle locale
sur cette machine. Le serveur écoute bien sur `0.0.0.0` et l'adresse annoncée est celle
de l'interface par défaut (`192.168.1.22`), mais **je n'ai pas fait l'essai entre deux
machines physiques** — un pare-feu ou une interface mal choisie reste possible. Le
paquet, lui, est réellement installé et son point d'entrée réellement exécuté.

### 13.5 Quatre erreurs, toutes de mon fait

1. **`soundfile` ne peut pas deviner le format d'un fichier `.tmp`.** L'écriture
   atomique passe par un fichier temporaire, et `sf.write("xxx.tmp", ...)` échoue sur
   « Unknown format » — extension inconnue. Six tests sont tombés d'un coup. Corrigé en
   passant `format="WAV"` explicitement. Le vrai enseignement est ailleurs : mon
   `except` renvoyait `False` en silence, donc l'échec n'apparaissait que dans les
   tests — sans eux, la réserve aurait paru fonctionner en ne gardant jamais rien.
2. **`charset=utf-8` collé sur `audio/wav`.** Voulant que les messages d'erreur
   français s'affichent (`synth\u00e9tiseur` → `synthétiseur`, corrigé par
   `ensure_ascii=False`), j'ai ajouté le charset à *toutes* les réponses. Un avertissement
   sur un type MIME audio est faux, et un test l'a attrapé.
3. **`TTSDistant` n'est pas un `KokoroTTS`.** Le vérificateur de types refusait
   l'affectation. Plutôt qu'un `type: ignore`, j'ai déclaré le contrat que les deux
   respectent (`Synthetiseur`, un `Protocol`) : ça documente ce qu'un synthétiseur doit
   fournir, et un test vérifie que le client le respecte.
4. **Mon test d'éviction était faux.** Il rafraîchissait la date d'une entrée puis en
   écrivait deux autres, en attendant que la première survive — sauf qu'après
   rafraîchissement elle restait la plus ancienne des trois, donc évincée à juste titre.
   Le test est refait dans le bon ordre (vieillir A → écrire B → relire A → écrire C) :
   c'est B qui doit partir, et c'est ce que vérifie le test.

### 13.6 Vérification

```
258 tests (46 de plus que la v0.6), dont 16 sur la réserve, 17 sur le serveur TTS
4 scripts de vérification exécutés pour de vrai : bench_cache, verif_config,
verif_distant, auxquels s'ajoutent les 9 contrôles de la v0.5 rejoués
```

| Fichier | Tests | Ce qu'il couvre |
|---|---|---|
| `tests/test_cache.py` | 16 | clé, atomicité, éviction, corrompu, branchement dans `synth` |
| `tests/test_distant.py` | 17 | routes, erreurs HTTP, client, aller-retour audio, contrat |
| `tests/test_config.py` | +12 | TOML, précédence, réglages par modèle, clés inconnues |

---

## 14. Confort d'usage (v1.1)

Quatre ajouts qui ne touchent ni au réseau, ni au GPU, ni au découpage du texte — donc
finissables et vérifiables en une soirée.

### 14.1 `--dire` : parler sans le modèle

```bash
voicechat --dire "Bonjour tout le monde."
voicechat --dire-fichier notes.md
voicechat --voice ff_siwis:3+ef_dora:1 --dire "Test de mélange."
```

Tout ce qui existe déjà est réutilisé : le nettoyage markdown, le découpage en phrases,
la file de synthèse, la réserve, le serveur TTS distant si `--tts-distant` est donné. Ce
qu'on entend par `--dire` est donc **exactement** ce qu'on entendrait d'une réponse.

Le texte passe d'abord par `split_sentences`, qui est fait pour un flux : le dernier
fragment, sans ponctuation finale, doit être réuni à la main, sinon un fichier s'arrête
à sa dernière phrase ponctuée.

La vérification ne se fie pas à l'oreille : elle construit la **clé de réserve** de la
phrase demandée et la cherche dans le dossier.

```
--- B. --dire : lecture directe, vérifiée par la réserve ---
  | Voix   : chargement de Kokoro « ff_siwis »…
  | Lecture de 1 phrase(s)…
  fichiers dans la réserve : 1
  durée du WAV déposé : 4.12 s
  clé attendue présente : True
```

### 14.2 `/cherche` : retrouver un passage

```
vous › /cherche tailscale
2 passage(s) pour « tailscale » :
  [reseau] 14/03 18:22 · vous · msg 1
      … on avait dit quoi sur le tailscale du portable ? …
  [reseau] 14/03 18:22 · ia · msg 2
      … le tailscale donne des adresses 100.x.y.z stables …
```

Deux points de conception :

**Insensible aux accents, dans les deux sens.** Chercher « resume » sans trouver
« résumé » serait un échec silencieux particulièrement pénible en français. Le texte est
normalisé en NFKD puis dépouillé de ses diacritiques — des deux côtés de la comparaison.
La longueur est préservée pour les lettres accentuées, ce qui permet de réutiliser la
position trouvée pour découper l'extrait ; les ligatures (`œ` → `oe`) allongent la
chaîne, d'où une borne pour ne jamais sortir du texte.

**Pas d'index, on relit les fichiers.** Ils sont petits, et un index qu'on oublie de
mettre à jour est pire que pas d'index du tout : il donnerait des résultats faux sans
prévenir. Une conversation abîmée est ignorée sans interrompre la recherche.

17 tests dans `tests/test_recherche.py`, dont « l'index de message est correct » (le
numéro doit désigner le message, pas la position dans le texte) et « conversation
abîmée n'empêche pas la recherche ».

### 14.3 `/resume` : ne plus perdre le début

Avant cette version, `_trim_history()` gardait les 24 derniers messages et **laissait
tomber les autres en silence**. Le modèle semblait devenir bête au fil d'une
conversation, alors qu'il avait simplement cessé de recevoir le début. Désormais :

* le débordement est **annoncé** une fois (« 4 message(s) ancien(s) ne sont plus
  envoyés ») avec la commande à lancer ;
* `/resume` demande au modèle de résumer les échanges anciens et les remplace par ce
  résumé, en gardant le prompt système et le dernier échange intacts.

```
--- D. /resume : compactage du contexte ---
  | [contexte] compactage de 4 message(s)…
  | [contexte] 233 caractères résumés en 158 ; 4 message(s) en contexte.
```

Côté sûreté : si l'appel échoue ou est interrompu, **l'historique n'est pas touché** —
mieux vaut un contexte long qu'un contexte perdu.

Ce que la mesure dit, honnêtement : ce modèle **n'est pas compressif**.

| Entrée | Résumé | Rapport | Tronqué ? |
|---|---|---|---|
| 855 caractères (échange dense, une info par phrase) | 785 | 1,1× | non (215 tokens produits) |
| 1572 caractères (réponses verbeuses) | 1001 | 1,6× | oui, au plafond |

D'où deux garde-fous ajoutés après mesure : un **plafond généreux** (512 à 2000 tokens,
proportionnel à l'entrée) et deux avertissements explicites — l'un quand le résumé n'est
pas plus court que l'original, l'autre quand il a atteint le plafond, car un résumé
tronqué a l'air complet tout en s'arrêtant au milieu d'une phrase. Le premier essai
coupait à 96 tokens, en plein « Concernant l'épingle d'av==18.1.0, » : c'est exactement
ce qu'on ne veut pas d'un mécanisme destiné à ne rien perdre.

### 14.4 `-q` : rendre le projet scriptable

```bash
voicechat -q "Résume les trois points de la réunion." --no-tts
voicechat -q "Il est l'heure de la pause."        # et il le dit à voix haute
```

C'est ce qui permet de brancher le projet ailleurs : un cron qui annonce un rappel, un
autre programme qui l'appelle. Le cas qui compte est donc l'absence de terminal — c'est
ainsi qu'il est testé, avec `stdin` sur `/dev/null` :

```
--- A. mode non-interactif, sans terminal (cas du cron) ---
  code de sortie : 0
  | Modèle : Ornith-1.5-35B-A3B-APEX-i-mini
  | Voix   : désactivée (--no-tts)
  | ia › bonjour.
```

Vérifié avant de s'appuyer dessus : `ClavierGeneration`, qui met le terminal en mode
brut pendant la génération, sort proprement quand l'entrée n'est pas un terminal. Sans
ça, ce mode aurait planté au premier lancement depuis un cron.

### 14.5 Intégration continue

La suite tourne sur chaque `push`, avec pour seul bagage `numpy`, `soundfile` et
`pytest` — pas de torch, pas de Kokoro, pas de GPU. Deux jobs, un par version de Python,
chacun en environ une minute :

```
✓ tests hors ligne (Python 3.11) in 1m10s
✓ tests hors ligne (Python 3.12) in 1m10s
```

Et ce que la CI a réellement exécuté, tel quel dans ses journaux :

```
Tests                                  271 passed, 4 skipped in 29.26s
La suite ne doit pas dépendre des paquets lourds
   blocage actif : torch, kokoro, faster_whisper, ctranslate2, av
                                       271 passed, 4 skipped in 28.82s
```

Ce second contrôle n'est pas décoratif : `tests/verif_sans_lourds.py` installe un
bloqueur d'import et échoue si un module se met à importer torch en haut de fichier.
Sans lui, quelqu'un ajouterait un `import torch` un jour, la CI deviendrait lente puis
serait désactivée, et plus rien ne serait vérifié. Le badge est en haut de ce README.

### 14.6 Quatre erreurs, toutes dans mes vérifications

Aucune n'est dans le code livré, mais elles expliquent pourquoi la mise au point a pris
plus de temps que prévu — et trois d'entre elles m'ont appris quelque chose sur le
programme lui-même.

1. **Attendre le prompt ne marche pas.** Mon pilote de pseudo-terminal attendait
   « vous › » pour savoir qu'un tour était fini. Or l'éditeur de ligne **réaffiche le
   prompt à chaque frappe** : il réapparaissait donc dès que la question était tapée,
   bien avant la réponse. Symptôme : `/resume` semblait ne rien faire. Corrigé par une
   détection de silence, avec un motif explicite quand la commande appelle le modèle sans
   rien afficher pendant plusieurs secondes.
2. **Texte et Entrée dans une seule écriture = collage.** L'éditeur attend alors une
   Entrée distincte avant d'envoyer — c'est exactement le comportement voulu depuis la
   v0.1 pour qu'un texte collé ne parte pas tout seul. Il faut deux écritures séparées,
   comme un vrai clavier (`tests/test_editor.py` le fait déjà).
3. **Mon bloqueur d'import levait la mauvaise exception.** Il levait `ImportError` ; un
   paquet réellement absent lève `ModuleNotFoundError`, et depuis pytest 8.2
   `importorskip` ne saute **que** sur celle-là. Le harnais échouait donc là où la vraie
   CI aurait ignoré le test — il mentait sur ce qu'il prétendait mesurer.
4. **Un test supposait faster-whisper installé.** `test_chargement_impossible_...`
   attend un message précis (« aucune configuration ») qui n'arrive que si la
   bibliothèque est présente. Sans elle, `charger()` s'arrête avant, sur un autre message
   tout aussi lisible. Le test est maintenant explicitement conditionné.

### 14.7 Vérification

```
$ .venv/bin/python tests/verif_confort.py
=== contrôles ===
  OK  `-q` répond sans terminal et sort en 0
  OK  `--dire` synthétise bien la phrase demandée
  OK  `/cherche` retrouve le passage
  OK  `/resume` compacte l'historique

>>> OK : les quatre apports de la v1.1 fonctionnent sur le vrai programme
```

| Fichier | Tests | Ce qu'il couvre |
|---|---|---|
| `tests/test_recherche.py` | 17 | accents, casse, extraits, limites, index, fichier abîmé |
| `tests/verif_sans_lourds.py` | — | la suite entière, paquets lourds interdits |
| `tests/verif_confort.py` | — | `-q`, `--dire`, `/cherche`, `/resume` sur le vrai CLI |

---

## 15. Licence

MIT — voir le fichier `LICENSE`. Faire ce qu'on veut, sans garantie.
