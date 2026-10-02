# voicechat ★

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
| Sélection GPU `auto/cuda/cpu` | ✅ **GPU opérationnel** — RTF 0,09 (cf. §8 pour l'obligation de build cu126 sur Pascal) |
| Serveur LLM `100.91.114.49:8080` (niko-1650-super) | ✅ **joignable** — Ornith-1.5-35B-A3B Q4_K_M |
| Suite de tests hors ligne | ✅ 72 tests passent |

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
kokoro        # TTS (tire torch automatiquement)
sounddevice   # lecture audio (PortAudio)
numpy
```

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
| `VOICECHAT_MODEL` | *(auto)* | nom du modèle ; si vide → détecté via `/v1/models` |
| `VOICECHAT_API_KEY` | *(vide)* | jeton éventuel |
| `VOICECHAT_VOICE` | `ff_siwis` | voix Kokoro (français) |
| `VOICECHAT_LANG` | `f` | code langue Kokoro (`a`=en, `f`=fr, `e`=es, `i`=it, `p`=pt, `j`=ja, `z`=zh, `h`=hi) |
| `VOICECHAT_SPEED` | `1.0` | vitesse de lecture |
| `VOICECHAT_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOICECHAT_SYSTEM` | *(court prompt FR)* | prompt système |
| `VOICECHAT_TTS` | `1` | `0` pour désactiver la voix |
| `VOICECHAT_DATA` | `~/.local/share/voicechat` | dossier des conversations sauvegardées |

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
| `/voice <nom>` | change la voix à chaud |
| `/lang <code>` | change la langue Kokoro |
| `/speed <x>` | change la vitesse (ex. `/speed 1.15`) |
| `/tts on\|off` | active/coupe la voix |
| `/model <nom>` | change de modèle pour les tours suivants |
| `/system <texte>` | remplace le prompt système |
| `/save [nom]` | enregistre la conversation (défaut : `derniere`) |
| `/load <nom>` | recharge une conversation sauvegardée |
| `/conversations` | liste les conversations sauvegardées |
| `/forget <nom>` | supprime une conversation sauvegardée |
| `/voices` | liste les voix Kokoro |
| `/device` | liste les sorties audio détectées |
| `/stats` | latences (TTFT, débit en tokens, RTF TTS) |
| `/debug` | bascule l'affichage des stats à chaque tour |

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
| `←` / `→` | déplacer le curseur |
| `Ctrl+U` / `Ctrl+W` | effacer la ligne / le mot précédent |
| `Ctrl+C` | effacer le brouillon (sur ligne vide : quitter) |
| `Ctrl+D` | quitter |

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

**Limites assumées** : pas de `Ctrl+R` (recherche dans l'historique) ; au-delà d'une
ligne écran le message est résumé au lieu d'être redessiné — l'édition (retour arrière,
`Ctrl+U`) reste exacte, seul l'affichage est condensé.

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
├── store.py        # conversations sauvegardées (JSON, écriture atomique)
└── cli.py          # boucle console, routage des commandes, orchestration
tests/              # pytest hors ligne + fake_llm_server.py (faux serveur OpenAI)
```

**Pourquoi cette découpe ?** Le pipeline voix doit se déclencher **pendant** la génération,
pas après : sinon on attend la fin de la réponse avant d'entendre le premier mot. Le flux est
donc découpé en phrases au vol (`text.split_sentences`), chaque phrase part dans une file
(`tts.SpeechPipeline`) consommée par un thread qui synthétise puis empile le son dans
`audio.Speaker`. Un seul thread de synthèse → les phrases restent dans l'ordre, et le GPU
n'est jamais appelé en concurrence.

### Tests (sans réseau, sans audio)

```bash
.venv/bin/python -m pytest tests/ -q
```

Pour valider la v0.1 sans dépendre de `niko-tv`, un faux serveur OpenAI-compatible est fourni :

```bash
python3 tests/fake_llm_server.py 8099          # terminal 1
.venv/bin/python -m voicechat --base-url http://127.0.0.1:8099/v1 --no-tts   # terminal 2
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

### v0.3 — confort restant
- [ ] Historique de saisie : recherche `Ctrl+R` et complétion des `/commandes` (Tab)
- [ ] Streaming audio par morceaux (moins de trou entre deux phrases)
- [ ] Profils de prompt système (`--profil brain-wash`) pour ne pas les retaper
- [ ] Export d'une conversation en markdown

### v0.4 — entrée vocale (mode mains libres)
- [ ] Capture micro (`sounddevice`) + VAD (détection d'activité vocale)
- [ ] `faster-whisper` en local pour la reconnaissance (GPU, `small`/`medium`)
- [ ] Boucle full-duplex : on parle → transcription → LLM → voix

### v0.5 — voix de meilleure qualité
- [ ] Mélange de voix Kokoro (mix de styles, cf. `kokoro.StyleTTS`)
- [ ] Sélection de voix par persona dans le prompt système
- [ ] Césures/abreviations FR affinées (nombres, sigles, unités)
- [ ] Normalisation des réponses markdown avant synthèse (listes, code, liens)

### v0.6 — robustesse réseau
- [ ] Reconnexion automatique + retry exponentiel si le serveur redémarre
- [ ] Bascule automatique vers un serveur de secours (ex. `niko-tv`, aujourd'hui éteint)
- [ ] Mode dégradé : réponses texte, voix mise en file puis rejouée au retour du GPU

### v1.0 — distribué
- [ ] Serveur TTS partagé (le poste léger envoie le texte, un poste GPU synthétise)
- [ ] Paquet `pip install ./voicechat` + point d'entrée `voicechat`
- [ ] Fichier de config TOML + profils (par modèle, par voix)
- [ ] Cache disque des phrases déjà synthétisées (le LLM se répète souvent)

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

`tests/test_editor.py` verrouille tout ça (25 tests) via de vrais pseudo-terminaux :
collage balisé, collage non balisé détecté par rafale, frappe lente qui ne doit **pas**
ressembler à un collage, retours arrière, `Ctrl+U`, historique, Échap pendant une
génération, et entrée non interactive.

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [100%]
72 passed in 7.84s
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
la voix en français sans changer de langue, voir la piste « mélange de styles » en v0.4 de la
roadmap. Les 53 autres voix servent aux langues ci-dessus.

---

## 10. Licence

MIT — faire ce qu'on veut, sans garantie.
