"""Lecture d'un message au clavier, avec gestion correcte du collage multi-lignes.

Pourquoi ne pas utiliser ``input()`` ? Parce que lorsqu'on colle un texte de plusieurs
lignes, le terminal livre chaque ligne comme si l'utilisateur les avait tapées une par
une : ``input()`` en consommait une, et la boucle de chat traitait toutes les suivantes
comme autant de nouveaux prompts.

Ce module lit le clavier en mode brut et distingue trois choses :

* une **frappe** normale — le texte s'affiche au fil de l'eau ;
* un **collage** — repéré par les marqueurs du « bracketed paste » que l'on active
  nous-mêmes (``\\x1b[?2004h``) : les sauts de ligne sont conservés *à l'intérieur* du
  message, et rien n'est envoyé ;
* la touche **Entrée**, seul et unique déclencheur d'envoi.

Repli pour les terminaux qui n'implémentent pas le bracketed paste : un saut de ligne
reçu moins de ``BURST_DELAY`` secondes après le caractère précédent est considéré comme
faisant partie d'un collage — un humain n'enchaîne pas aussi vite.

Limites assumées : pas d'historique Ctrl+R, et au-delà d'une ligne écran le message est
résumé (``[12 lignes, 842 car.]…``) au lieu d'être redessiné à chaque frappe.
"""

from __future__ import annotations

import codecs
import os
import select
import shutil
import sys
import termios
import time

PROMPT = "vous › "

PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"

BURST_DELAY = 0.03  # s — au-delà, un saut de ligne est une vraie touche Entrée
ESC_TIMEOUT = 0.05  # s — au-delà, une séquence \x1b incomplète est abandonnée

# Séquences de touches reconnues (flèches, début/fin, suppr).
_KEYS = {
    "\x1b[A": "haut",
    "\x1b[B": "bas",
    "\x1b[C": "droite",
    "\x1b[D": "gauche",
    "\x1b[H": "debut",
    "\x1b[F": "fin",
    "\x1bOH": "debut",
    "\x1bOF": "fin",
    "\x1b[1~": "debut",
    "\x1b[4~": "fin",
    "\x1b[3~": "suppr",
}

# Les plus longues d'abord : « \x1b[200~ » ne doit pas être confondu avec « \x1b[2 ».
_SEQUENCES = tuple(sorted((*_KEYS, PASTE_START, PASTE_END), key=len, reverse=True))


def _est_prefixe(s: str) -> bool:
    """``s`` est-il le début (incomplet) d'une séquence connue ?"""
    return any(seq.startswith(s) for seq in _SEQUENCES)


def _jeton(s: str) -> tuple[str | None, str]:
    """Découpe le début de ``s`` en un jeton.

    Retourne ``(jeton, reste)``. ``jeton`` vaut ``None`` quand ``s`` est le début
    d'une séquence encore incomplète : il faut attendre d'autres octets.
    """
    for seq in _SEQUENCES:
        if s.startswith(seq):
            return seq, s[len(seq) :]
    if s.startswith("\x1b") and _est_prefixe(s):
        return None, s
    return s[0], s[1:]


class LineEditor:
    """Lit un message. Entrée envoie ; un collage ne déclenche jamais rien."""

    def __init__(self, prompt: str = PROMPT, history_limit: int = 200) -> None:
        self.prompt = prompt
        self.history: list[str] = []
        self.history_limit = history_limit
        # Passe à True dès qu'un marqueur de collage est vu : à partir de là on peut
        # faire confiance aux marqueurs et ignorer l'heuristique de timing.
        self._balise_paste = False

    # ------------------------------------------------------------------ API
    def read(self) -> str | None:
        """Lit un message complet. ``None`` signifie fin d'entrée (Ctrl+D)."""
        if not sys.stdin.isatty():
            return self._read_non_interactif()
        fd = sys.stdin.fileno()
        try:
            ancien = termios.tcgetattr(fd)
        except termios.error:
            return self._read_non_interactif()

        buf: list[str] = []
        curseur = 0
        histo = len(self.history)
        brouillon = ""
        collage = False
        en_attente = ""
        precedent = 0.0
        decodeur = codecs.getincrementaldecoder("utf-8")("replace")

        try:
            self._mode_brut(fd, ancien)
            sys.stdout.write("\x1b[?2004h\n")  # active le bracketed paste
            self._afficher(buf, curseur)

            while True:
                timeout = ESC_TIMEOUT if (en_attente and _est_prefixe(en_attente)) else None
                pret, _, _ = select.select([fd], [], [], timeout)
                if not pret:
                    en_attente = ""  # séquence \x1b incomplète et rien n'arrive
                    continue

                data = os.read(fd, 4096)
                if not data:  # tube fermé / terminal disparu
                    sys.stdout.write("\n")
                    return None

                maintenant = time.monotonic()
                rafale = len(data) > 1 or (maintenant - precedent) < BURST_DELAY
                precedent = maintenant
                en_attente += decodeur.decode(data)

                while en_attente:
                    jeton, reste = _jeton(en_attente)
                    if jeton is None:
                        break  # séquence incomplète : on attend la suite
                    en_attente = reste

                    # ---------------------------------------------- collage
                    if jeton == PASTE_START:
                        collage = True
                        self._balise_paste = True
                        continue
                    if jeton == PASTE_END:
                        collage = False
                        self._afficher(buf, curseur)
                        continue

                    # ---------------------------------------------- envoi
                    if jeton in ("\r", "\n"):
                        if collage:
                            curseur = self._inserer(buf, curseur, "\n")
                        elif self._balise_paste or not rafale:
                            return self._soumettre(buf)
                        else:
                            # Repli : saut de ligne reçu en rafale = collage.
                            curseur = self._inserer(buf, curseur, "\n")
                        continue

                    # ---------------------------------------------- contrôle
                    if jeton == "\x03":  # Ctrl+C
                        if buf:
                            buf.clear()
                            curseur = 0
                            self._afficher(buf, curseur)
                            continue
                        raise KeyboardInterrupt
                    if jeton == "\x04":  # Ctrl+D
                        if not buf:
                            sys.stdout.write("\n")
                            return None
                        if curseur < len(buf):
                            del buf[curseur]
                    elif jeton == "\x15":  # Ctrl+U — effacer la ligne
                        del buf[:]
                        curseur = 0
                    elif jeton == "\x17":  # Ctrl+W — effacer le mot précédent
                        while buf and curseur and buf[curseur - 1] == " ":
                            del buf[curseur - 1]
                            curseur -= 1
                        while buf and curseur and buf[curseur - 1] != " ":
                            del buf[curseur - 1]
                            curseur -= 1
                    elif jeton in ("\x7f", "\x08"):  # Retour arrière
                        if curseur:
                            del buf[curseur - 1]
                            curseur -= 1
                    elif jeton == "\x0c":  # Ctrl+L — nettoyer l'écran
                        sys.stdout.write("\x1b[2J\x1b[H")
                    elif jeton in _KEYS:
                        action = _KEYS[jeton]
                        if action == "gauche":
                            curseur = max(0, curseur - 1)
                        elif action == "droite":
                            curseur = min(len(buf), curseur + 1)
                        elif action == "debut":
                            curseur = 0
                        elif action == "fin":
                            curseur = len(buf)
                        elif action == "suppr":
                            if curseur < len(buf):
                                del buf[curseur]
                        elif action == "haut" and histo > 0:
                            if histo == len(self.history):
                                brouillon = "".join(buf)
                            histo -= 1
                            texte = self.history[histo]
                            buf[:] = list(texte)
                            curseur = len(texte)
                        elif action == "bas" and histo < len(self.history):
                            histo += 1
                            texte = brouillon if histo == len(self.history) else self.history[histo]
                            buf[:] = list(texte)
                            curseur = len(texte)
                    elif jeton.isprintable():
                        curseur = self._inserer(buf, curseur, jeton)

                    self._afficher(buf, curseur)
        finally:
            sys.stdout.write("\x1b[?2004l")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, ancien)

    # -------------------------------------------------------------- affichage
    def _afficher(self, buf: list[str], curseur: int) -> None:
        texte = "".join(buf)
        largeur = shutil.get_terminal_size().columns

        if "\n" not in texte and len(self.prompt) + len(texte) < largeur:
            # Tient sur la ligne courante : on redessine et on replace le curseur.
            sys.stdout.write("\r\x1b[J" + self.prompt + texte)
            reste = len(texte) - curseur
            if reste:
                sys.stdout.write(f"\x1b[{reste}D")
        else:
            lignes = texte.count("\n") + 1
            indication = "⏎ envoyer · Ctrl+C annuler"
            plat = " ".join(texte.split())
            base = f"{self.prompt}[{lignes} lignes, {len(texte)} car.] "
            marge = max(10, largeur - len(base) - len(indication) - 3)
            apercu = plat[:marge]
            if len(plat) > marge:
                apercu = apercu.rstrip() + "…"
            sys.stdout.write(f"\r\x1b[J{base}{apercu}  {indication}")
        sys.stdout.flush()

    # --------------------------------------------------------------- outils
    def _inserer(self, buf: list[str], curseur: int, texte: str) -> int:
        for c in texte:
            buf.insert(curseur, c)
            curseur += 1
        return curseur

    def _soumettre(self, buf: list[str]) -> str:
        texte = "".join(buf).strip()
        sys.stdout.write("\n")
        sys.stdout.flush()
        if texte:
            self.history.append(texte)
            if len(self.history) > self.history_limit:
                del self.history[0]
        return texte

    @staticmethod
    def _mode_brut(fd: int, ancien) -> None:
        """Mode brut ciblé : ni canonique, ni écho, ni signaux — mais OPOST conservé.

        Garder ``OPOST`` est essentiel : sans lui, un « \\n » écrit à l'écran ne
        revient pas en colonne 0 et l'affichage part en escalier.
        """
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)

    def _read_non_interactif(self) -> str | None:
        """Entrée non interactive (tube, redirection) : une ligne = un message.

        On garde volontairement l'ancien comportement ici : un script qui écrit
        plusieurs lignes attend plusieurs tours de conversation, pas un collage.
        """
        ligne = sys.stdin.readline()
        if ligne == "":
            return None  # fin de fichier
        return ligne.strip()
