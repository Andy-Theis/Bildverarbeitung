# Bildverarbeitung
# Anisotrope Diffusion (Perona-Malik-Algorithmus) in Python

Eine effiziente NumPy-Implementierung der anisotropen Diffusion nach dem **Perona-Malik-Modell** zur kantenerhaltenden Bild-Entrauschung.

---

## Über das Projekt

Klassische Filter (wie der Gauß-Filter) glätten ein Bild gleichmäßig und verschmieren dabei wichtige Kanten und Objektgrenzen. Die **anisotrope Diffusion** löst dieses Problem: Sie reduziert Rauschen in homogenen Bildbereichen stark, während Kanten und Feinstrukturen erhalten bleiben.

### Funktionsweise
Der Algorithmus berechnet lokal die Bildgradienten in vier Richtungen (**N**ord, **S**üd, **O**st, **W**est) und steuert den Diffusionsfluss über eine Gauss-ähnliche Leitwert-Funktion:

$$g(\nabla u) = \exp\left(-\left(\frac{\nabla u}{\kappa}\right)^2\right)$$

* **Kleine Gradienten** ($\nabla u \ll \kappa$): $g \approx 1 \rightarrow$ Starke Glättung (Rauschreduktion).
* **Große Gradienten** ($\nabla u \gg \kappa$): $g \approx 0 \rightarrow$ Keine Glättung (Kanten bleiben scharf).

---

## Quickstart

### 1. Voraussetzungen

Erstelle eine virtuelle Python-Umgebung und installiere die benötigten Pakete:

```bash
# Virtuelle Umgebung erstellen & aktivieren
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Abhängigkeiten installieren
pip install numpy matplotlib pillow
