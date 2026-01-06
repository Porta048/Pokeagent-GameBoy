# PokéAgent GameBoy

Un agente AI autonomo progettato per giocare a Pokémon Rosso sul Game Boy utilizzando **PyBoy** per l'emulazione e **LLM (Large Language Models)** locali (tramite Ollama) per il ragionamento decisionale.

L'agente utilizza un approccio ibrido che combina:
- **Visione Computerizzata**: Analisi dello schermo di gioco.
- **Ragionamento LLM (Chain of Thought)**: Pianificazione strategica basata sul contesto.
- **Knowledge Base**: Database di conoscenze sul gioco (mappe, obiettivi, debolezze tipi).
- **Memoria Ibrida**: Memoria a breve termine per il contesto immediato e a lungo termine per i progressi.

## Caratteristiche Principali

*   **Architettura a 3 Fasi**:
    *   **Planning**: Formula obiettivi a breve termine basati sullo stato attuale e sulla Knowledge Base.
    *   **Execution**: Traduce gli obiettivi in sequenze di azioni precise, gestendo movimenti e menu.
    *   **Critique**: Valuta periodicamente il successo delle azioni e adatta la strategia.
*   **Integrazione LLM Locale**: Supporto per modelli via Ollama (default: `qwen2.5:0.5b` per velocità ed efficienza).
*   **Chain of Thought (CoT)**: L'agente "pensa" prima di agire, spiegando il motivo delle sue scelte (visibile nei log).
*   **Knowledge Base Semantica**: Il sistema conosce la mappa del gioco, le connessioni tra le aree e gli obiettivi principali.
*   **Gestione Intelligente dei Fallback**: Se l'LLM è lento o non disponibile, l'agente usa logiche euristiche per evitare di bloccarsi.
*   **Sincronizzazione GUI**: Ottimizzato per sincronizzare le azioni dell'agente con le animazioni del gioco.

## Logica di Gioco del Modello

### Come l'Agente Prende Decisioni

L'agente opera attraverso un ciclo continuo di **percezione → pianificazione → azione → riflessione**:

1. **Analisi del Contesto**: Ogni 25 step, l'agente analizza:
   - Posizione del giocatore nella mappa corrente
   - Stato della squadra Pokémon (HP, livelli, condizioni)
   - Modalità di gioco attiva (esplorazione, battaglia, menu, dialogo)
   - Obiettivi a lungo termine (es. "Sconfiggi la Lega Pokémon")

2. **Formulazione degli Obiettivi**: Basandosi sul contesto, l'agente genera obiettivi specifici come:
   - *"Esplora l'area per trovare oggetti utili"*
   - *"Cura i Pokémon feriti al Centro Pokémon"*
   - *"Affronta l'Allenatore sulla route 24"*

3. **Selezione delle Azioni**: L'agente considera 9 azioni possibili:
   - Movimento: su, giù, sinistra, destra
   - Interazioni: A, B, START, SELECT
   - Nessuna azione (attesa)

### Strategie di Gioco Intelligenti

#### Riconoscimento delle Modalità
L'agente adatta il suo comportamento in base alla situazione:

- **Modalità Esplorazione**: Movimenti variati, ricerca di oggetti, interazione con NPC
- **Modalità Battaglia**: Strategie di tipo, gestione HP, cambi Pokémon
- **Modalità Menu**: Navigazione efficiente tra le opzioni
- **Modalità Dialogo**: Progressione attraverso le conversazioni

#### Anti-Loop System
Per evitare di rimanere bloccati, l'agente implementa:
- **Pattern Detection**: Riconosce sequenze ripetitive (es. su-giù-su-giù)
- **Circular Movement Detection**: Identifica movimenti circolari che indicano essere bloccati
- **Alternating Pattern Prevention**: Evita pattern A-B-A-B che non portano progressione
- **Exploration Boost**: Dopo 3-4 tentativi falliti, aumenta la varietà delle azioni

#### Sistema di Fallback Ibrido
Quando l'LLM non è disponibile o lento:

1. **Strategia Contestuale**: Seleziona azioni appropriate alla modalità
   - In battaglia: priorità a "A" per attaccare, "su/giù" per selezionare mosse
   - Nei menu: "B" per tornare indietro, navigazione direzionale
   - In esplorazione: movimenti cardinali con preferenza per nuove direzioni

2. **Memoria di Pattern**: Ricorda quali azioni hanno funzionato in contesti simili

3. **Euristica di Esplorazione**: Preferisce direzioni non esplorate recentemente

### Timing e Sincronizzazione

L'agente rispetta i tempi del gioco Game Boy:
- **Frame Rate**: 59.73 FPS (la velocità originale del Game Boy)
- **Durata Animazioni**: Aspetta il completamento delle animazioni (16 frame per passo)
- **Input Windows**: Mantiene i pulsanti premuti per 3-6 frame per garantire registrazione
- **Safety Factor**: 25% di margine per compensare il lag dell'emulazione

### Apprendimento Adattivo

L'agente migliora nel tempo attraverso:
- **Memoria di Successo**: Ricorda quali sequenze hanno portato a progressi
- **Pattern Efficaci**: Identifica strategie vincenti per situazioni specifiche
- **Critica Periodica**: Ogni 50 step, valuta se l'approccio sta funzionando
- **Adattamento Strategico**: Modifica obiettivi se non sta facendo progressi

### Gestione degli Errori

Il sistema include robustezza attraverso:
- **Rate Limiting**: Massimo 600 chiamate LLM al minuto per evitare sovraccarico
- **Timeout Management**: 60 secondi massimi per risposte LLM
- **Fallback Gerarchico**: 4 livelli di fallback (LLM → Template → Euristiche → Azione casuale)
- **Recovery da Crash**: Ripristino dello stato dopo errori critici

## Requisiti

*   **Python 3.10+**
*   **Ollama**: Installato e in esecuzione localmente.
*   **ROM di Pokémon Rosso**: File `.gb` (da posizionare in `roms/`).

## Installazione

1.  **Clona il repository**:
    ```bash
    git clone https://github.com/tuo-username/Pokeagent-GameBoy.git
    cd Pokeagent-GameBoy
    ```

2.  **Crea un ambiente virtuale (opzionale ma consigliato)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Installa le dipendenze**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepara Ollama**:
    Assicurati di avere Ollama installato e scarica il modello richiesto (o modificalo in `config.py`):
    ```bash
    ollama pull qwen2.5:0.5b
    ```

5.  **Aggiungi la ROM**:
    Copia la tua ROM di Pokémon Rosso (es. `Pokemon Red.gb`) nella cartella `roms/`.

## Utilizzo

Assicurati che il server Ollama sia attivo, poi avvia l'agente:

```bash
python3 main.py
```

### Opzioni da riga di comando (non implementate nel main attuale ma configurabili in `config.py`)
Attualmente la configurazione principale avviene tramite il file `config.py`.

## Configurazione Avanzata (`config.py`)

Puoi personalizzare il comportamento dell'agente modificando `config.py`:

*   `LLM_MODEL`: Modello Ollama da utilizzare (default: `qwen2.5:0.5b`).
*   `EMULATION_SPEED`: Velocità di emulazione (default: `1`x).
*   `HEADLESS`: Imposta a `True` per eseguire senza finestra grafica.
*   `RENDER_EVERY_N_FRAMES`: Frequenza di rendering (per velocizzare in modalità headless).
*   `LLM_MAX_CALLS_PER_MINUTE`: Limite chiamate API per evitare sovraccarico.

## Contribuire

Sentiti libero di aprire Issue o Pull Request per migliorare l'agente, aggiungere nuove funzionalità alla Knowledge Base o ottimizzare i prompt!

## Licenza

[Inserisci qui la tua licenza, es. MIT]