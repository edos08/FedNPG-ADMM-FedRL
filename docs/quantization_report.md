# Quantizzazione in FedNPG-ADMM — come funziona

Questo documento descrive il pacchetto `src/fedrl_comm/quantization`:
cosa viene quantizzato, con quali schemi, come si misurano i byte e come si
lancia lo sweep sui bit-width. Non è nel paper originale (che cita Q-GADMM [9]
solo come lavoro correlato), quindi non c'è una figura di riferimento.

---

## 1. L'idea in una frase

Ogni tensore che un agente manda al server passa per un `Channel` simulato.
La quantizzazione è una **funzione `transform`** passata al canale: prende il
tensore, lo quantizza **e lo dequantizza subito** (andata e ritorno in un'unica
chiamata), e dichiara quanti bit per elemento sarebbero serviti sul "filo".
Il server aggrega la versione degradata; il `ByteMeter` conta i byte usando i
bit dichiarati.

Nessun file dell'algoritmo (`agent.py`, `server.py`, `fisher.py`, `cg.py`) sa
che la quantizzazione esiste.

---

## 2. Dove si aggancia al training

### 2.1 Cosa viaggia sul canale

Nel ciclo di [train.py](../src/fedrl_comm/fednpg_admm/src/fednpg_admm/train.py),
a ogni iterazione `k`, ogni agente `i` invia (solo in uplink, agente → server):

| Tensore | Dimensione | Quante volte per iterazione | Note |
|---------|-----------|------------------------------|------|
| `g_i` (gradiente locale della policy) | `d` | 1 | non cambia fra i round ADMM, quindi viene inviato e conteggiato una sola volta |
| `y_i` (direzione locale ADMM) | `d` | `--admm-rounds` (default 1) | ricalcolato a ogni round |

`d` è il numero di parametri della policy. Il downlink (broadcast di `θ` e `y`
dal server) **non** viene quantizzato né conteggiato.

```python
local_grads.append(channel.send("g_i", g_i, iteration=k))
...
y_list.append(channel.send("y_i", y_i, iteration=k))
```

Si usa sempre il **valore restituito** da `send()`: è quello che "arriva" al
server. Quindi `compute_candidate_step(y_list, g_list, ...)` e la media dei
`y_i` lavorano sui tensori già quantizzati: l'errore di quantizzazione entra
davvero nell'aggiornamento di `θ`.

### 2.2 Il gancio: `channel_factory`

`train(args, channel_factory=...)` accetta una funzione `num_agents -> list[Channel]`.

- Di default è `make_channels_for_agents`: canali con `identity_transform`, cioè
  nessuna quantizzazione, contabilizzati a 64 bit/elemento (convenzione del
  paper: "double precision").
- Lo sweep passa invece `make_channel_factory(transform)`, che crea
  `num_agents` canali che condividono lo stesso `transform`, ognuno con il
  **proprio** `ByteMeter`.

### 2.3 Flusso completo

```
agente i                          Channel.send()                            server
────────                          ──────────────                            ──────
g_i, y_i (float32) ──► transform(t) ─► (t̂ degradato, bits) ─► ByteMeter.record(numel, bits)
                                         │
                                         └──────────────────────────────► aggrega t̂
```

---

## 3. I due schemi

La scelta si fa con `--scheme` ([quantizers.py](../src/fedrl_comm/quantization/src/quantization/quantizers.py)).

### 3.1 `stochastic`: quantizzazione uniforme stocastica (tipo QSGD / Q-GADMM)

Per ogni tensore `x` con `b` bit:

1. `t_min = min(x)`, `t_max = max(x)` (una coppia per tensore)
2. `L = 2^b − 1` livelli, passo `s = (t_max − t_min) / L`
3. normalizzazione: `u = (x − t_min) / s ∈ [0, L]`
4. arrotondamento **stocastico**: `q = ⌊u⌋ + Bernoulli(u − ⌊u⌋)`
5. dequantizzazione: `x̂ = t_min + q · s`

**Proprietà chiave: non è distorto.** `E[x̂] = x`, perché ogni elemento va al
livello superiore con probabilità pari alla distanza frazionaria. In media quindi
il server riceve il gradiente e la direzione ADMM corretti, con varianza extra
per elemento ≤ `s²/4`. Con l'arrotondamento al più vicino l'errore sarebbe
sistematico e si accumulerebbe fra le iterazioni.

Dettagli di implementazione:

- **`b = 1` è valido:** ogni elemento diventa `t_min` oppure `t_max`, con la
  probabilità giusta.
- **Tensore costante** (`t_max == t_min`, per esempio un gradiente tutto zero):
  `scale = 0` ed errore nullo.
- **`clamp(0, 1)` su `frac`:** per errori di arrotondamento float32, `frac` può
  superare 1 di pochissimo e `torch.bernoulli` andrebbe in errore.
- **RNG isolato:** se si passa `seed`, il transform crea **un solo**
  `torch.Generator` dedicato. Viene creato in modo lazy alla prima chiamata, sul
  device del tensore. Così i bit casuali della quantizzazione **non consumano**
  l'RNG globale usato per campionare le azioni: run con bit-width diversi vedono
  la stessa sequenza di campionamento delle azioni, e il confronto è più pulito.
  Il generatore è condiviso fra tutti gli agenti (stesso transform) e avanza a
  ogni chiamata.

Errore relativo indicativo, con `s = range/(2^b − 1)`:

| b | livelli | passo `s` in frazione del range |
|---|---------|----------------------------------|
| 1 | 1 | 1 |
| 2 | 3 | 0.33 |
| 4 | 15 | 0.067 |
| 8 | 255 | 0.0039 |
| 16 | 65535 | 1.5e-5 |

### 3.2 `float`: formati float a precisione ridotta

Questo schema è **deterministico** (arrotondamento al valore rappresentabile più
vicino) e ha una spaziatura **non uniforme** (esponenziale): è molto preciso
vicino allo zero e grossolano sui valori grandi. È una filosofia diversa da
`stochastic`, non la stessa cosa con un altro nome.

Bit ammessi: **4, 8, 16, 32** (`FLOAT_ALLOWED_BITS`).

| bits | Implementazione | Note |
|------|-----------------|------|
| 32 | passthrough, dichiarato **32** bit | di proposito *non* `identity_transform`, che verrebbe contato a 64 bit falsando il punto "32" nel grafico byte-vs-bit |
| 16 | cast `float16` e ritorno a `float32` | nativo torch |
| 8 | cast `float8_e4m3fn` e ritorno a `float32` | nativo torch (4 bit di esponente, 3 di mantissa, max ≈ 448) |
| 4 | **simulato** (`_simulate_fp4`) | torch non ha un dtype a 4 bit |

**FP4 simulato (stile E2M1):** 1 bit di segno, 2 di esponente, 1 di mantissa,
bias 1. Le magnitudini rappresentabili sono `{0, 1, 1.5, 2, 3, 4, 6}`. Siccome
gradienti e `y_i` sono tipicamente molto più piccoli di 1, senza un fattore di
scala verrebbero azzerati tutti. Per questo si usa una **scala per tensore**
(nello spirito dei block-scale hardware, per esempio NVFP4/MX):

```
scale = max|x| / 6
x̂ = sign(x) · nearest_level(|x| / scale) · scale
```

Non è bit-exact rispetto a E2M1 reale: niente subnormali (vanno a zero), niente
inf/NaN. Per studiare il compromesso reward/byte basta una perdita di precisione
reale e monotonamente più grossolana di fp8.

⚠️ Attenzione: fp16 e fp8 **non** hanno una scala per tensore. Valori molto
piccoli possono andare in underflow (in fp8 e4m3 il minimo subnormale è ≈ 2e-3,
il minimo normale ≈ 1.6e-2). Se i gradienti hanno componenti di ordine 1e-3 o
più piccole, fp8 ne azzera molte: a 8 bit fp8 può quindi andare **peggio** di
fp4 scalato. È un comportamento reale del formato, non un bug, ma va tenuto
presente quando si leggono i risultati.

---

## 4. Contabilità dei byte

[`ByteMeter`](../src/fedrl_comm/common/src/common/comms/meter.py) non guarda
`tensor.element_size()`: in memoria il tensore resta sempre float32. Registra
`(numel, bits_per_element)` e calcola:

```
byte = numel · bits_per_element / 8
```

Per ogni iterazione il training logga:

- `comm_bytes_this_iteration` = somma su tutti i canali dei byte dell'iterazione `k`
- `cumulative_comm_bytes` = somma su tutti i canali dei byte totali

Per iterazione, con `R = --admm-rounds`, `N` agenti e `b` bit:

```
byte/iter = N · (1 + R) · d · b / 8
```

Il rapporto di compressione rispetto al baseline non quantizzato (64 bit) è
`64 / b`: 2× a 32 bit, 8× a 8 bit, 64× a 1 bit.

### 4.1 Cosa NON viene conteggiato (scelta deliberata)

**Metadati di quantizzazione per tensore.** In uno scenario reale, a ogni invio
di `g_i` o `y_i` l'agente dovrebbe trasmettere, oltre agli indici quantizzati,
anche i parametri per ricostruirli, perché sono ricalcolati sul tensore corrente
e cambiano a ogni invio:

| Schema | Metadati per invio | Overhead reale |
|--------|--------------------|----------------|
| `stochastic` | `t_min`, `scale` (equivalenti a min e max del tensore) | 2 × float32 = 64 bit |
| `float`, 4 bit (fp4 simulato) | `scale = max\|x\| / 6` | 1 × float32 = 32 bit |
| `float`, 8/16/32 bit | nessuno | 0 |

Il codice usa questi valori per la dequantizzazione simulata, ma il transform
dichiara al `ByteMeter` solo `bits_per_element = b`. Quindi i byte loggati sono
`d · b / 8` per invio e **l'overhead dei metadati è escluso**. L'omissione è
voluta: rispetto al payload (`d · b` bit) l'overhead vale `64 / (d · b)`, cioè
circa l'1–3% nel caso peggiore (`b = 1`, `d` di qualche migliaio) e meno dello
0.5% già a 8 bit. Non cambia quindi le conclusioni dei grafici byte-vs-bit.

Il byte count esatto per invio sarebbe:

```
stochastic:  d · b / 8 + 8   byte
fp4:         d · 4 / 8 + 4   byte
```

**Downlink.** Anche il downlink server → agenti (broadcast di `θ` e `y`) non
viene conteggiato.

Nota: i docstring del meter parlano di convenzione "per agente" (Fig. 3 del
paper), ma `train.py` logga la **somma su tutti gli agenti**. Per confrontarsi
con il paper, dividi per `N`.

---

## 5. Lo sweep: `fedrl-quantize-sweep`

[sweep.py](../src/fedrl_comm/quantization/src/quantization/sweep.py) lancia
`fednpg_admm.train.train()` **una volta per ogni bit-width**, in sequenza, con
gli stessi seed e iperparametri. Cambia solo il `channel_factory`.

```bash
# stocastico
fedrl-quantize-sweep --algo fednpg-admm --env swimmer --scheme stochastic --bits 1,2,4,8,32

# float
fedrl-quantize-sweep --algo fednpg-admm --env swimmer --scheme float --bits 4,8,16,32
```

Flag disponibili: `--algo` (per ora solo `fednpg-admm`), `--env`, `--scheme`,
`--bits` (obbligatorio), `--num-agents`, `--iterations`, `--timesteps`,
`--device`, `--seed`, `--results-path`. Gli altri iperparametri (per esempio
`--admm-rounds`, `--rho`) restano ai default di `fedrl-train-admm` e lo sweep
non li espone.

Validazione di `--bits`:

- `stochastic`: interi ≥ 1;
- `float`: solo {4, 8, 16, 32};
- i duplicati vengono rimossi e l'elenco ordinato.

Per ogni run:

- run id: `fednpg_admm_quant_<scheme>_<bits>bit_<env>_<N>agents_<timestamp>`
- a fine training, `metadata.json` viene riletto e arricchito con
  `quant_scheme` e `quant_bits`: `train()` non ha un gancio per aggiungere
  chiavi, quindi la scrittura avviene *dopo* che `train()` ha finito, senza race.

I run finiscono in `results/` e si vedono nella dashboard
(`fedrl-results-webapp`), come quelli normali.

---

## 6. Punti aperti e cose da sistemare

- `--algo fednpg-standard` non è ancora supportato: serve `baselines/fednpg_standard.py`
  con un `train(args, channel_factory=...)` equivalente. *(Al 2026-10-09
  `baselines/fednpg_standard.py` esiste, ma `train()` in `baselines/train.py`
  non accetta ancora un `channel_factory` e lo sweep ammette solo
  `--algo fednpg-admm`; analiticamente: a $b$ bit il metodo
  standard pareggia ADMM non quantizzato solo per $b \approx 128/d$, cioè
  ~0.03 bit su Swimmer.)*
- Con `--scheme stochastic --bits 32`, `2^32 − 1` livelli in aritmetica float32
  superano la precisione della mantissa. La parte frazionaria è di fatto 0, quindi
  l'arrotondamento diventa deterministico ed equivale circa a float32. Va bene
  come punto di riferimento "non quantizzato", ma non è stocastico in senso
  stretto.
- Lo sweep non espone `--admm-rounds`: con `R > 1` i byte di `y_i` crescono
  linearmente con `R`.

## 7. Note quando si usa troppa quantization:
[fednpg_admm_quant_stochastic_6bit_swimmer_8agents_20260924_142800] iter 851: step skipped ((sum g_i)^T y = -0.00211088 is non-positive -- the trust-region step size sqrt(2*N*delta/denom) is undefined. The aggregated ADMM direction isn't an ascent direction this round. Algorithm 1 as written has no fallback for this (no line search). Most common early in training with a badly-fit critic producing noisy advantages -- check critic_loss in the results before assuming the ADMM/CG code is wrong.)

## 8. Risultati (aggiornato 2026-10-09)

Run quantizzati (`scripts/run_grid.py`) confrontati con i run non quantizzati
della stessa configurazione e degli stessi seed (`run_sweep.sh`; i run Swimmer a N=1 delle due codebase coincidono cifra per cifra).
Metrica: media delle ultime 100 iterazioni di `mean_episode_return`
per seed, poi media ± deviazione standard campionaria sui 5 seed.

### 8.1 Griglia a 5 seed (stochastic, 6 e 8 bit)

| Ambiente | N | 64 bit | 8 bit | 6 bit |
|---|---|---|---|---|
| Swimmer | 2 | 126.0 ± 1.3 | 126.1 ± 0.8 | 128.9 ± 7.4 |
| Swimmer | 4 | 127.0 ± 0.8 | 127.1 ± 0.5 | 125.8 ± 3.4 |
| Swimmer | 8 | 128.4 ± 0.8 | 128.6 ± 0.4 | 127.8 ± 0.7 |
| Humanoid | 2 | 3391 ± 796 | 2956 ± 593 | 497 ± 34 |
| Humanoid | 4 | 5343 ± 121 | 5244 ± 237 | 676 ± 73 |
| Humanoid | 8 | 6329 ± 334 | 5997 ± 259 | 1514 ± 526 |
| Hopper | 2 | 1636 ± 282 | 2052 ± 389 | 1941 ± 461 |
| Hopper | 4 | 2536 ± 116 | 2197 ± 533 | 2096 ± 243 |
| Hopper | 8 | 2656 ± 286 | 2581 ± 466 | 2590 ± 338 |

- **8 bit**: indistinguibile da 64 bit su Swimmer; su Humanoid entro circa una
  deviazione standard. Più agenti continuano ad apprendere più in fretta a ogni
  checkpoint (Humanoid, iterazione 1000: 580/905/3428 per N=2/4/8, contro
  602/944/3863 non quantizzato). Riduzione ulteriore di 8x sull'uplink.
- **6 bit**: Humanoid crolla a ogni N (5 seed, quindi non è rumore tra seed).
  Swimmer tiene il reward finale ma con più varianza a N=2/4, e all'inizio
  N=4 apprende più lentamente di N=2 (iterazione 100: 75 contro 95).
- **5 bit** (Humanoid, 1 seed): 303 / 349 / 427 a N=2/4/8.

### 8.2 Scansione a 1 seed sui bit (N=8)

Swimmer: 94.4 a 5 bit, poi 127.7-128.7 da 6 a 32 bit. Humanoid: 1991 a 6 bit,
5543 a 7, 6073 a 24. La soglia sale con $d$ (Humanoid, $d \approx 7.3\times10^5$,
richiede un bit in più di Swimmer, $d = 4868$), coerente con un quantizzatore
non distorto la cui varianza entra in una somma $d$-dimensionale. Punto di
lavoro che regge ovunque: **8 bit**.

### 8.3 Hopper

I run Hopper usano la configurazione ridotta di Hopper (normalizzazioni,
line search e clamp disattivati; vedi `docs/hopper_agent_scaling_findings.md`),
sia quantizzati (`./final_sweep.sh hopper-quant`, `results/hopper-quant/`) sia
non quantizzati (`final-hopper-*`), con gli stessi seed.

- A N=8 entrambe le larghezze restano entro la variabilità tra seed del run non
  quantizzato (2581 e 2590 contro 2656); a 6 bit non c'è il crollo di Humanoid.
- A N=4 entrambe finiscono più in basso (2197 e 2096 contro 2536): circa una
  deviazione standard a 8 bit, di più a 6. A N=2 finiscono più in alto.
- Il reward finale cresce con N a entrambe le larghezze (8 bit: 2052/2197/2581;
  6 bit: 1941/2096/2590), quindi lo scaling con il numero di agenti regge.
  Ritorno smussato all'iterazione 100 a 8 bit: 857/914/1195 per N=2/4/8.

