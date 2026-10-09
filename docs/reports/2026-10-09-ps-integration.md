# PhotoCraft: integrazione del piano Photoshop

Base: `be4168c`. Worktree integrato: `E:\pc-ps-integrated`, branch `perf/ps-implementation`. Il checkout `E:\photocraft-main` resta separato. Tre lane hanno lavorato su copie isolate su E:, con compilazioni serializzate nel target condiviso `E:\photocraft-target-main` e temporanei su E:.

Il documento `implementazione ps.md` e la richiesta esplicita dell'utente escludono test nuovi, esecuzione dei test, benchmark, misure e confronti visivi. Le verifiche di compilazione sono riportate separatamente: non dimostrano correttezza dei pixel, fluidita, latenza, tempi di apertura o comportamento della UI.

| Lavoro | Codice integrato | Limiti |
| --- | --- | --- |
| W1 | Snapshot/commit engine, worker dei comandi, progresso, Esc, busy per documento, import/export asincroni nativi, protezione delle revisioni e della tab attiva. | Annullamento cooperativo: alcune operazioni lo controllano solo ai confini di fase. Wasm e automazione mantengono il percorso sincrono. |
| W2 | Tile solidi condivisi, COW, fill/clear rettangolari, no-op su tinta identica, deduplica pack per Arc; Canvas Size riempie solo le nuove bande e trasla per tile. | Il caricamento GPU condiviso dei solidi e opzionale e non implementato. |
| W3 | Due worker dedicati, richieste visibili limitate, risultati limitati, chiavi documento/livello, box average e mip CPU. | Wasm elabora al massimo due miniature per frame. |
| W4 | Mip native CPU on demand, invalidazione degli antenati, priorita scratch, persistenza facoltativa LZ4/CRC, consumatori canvas/navigatore/miniature. | Le mip mancanti del canvas possono essere calcolate nel frame. I vecchi pack senza bounds persistiti possono richiedere una scansione iniziale. |
| W5 | Output GPU limitato alla vista, sorgenti mip allo zoom corretto, ripiego CPU della vista, prefix cache CPU, mappe effetti/maschere con halo, cache separate per viewport e limitate per documento. | La preparazione delle mip e alcune fasi del compositor restano sincrone. Nessun criterio di selezione CPU/GPU basato su misure e stato introdotto. |
| W6 | Worker write-behind sopra l'80% del budget, invalidazione degli slot alle scritture, precedenza ai reload, eviction esistente come ripiego. | Nessuna misura del traffico scratch o della latenza. |
| W7 | Shape e campi esatti CPU su due worker limitati; upload GPU sul thread render; invalidazione display alla consegna; jump flooding euclideo GPU attivabile con `PHOTOCRAFT_GPU_DISTANCE=1`. | Il default resta esatto CPU. Il jump flooding e approssimato e senza verifica di precisione; stroke, export, gruppi annidati, wasm e lavoro non supportato conservano il percorso esatto. |
| W8 | Parser reader nativo senza Vec dell'intero file, decode parallelo deterministico, bande Raw/PackBits con buffer riusati, anteprima composita prima del decode dei livelli. | I payload codificati restano in memoria; ZIP usa piani completi. L'anteprima segue il parsing dei payload, senza seek anticipato alla composita. |
| W9 | Proxy persistente; per cinque filtri locali su raster: vista+alone, mip, worker latest-wins e pipeline GPU con ripiego CPU. | Gli altri dialoghi conservano il proxy generale. I comandi distruttivi usano il job CPU. La tolleranza di 1/255 delle pipeline GPU non e verificata. |
| W10 | Bande Rayon per dab grandi, accumulo/composito per tile, cache per timbri statici compatibili. | Identita bit per bit e tempi dei frame non misurati. |
| W11 | Rimandato: facoltativo nel documento. | Nessuna foreground cache; modalita separabile da sola non garantisce che uno stack sia precomponibile indipendentemente dal fondale. |

Le variabili di disattivazione e i dettagli dei file sono nei report delle singole lane: `2026-10-09-w1-jobs.md`, `W2-W4-W6-lane-b.md`, `W3-background-thumbnails.md`, `W5-W9-viewport-2026-10-09.md`, `W5-cpu-prefix-2026-10-09.md`, `W7-effects-2026-10-09.md`, `2026-10-09-w8-psd.md`, `W9-gpu-filters-2026-10-09.md`, `w9-w10-preview-brush.md`.

## Verifiche dell'integrazione

Sul tree finale sono passati:

- `cargo build --workspace --exclude photocraft-web`: desktop, CLI e default-members compilati.
- `cargo xtask layers`: 26 crate, nessuna violazione.
- `cargo xtask wasm`: tutti i 20 crate previsti compilano per `wasm32-unknown-unknown`.

Il gate wasm termina con warning, tra cui import e varianti del ciclo I/O nativo non usati su wasm; non e una build priva di warning. Il percorso wasm dei filtri ora restituisce subito Unsupported prima di creare pipeline native e prosegue sul filtro CPU. Helper nativi di pennello e firma dei job non vengono compilati su wasm. I gate sono stati ripetuti dopo questo ultimo follow-up.

Naga 30 ha completato parsing e validazione statica di `filters.wgsl` e `distance.wgsl`. Non sono stati eseguiti shader sulla GPU. La prima compilazione statica ha rilevato un identificatore riservato in `filters.wgsl`, corretto prima della build finale.

Binario desktop compilato: `E:\photocraft-target-main\debug\photocraft.exe`. Non e stato avviato durante questa sessione.

`../craftrules/README.md` non era disponibile nei percorsi locali consultati. Nessun file di memoria e stato modificato.
