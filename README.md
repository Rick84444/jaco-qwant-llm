# JACO Qwant LLM

Aktiveringsgleshet ovanpå kvantiserade vikter (Q8/Q4) för on-device-LLM på ARM. Kärnan hoppar över hela viktrader för inaktiva kanaler. Varje påstående mäts mot den snabbaste täta kvantiserade kärnan: fel, minnestrafik, tid och batterienergi.

## Start (Termux)

```
pkg install git
git clone https://github.com/<ditt-konto>/jaco-qwant-llm
cd jaco-qwant-llm
./start.sh
```

`start.sh` installerar clang, python och numpy om de saknas. Den bygger kärnan, kör självtestet och kör sedan benchmarken. Resultatet skrivs till `report-<tid>.json`.

- `./start.sh acts`: kör på riktiga aktiveringar som fångats på PC.
- `./start.sh acts energy`: samma, plus energi per token. Kör ur laddaren.


## Filer

| Fil | Roll |
|---|---|
| `acfq_kernel.c` | Kvantiserad kärna: fp32, Q8, Q4 samt Q8h/Q4h (fp16-aritmetik). Hela indatarader kan hoppas över. NEON med 4-radsfusion. |
| `acfq_rig.py` | Huvudriggen. Fel mot float64, MB/token, interfolierad tidtagning och energi via batterisensorn. |
| `test_acfq.c` | Självtest av kärnan mot float64-referens, alla format, tät och gles. |
| `capture_activations.py` | Körs på PC. `capture` dumpar riktiga aktiveringar. `eval` mäter perplexitet med gleshet, och med `--wquant q4` även Q4-vikter i exakt kärnans layout. |
| `acf_kernel.c`, `acf_rig.py` | Första fp32-riggen, inklusive originalets blockenergimask. |
| `build.sh` | Bygger båda biblioteken och kör självtestet. |
| `start.sh` | Installerar beroenden, bygger, testar och kör. |
| `.github/workflows/test.yml` | CI: skalär, NEON och NEON+fp16 (via qemu) samt rök-körning av riggarna. |

## Layout

Rad i (indatakanal i) lagras som `[N/32 fp16-skalor][kvantiserad nyttolast]`. Grupper om 32 löper längs utdata.

- Q4 tar 4,5 bitar per vikt, samma som Q4_0.
- En inaktiv kanal hoppar över hela sin rad, både skalor och nyttolast.
- q8h/q4h använder samma bytes men dekvantiserar via int16 → fp16 och ackumulerar 4 rader i fp16 innan värdet breddas till fp32. Det kräver ARMv8.2 FP16, vilket Pixel 10 har.

## Manuell körning i Termux

```
./build.sh                                        # ska sluta med ALL PASS
python3 acfq_rig.py --synthetic                   # snabb översikt
python3 acfq_rig.py --data acts                   # riktiga aktiveringar från PC:n
python3 acfq_rig.py --data acts --energy          # ur laddaren, skärmen på konstant ljusstyrka
```

## Körning på PC

```
python capture_activations.py eval --model Qwen/Qwen2.5-0.5B --wquant q4 --sparsities 0.3 0.4 0.5
python capture_activations.py capture --model Qwen/Qwen2.5-0.5B --out acts
```

## Läsa resultatet

- **opponent**: den snabbaste täta kvantiserade varianten på just din telefon.
- **win**: snabbare än opponent, och felet ökar högst `--err-budget` (standard 0,03).
- **MB/tok**: deterministiskt mått på minnestrafik. Det är den största posten i energiåtgången.
- **mJ/tok**: nettoenergi per token över tomgång. Kräver att telefonen går på batteri. Kör gärna 2–3 gånger, sensorn brusar.
- **eval**-raden `dense q4` är referensen. Gleshetsraderna under den ska jämföras med den, inte med fp.

## Licens

MIT
