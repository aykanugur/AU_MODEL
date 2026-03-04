# Epic 1 — Tokenizer

| Field | Value |
|-------|-------|
| **Branch** | `epic/01-tokenizer` |
| **Base branch** | `main` |
| **Merge target** | `main` |
| **PRD refs** | F-01, G1, M1 |
| **Depends on** | nothing — start here |
| **Status** | ⬜ Not started |
| **Output files** | `tokenizer/train_tokenizer.py`, `tokenizer/turkish_bpe.model`, `tokenizer/turkish_bpe.vocab` |

---

## Goal

Train a 64k-vocabulary Turkish-native SentencePiece BPE tokenizer (`character_coverage=0.9999`) on Turkish Wikipedia + 5 GB of OSCAR 23.01 Turkish, then validate that it meets fertility and coverage targets before any other epic can start.

---

## Tasks

- [ ] **Corpus downloader** — stream Turkish Wikipedia (`datasets.load_dataset('wikimedia/wikipedia', '20231101.tr', streaming=True)`) and first 5 GB of OSCAR 23.01 Turkish (`datasets.load_dataset('oscar-corpus/OSCAR-2301', 'tr', streaming=True)`) into `data/raw/tokenizer_corpus.txt`; apply NFC normalization; filter documents < 200 chars
- [ ] **SentencePiece training script** — call `spm.SentencePieceTrainer.train(input='data/raw/tokenizer_corpus.txt', model_prefix='tokenizer/turkish_bpe', vocab_size=64000, character_coverage=0.9999, model_type='bpe', pad_id=0, unk_id=1, bos_id=2, eos_id=3)`
- [ ] **Fertility validator** — tokenize 10,000 Turkish sentences from Wikipedia held-out set; compute `avg_tokens / avg_words`; assert result ≤ 1.4
- [ ] **Coverage checker** — for each of `['ç','ğ','ı','İ','ö','ş','ü','Ü','Ö','Ç','Ğ','Ş']` assert `sp.piece_to_id(char) != sp.unk_id()` (char has its own token, not byte-fallback)
- [ ] **`TurkishTokenizer` wrapper class** — `encode(text: str) → List[int]`, `decode(ids: List[int]) → str`, `vocab_size() → int`, `special_ids() → dict` (returns BOS/EOS/PAD/UNK/system/user/assistant IDs)

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Fertility on 10,000 sentences | ≤ 1.4 tokens/word |
| Turkish char coverage | All 12 chars have dedicated token ID (≠ UNK) |
| `sp.encode("merhaba")` | Returns exactly 1 token ID |
| `sp.decode(sp.encode(text)) == text` | True for any Turkish Unicode input |
| `vocab_size` | 64,000 exactly |

---

## Unlocks

- **Epic 2** (Model Architecture) — needs `vocab_size=64000` confirmed
- **Epic 3** (Data Pipeline) — needs trained tokenizer to tokenize corpus

---

_Last updated: 4 Mart 2026_
