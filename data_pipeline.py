from __future__ import annotations
import os
import re
import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns
import textstat
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, cmudict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wordfreq import zipf_frequency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "cmudict", "universal_tagset"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
NEGATIONS  = {"not","no","never","neither","nor","nobody","nothing",
              "nowhere","hardly","scarcely","barely","doesn't","isn't",
              "wasn't","shouldn't","wouldn't","couldn't","won't","can't","don't"}

try:
    CMU = cmudict.dict()
except Exception:
    CMU = {}

DATA_DIR   = Path("data")
SCALER_PATH = Path("scaler.pkl")
FEATURES_PATH = Path("data/features.csv")


def count_syllables(word: str) -> int:
    w = word.lower()
    if w in CMU:
        return sum(1 for ph in CMU[w][0] if ph[-1].isdigit())
    vowel_groups = re.findall(r'[aeiouy]+', w)
    n = len(vowel_groups)
    if w.endswith('e') and len(w) > 2:
        n = max(1, n - 1)
    return max(1, n)


def extract_features(text: str) -> dict[str, float]:
   
    if not text or not text.strip():
        return {f: 0.0 for f in FEATURE_NAMES}

    sentences = sent_tokenize(text)
    words_raw  = word_tokenize(text)
    words_alpha = [w for w in words_raw if w.isalpha()]
    words_lower = [w.lower() for w in words_alpha]

    n_sents = max(1, len(sentences))
    n_words = max(1, len(words_alpha))

    #readibility
    flesch     = textstat.flesch_reading_ease(text)
    fk_grade   = textstat.flesch_kincaid_grade(text)
    avg_wlen   = np.mean([len(w) for w in words_alpha]) if words_alpha else 0.0
    avg_slen   = n_words / n_sents
    syl_counts = [count_syllables(w) for w in words_lower]
    syl_density = np.sum(syl_counts) / n_words   # avg syllables per word

    #vocab
    zipf_scores  = [zipf_frequency(w, "en") for w in words_lower]
    rare_word_pct = sum(1 for z in zipf_scores if z < 4.0) / n_words
    type_token    = len(set(words_lower)) / n_words
    long_word_pct = sum(1 for w in words_alpha if len(w) > 7) / n_words

    #syntax
    try:
        pos_tags = nltk.pos_tag(words_alpha, tagset="universal")
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        noun_ratio = pos_counts.get("NOUN", 0) / n_words
        verb_ratio = pos_counts.get("VERB", 0) / n_words
        adj_ratio  = pos_counts.get("ADJ",  0) / n_words
    except Exception:
        noun_ratio = verb_ratio = adj_ratio = 0.0

    # Passive voice:
    passive_count = len(re.findall(
        r'\b(was|were|been|is|are|be)\s+\w+ed\b', text.lower()
    ))

    #cog load
    negation_count   = sum(1 for w in words_lower if w in NEGATIONS) / n_words
    punct_density    = sum(1 for c in text if c in ",.;:!?()[]{}") / max(1, len(text))
    # Clause density
    clause_markers   = {"although","because","since","while","when","if","unless",
                        "until","after","before","though","whether","which","that","who"}
    clause_density   = sum(1 for w in words_lower if w in clause_markers) / n_words

    return {
        "flesch_score":       float(np.clip(flesch, 0, 100)),
        "fk_grade":           float(np.clip(fk_grade, 0, 18)),
        "avg_word_len":       float(avg_wlen),
        "avg_sent_len":       float(avg_slen),
        "syllable_density":   float(syl_density),
        "rare_word_pct":      float(rare_word_pct),
        "type_token_ratio":   float(type_token),
        "long_word_pct":      float(long_word_pct),
        "noun_ratio":         float(noun_ratio),
        "verb_ratio":         float(verb_ratio),
        "adj_ratio":          float(adj_ratio),
        "passive_voice_count":float(passive_count),
        "negation_count":     float(negation_count),
        "punct_density":      float(punct_density),
        "clause_density":     float(clause_density),
    }


FEATURE_NAMES = list(extract_features("sample").keys())


def assign_difficulty(flesch: float, fk_grade: float) -> int:
    """Map readability scores to 4-class difficulty label."""
    if flesch >= 70 or fk_grade <= 6:
        return 0   # Easy
    elif flesch >= 50 or fk_grade <= 9:
        return 1   # Moderate
    elif flesch >= 30 or fk_grade <= 12:
        return 2   # Hard
    else:
        return 3   # Very Hard


def assign_emotional_load(row: pd.Series) -> float:

    score = (
        0.35 * (row["difficulty"] / 3.0) +
        0.25 * min(1.0, row["rare_word_pct"] * 2) +
        0.15 * min(1.0, row["negation_count"] * 5) +
        0.15 * min(1.0, row["clause_density"] * 5) +
        0.10 * min(1.0, row["passive_voice_count"] / 3)
    )
    return float(np.clip(score, 0.0, 1.0))

def load_wikilarge(path: str = "data/wikilarge.txt") -> list[str]:

    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 30]
    logger.info("Loaded %d sentences from WikiLarge", len(lines))
    return lines[:50000]


def generate_synthetic_dataset(n: int = 2000) -> list[str]:
    import random
    random.seed(42)

    easy = [
        "The cat sat on the mat and looked at the bird.",
        "Dogs like to play and run in the park.",
        "The sun is hot and bright in the sky.",
        "She went to the store to buy some milk.",
        "He eats lunch at school every day.",
        "The boy kicked the ball into the goal.",
        "Birds can fly high in the blue sky.",
        "We read books and draw pictures at school.",
        "The fish swim in the clean blue water.",
        "My friend likes to eat apples and oranges.",
        "The baby slept in the warm soft bed.",
        "Rain falls from the clouds onto the ground.",
        "The teacher reads a story to the class.",
        "We can see stars in the sky at night.",
        "The puppy ran fast and wagged its tail.",
    ]

    moderate = [
        "Children with dyslexia often find reading more difficult than their peers.",
        "The teacher prepared a lesson plan to help students understand fractions.",
        "Scientists study the behaviour of animals in their natural habitats.",
        "The library contains thousands of books on many different subjects.",
        "Learning a new language requires practice and dedication every day.",
        "The doctor explained that regular exercise improves mental health.",
        "Students should review their notes before taking an important exam.",
        "The community came together to clean the local park on Saturday.",
        "Technology has changed the way people communicate with each other.",
        "The experiment showed that plants grow faster in sunlight than shade.",
        "Reading aloud helps children develop their vocabulary and fluency.",
        "The museum displayed artefacts from ancient civilisations around the world.",
        "Healthy eating habits can reduce the risk of many chronic diseases.",
        "The school introduced a new programme to support children with learning difficulties.",
        "Parents play an important role in encouraging children to read at home.",
    ]

    hard = [
        "The implementation of evidence-based interventions requires systematic evaluation of cognitive outcomes.",
        "Phonological processing deficits constitute a primary underlying mechanism of dyslexic reading difficulties.",
        "Multisensory instructional approaches have demonstrated considerable efficacy in remediating reading disorders.",
        "The neurobiological correlates of reading acquisition involve distributed cortical networks.",
        "Socioeconomic factors significantly moderate the relationship between early literacy exposure and academic achievement.",
        "Adaptive learning systems must incorporate real-time assessment mechanisms to optimise instructional scaffolding.",
        "The lexical decision paradigm has been extensively utilised to investigate word recognition processes.",
        "Longitudinal cohort studies suggest that early identification of phonemic awareness deficits predicts later reading outcomes.",
        "Computational models of reading acquisition have advanced our understanding of orthographic mapping processes.",
        "The heterogeneous nature of dyslexia necessitates individualised intervention protocols.",
        "Metacognitive strategy instruction facilitates the development of self-regulated reading comprehension.",
        "Neuroimaging investigations have revealed atypical activation patterns in the temporo-parietal cortex of dyslexic readers.",
        "The ecological validity of laboratory-based reading measures remains a subject of ongoing methodological debate.",
        "Differential diagnosis requires comprehensive assessment across phonological, orthographic, and morphological domains.",
        "The interplay between genetic predisposition and environmental factors in dyslexia aetiology remains incompletely characterised.",
    ]

    very_hard = [
        "The ontogenetic trajectory of phonological awareness development exhibits considerable heterogeneity, necessitating nuanced taxonomic frameworks that transcend simplistic categorical distinctions.",
        "Epistemological considerations pertaining to the construct validity of psychometric instruments employed in the differential diagnosis of specific learning disabilities warrant systematic scrutiny.",
        "The neurophysiological underpinnings of rapid automatised naming deficits implicate both cerebellar and magnocellular pathway dysfunction in the aetiopathogenesis of developmental dyslexia.",
        "Corticostriatal circuitry mediating procedural learning has been hypothesised to contribute substantively to the automatisation deficits characteristically observed in individuals with dyslexic symptomatology.",
        "Multivariate statistical paradigms incorporating latent variable modelling frameworks provide methodologically superior approaches to elucidating the dimensional structure of reading disability phenotypes.",
        "The epistemological foundations of evidence-based practice in educational psychology necessitate rigorous interrogation of the methodological assumptions underlying randomised controlled trial designs.",
        "Morphosyntactic processing anomalies in developmental dyslexia may reflect broader disruptions to neural oscillatory mechanisms subserving the temporal sampling of hierarchically structured linguistic input.",
        "The psycholinguistic grain size theory posits that orthographic granularity mediates cross-linguistic variability in the manifestation of phonological decoding deficits.",
        "Transcranial magnetic stimulation paradigms have elucidated the causal contributions of left hemisphere perisylvian regions to grapheme-phoneme conversion processes.",
        "Computationally instantiated connectionist architectures have provided mechanistic accounts of the emergent properties characterising skilled word recognition in both typical and atypical readers.",
    ]

    corpus = []
    for _ in range(n // 4):
        corpus.append(random.choice(easy))
        corpus.append(random.choice(moderate))
        corpus.append(random.choice(hard))
        corpus.append(random.choice(very_hard))

    random.shuffle(corpus)
    return corpus


#pipeline
def build_dataset(
    wikilarge_path: Optional[str] = None,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:

    DATA_DIR.mkdir(exist_ok=True)

    # ── 1. Load texts ──
    texts = []
    if wikilarge_path:
        texts = load_wikilarge(wikilarge_path)
    if not texts:
        logger.info("WikiLarge not found — generating synthetic dataset …")
        texts = generate_synthetic_dataset(n=2000)

    logger.info("Total texts: %d", len(texts))

    # ── 2. Feature extraction ──
    logger.info("Extracting features …")
    records = []
    for i, text in enumerate(texts):
        if i % 200 == 0:
            logger.info("  %d / %d", i, len(texts))
        feats = extract_features(text)
        feats["text"] = text
        records.append(feats)

    df = pd.DataFrame(records)

    # ── 3. Labels ──
    df["difficulty"]     = df.apply(lambda r: assign_difficulty(r["flesch_score"], r["fk_grade"]), axis=1)
    df["emotional_load"] = df.apply(assign_emotional_load, axis=1)

    logger.info("Class distribution:\n%s", df["difficulty"].value_counts().sort_index().to_string())
    logger.info("Emotional load stats:\n%s", df["emotional_load"].describe().to_string())

    if save:
        df.to_csv(FEATURES_PATH, index=False)
        logger.info("Saved features to %s", FEATURES_PATH)

    # ── 4. Split ──
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_diff = df["difficulty"].values.astype(np.int64)
    y_emo  = df["emotional_load"].values.astype(np.float32)

    X_tv, X_test, yd_tv, yd_test, ye_tv, ye_test = train_test_split(
        X, y_diff, y_emo, test_size=0.15, random_state=42, stratify=y_diff
    )
    X_train, X_val, yd_train, yd_val, ye_train, ye_val = train_test_split(
        X_tv, yd_tv, ye_tv, test_size=0.176, random_state=42, stratify=yd_tv
    )

    logger.info("Split: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    # ── 5. Scale ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    if save:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Saved scaler to %s", SCALER_PATH)

    return X_train, X_val, X_test, yd_train, yd_val, yd_test, ye_train, ye_val, ye_test, scaler


def load_scaler() -> StandardScaler:
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_dataset(wikilarge_path="data/wikilarge.txt.src", save=True)
