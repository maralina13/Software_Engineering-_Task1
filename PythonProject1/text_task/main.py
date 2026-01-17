from transformers import pipeline

def main():
    clf = pipeline(
        task="text-classification",
        model="cointegrated/rubert-tiny-sentiment-balanced",
        tokenizer="cointegrated/rubert-tiny-sentiment-balanced",
        device=-1,
        truncation=True
    )

    samples = [
        "Мне очень понравился курс, всё понятно!",
        "Это ужасно, ничего не работает и объяснений нет.",
        "Ну, нормально. Могло быть лучше."
    ]

    for s in samples:
        res = clf(s)[0]
        print(f"TEXT: {s}")
        print(f" -> label={res['label']}, score={res['score']:.3f}")
        print()

if __name__ == "__main__":
    main()

