import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

LABELS = ("spam", "ham")
MODEL_FILE = Path("spam_model.json")

SAMPLE_DATA = [
    ("spam", "WINNER!! You have been selected to receive a $1000 gift card. Click http://claim-prize.example now"),
    ("spam", "Congratulations! You won a free iPhone. Reply YES to claim your prize today"),
    ("spam", "URGENT: your bank account is suspended. Verify your details at www.secure-bank.example immediately"),
    ("spam", "Free entry in our weekly draw! Text WIN to 80085 to get cash rewards"),
    ("spam", "Limited time offer: get 90% off all pills, order now, no prescription needed"),
    ("spam", "You have been pre-approved for a $5000 loan. Apply now, no credit check"),
    ("spam", "Claim your free vacation to the Bahamas, call 0800 555 0199 now"),
    ("spam", "Make money fast from home! Earn $500 per day, click here to start"),
    ("spam", "Your parcel could not be delivered. Pay a small fee at http://track-parcel.example to reschedule"),
    ("spam", "Hot singles in your area are waiting. Click now for free access"),
    ("spam", "Final notice: you owe unpaid taxes. Pay immediately to avoid arrest, call 0800 555 0100"),
    ("spam", "Exclusive deal just for you! Buy one get one free, offer expires tonight, click now"),
    ("spam", "Congratulations, you are our lucky winner of 1000 pounds cash. Call now to claim"),
    ("spam", "Get a free credit report and boost your score today. Limited offer, act now"),
    ("spam", "Your account has been compromised. Click the link to reset your password urgently"),
    ("spam", "FREE ringtones! Text TONE to 80100 and win a prize every week"),
    ("ham", "Are we still meeting for lunch tomorrow at 1?"),
    ("ham", "Can you send me the notes from today's class?"),
    ("ham", "I'll be home late tonight, don't wait for dinner"),
    ("ham", "Happy birthday! Hope you have an amazing day with family and friends"),
    ("ham", "Did you finish the assignment? The deadline is on Friday"),
    ("ham", "Mom called, she wants you to call her back when you get a chance"),
    ("ham", "Let's study together at the library this evening"),
    ("ham", "Thanks for helping me with the project, I really appreciate it"),
    ("ham", "What time does the movie start? I'll meet you at the entrance"),
    ("ham", "Please bring the charger when you come over, mine stopped working"),
    ("ham", "The meeting got moved to 3pm, see you in the conference room"),
    ("ham", "I'm on the bus, will reach in ten minutes"),
    ("ham", "Can you pick up some milk and bread on your way back?"),
    ("ham", "Great game last night! We should play again next weekend"),
    ("ham", "Sorry I missed your call, I was in the lab. Talk to you later?"),
    ("ham", "Don't forget to submit the form before the office closes today"),
]

def tokenize(text: str) -> list[str]:
    """Lowercase words, plus special tokens for links, numbers, money and '!'.

    Those special tokens matter: spam is full of links, phone numbers and prices,
    and the exact values don't matter, only that they appear.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\d+", " <num> ", text)
    text = re.sub(r"[$£€]", " <money> ", text)
    text = text.replace("!", " <excl> ")
    return re.findall(r"<url>|<num>|<money>|<excl>|[a-z']+", text)


class NaiveBayes:
    """Multinomial Naive Bayes with add-alpha (Laplace) smoothing.

    It scores each label as: log P(label) + sum of log P(word | label), assuming
    words are independent given the label (the "naive" part), then compares them.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.doc_counts: Counter = Counter()
        self.word_counts = {label: Counter() for label in LABELS}

    def train(self, messages: list[tuple[str, str]]) -> None:
        for label, text in messages:
            self.doc_counts[label] += 1
            self.word_counts[label].update(tokenize(text))

    @property
    def vocab(self) -> set[str]:
        return set(self.word_counts["spam"]) | set(self.word_counts["ham"])

    def _log_word_prob(self, word: str, label: str, vocab_size: int) -> float:
        total = sum(self.word_counts[label].values())
        return math.log((self.word_counts[label][word] + self.alpha) / (total + self.alpha * vocab_size))

    def spam_probability(self, text: str) -> float:
        vocab = self.vocab
        total_docs = sum(self.doc_counts.values())
        scores = {}
        for label in LABELS:
            prior = (self.doc_counts[label] + 1) / (total_docs + len(LABELS))
            score = math.log(prior)
            for word in tokenize(text):
                if word in vocab:  
                    score += self._log_word_prob(word, label, len(vocab))
            scores[label] = score
        best = max(scores.values())  
        weights = {label: math.exp(score - best) for label, score in scores.items()}
        return weights["spam"] / sum(weights.values())

    def explain(self, text: str, top: int = 3) -> tuple[list[str], list[str]]:
        """Return the words pushing hardest towards spam and towards ham."""
        vocab = self.vocab
        evidence = {
            word: self._log_word_prob(word, "spam", len(vocab)) - self._log_word_prob(word, "ham", len(vocab))
            for word in set(tokenize(text))
            if word in vocab
        }
        ranked = sorted(evidence, key=evidence.get)
        spammy = [w for w in reversed(ranked) if evidence[w] > 0][:top]
        hammy = [w for w in ranked if evidence[w] < 0][:top]
        return spammy, hammy

    def save(self, path: Path) -> None:
        data = {
            "alpha": self.alpha,
            "doc_counts": dict(self.doc_counts),
            "word_counts": {label: dict(counts) for label, counts in self.word_counts.items()},
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "NaiveBayes":
        data = json.loads(path.read_text())
        model = cls(data["alpha"])
        model.doc_counts = Counter(data["doc_counts"])
        model.word_counts = {label: Counter(data["word_counts"][label]) for label in LABELS}
        return model


def load_dataset(path: Path) -> list[tuple[str, str]]:
    """Read 'label<TAB>message' lines (a comma also works). Bad lines are skipped."""
    messages = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        label, _, text = line.partition("\t")
        if not text:
            label, _, text = line.partition(",")
        label = label.strip().lower()
        if label in LABELS and text.strip():
            messages.append((label, text.strip()))
    return messages


def evaluate(messages: list[tuple[str, str]], test_fraction: float = 0.2,
             seed: int = 42, threshold: float = 0.5) -> dict:
    """Train on most of the data, then test on messages the model has never seen."""
    if len(messages) < 10:
        raise ValueError("Need at least 10 messages to train and test on.")
    shuffled = messages[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    train, test = shuffled[:cut], shuffled[cut:]

    model = NaiveBayes()
    model.train(train)
    tp = fp = tn = fn = 0
    for label, text in test:
        is_spam = model.spam_probability(text) >= threshold
        if is_spam and label == "spam":
            tp += 1
        elif is_spam:
            fp += 1
        elif label == "spam":
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tested": len(test), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(test), "precision": precision, "recall": recall,
    }


    if path is None:
        return SAMPLE_DATA
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    messages = load_dataset(path)
    if not messages:
        raise SystemExit(f"No usable 'label<TAB>message' lines found in {path}")
    return messages


def get_model(path: Path) -> NaiveBayes:
    if path.exists():
        return NaiveBayes.load(path)
    print("(No saved model yet, so using the tiny built-in sample. Run 'train --data FILE' for better results.)")
    model = NaiveBayes()
    model.train(SAMPLE_DATA)
    return model


def show_verdict(model: NaiveBayes, text: str) -> None:
    probability = model.spam_probability(text)
    verdict = "SPAM" if probability >= 0.5 else "NOT SPAM"
    print(f"{verdict}  ({probability:.1%} chance of spam)")
    spammy, hammy = model.explain(text)
    if spammy:
        print("  spammy words:", ", ".join(spammy))
    if hammy:
        print("  normal words:", ", ".join(hammy))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    train_cmd = sub.add_parser("train", help="train on a dataset and save the model")
    eval_cmd = sub.add_parser("evaluate", help="measure accuracy on messages held out from training")
    predict_cmd = sub.add_parser("predict", help="classify a message (no text = interactive mode)")
    for command in (train_cmd, eval_cmd):
        command.add_argument("--data", type=Path, help="dataset file (default: built-in sample)")
    for command in (train_cmd, predict_cmd):
        command.add_argument("--model", type=Path, default=MODEL_FILE)
    eval_cmd.add_argument("--threshold", type=float, default=0.5, help="spam cutoff, default 0.5")
    predict_cmd.add_argument("text", nargs="*", help="the message to check")
    args = parser.parse_args()

    if args.cmd == "train":
        messages = get_data(args.data)
        model = NaiveBayes()
        model.train(messages)
        model.save(args.model)
        spam = model.doc_counts["spam"]
        print(f"Trained on {len(messages)} messages ({spam} spam, {len(messages) - spam} ham), "
              f"{len(model.vocab)} distinct words. Saved to {args.model}")
    elif args.cmd == "evaluate":
        result = evaluate(get_data(args.data), threshold=args.threshold)
        print(f"Tested on {result['tested']} unseen messages")
        print(f"  accuracy:  {result['accuracy']:.1%}")
        print(f"  precision: {result['precision']:.1%}  (of messages flagged as spam, how many really were)")
        print(f"  recall:    {result['recall']:.1%}  (of real spam, how much was caught)")
        print(f"  caught spam: {result['tp']}, missed spam: {result['fn']}, "
              f"false alarms: {result['fp']}, correct ham: {result['tn']}")
    else:
        model = get_model(args.model)
        if args.text:
            show_verdict(model, " ".join(args.text))
        else:
            print("Type a message to check (empty line to quit).")
            try:
                while (line := input("> ").strip()):
                    show_verdict(model, line)
            except (EOFError, KeyboardInterrupt):
                print()


if __name__ == "__main__":
    main()
