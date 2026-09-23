import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from model import build_siamese
from losses import contrastive_loss
from preprocessing import preprocess_image
from create_pairs import create_pairs
from threshold import calibrate, save_config


def arrays(pairs, augment=False):
    a, b, y = [], [], []
    for left, right, label in pairs:
        a.append(preprocess_image(left, augment)); b.append(preprocess_image(right, augment)); y.append(label)
    return np.asarray(a), np.asarray(b), np.asarray(y, dtype="float32")


class PairSequence(tf.keras.utils.Sequence):
    """Load only one batch of pairs at a time to keep training RAM-safe."""

    def __init__(self, pairs, batch_size, augment=False):
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.augment = augment
        self.order = np.arange(len(self.pairs))

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        xa, xb, y = [], [], []
        for item in indices:
            left, right, label = self.pairs[item]
            xa.append(preprocess_image(left, self.augment))
            xb.append(preprocess_image(right, self.augment))
            y.append(label)
        return (np.asarray(xa), np.asarray(xb)), np.asarray(y, dtype="float32")

    def on_epoch_end(self):
        np.random.shuffle(self.order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--model-dir", default="../backend/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pairs-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-writers", default=None, help="Comma-separated writer folder names")
    parser.add_argument("--val-writers", default=None, help="Comma-separated writer folder names")
    args = parser.parse_args()
    train_writers = args.train_writers.split(",") if args.train_writers else None
    val_writers = args.val_writers.split(",") if args.val_writers else train_writers
    train_pairs = create_pairs(args.data, writers=train_writers, pairs_per_class=args.pairs_per_class, seed=42)
    val_pairs = create_pairs(args.data, writers=val_writers, pairs_per_class=max(1, args.pairs_per_class // 5), seed=43)
    if not train_pairs or not val_pairs: raise SystemExit("No pairs found. Expected data/<writer>/{genuine,forged}/ images.")
    train_sequence = PairSequence(train_pairs, args.batch_size, augment=True)
    va, vb, vy = arrays(val_pairs)
    model = build_siamese(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=contrastive_loss)
    out = Path(args.model_dir); out.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(out / "siamese_signature_model.keras", monitor="val_loss", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.5)]
    history = model.fit(train_sequence, validation_data=([va, vb], vy), epochs=args.epochs, callbacks=callbacks)
    model.save(out / "siamese_signature_model_final.keras")
    distances = model.predict([va, vb], batch_size=args.batch_size, verbose=0).ravel(); selected = calibrate(distances, vy)
    save_config(out / "model_config.json", selected["threshold"])
    Path("results").mkdir(exist_ok=True); Path("results/training_history.json").write_text(json.dumps(history.history, indent=2)); Path("results/validation_metrics.json").write_text(json.dumps(selected, indent=2))
    print("Calibrated validation threshold:", selected)


if __name__ == "__main__": main()
