import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from model import build_siamese
from preprocessing import preprocess_image
from create_pairs import create_pairs

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", default="data"); p.add_argument("--model", default="../backend/models/siamese_signature_model.keras"); p.add_argument("--threshold", type=float, required=True); args = p.parse_args()
    pairs = create_pairs(args.data); a=np.asarray([preprocess_image(x[0]) for x in pairs]); b=np.asarray([preprocess_image(x[1]) for x in pairs]); y=np.asarray([x[2] for x in pairs]); d=tf.keras.models.load_model(args.model, compile=False).predict([a,b], verbose=0).ravel(); pred=(d<=args.threshold).astype(int)
    results=Path("results"); results.mkdir(exist_ok=True)
    report=classification_report(y,pred,output_dict=True,zero_division=0); report["confusion_matrix"]=confusion_matrix(y,pred).tolist(); Path(results/"metrics.json").write_text(json.dumps(report,indent=2))
    plt.figure(); plt.hist(d[y==1], bins=30, alpha=.65, label="Genuine"); plt.hist(d[y==0], bins=30, alpha=.65, label="Forged"); plt.axvline(args.threshold, color="black", linestyle="--", label="Threshold"); plt.legend(); plt.xlabel("Euclidean distance"); plt.savefig(results/"distance_distribution.png"); plt.close()
    thresholds=np.linspace(float(d.min()), float(d.max()), 100); accuracies=[np.mean((d<=t)==y) for t in thresholds]; plt.figure(); plt.plot(thresholds,accuracies); plt.axvline(args.threshold, color="black", linestyle="--"); plt.xlabel("Threshold"); plt.ylabel("Accuracy"); plt.savefig(results/"threshold_analysis.png"); plt.close()
    false_positive, true_positive, _ = roc_curve(y, -d); plt.figure(); plt.plot(false_positive,true_positive,label=f"AUC {auc(false_positive,true_positive):.3f}"); plt.plot([0,1],[0,1],"--"); plt.xlabel("False acceptance rate"); plt.ylabel("True acceptance rate"); plt.legend(); plt.savefig(results/"roc_curve.png"); plt.close()
    print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
