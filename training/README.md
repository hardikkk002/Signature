# Training

Expected dataset layout: `data/<writer>/{genuine,forged}/*.(png|jpg|jpeg)`. Keep writer identities disjoint between train, validation, and test runs; configure explicit writer lists in `create_pairs.py` or call it with filtered roots. Run `python train.py --data data --pairs-per-class 2000`. This writes the best Keras model and validation-calibrated threshold to `backend/models/`. Run evaluation with `python evaluate.py --data data/test --model ../backend/models/siamese_signature_model.keras --threshold <calibrated-threshold>`.
