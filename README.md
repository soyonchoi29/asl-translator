# asl-translator

**Have you ever wished you could understand sign language?**

A real-time translation software that detects user's hand and predicts what ASL letter they are holding up.
Utilizes CNNs mostly trained on [this dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/metadata?resource=download).
Run main.py for a demo!

Implemented:
- CNN on raw ASL data from online source
- CNN on raw self-collected webcam data
- CNN on coordinates of hand given by processing raw online data through MediaPipe
- Model fine-tuning (online source model was fine-tuned with self-collected data)
- Computer vision using OpenCV and MediaPipe hand detection library
- Basic spell check using SpellChecker

Demo video:

https://github.com/soyonchoi29/asl-translator/assets/100095847/cdc352a7-478c-4ed6-9acb-a32d02c65532


