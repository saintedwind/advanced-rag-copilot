# Optional local extractive reader attribution

The reader uses deepset's tinyroberta-squad2, converted and quantized for ONNX by the ONNX Community. Model repository: https://huggingface.co/onnx-community/tinyroberta-squad2-ONNX . Original model: https://huggingface.co/deepset/tinyroberta-squad2 . Model license: CC BY 4.0, as identified in the upstream model card: https://creativecommons.org/licenses/by/4.0/ . This project does not modify the model weights. Weights are downloaded into the ignored runtime cache and are not part of this Git repository.

Pinned conversion revision: 7c9f69b7e6228375169a4553bcfa6639152e3a69.
Quantized ONNX SHA-256: 4db70e7a019e32bf652fb9d30a5c24d18b6949fae2f1dd00d9fe6d43052b77cd.
Tokenizer SHA-256: 7b62d797c95d1563d3467f48daad0eca3e150f2d587e7e687b557c9ef1b25473.

SQuAD2 training does not establish performance on enterprise documents. The answer-versus-null logit margin used here is an uncalibrated heuristic, not a probability or entailment guarantee. The engine renders existing source passages, never text invented by this reader.
