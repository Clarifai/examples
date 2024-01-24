import numpy as np
from inference import InferenceModel

inference_model = InferenceModel()

batch = (255*np.random.rand(1, 512, 512, 3)).astype(np.uint8)
outputs = inference_model.predict(batch)
print(outputs)

for pred in outputs:
    print(pred.predicted_bboxes)
    print(pred.predicted_labels)
    print(pred.predicted_scores)
