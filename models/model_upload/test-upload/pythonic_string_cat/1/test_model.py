# Runner test your implementation locally
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import PythonicStringCat

if __name__ == "__main__":
  model = PythonicStringCat()
  model.load_model()
  prompt = "Write 2000 word story?"

  res = model.predict(prompt)
  assert res == "Write 2000 word story?Hello World", 'predict failed'

  res = model.generate(prompt)
  for i, r in enumerate(res):
    assert r == f"Write 2000 word story?Generate Hello World {i}", 'generate failed'

  res = model.stream(iter([prompt] * 5))
  for i, r in enumerate(res):
    assert r == f"Write 2000 word story?Stream Hello World {i}", 'stream failed'
