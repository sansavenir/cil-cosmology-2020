import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.ylabel('average score')
plt.ylim(top=6)
plt.ylim(bottom=0)

models = ['naive', 'styleGAN', 'layered']
scores = [0.27599349734828577, 1.2304153329772458, 2.0083818784081973]

errors = [[0.2565912032318418, 1.3654140043492544],
          [0.9294262880726123, 1.1414445327904437],
          [1.6416210477554136, 2.1324531635916197]]
errors = np.asarray(errors).T

plt.bar(models, scores, yerr=errors, capsize=3)

for i, v in enumerate(scores):
    plt.text(i+0.1, v + 0.2, str(v)[:4], fontweight='bold')

plt.show()