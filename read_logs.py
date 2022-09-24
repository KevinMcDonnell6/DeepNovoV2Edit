import matplotlib.pyplot as plt
import re

file  = "/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter/DeepNovo.log"
file  = "/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter (copy)/DeepNovo.log"

files= [
	"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter/DeepNovo.log",
        "/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter (copy)/DeepNovo.log"
]

for file in files:
	train_loss = []
	val_loss = []

	with open(file,"r") as file_handle:

		line = file_handle.readline()
		
		while line:
			if "perplexity" in line:
			
				tp,vp = re.findall("perplexity: (\d+\.\d+)",line)
				train_loss.append(float(tp))
				val_loss.append(float(vp))
			line = file_handle.readline()
	plt.plot(train_loss)
	plt.plot(val_loss)
plt.show()
