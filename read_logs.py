import matplotlib.pyplot as plt
import re

file  = "/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter/DeepNovo.log"
file  = "/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter (copy)/DeepNovo.log"

files= [

        #"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter (copy)/DeepNovo.log",
	#"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter/DeepNovo.log",
	#"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilter/console_log.log",
	#"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA (wrong) - noNorm and zeroGrad/DeepNovo.log",
	"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA/DeepNovo.log",
	"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA_MZ1_INT0_Jit1/DeepNovo.log",
	#"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA_MZ0_INT1_Jit0/DeepNovo.log"
	"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA_MZ1/DeepNovo.log",
	"/home/kevin/Python/DeepNovoV2/DeepNovoV2-DeepNovoV2/trainExcludeYeastRealFilterGA_MZ1 (wrong peak match)/DeepNovo.log"
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
			
			## NB learning rate not in print to file only console
			#if "learning" in line:
			#if "best" in line:
				#train_loss.append(min(train_loss))
				#val_loss.append(max(train_loss))
				#val_loss.append(float(vp)-.1)
				#plt.vlines(len(val_loss),val_loss[-1],val_loss[0])	
#	plt.plot(train_loss)
	plt.plot(val_loss)
	#plt.hlines(min(val_loss),0,len(val_loss))
	#plt.vlines(val_loss.index(min(val_loss)),min(val_loss)-.1,min(val_loss))
plt.show()
