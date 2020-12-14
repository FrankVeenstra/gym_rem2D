import Main
import pickle
from Main import Individual
from Main import Encoding_Type
import OpenGL

def load_best():
	directory = ""
	run = ""
	id = 0
	filename = "results/cluster/work/jobs/418036/result_417852-107/s_elite20"
	filename = "results/2020-03-05/result_nr/s_elite20"	#filename = "results\cluster\work\jobs\417853\result_417852-0"
	elitenr = 0
	print("Loading best")
	#individual = pickle.load(open(filename,"rb"))
	runid = 0
	max = 0.0
	originalfit = 0.0
	interval = 1
	dir = 'D:/results/cppn_final/'
	dir = 'C:/result_temp_final/'
	#dir = 'D:/results/cppn_final/'
	for i in range(500):
		filename = dir + "result_" + str(runid)+"/s_elite"+str(id)
		try :
			individual = pickle.load(open(filename,"rb"))
			originalfit = individual.fitness
		except: 
			runid+=1
			id = 0
			continue
		#Main.M2D.MAX_PERTURBANCE_TERRAIN = 0.0
		fit = Main.evaluate(individual,HEADLESS = False,INTERVAL =5)
		if fit > max:
			max = fit
		print("run, id ",runid,id, " , fit : ", fit, " ,original: ",originalfit, ", max ", max)
		id+=500

if __name__ == "__main__":
	load_best()

	