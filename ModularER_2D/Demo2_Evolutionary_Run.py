import REM2D_main as r2d

if __name__=="__main__":
	# Read a config file and get a working directort
	# Note: the --file of the argument parser should be directed to the file specifying any configurations. 
	#       the working directory will be set to the path specified. By default, the working directoy is
	#       the location of REM2D.py
	config, dir = r2d.setup()
	experiment = r2d.run2D(config,dir)
	experiment.run(config)
