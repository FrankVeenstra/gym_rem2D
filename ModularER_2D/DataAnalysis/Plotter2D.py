import NeuralNetwork.NEAT_NN as NE
import Cellular_Encoding as CE
import matplotlib.pyplot as plt
import pygame

im = {}
from pygame import gfxdraw

def plotMAT(ax, array):
	#plt.hist2d(array)
	global im
	if (ax not in im):
		im[ax] = ax.imshow(array,cmap = "viridis")
	else:
		im[ax].set_data(array)

def plot(gameWindow, array):
	#plt.hist2d(array)
	global im
	if (ax not in im):
		im[ax] = ax.imshow(array,cmap = "viridis")
	else:
		im[ax].set_data(array)

width = 400
height = 400
scale = 20
def createNew(enc,ax,ax2,plot2,timestep):
	width = 8
	height = 8
	ax2.clear();
	#ax.clear()
	#if (plot2):
	#	ax2.clear()
	#enc2 = NE.CPPN(2,1)
	array = []
	for y in range(height):
		row = []
		for x in range(width):
			input = [x/(width/2)-1.0,y/(height/2)-1.0,timestep]
			output = enc.update(input)
			s_output = 0.0;
			for out in output:
				s_output+=out

			#print(output,input)
			row.append(s_output)
		array.append(row)
	plotMAT(ax,array)
	if (plot2):
		enc.display(ax2)
	return

def updateGame(enc,screen,timestep,screenNr,cmap):
	#pxarray = pygame.PixelArray(screen)
	#pxarray[width:height] = (255,0,255)
	for y in range(0,height-1,scale):
		row = []
		for x in range(0,width-1,scale):
			input = [x/(width/2)-1.0,y/(height/2)-1.0,(timestep*2)-1]
			#print(input)
			output = enc.update(input)
			s_output = 0.0;
			for i,out in enumerate(output):
				s_output+=out
				if (output[i] > 1):
					output[i] = 1
				elif(output[i] < -1):
					output[i] = -1
			act = int(123*(1+output[0]))
			c = cmap(act/255)
			color = (c[0]*255,c[1]*255,c[2]*255)
			if (len(output)>2):
				color = ((1+output[0])*122,(1+output[1])*122,(1+output[2])*122)
			if (screenNr == 1):
				pygame.draw.rect(screen,color,(x+width,y,scale,scale))
			else:
				pygame.draw.rect(screen,color,(x,y,scale,scale))
			#screen.set_at((x, y), color)
	pygame.display.update()
	pygame.event.get()

if __name__ == "__main__":
	#fig = plt.figure()
	#ax = fig.add_subplot(1,2,1)
	#ax2 = fig.add_subplot(1,2,2)
	#fig3 = plt.figure()
	#ax3 = fig3.add_subplot(1,2,1)
	#ax4 = fig3.add_subplot(1,2,2)
	xsize = 1
	ysize = 2
	ax = None
	ax2 = None
	ax3 = None
	ax4 = None
	axes = None
	axes2 = None
	usematplotlib = False
	screen1 = None
	screen2 = None
	cmap = plt.get_cmap('viridis')
	if usematplotlib:
		if (xsize == 1):
			fig = plt.figure()
			ax = fig.add_subplot(2,2,1)
			#fig2 = plt.figure()
			ax2 = fig.add_subplot(2,2,2)
			#fig3 = plt.figure()
			ax3 = fig.add_subplot(2,2,3)
			#fig4 = plt.figure()
			ax4 = fig.add_subplot(2,2,4)
		else:
			fig, axes = plt.subplots(xsize, 2,figsize=(10, 5))
			fig2, axes2 = plt.subplots(ysize, 2,figsize=(10, 5))
	else:
		screen1 = pygame.display.set_mode((int(width*2),height))
#		screen2 = pygame.display.set_mode((width,height))
	test = "CPPN"
	if not usematplotlib:
		test = "game"
		test = "rgb"
	shouldContinue = True
	mutation_steps = 2
	time_steps = 40
	if test == "game":
		while shouldContinue == True:
			#for x in range(xsize):
			enc = NE.CPPN(3,1)
			enc2 = CE.CE()
			enc2.create() # move
			for y in range(ysize-1):
				for m in range(mutation_steps):
					enc.mutate()
					enc2.mutate(0.1,0.1)
					enc2.reset()
					for t in range(time_steps):						
						updateGame(enc,screen1,t/time_steps,0,cmap)
						updateGame(enc2,screen1,t/time_steps,1,cmap)
	if test == "rgb":
		while shouldContinue == True:
			#for x in range(xsize):
			enc = NE.CPPN(3,3)
			enc2 = CE.CE()

			enc2.create() # move
			for y in range(ysize-1):
				for m in range(mutation_steps):
					enc.mutate()
					enc2.mutate(0.1,0.1,0.1)
					enc2.reset()
					for t in range(time_steps):						
						updateGame(enc,screen1,t/time_steps,0,cmap)
						#updateGame(enc2,screen1,t/time_steps,1,cmap)
	
	if test == "CPPN":
		# CPPN
		while shouldContinue == True:
			
			for x in range(xsize):
				enc = NE.CPPN(3,1)
				enc2 = CE.CE()
				enc2.create() # move
				for y in range(ysize-1):
					for m in range(mutation_steps):
						enc.mutate()
						enc2.mutate(0.1,0.1)
						enc2.reset()
						
						for t in range(time_steps):

							plotNet = True
							#if n == 0:
							#plotNet = True
						
							if xsize == 1:
								createNew(enc,ax,ax3,plotNet,t/time_steps)
								createNew(enc2,ax2,ax4,plotNet,t/time_steps)
							else:
								createNew(enc,axes[x,y],axes[x,len(axes[0])-1],plotNet,t)
								createNew(enc2,axes2[x,y],axes2[x,len(axes[0])-1],plotNet,t)
							fig.canvas.draw()
							fig.canvas.flush_events()
							#fig2.canvas.draw()
							#fig2.canvas.flush_events()
	
							plt.pause(0.01)
							plt.ion()
	else:
		# Cellular Encoding
		for i in range(100):
			enc = CE.CE()
			enc.create() # move
			createNew(enc,ax,ax2)
	plt.show()
