#######################################################
##### PARTICLE SWARM OPTIMIZATION (PSO) ALGORITHM #####
############################### -Siddharth Yadav- #####

####======== plot.py ========####
# Contains the source code for ploting the results of PSO algorithm
# The 'plot_pso' function can be imported to 'main.py'

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from tkinter.filedialog import askopenfilename
 
# Defining the ploting function
def plot_pso(dimension_limit=(15,15), speed=3, skip=None,show_all_positions=False, show_origin=False, function_name=None):
    '''Function to plot the results of the particle swarm optmization algorithm
    -->dimension_limit = a tuple or list of two numbers which specify the x and y coordinate limits for the graph
    -->speed = a number between 0 (slowest animation) and 5 (fast animation)
    -->skip = the number of frames to skip between two displayed frames
    -->show_all_positions = if True; will display the position of all particles in every iteration without clearing them
    -->show_origin = if True; will display x and y axis originating from the origin
    -->function_name = provide the name of the function being optimized'''

    # Prompt the user to specify the local files generated during the PSO run to obtain particle positions
    # Collection of the positions of all particles in the swarm per iteration
    collection_filename = askopenfilename(title="Select the swarm collection file", filetype=[("Joblib files", "*.joblib")])
    # The global best solution discovered during the PSO run
    bestposition_filename = askopenfilename(title="Select the best position file", filetype=[("Joblib files", "*.joblib")])

    # Loading the swarm collection and best position coordinates from the specified files
    swarm_collection = load(collection_filename) 
    best_x, best_y = load(bestposition_filename)
    
    # Calculating and displaying total iterations and total particles
    total_iterations = len(swarm_collection)-1
    total_particles = len(swarm_collection[0])
    print(f"\n------> Swarm collection file loaded:\n\tNumber of iterations = {total_iterations}\n\tNumber of particles = {total_particles}")

    # Prompting the use to provide a skip step for the animation
    if len(swarm_collection) >= 500:
        skip= int(input(f"\nThe animations can be slow for algorithm runs exceeding 500 iterations (current number of iterations = {len(swarm_collection)}).\nA skip factor can be provided to short the animation length.\n\t0---No skip\n\t1---Skip every alternate frame\n\t2---Skip 2/3rd of all frames\n\t3---Skip 3/4th of all frame (..and so on, Maximum skip=10)\n------> "))
        
    if skip: # Removing frames if skip step was provided
        swarm_collection = swarm_collection[::skip+1]
            
    # Seting up the figure for animation
    plt.style.use('seaborn-darkgrid') # Use a seaborn dark grid style
    fig, ax = plt.subplots(figsize=(10, 10), dpi=250) # Size and dpi of the figure
    if function_name: # Set the title of the figure according to the provided function name
        ax.set_title(f'Particle swarm optimization for the {function_name} function', fontweight='bold')
    else:
        ax.set_title('Particle swarm optimization', fontweight='bold')
    ax.set_xlim(-dimension_limit[0], dimension_limit[0]) # Limiting the displayed x-axis
    ax.set_ylim(-dimension_limit[1], dimension_limit[1]) # Limiting the displayed y-axis
    plt.xticks(fontsize=5) # Adjusting the x axis tick label size
    plt.yticks(fontsize=5) # Adjusting the y axis tick label size
    point_size = 5 # The size of individual particle
    color_idx = np.linspace(0, 1, len(swarm_collection)) # Getting evenly spaced numbers to sequencial color the animation
    if show_origin: # Displaying the x and y axis from the origin
        ax.axhline(0)
        ax.axvline(0)
        
    pause_time = 1/(10**speed) # Calculating the delay (in milliseconds) between each frame
        
    # Iterating through the swarm collection and ploting a frame for each step
    for idx, particles in enumerate(swarm_collection):
        if idx == 0: # Displaying the initial swarm and pausing the animationm for a few seconds
            particles_xs = [x for x, y in particles] # Extracting the x coordinates for all the particles in the initial swarm
            particles_ys = [y for x, y in particles] # Extracting the y coordinates for all the particles in the initial swarm
            # Ploting the initial swarm
            ax.scatter(particles_xs, particles_ys,
                    color=plt.get_cmap("rainbow")(color_idx[idx]),
                    s=point_size, alpha=0.5)
            plt.pause(3) # Pausing the figure before the animation
            continue

        if not show_all_positions: # Clear the axis if 'show_all_positions' is False and resetting all the figure parameters
            ax.clear()
            if function_name:
                ax.set_title(f'Particle swarm optimization for the {function_name} function', fontweight='bold')
            else:
             ax.set_title('Particle swarm optimization', fontweight='bold')
            ax.set_xlim(-dimension_limit[0], dimension_limit[0])
            ax.set_ylim(-dimension_limit[1], dimension_limit[1])
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            if show_origin:
                ax.axhline(0)
                ax.axvline(0)
        
        particles_xs = [x for x, y in particles] # Extracting the x coordinates for all the particles in the current swarm
        particles_ys = [y for x, y in particles] # Extracting the x coordinates for all the particles in the current swarm
        
        # Ploting the initial swarm
        ax.scatter(particles_xs, particles_ys,
                color=plt.get_cmap("rainbow")(color_idx[idx]), s=point_size, alpha=0.5)
        plt.pause(pause_time)  # Pausing before the next figure generation to simulate animation in a GUI window

    # Plotint the global best solution as a black particle at the end of the animation
    ax.scatter(best_x, best_y,
            color='black', s=point_size+5, label=f'Global best solution\n({best_x:+.3f}, {best_y:+.3f})')

    # Setting the legend
    plt.legend(frameon=True, facecolor='white', shadow=True, prop={'size': 8, 'weight': 'bold'})
    
    # Displaying the figure
    plt.show()

if __name__ == '__main__':
    plot_pso()
