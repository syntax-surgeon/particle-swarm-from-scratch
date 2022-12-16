# particle-swarm-from-scratch

> Designing a simple and intuitive particle swarm optimization algorithm from scratch, without the use of external ML libraries.<br>_(current version: 1.0)_
>
> #### Author: _Siddharth Yadav (syntax-surgeon)_

### main.py

- Contains version-1.0 of the main program which tries to implement a **particle swarm optimization algorithm** to optimize several classic functions
- The available funtions include: **Booth, Bukin No.6, Himmelblau and Goldstein-Price**
- The core of the optimization algorithm depends on **scaled vector addition** influenced by personal and global best solutions
- The code follows a **strong object-oriented style** in which both the particle and the swarm are implemented created via their respective classes
- Advanced plotting is included to visualize the results of the optimization
- An option to **save the results** of a particular run is provided (_see 'dump_swarm' parameter of 'Swarm' class objects_)

### plot.py

- Contains the **plotting functionality** for the results obtained from run the PSO algorithm from main.py
- Two styles of graphs are available: a **collated graph** which includes all the positions sampled by the swarm particles (_figure 1_) and an **animation** which shows the movement of particles through the run (_figure 2_)
- Generally **imported as a module** from main.py to utilize the 'plot_pso' function
- Several options are available to customize the animation such as speed, skip, dimension limits etc (_see the documentation of the 'plot_pso' function_)

> #### Collated graph showing all particle positions (_figure 1_)
>
> ![alt text](https://github.com/syntax-surgeon/particle-swarm-from-scratch/blob/main/readme_assets/bukin_4.png?raw=true)
>
> #### Particle movement animation (_figure 2_)
>
> ![alt text](https://github.com/syntax-surgeon/particle-swarm-from-scratch/blob/main/readme_assets/Bukin_run.gif?raw=true)
