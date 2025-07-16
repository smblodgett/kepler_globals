# radius_grid_array = [0.0,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0,1000000000.0] # earth radii
radius_grid_array = [0.1,0.5,1.0,2.0,5.0,10.0,20] # earth radii

# period_grid_array = [0.0,0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,500.0,1000000000.0] # days
period_grid_array = [0.2,1.0,2.0,5.0,20.0,50.0,100.0,200.0,500.0]#,8.0,1000] dipping below 0.2 causes things to break so don't do that 

# mass_grid_array = [0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,128.0,256.0,1000000000.0] # earth mass #### Change these bins, we want to focus on the 1-8 size especially...maybe 256 + is the last bin?
mass_grid_array = [0.1,10.0]#,8.0,100,10000] # earth mass #### Change these bins, we want to focus on the 1-8 size especially...maybe 256 + is the last bin?


# eccentricity_grid_array = [0.0,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0] # unitless
eccentricity_grid_array = [0.0,0.99] # unitless ecc of 1 causes things to break

omega_grid_array = [0.0,360.0] # degrees