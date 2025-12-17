radius_grid_array = [0.1,0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,2.0,2.5,3.0,4.0,8.0,12.0,16.0,20] # len=18 earth radii !!!!
# radius_grid_array = [0.1,0.5,1.0,1.5,2.0,3.0,5.0,10.0,15.0,20] # earth radii
# radius_grid_array = [0.1,16,20] ### TESTING


# period_grid_array = [0.2,16.0,500.0] #### TESTING
period_grid_array = [0.2,0.5,0.75,1.0,1.5,2.0,4.0,8.0,12.0,16.0,32.0,64.0,128.0,256.0,500.0] # days len=15
# period_grid_array = [0.2,1.0,2.0,5.0,20.0,50.0,100.0,200.0,500.0]#,8.0,1000] dipping below 0.2 causes things to break so don't do that 

# mass_grid_array = [0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,128.0,256.0,1000,10000] # earth mass 
mass_grid_array = [0.1,0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,2.0,3.0,4.0,6.0,8.0,12.0,16.0,24.0,32.0,64.0,128.0,256.0,512.0,10000] # earth mass len=24
# mass_grid_array = [0.1,128,10000] #### TESTING

eccentricity_grid_array = [0.0,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,0.7,0.99] # unitless  len=12 more ecc in lower esp.
# eccentricity_grid_array = [0.0,0.1,0.5,0.99] # TESTING

omega_grid_array = [0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0,360.0] # at least 30 degrees   len=9