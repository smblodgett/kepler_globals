import os
import sys
import commentjson as json
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from kg_grid_boundary_arrays import radius_grid_array, period_grid_array, mass_grid_array
from kg_griddefiner import *


# object to read in plotprops file
class ReadJson:
    def __init__(self, filename):
        print('read the plotprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data


def heatmap_plot(rpm_grid,results_folder, make_gifs=True, verbose=False, fps=0.5):

    heatmap_df = get_heatmap_df(rpm_grid)
    make_mass_histograms(rpm_grid,results_folder,heatmap_df,make_gifs=make_gifs,verbose=verbose,fps=fps)
    make_radius_histograms(rpm_grid,results_folder,heatmap_df,make_gifs=make_gifs,verbose=verbose,fps=fps)
    make_period_histograms(rpm_grid,results_folder,heatmap_df,make_gifs=make_gifs,verbose=verbose,fps=fps)

def get_heatmap_df(rpm_grid):
    heatmap_df = pd.DataFrame()

    for i in rpm_grid.voxel_array:
        for j in i:
            for k in j:
                df_selected = k.df[['R_pE', 'Period_days', 'M_pE', 'mass_divided_weights']]
                heatmap_df = pd.concat([heatmap_df, df_selected], ignore_index=True)

    return heatmap_df

def make_mass_histograms(rpm_grid,results_folder,histogram_df,make_gifs=True, verbose=False, fps=0.5):

    heatmap_folder = os.path.join(results_folder, "plots", "heatmaps","mass")
    os.makedirs(heatmap_folder, exist_ok=True)

    vmax = 0
    for i in range(len(rpm_grid.mass_grid_array) - 1):
        lower_mass = rpm_grid.mass_grid_array[i]
        upper_mass = rpm_grid.mass_grid_array[i + 1]


        mass_filtered_df = histogram_df[(histogram_df["M_pE"] > lower_mass) & (histogram_df["M_pE"] <= upper_mass)]
        period_data = mass_filtered_df["Period_days"].values
        radius_data = mass_filtered_df["R_pE"].values
        weights = mass_filtered_df["mass_divided_weights"].values # I think I need to split this the other way too...

        hist, xedges, yedges = np.histogram2d(radius_data, period_data, 
                                              bins=[rpm_grid.radius_grid_array, rpm_grid.period_grid_array], 
                                              weights=weights)  
        
        hist /= 1000
        
        vmax = max(vmax, np.nanmax(hist))

    for i in range(len(rpm_grid.mass_grid_array) - 1):
        lower_mass = rpm_grid.mass_grid_array[i]
        upper_mass = rpm_grid.mass_grid_array[i + 1]

        mass_filtered_df = histogram_df[(histogram_df["M_pE"] > lower_mass) & (histogram_df["M_pE"] <= upper_mass)]
        period_data = mass_filtered_df["Period_days"].values
        radius_data = mass_filtered_df["R_pE"].values
        weights = mass_filtered_df["mass_divided_weights"].values 

        R_30_prior_upper = (((upper_mass)*MEG)/((4/3)*np.pi*30))**(1/3) / RECM
        R_30_prior_lower = (((lower_mass)*MEG)/((4/3)*np.pi*30))**(1/3) / RECM          


        R_001_prior_upper = (((upper_mass)*MEG)/((4/3)*np.pi*0.01))**(1/3) / RECM
        R_001_prior_lower = (((lower_mass)*MEG)/((4/3)*np.pi*0.01))**(1/3) / RECM

        hist, xedges, yedges = np.histogram2d(radius_data, period_data, 
                                              bins=[rpm_grid.radius_grid_array, rpm_grid.period_grid_array], 
                                              weights=weights)

        hist /= 1000
        
        plt.figure(figsize=(8, 6), dpi=200)

        ax = sns.heatmap(hist, annot=True, fmt=".4f", cbar=False, 
                         cmap=plt.cm.Spectral, vmin=0, vmax=vmax, annot_kws={"size": 5})
        # put a line to denote where it's hard to detect planets?

        for prior_line in [R_30_prior_upper,R_30_prior_lower,R_001_prior_upper,R_001_prior_lower]:
            heatmap_prior_lines(ax,rpm_grid.period_grid_array,rpm_grid.radius_grid_array,prior_line)

        ax.set_xticks(np.arange(len(yedges)))
        ax.set_xticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in yedges], 
                            fontsize=5)

        ax.set_yticks(np.arange(len(xedges)))
        ax.set_yticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in xedges],rotation=90, fontsize=5)

        ax.invert_yaxis()

        plt.xlabel('Period [days]')
        plt.ylabel('Radius [$R_{Earth}$]')

        plt.suptitle("Occurrence-Weighted Fraction of PhoDyMM Kepler Systems With Given Radii vs Periods", fontsize=22)
        plt.title(f"M={lower_mass}-{upper_mass}")

        plt.savefig(os.path.join(heatmap_folder, f"M{lower_mass}-{upper_mass}_heatmap.png"), dpi=200)
        if verbose: 
            print(f"made M{lower_mass}-{upper_mass}_heatmap.png")
        plt.close()

    if make_gifs:
        os.makedirs(os.path.join(results_folder, "plots", "animations"), exist_ok=True)
        output_gif_path = os.path.join(results_folder, "plots", "animations", "mass_heatmap_animation.gif")
        make_gif_from_pngs(heatmap_folder, output_gif_path, fps=fps)


def make_radius_histograms(rpm_grid,results_folder, histogram_df, make_gifs=True, verbose=False, fps=0.5):

        heatmap_folder = os.path.join(results_folder, "plots", "heatmaps","radius")
        os.makedirs(heatmap_folder, exist_ok=True)

        vmax = 0
        for i in range(len(rpm_grid.radius_grid_array) - 1):
            lower_radius = rpm_grid.radius_grid_array[i]
            upper_radius = rpm_grid.radius_grid_array[i + 1]



            radius_filtered_df = histogram_df[(histogram_df["R_pE"] > lower_radius) & (histogram_df["R_pE"] <= upper_radius)]
            period_data = radius_filtered_df["Period_days"].values
            mass_data = radius_filtered_df["M_pE"].values
            weights = radius_filtered_df["mass_divided_weights"].values

            hist, xedges, yedges = np.histogram2d(mass_data, period_data, 
                                                  bins=[rpm_grid.mass_grid_array, rpm_grid.period_grid_array], 
                                                  weights=weights)  
            hist /= 1000
            
            vmax = max(vmax, np.nanmax(hist))

        for i in range(len(rpm_grid.radius_grid_array) - 1):
            lower_radius = rpm_grid.radius_grid_array[i]
            upper_radius = rpm_grid.radius_grid_array[i + 1]

            radius_filtered_df = histogram_df[(histogram_df["R_pE"] > lower_radius) & (histogram_df["R_pE"] <= upper_radius)]
            period_data = radius_filtered_df["Period_days"].values
            mass_data = radius_filtered_df["M_pE"].values
            weights = radius_filtered_df["mass_divided_weights"].values

            M_30_prior_upper = ((4/3)*np.pi*30/MEG)*(upper_radius * RECM)**3
            M_30_prior_lower = ((4/3)*np.pi*30/MEG)*(lower_radius * RECM)**3         

            M_001_prior_upper = ((4/3)*np.pi*0.01/MEG)*(upper_radius * RECM)**3
            M_001_prior_lower = ((4/3)*np.pi*0.01/MEG)*(lower_radius * RECM)**3

            hist, xedges, yedges = np.histogram2d(mass_data, period_data, 
                                                  bins=[rpm_grid.mass_grid_array, rpm_grid.period_grid_array], 
                                                  weights=weights)

            hist /= 1000
            # put a line denoting the density prior (don't forget the lower bound) 

            plt.figure(figsize=(8, 6), dpi=200)

            ax = sns.heatmap(hist, annot=True, fmt=".4f", cbar=False, 
                             cmap=plt.cm.Spectral, vmin=0, vmax=vmax, annot_kws={"size": 5})


            for prior_line in [M_30_prior_upper,M_30_prior_lower,M_001_prior_upper,M_001_prior_lower]:
                heatmap_prior_lines(ax,rpm_grid.period_grid_array,rpm_grid.mass_grid_array,prior_line)

            ax.set_xticks(np.arange(len(yedges)))
            ax.set_xticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in yedges], 
                                fontsize=5)

            ax.set_yticks(np.arange(len(xedges)))
            ax.set_yticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in xedges],rotation=90, fontsize=5)

            ax.invert_yaxis()

            plt.xlabel('Period [days]')
            plt.ylabel('Mass [$M_{Earth}$]')

            plt.suptitle("Occurrence-Weighted Fraction of PhoDyMM Kepler Systems With Given Mass vs Period", fontsize=22)
            plt.title(f"R={lower_radius}-{upper_radius}")

            plt.savefig(os.path.join(heatmap_folder, f"R{lower_radius}-{upper_radius}_heatmap.png"), dpi=200)
            if verbose: 
                print(f"made R{lower_radius}-{upper_radius}_heatmap.png")
            plt.close()

        if make_gifs:
            os.makedirs(os.path.join(results_folder, "plots", "animations"), exist_ok=True)
            output_gif_path = os.path.join(results_folder, "plots", "animations", "radius_heatmap_animation.gif")
            make_gif_from_pngs(heatmap_folder, output_gif_path, fps=fps)      


def make_period_histograms(rpm_grid,results_folder, histogram_df, make_gifs=True, verbose=False, fps=0.5):

        heatmap_folder = os.path.join(results_folder, "plots", "heatmaps","period")
        os.makedirs(heatmap_folder, exist_ok=True)

        vmax = 0
        for i in range(len(rpm_grid.period_grid_array) - 1):
            lower_period = rpm_grid.period_grid_array[i]
            upper_period = rpm_grid.period_grid_array[i + 1]
            period_filtered_df = histogram_df[(histogram_df["Period_days"] > lower_period) & (histogram_df["Period_days"] <= upper_period)]
            radius_data = period_filtered_df["R_pE"].values
            mass_data = period_filtered_df["M_pE"].values
            weights = period_filtered_df["mass_divided_weights"].values

            hist, xedges, yedges = np.histogram2d(radius_data, mass_data,
                                                  bins=[rpm_grid.radius_grid_array, rpm_grid.mass_grid_array], 
                                                  weights=weights)  
            hist /= 1000
            vmax = max(vmax, np.nanmax(hist))

        for i in range(len(rpm_grid.period_grid_array) - 1):
            lower_period = rpm_grid.period_grid_array[i]
            upper_period = rpm_grid.period_grid_array[i + 1]

            period_filtered_df = histogram_df[(histogram_df["Period_days"] > lower_period) & (histogram_df["Period_days"] <= upper_period)]
            radius_data = period_filtered_df["R_pE"].values
            mass_data = period_filtered_df["M_pE"].values
            weights = period_filtered_df["mass_divided_weights"].values

            hist, xedges, yedges = np.histogram2d(radius_data, mass_data,
                                                  bins=[rpm_grid.radius_grid_array, rpm_grid.mass_grid_array], 
                                                  weights=weights)

            hist /= 1000
            plt.figure(figsize=(8, 6), dpi=200)

            ax = sns.heatmap(hist, annot=True, fmt=".4f", cbar=False, 
                             cmap=plt.cm.Spectral, vmin=0, vmax=vmax,annot_kws={"size": 5})

            ax.set_xticks(np.arange(len(yedges)))
            ax.set_xticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in yedges], 
                                fontsize=5)

            ax.set_yticks(np.arange(len(xedges)))
            ax.set_yticklabels([f"{edge:.1f}" if edge < 1e5 else f"{edge:.1e}" for edge in xedges],rotation=90, fontsize=5)

            ax.invert_yaxis()

            plt.ylabel('Radius [$R_{Earth}$]')
            plt.xlabel('Mass [$M_{Earth}$]')

            plt.suptitle("Occurrence-Weighted Fraction of PhoDyMM Kepler Systems With Given Mass vs Radius", fontsize=22)
            plt.title(f"P={lower_period}-{upper_period}")

            plt.savefig(os.path.join(heatmap_folder, f"P{lower_period}-{upper_period}_heatmap.png"), dpi=200)
            if verbose: 
                print(f"made P{lower_period}-{upper_period}_heatmap.png")
            plt.close()

        if make_gifs:
            os.makedirs(os.path.join(results_folder, "plots", "animations"), exist_ok=True)
            output_gif_path = os.path.join(results_folder, "plots", "animations", "period_heatmap_animation.gif")
            make_gif_from_pngs(heatmap_folder, output_gif_path, fps=fps)   

def heatmap_prior_lines(ax,x_bins,y_bins,prior_line_y):
    if prior_line_y > np.max(y_bins):
        return 

    # Find the bin **edges** that enclose `target_y`
    bin_idx = np.searchsorted(y_bins, prior_line_y) - 1  # Find the lower edge bin index

    # Linear interpolation between bin edges
    y_lower, y_upper = y_bins[bin_idx], y_bins[bin_idx + 1]
    frac = (prior_line_y - y_lower) / (y_upper - y_lower)  # Fraction between bins

    # Map it to heatmap grid space
    mapped_y = bin_idx + frac  # Precise location inside the bin

    # Plot precise horizontal line
    ax.hlines(y=mapped_y, xmin=0, xmax=len(x_bins)-1, colors='k', linestyles='dashed')



def make_gif_from_pngs(input_dir, output_gif_path, fps=1,verbose=True):
    
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    file_numbers_dict = {}
    for file in png_files:
        m = re.search('\d+\.\d+',file)
        value = float(m.group(0))
        file_numbers_dict[value] = file
    
    images = []
    for key in sorted(file_numbers_dict):
        image_path = os.path.join(input_dir, file_numbers_dict[key])
        image = Image.open(image_path)
        images.append(image)
    
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=True, duration=1000/fps, loop=0)

    if verbose: print(f"GIF saved to {output_gif_path}")
        
        
 # some kind of best fit plot        
 # residuals (best fit - data)       
        
        
def main():
    
    cwd = os.getcwd()

        # find plotprops path
    if 'src' in cwd:
        plotprops_filename = "../runs/plotprops.txt"
    elif 'runs' in cwd:
        plotprops_filename = "plotprops.txt"
    elif 'results' in cwd:
        plotprops_filename = "plotprops.txt"
    else:
        print('you are not starting from a proper directory. you should run kg_run.py from a src, runs, or a results directory.')
        sys.exit()
        
    # get plotprops loaded in
    getData = ReadJson(plotprops_filename)
    plotprops = getData.outProps()
    verbose = plotprops.get("verbose")
    plottype = plotprops.get("plottype")
    input_data_filename = plotprops.get("input_data_filename")
    results_folder = plotprops.get("results_folder")
    make_gifs = plotprops.get("make_gifs")
    fps = plotprops.get("fps")
    print(plottype, type(plottype))
    
    plot_all = plottype == "all"
    if plot_all or plottype == "heatmap":
        voxel_grid = RPM_Grid(radius_grid_array,period_grid_array,mass_grid_array)
        df = pd.read_csv(input_data_filename)
        voxel_grid.setup_dataframes(df.columns)
        voxel_grid.add_data(df)
        voxel_grid.make_mass_divided_weights()
        heatmap_plot(voxel_grid,'../results',make_gifs=make_gifs,verbose=verbose,fps=fps)
    
    
if __name__ == "__main__":
    main()

