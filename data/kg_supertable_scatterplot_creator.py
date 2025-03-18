import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


graphing_columns = ["Period_days","T_0","MES_rowe","chi2W_rowe","chi2WO_rowe","Brho*_rowe","BTeff_rowe","BR*_rowe","BM*_rowe", "Blog(g)*_rowe", "BZ*_rowe",
    "sqrt(e) cos(omega)","sqrt(e) sin(omega)","i",
"M_pJ","R_p/R_s","M_s","R_s","c_1","c_2","chisq","chisq_rank","R_pE","M_pE","rho_p","rho_s",
"M_p/M_s","e","omega","true_anomaly","eccentric_anomaly","mean_anomaly","mean_longitude",                     
"a_AU","a_R_s","peri_AU","peri_R_s","apo_AU","apo_R_s","d_AU","d_R_s","b_trans","b_occ",
 "p_trans","p_occ", "T_total_hr","T_full_hr","K_RV","occurrence_rate_hsu","multiplicity",                  
"P/Pin","P/Pout","Tdur/Tdurin","Tdur/Tdurout","R/Rin","R/Rout","M/Min","M/Mout","rho/rhoin",
"rho/rhoout","i-iin","iout-i","xiin","xiout","distin_hillrad","distout_hillrad","distin_hillrad_e",
"distout_hillrad_e","e/ein","eout/e","omega-omegain","omegaout-omega","Period_days_rowe",                    
"T0_rowe","Rp/R*_rowe","b_rowe","rho*M_rowe","u1_rowe","u2_rowe","TDepth_rowe","TDur_rowe",
"ATDur_rowe","S/N_rowe","a/R*_rowe","Inc_rowe",
"Rp_rowe", "S0_rowe", "Kmag_rowe", "rho*_rowe", "Teff_rowe", "R*_rowe","M*_rowe", "log(g)*_rowe", 
"Z*_rowe", "BRp_rowe", "BS0_rowe","planet",]





def make_scatterplots():
    
    df = pd.read_csv('thinned/all_thin.csv', index_col=0)

    completed_comparisons = set()

    for x_column_name in graphing_columns:
        for y_column_name in graphing_columns:
            try:
                if x_column_name == y_column_name:
                    continue

                if frozenset([x_column_name,y_column_name]) in completed_comparisons:
                    continue

                if "rowe" in x_column_name and "rowe" in y_column_name:
                    continue



                is_rowe_column = "rowe" in x_column_name or "rowe" in y_column_name

                print("is rowe column here?", is_rowe_column)

                completed_comparisons.add(frozenset([x_column_name,y_column_name]))

                print(x_column_name+" vs "+y_column_name)

                x_column = df[x_column_name]
                y_column = df[y_column_name]

                if x_column_name == "sqrt(e) cos(omega)" or x_column_name == "sqrt(e) sin(omega)":
                    x_column += 1 
                    x_column_name += " + 1"

                if y_column_name == "sqrt(e) cos(omega)" or y_column_name == "sqrt(e) sin(omega)":
                    y_column += 1 
                    y_column_name += " + 1"

                if x_column_name == "Z*_rowe" or x_column_name == "BZ*_rowe":
                    x_column = 10 ** x_column 
                    x_column_name = "10^"+x_column_name

                if y_column_name == "Z*_rowe" or y_column_name == "BZ*_rowe":
                    y_column = 10 ** y_column 
                    y_column_name = "10^"+y_column_name             

                # Create a 2x2 subplot
                fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

                unweighted_alpha = 0.05
                weighted_alpha = df['occurrence_rate_hsu'] if not is_rowe_column else df['occurrence_rate_hsu']/5
                marker_size = 2.5

                # Linear scale plots
                axes[0, 0].scatter(x_column,y_column,s=marker_size,alpha=unweighted_alpha)
                axes[0, 0].set_title(x_column_name + ' vs ' + y_column_name)
                axes[0, 0].set_xscale('linear')
                axes[0, 0].set_yscale('linear')

                axes[0, 1].scatter(x_column,y_column, color='lightseagreen',alpha=weighted_alpha,s=marker_size)
                axes[0, 1].set_title(x_column_name + ' vs ' + y_column_name + " occurrence weighted")
                axes[0, 1].set_xscale('linear')
                axes[0, 1].set_yscale('linear')

                # Log-log scale plots
                axes[1, 0].scatter(x_column,y_column, color='indigo',s=marker_size,alpha=unweighted_alpha)  # Using index as x-values
                axes[1, 0].set_title(x_column_name + ' vs ' + y_column_name + " loglog")
                axes[1, 0].set_xscale('log')
                axes[1, 0].set_yscale('log')

                axes[1, 1].scatter(x_column,y_column, color='darkolivegreen',alpha=weighted_alpha,s=marker_size)  # 
                axes[1, 1].set_title(x_column_name + ' vs ' + y_column_name + " loglog" + " occurrence weighted")
                axes[1, 1].set_xscale('log')
                axes[1, 1].set_yscale('log')

                plt.tight_layout(rect=[0, 0, 1, 0.92])
                fig.suptitle(f"Scatter for {x_column_name} vs {y_column_name}", fontsize=20, fontweight='bold')


                fig.text(0.5, 0.01, x_column_name, ha='center', va='center', fontsize=14) 
                fig.text(0.01, 0.5, y_column_name, ha='center', va='center', rotation='vertical', fontsize=14) 

                safe_x_column_name = x_column_name.replace("/", "_")
                safe_y_column_name = y_column_name.replace("/", "_")

                plt.savefig('plots/scatters/'+safe_x_column_name+'_vs_'+safe_y_column_name+'.png',dpi=120)
            
            except Exception as e:
                print(e)
                with open("scatterplot_creator_error_log.txt", "a") as file:
                    file.write(x_column_name+","+y_column_name+"\n")
            
            
if __name__ == "__main__":
    make_scatterplots()