import numpy as np
import os
import json
import emcee
from kg_utilities import ReadJson


def save_best_model(best_guess_filename,model_run_dir,backend):
    """Saves the best model parameters found during the MCMC run."""

    # get samples and log probabilities from backend
    samples = backend.get_chain(flat=True)
    log_prob = backend.get_log_prob(flat=True)

    # find best sample in current samples
    best_idx = np.argmax(log_prob)
    best_logp = log_prob[best_idx]
    best_params = samples[best_idx].tolist()  # convert to list for JSON

    # load an existing best guess, if it exists
    if os.path.exists(best_guess_filename):
        with open(best_guess_filename, "r") as f:
            saved = json.load(f)
        saved_logp = saved["log_prob"]
        saved_params = saved["params"]
    else:
        saved_logp = -np.inf
        saved_params = None

    # save the very best parameters from this run in the run output directory
    with open(model_run_dir + '/best_fit.json', "w") as f:
        json.dump({"log_prob": best_logp, "params": best_params}, f, indent=2)

    # compare and update if better in the all-time best guess file
    if best_logp > saved_logp:
        with open(best_guess_filename, "w") as f:
            json.dump({"log_prob": best_logp, "params": best_params}, f, indent=2)
        print("New best parameters saved.")
    else:
        print("Existing best parameters are better. No update made.")

    # save the rng metadata for the best run from each rank
    for rng_metadata_file in os.listdir(model_run_dir+"/rank_metadata"):
        with open(model_run_dir+"/rank_metadata/"+rng_metadata_file,'r') as f:
            rng_metadata = json.load(f)
        logP = rng_metadata["logProb"]

        # this specifically saves the very best run's metadata into the run output directory
        if logP == best_logp:
            with open(model_run_dir+"/rng_metadata.json",'w') as f:
                json.dump(rng_metadata, f)
            print("yay! found the metadata for the best run!")
            break


def main():
    plotprops = ReadJson("../runs/plotprops.txt").outProps()
    model_id = plotprops['model_id']
    model_run_dir = "../results/param_runs/" + 'model_' + str(model_id) + '/' + plotprops['model_run_folder']
    backend_folder = model_run_dir
    os.makedirs(backend_folder, exist_ok=True)
    backend_filename = backend_folder + f'/model_{model_id}.h5'
    print(backend_filename)
    best_guess_filename = '../runs/' + f'best_model_{model_id}.json'
    backend = emcee.backends.HDFBackend(backend_filename)    
    save_best_model(best_guess_filename,model_run_dir,backend)

if __name__ == "__main__":
    main()