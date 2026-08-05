### NEED TO VERIFY WITH PHODYMM THAT ALL THE CONSTANTS ARE THE RIGHT VALUES -- CURRENTLY 28.02 is max density


### PhoDyMM Values:::
#define G 2.9591220363e-4         ///*  Newton's constant, AU^3*days^-2 */
#define RSUNAU  0.0046491            ///* solar radius in au */
#define REARTHORSUN 0.009171    ///* earth radius divided by solar radius */
#define MPSTOAUPD 5.77548327363993e-7    ///* meters per second to au per day conversion factor */
#define MSOMJ 1.04737701464237e3  ///* solar mass in terms of jupiter masses */ 
#define CAUPD 173.144632674240  ///* speed of light in AU per day */




METOMS=3.0024584*10**-6 # earth mass to solar mass
METOMJ=0.003146336   # earth mass to jupiter mass 
MEG=5.9736*10**27 # earth mass in grams
MEKG=MEG/1000 # earth mass in kilograms
MSKG=1.9891*10**30 # solar mass in kilograms
MSTOMJ=1.04737701464237e3 # solar mass in jupiter masses

RECM=6.371*10**8  # earth radius in cm    ### SHOULD BE: volumetric radius (6.371*10**8)
RETORS=0.009171 # earth radius to solar radius
RJTORE=10.973 # jupiter radius to earth radius
RJTORS=0.1004901538 # jupiter radius to solar radius
RSAU=0.0046491  # solar radius in AU
RJAU=4.676*10**-4 # jupiter radius in AU
RSCM=6.96*10**10 # solar radius in cm  

RHOS = 1408 # solar average density in kg/m^3

DTOS=24*3600 # days to seconds
MTOAU=6.68458712*10**-12 # meters to AU

G=6.6743*10**-11 # gravitational constant in sci units
GAU=2.9591220363e-4        # Newton's constant, AU^3*days^-2

N_HSU_STARS = 80006 
N_PHODYMM_SYSTEMS = 661
N_PHODYMM_PLANETS = 1665
N_PHODYMM_STARS = 198709 # what is the correct number on this?