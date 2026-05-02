# GTC_PypeIt

Repo for Lara's pipeline for reducing GTC data with PypeIt. 

So far, only for OSIRIS.


# NOTES (LG)

create rawdir01 and copy all files
create ob1 and cd ob1
pypeit_setup -s gtc_osiris_new -r ../rawdir01 -b -c all
edit .pypeit files
add:
[reduce]
    [[findobj]]
        maxnumber_sci = 1
copy bias lines from B to A (or A to B) and change calib column
run_pypeit gtc_osiris_plus_A/gtc_osiris_plus_A.pypeit (then B)
if bad, add manual column and the right column number.... if ok keep going
...
if there is hostgalaxy contamination add:
    [[extraction]]
         use_2dmodel_mask = False   
...
