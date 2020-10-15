set -e
#compile
cd /home/mkarrer/Dokumente/McSnow_checks
#make release

#check
cd /home/mkarrer/Dokumente/McSnow_checks/run
#bash check

#plot the collision efficiency and integrate numerically and compare it with the parameterization in the SB scheme
cd /home/mkarrer/Dokumente/cloud-microphysics/
python bulk_colleffi_riming.py
