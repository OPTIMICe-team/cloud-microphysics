set -e
#compile
cd /home/mkarrer/Dokumente/McSnow_checks
make release

#check
cd /home/mkarrer/Dokumente/McSnow_checks/run
bash check

#plot the collision efficiency and integrate numerically and compare it with the parameterization in the SB scheme
cd /home/mkarrer/Dokumente/cloud-microphysics/
python bulk_colleffi_riming.py -col 1 -m 0 -fr 0
python bulk_colleffi_riming.py -col 1 -m 1 -fr 0
python bulk_colleffi_riming.py -col 1 -m 0 -fr 1
python bulk_colleffi_riming.py -col 1 -m 1 -fr 1
