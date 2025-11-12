#!/bin/bash

# prepare original sets for 0...4 and 5...9
python ./generate_data_orig.py

# join by copying
cd ./generated_data/
mkdir origin
cd origin
cp ../origin1/first/*.* ./
cp ../origin1/second/*.* ./
cd ../../

# compute FIDS
mkdir fids

for i in 0 1 2 3 4 5 6 7 8 9;
do

echo Doing $i-th model

# the model learned on 0...4
python3 generate_data.py --id mt0_$i --nz0 30 --nz1 100;

# to all
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin/ ./generated_data/generated/ > ./fids/out0_"$i"_a.txt 2> fids/err0_"$i"_a.txt;

# to self
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin1/first/ ./generated_data/generated/ > ./fids/out0_"$i"_s.txt 2> fids/err0_"$i"_s.txt;

# to other
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin1/second/ ./generated_data/generated/ > ./fids/out0_"$i"_o.txt 2> fids/err0_"$i"_o.txt;

# the model learned on 5...9
python3 generate_data.py --id mt1_$i --nz0 30 --nz1 100;

# to all
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin/ ./generated_data/generated/ > ./fids/out1_"$i"_a.txt 2> fids/err1_"$i"_a.txt;

# to self
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin1/second/ ./generated_data/generated/ > ./fids/out1_"$i"_s.txt 2> fids/err1_"$i"_s.txt;

# to other
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin1/first/ ./generated_data/generated/ > ./fids/out1_"$i"_o.txt 2> fids/err1_"$i"_o.txt;

done

# the baseline model
python3 generate_data.py --id mto_baseline --nz0 30 --nz1 100;
python3 -m pytorch_fid --device cuda:2 ./generated_data/origin/ ./generated_data/generated/ > ./fids/out_baseline.txt 2> fids/err_baseline.txt;
