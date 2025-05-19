#!/bin/bash

for i in 05 09 13 18 21
do
  echo $i
  python3 ../plot_details.py $i ms sf10/$i ./ hj_sf10/ 'HJ' ssmj_sf10/ 'SSMJ' ssmj_sf10_nb/ 'SSMJ w/o BF'
done

for i in 05 09 13 18 21
do
  echo $i
  python3 ../plot_details.py $i ms sf50/$i ./ hj_sf50/ 'HJ' ssmj_sf50/ 'SSMJ' ssmj_sf50_nb/ 'SSMJ w/o BF'
done

for i in '05' '05b' '09' '13' '18' '21'
do
  echo $i
  python3 ../plot_details.py $i s sf100/$i ./ hj_sf100/ 'HJ' ssmj_sf100/ 'SSMJ' ssmj_sf100_nb/ 'SSMJ w/o BF'
done

