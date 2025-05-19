#!/bin/bash

for i in 16a # 16b 16c 16d 17e 6f 8c
do
  echo $i
  python3 ../plot_details.py $i ms $i ./ hj/ 'HJ' ssmj/ 'SSMJ' ssmj_no_bf/ 'SSMJ w/o BF'
done