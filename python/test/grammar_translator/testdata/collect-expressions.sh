#!/usr/bin/env bash
ROOT_DIR=/home/amd/work/dominic/q-e-gpu
for subfolder in FFTXlib LAXlib; do
  grep -r -h "\s*=\s*" $(find $ROOT_DIR/$subfolder -name *.f90) | grep -v "!" | grep -e "+" -e "*" -e "/" -e "-" |\
     grep -v -i -e WRITE -e PRINT -e FORMAT -e OPEN -e READ -e IF -e THEN -e DO -e CALL -e ALLOCATE | tr -d ' ' | grep -v -e "^[0-9]\+" |\
     grep -v -e ">\|<" -e "::" |\
     grep -v -e "&\s*$" | grep "^\w\+" | sort -u  > assignment-$subfolder.txt
done


#function attributesAndDeclarations() {
#  subfolder=$1
#  grep -r -h "\s*::\s*" $(find $ROOT_DIR/$subfolder -name *.f90) | grep -v "!" |\
#     tr -d ' ' | grep -v -e "^[0-9]\+" -e "IMPORT" -e "PUBLIC" |\
#     grep -v -e "&\s*$" | grep "^\w\+" |  sort -u 
#}
#
#for subfolder in PW; do
#  attributesAndDeclarations "$subfolder" | grep -v "attributes" -i > declarations-$subfolder.txt
#  attributesAndDeclarations "$subfolder" | grep "attributes" -i > attributes-$subfolder.txt
#done
