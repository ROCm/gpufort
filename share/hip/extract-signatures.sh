#!/usr/bin/env bash
#$1 - file
#$2 - lines

file=$1
lines=$2

if [ -f $file ]; then
  result=$(grep -i "subroutine\s*launch_krnl_.\+_auto" $file)
  result=$(grep "subroutine\s*launch_krnl_.\+_auto" $file -A${lines} | grep -v "\<use \|implicit none\|::\|end subroutine")
  if [ ! -z "$result" ]; then
    result=$(printf "$result" \
      | sed "s,stream,c_null_ptr,g" \
      | sed "s/)\s*bind(c,.\+)$/)/g" \
      | sed "s/\(\w\+\)_n\([0-9]\)/size(\1,\2)/g" \
      | sed "s/\(\w\+\)_lb\([0-9]\)/lbound(\1,\2)/g" \
      | sed "s,sharedMem,0,g" \
      | sed "s,subroutine,CALL,g")

    result=$(printf "$result" | tr '\n' '#' | sed "s/\(\w\+\)\(,&#\s*size(\)/\1_d\2/g" | tr '#' '\n')

    result=$(printf "$result\n" | tr '\n' '#' | sed "s,&#\s*, ,g" | tr '#' '\n' | grep "auto" )
    printf "$result\n"
  fi
fi
