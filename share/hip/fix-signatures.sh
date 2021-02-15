#!/usr/bin/env bash
#$1 target
#$2 kernel module

target=$1
module=$2
apply=${3:--c}

if [ -f "$target" ]; then
  if [ -f "$module" ]; then
    new_sigs=$(./extract-signatures.sh $module 70 | sed "s,^\s\+,,g")
    anchors=$(printf "$new_sigs" | grep -o "CALL\s*launch_krnl_\w\+_auto")
    SAVEIFS=$IFS   # Save current IFS
    IFS=$'\n'      # Change IFS to new line
    new_sigs=($new_sigs)
    anchors=($anchors)

    c=0
    for a in ${anchors[@]}; do
      pattern=${anchors[c]}
      subst=${new_sigs[c]}
      grep -Hn "$pattern" $target | multisub -p "${pattern}.+$" "${subst}" ${apply}
      let c=c+1
    done
    
    IFS=$SAVEIFS   # Restore IFS
  fi
fi
