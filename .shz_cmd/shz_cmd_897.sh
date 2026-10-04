cd /ssd/Strata && grep -oE "n_experts?[ =:]+[0-9]+|[0-9]+ experts per layer|experts: [0-9]+" strata-iq3_s.log | sort -u | head -8; echo "--- arena/expert math ---"; python3 -c "
arena=46.84*2**30; blob=14.75*2**30/7777
print('total experts in the pack:', round(arena/blob))
print('per layer (48):', round(arena/blob/48))
print('resident per layer:', round(7777/48), '=', round(100*(7777/48)/(arena/blob/48),1), '% of the layer')
"
