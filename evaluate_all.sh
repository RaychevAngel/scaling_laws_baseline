#evaluate_all.sh

be_even='[(1,[4]),(2,[6,8,10]),(3,[9,12,15]),(5,[10,15,20]),(8,[16,24,32]),(13,[26,39,52])]'
be_odd='[(21,[42,63,84])]'

for port in 0 1; do
  for gpu in {1..7}; do
    iter=$gpu

    if [ $port -eq 0 ]; then
      be=$be_even
    else
      be=$be_odd
    fi

    tmux new-session -d -s "eval_iter${iter}_port${port}" \
      "python -m scripts.evaluate --iter ${iter} --gpu ${gpu} --port ${port} --be '${be}'"
  done
done