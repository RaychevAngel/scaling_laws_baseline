#evaluate_all.sh

be_even='[(34,[68,102,136])]'
be_odd='[(55,[110,165,220])]'

for port in 0 1; do
  for gpu in {1..7}; do
    iter=$gpu

    if [ $port -eq 0 ]; then
      be=$be_even
    else
      be=$be_odd
    fi

    tmux new-session -d -s "eval_iter${iter}_port${port}" \
      "python -m scripts.evaluate --iter ${iter} --gpu ${gpu} --port ${port} --be '${be}' ; bash"
  done
done