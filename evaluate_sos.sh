for port in 0 1 2; do
  for gpu in 0 1 2 3 4 5 6; do
    iter=1
    tmux new-session -d -s "eval_sos_gpu${gpu}_port${port}" \
      "python -m scripts_sos.sos --iter ${iter} --gpu ${gpu} --port ${port} --mode eval_dev; bash"
  done
done

