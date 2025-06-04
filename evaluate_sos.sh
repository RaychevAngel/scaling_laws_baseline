for port in 3; do
  for gpu in 4; do
    iter=2
    tmux new-session -d -s "eval_sos_gpu${gpu}_port${port}" \
      "python -m scripts_sos.sos --iter ${iter} --gpu ${gpu} --port ${port} --mode eval_dev; bash"
  done
done

