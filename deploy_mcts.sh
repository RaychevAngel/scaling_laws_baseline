#deploy_all.sh
sleep_seconds=30

for port in 0 1; do
  for gpu in 0 1 2 3 4 5 6 7; do
    policy_iter=7
    # policy deploy
    tmux new-session -d -s "policy_gpu${gpu}_port${port}" \
      "python -m scripts_mcts.deploy_policy --iter ${policy_iter} --gpu ${gpu} --port ${port}"
  done
  sleep ${sleep_seconds}
  for gpu in 0 1 2 3 4 5 6 7; do
    value_iter=7
    # value deploy
    tmux new-session -d -s "value_gpu${gpu}_port${port}" \
      "python -m scripts_mcts.deploy_value  --iter ${value_iter} --gpu ${gpu} --port ${port}"
  done
  sleep ${sleep_seconds}
done